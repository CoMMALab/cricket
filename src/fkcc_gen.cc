#include <cricket/codegen.hh>
#include <cricket/robot_info.hh>

#include <Eigen/Core>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <inja/inja.hpp>
#include <cxxopts.hpp>

#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    cxxopts::Options options(argv[0], "Tracing compiler for forward kinematics and collision checking");

    options.positional_help("[JSON configuration filename]").show_positional_help();

    options.add_options()                                                                       //
        ("f,configuration_file", "JSON configuration filename", cxxopts::value<std::string>())  //
        ("o,output_filename", "Output JSON filename", cxxopts::value<std::string>())            //
        ("t,output_template",
         "Output template filename (override configuration file)",
         cxxopts::value<std::string>())  //
        ("h,help", "Print usage")        //
        ;

    options.parse_positional({"configuration_file"});

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (not result.count("configuration_file"))
    {
        throw std::runtime_error(fmt::format("Must provide configuration file!"));
    }

    std::filesystem::path json_path(result["configuration_file"].as<std::string>());
    auto parent_path = json_path.parent_path();

    if (not std::filesystem::exists(json_path))
    {
        throw std::runtime_error(fmt::format("JSON file {} does not exist!", json_path.string()));
    }

    std::ifstream json_file(json_path);
    nlohmann::json data;

    try
    {
        data = nlohmann::json::parse(json_file);
    }
    catch (std::exception &e)
    {
        throw std::runtime_error(fmt::format("Failed to parse JSON file! Error: \n{}", e.what()));
    }

    std::optional<std::filesystem::path> srdf_path = {};
    if (data.contains("srdf"))
    {
        srdf_path = parent_path / data["srdf"];
    }

    std::optional<std::string> end_effector_name = {};
    if (data.contains("end_effector"))
    {
        end_effector_name = data["end_effector"];
    }

    std::string language = "c++";
    if (data.contains("language"))
    {
        language = data["language"];
    }

    std::optional<cricket::Bounds> bounds;
    if (data.contains("bounds"))
    {
        const auto &bd = data["bounds"];
        if (not bd.contains("lower") or not bd.contains("upper"))
        {
            throw std::runtime_error("bounds must contain both 'lower' and 'upper' arrays");
        }
        const auto lower = bd["lower"].get<std::vector<double>>();
        const auto upper = bd["upper"].get<std::vector<double>>();
        if (lower.size() < 2 or lower.size() > 3 or upper.size() < 2 or upper.size() > 3)
        {
            throw std::runtime_error("bounds arrays must have 2 or 3 elements");
        }
        cricket::Bounds b;
        b.lower = Eigen::Vector3d(lower[0], lower[1], lower.size() == 3 ? lower[2] : 0.0);
        b.upper = Eigen::Vector3d(upper[0], upper[1], upper.size() == 3 ? upper[2] : 0.0);
        bounds = b;
    }

    cricket::RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_name);

    // Preserve compact_collisions across robot.json() merge; default false.
    const bool compact_collisions = data.value("compact_collisions", false);
    data.update(robot.json(bounds));
    data["compact_collisions"] = compact_collisions;

    auto traced_eefk_code = cricket::trace_sphere_cc_fk(robot, language, false, false, true);
    data["eefk_code"] = traced_eefk_code.code;
    data["eefk_code_vars"] = traced_eefk_code.temp_variables;
    data["eefk_code_output"] = traced_eefk_code.outputs;

    auto traced_spherefk_code = cricket::trace_sphere_cc_fk(robot, language, true, false, false);
    data["spherefk_code"] = traced_spherefk_code.code;
    data["spherefk_code_vars"] = traced_spherefk_code.temp_variables;
    data["spherefk_code_output"] = traced_spherefk_code.outputs;

    auto traced_ccfk_code = cricket::trace_sphere_cc_fk(robot, language, true, true, false);
    data["ccfk_code"] = traced_ccfk_code.code;
    data["ccfk_code_vars"] = traced_ccfk_code.temp_variables;
    data["ccfk_code_output"] = traced_ccfk_code.outputs;

    auto traced_ccfkee_code = cricket::trace_sphere_cc_fk(robot, language, true, true, true);
    data["ccfkee_code"] = traced_ccfkee_code.code;
    data["ccfkee_code_vars"] = traced_ccfkee_code.temp_variables;
    data["ccfkee_code_output"] = traced_ccfkee_code.outputs;

    auto mapconfig = cricket::trace_map_to_configuration(robot.model, language, bounds);
    data["mapconfig_code"] = mapconfig.code;
    data["mapconfig_code_vars"] = mapconfig.temp_variables;
    data["mapconfig_code_output"] = mapconfig.outputs;

    auto interp = cricket::trace_interpolate(robot.model, language);
    data["interpolate_code"] = interp.code;
    data["interpolate_code_vars"] = interp.temp_variables;

    auto interp_block = cricket::trace_interpolate_block(robot.model, language);
    data["interpolate_block_code"] = interp_block.code;
    data["interpolate_block_code_vars"] = interp_block.temp_variables;

    auto dist = cricket::trace_distance(robot.model, language);
    data["distance_code"] = dist.code;
    data["distance_code_vars"] = dist.temp_variables;

    if (data.contains("flask"))
    {
        const auto &fl = data["flask"];
        if (not fl.contains("rho"))
        {
            throw std::runtime_error("flask configuration must specify 'rho' (LQMT time-effort weight)");
        }

        const double rho = fl["rho"].get<double>();
        if (not (rho > 0.))
        {
            throw std::runtime_error("flask 'rho' must be positive");
        }
        data["rho"] = rho;

        // Python/C++ module identifier; must match the registered binding module name
        std::string module_name = data["name"].get<std::string>();
        for (auto &c : module_name)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        data["module_name"] = fl.value("module_name", module_name);

        // URDF limits emitted by RobotInfo, overridable from the flask block
        if (fl.contains("velocity_limits"))
        {
            data["velocity_limits"] = fl["velocity_limits"];
        }
        if (fl.contains("effort_limits"))
        {
            data["effort_limits"] = fl["effort_limits"];
        }

        const auto nq = static_cast<std::size_t>(robot.model.nq);
        const auto velocity_limits = data["velocity_limits"].get<std::vector<double>>();
        const auto effort_limits = data["effort_limits"].get<std::vector<double>>();
        if (velocity_limits.size() != nq or effort_limits.size() != nq)
        {
            throw std::runtime_error(
                fmt::format("flask velocity/effort limits must have {} entries", nq));
        }

        for (std::size_t i = 0; i < nq; ++i)
        {
            if (not std::isfinite(velocity_limits[i]) or velocity_limits[i] <= 0. or
                not std::isfinite(effort_limits[i]) or effort_limits[i] <= 0.)
            {
                throw std::runtime_error(
                    fmt::format(
                        "flask limits must be finite and positive (joint {}: velocity {}, effort "
                        "{}); override via the 'flask' block if the URDF lacks them",
                        i,
                        velocity_limits[i],
                        effort_limits[i]));
            }
        }

        data["n_z"] = 2 * nq;
        data["n_x"] = 3 * nq;

        // Flat-state box: z = (q, qdot) in [q_lower, q_upper] x [-v_max, v_max]
        const auto q_lower = data["lower"].get<std::vector<double>>();
        const auto q_upper = data["upper"].get<std::vector<double>>();

        std::vector<double> z_lower(2 * nq);
        std::vector<double> z_upper(2 * nq);
        std::vector<double> z_range(2 * nq);
        std::vector<double> z_descale(2 * nq);
        double z_measure = 1.;
        for (std::size_t i = 0; i < nq; ++i)
        {
            z_lower[i] = q_lower[i];
            z_upper[i] = q_upper[i];
            z_lower[nq + i] = -velocity_limits[i];
            z_upper[nq + i] = velocity_limits[i];
        }

        for (std::size_t i = 0; i < 2 * nq; ++i)
        {
            z_range[i] = z_upper[i] - z_lower[i];
            z_descale[i] = 1. / z_range[i];
            z_measure *= z_range[i];
        }

        data["z_lower"] = z_lower;
        data["z_upper"] = z_upper;
        data["z_range"] = z_range;
        data["z_descale"] = z_descale;
        data["z_measure"] = z_measure;

        auto z_joint_names = data["joint_names"].get<std::vector<std::string>>();
        for (std::size_t i = 0; i < nq; ++i)
        {
            z_joint_names.emplace_back(z_joint_names[i] + "_vel");
        }
        data["z_joint_names"] = z_joint_names;

        auto flask_interp = cricket::trace_flask_interpolate(robot.model, language);
        data["flask_interpolate_code"] = flask_interp.code;
        data["flask_interpolate_code_vars"] = flask_interp.temp_variables;

        auto flask_interp_block = cricket::trace_flask_interpolate_block(robot.model, language);
        data["flask_interpolate_block_code"] = flask_interp_block.code;
        data["flask_interpolate_block_code_vars"] = flask_interp_block.temp_variables;

        auto flask_rnea = cricket::trace_flask_rnea(robot.model, language);
        data["flask_rnea_code"] = flask_rnea.code;
        data["flask_rnea_code_vars"] = flask_rnea.temp_variables;

        auto flask_rnea_block = cricket::trace_flask_rnea_block(robot.model, language);
        data["flask_rnea_block_code"] = flask_rnea_block.code;
        data["flask_rnea_block_code_vars"] = flask_rnea_block.temp_variables;
    }

    inja::Environment env;

    for (const auto &subt : data["subtemplates"])
    {
        inja::Template temp = env.parse_template(parent_path / subt["template"]);
        env.include_template(subt["name"], temp);
    }

    std::string output_template;
    if (result.count("output_template"))
    {
        output_template = result["output_template"].as<std::string>();
    }
    else
    {
        output_template = data["output"];
    }

    inja::Template temp = env.parse_template(parent_path / data["template"]);
    env.write(temp, data, output_template);

    std::string output_filename;
    if (result.count("output_filename"))
    {
        output_filename = result["output_filename"].as<std::string>();
    }
    else
    {
        output_filename = "output.json";
    }

    std::ofstream output_file(output_filename);
    output_file << data.dump();
    output_file.close();

    return 0;
}

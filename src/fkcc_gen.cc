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

    std::vector<std::string> end_effector_names;
    if (data.contains("end_effector"))
    {
        if (data["end_effector"].is_array())
        {
            end_effector_names = data["end_effector"].get<std::vector<std::string>>();
        }
        else
        {
            end_effector_names.push_back(data["end_effector"].get<std::string>());
        }
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

    cricket::RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_names);

    // Preserve compact_collisions across robot.json() merge; default false.
    const bool compact_collisions = data.value("compact_collisions", false);
    data.update(robot.json(bounds));
    data["compact_collisions"] = compact_collisions;

    // Python/C++ module identifier; must match the registered binding module name.
    // Overridable from the top-level JSON (and the flask block below).
    if (not data.contains("module_name"))
    {
        std::string module_name = data["name"].get<std::string>();
        for (auto &c : module_name)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        data["module_name"] = module_name;
    }

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

    const auto set_trace = [&data](const cricket::Traced &traced, const std::string &key)
    {
        data[key + "_code"] = traced.code;
        data[key + "_code_vars"] = traced.temp_variables;
        data[key + "_code_output"] = traced.outputs;
    };

    using cricket::ProjMethod;

    const bool constraints = data.value("constraints", false);
    data["has_constraints"] = constraints;
    if (constraints)
    {
        set_trace(cricket::trace_tsr_error(robot, language), "tsr_error");
        set_trace(
            cricket::trace_solve_tsr(robot, language, ProjMethod::InnerLM),
            "solve_tsr_error_lm_inner");
        set_trace(
            cricket::trace_solve_tsr(robot, language, ProjMethod::OuterLM),
            "solve_tsr_error_lm_outer");
        set_trace(
            cricket::trace_solve_tsr(robot, language, ProjMethod::GradDesc),
            "solve_tsr_error_gradient_descent");

        if (robot.end_effector_indexes.size() > 1)
        {
            set_trace(cricket::trace_tsr_bimanual_error(robot, language), "tsr_bimanual_error");
            set_trace(
                cricket::trace_solve_tsr(robot, language, ProjMethod::InnerLM, true),
                "solve_tsr_relative_error_lm_inner");
            set_trace(
                cricket::trace_solve_tsr(robot, language, ProjMethod::OuterLM, true),
                "solve_tsr_relative_error_lm_outer");
            set_trace(
                cricket::trace_solve_tsr(robot, language, ProjMethod::GradDesc, true),
                "solve_tsr_relative_error_gradient_descent");
        }
    }

    // Center-of-mass kinematics: "com": true (world frame) or an object with optional
    // "reference_frames" (com expressed relative to the mean position of those frames).
    // The support-polygon error consuming these is 2D (xy), hence err_size 2 solvers.
    bool has_com = false;
    std::vector<std::string> com_reference_frames;
    if (data.contains("com"))
    {
        const auto &cm = data["com"];
        if (cm.is_boolean())
        {
            has_com = cm.get<bool>();
        }
        else
        {
            has_com = true;
            if (cm.contains("reference_frames"))
            {
                com_reference_frames = cm["reference_frames"].get<std::vector<std::string>>();
            }
        }
    }

    data["has_com"] = has_com;
    if (has_com)
    {
        data["com_reference_frames"] = com_reference_frames;
        set_trace(cricket::trace_com_jacobian(robot, com_reference_frames, language), "com_jacobian");
        set_trace(
            cricket::trace_solve_jacobian(robot, language, ProjMethod::InnerLM, 2),
            "solve_com_error_lm_inner");
        set_trace(
            cricket::trace_solve_jacobian(robot, language, ProjMethod::OuterLM, 2),
            "solve_com_error_lm_outer");
        set_trace(
            cricket::trace_solve_jacobian(robot, language, ProjMethod::GradDesc, 2),
            "solve_com_error_gradient_descent");
    }

    // Loop-closure distance constraints: "closed_loops" is a list of
    // {"start_frame", "end_frame", "length"} objects.
    const bool has_closed_loops = data.contains("closed_loops");
    data["has_closed_loops"] = has_closed_loops;
    if (has_closed_loops)
    {
        std::vector<cricket::ClosedLoop> loops;
        for (const auto &cl : data["closed_loops"])
        {
            loops.push_back(
                {cl["start_frame"].get<std::string>(),
                 cl["end_frame"].get<std::string>(),
                 cl["length"].get<double>()});
        }

        data["num_closed_loops"] = loops.size();
        set_trace(cricket::trace_closed_loop_error(robot, loops, language), "closed_loop_error");
        set_trace(
            cricket::trace_solve_jacobian(robot, language, ProjMethod::InnerLM, loops.size()),
            "solve_closed_loop_error_lm_inner");
        set_trace(
            cricket::trace_solve_jacobian(robot, language, ProjMethod::OuterLM, loops.size()),
            "solve_closed_loop_error_lm_outer");
        set_trace(
            cricket::trace_solve_jacobian(robot, language, ProjMethod::GradDesc, loops.size()),
            "solve_closed_loop_error_gradient_descent");
    }

    inja::Environment env;

    for (const auto &subt : data["subtemplates"])
    {
        inja::Template temp = env.parse_template(parent_path / subt["template"]);
        env.include_template(subt["name"], temp);
    }

    // FLASK flat-system (z-robot) sibling: rendered as a nested `Flask` struct inside the
    // geometric robot struct, so a single generated header carries both. The parent robot
    // is always the ambient position-space robot for chart-based constrained planning.
    const bool has_flask = data.contains("flask");
    data["has_flask"] = has_flask;
    if (has_flask)
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

        // Pre-render the nested struct so the main template just splices in finished code;
        // a parse-time `{% include %}` would fail for non-flask robots.
        const auto flask_template =
            fl.value("template", std::string("templates/flask_template.hh"));
        inja::Template flask_temp = env.parse_template(parent_path / flask_template);
        data["flask_struct"] = env.render(flask_temp, data);
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

#include <cricket/codegen.hh>
#include <cricket/robot_info.hh>

#include <Eigen/Core>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <inja/inja.hpp>
#include <cxxopts.hpp>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
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

    cricket::derive_constraint_traces(robot, data, language);

    // FLASK flat-system (z-robot) sibling: the offline path reads the flask template from
    // the recipe directory (overridable via the flask block's "template" key), so template
    // edits take effect without rebuilding cricket.
    std::string flask_template;
    if (data.contains("flask"))
    {
        const auto template_path =
            parent_path /
            data["flask"].value("template", std::string("templates/flask_template.hh"));
        std::ifstream template_file(template_path);
        if (not template_file)
        {
            throw std::runtime_error(
                fmt::format("flask template {} does not exist!", template_path.string()));
        }
        std::stringstream buffer;
        buffer << template_file.rdbuf();
        flask_template = buffer.str();
    }
    cricket::derive_flask_traces(robot, data, language, flask_template);

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

#include "types.hh"
#include "joint_utils.hh"
#include "robot_info.hh"
#include "tracing.hh"

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <inja/inja.hpp>
#include <cxxopts.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

using namespace cricket;

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

    // Parse bounds if provided
    std::optional<Bounds> bounds = std::nullopt;
    if (data.contains("bounds"))
    {
        Bounds b;
        auto &bd = data["bounds"];
        if (!bd.contains("lower") || !bd.contains("upper"))
        {
            throw std::runtime_error(
                "bounds must contain both 'lower' and 'upper' arrays");
        }
        auto lower = bd["lower"].get<std::vector<double>>();
        auto upper = bd["upper"].get<std::vector<double>>();
        if (lower.size() < 2 || lower.size() > 3 || upper.size() < 2 || upper.size() > 3)
        {
            throw std::runtime_error("bounds arrays must have 2 or 3 elements");
        }
        b.lower = Eigen::Vector3d(lower[0], lower[1], lower.size() == 3 ? lower[2] : 0.0);
        b.upper = Eigen::Vector3d(upper[0], upper[1], upper.size() == 3 ? upper[2] : 0.0);
        bounds = b;
    }

    RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_name);

    if (data.contains("mimics"))
    {
        auto mimics = data["mimics"];
        for (const auto &mimic : mimics)
        {
            robot.add_mimic_joint(mimic["name"], mimic["joint"], mimic["multiplier"], mimic["offset"]);
        }
    }

    data.update(robot.json());

    auto traced_eefk_code = trace_sphere_cc_fk(robot, language, false, false, true);
    data["eefk_code"] = traced_eefk_code.code;
    data["eefk_code_vars"] = traced_eefk_code.temp_variables;
    data["eefk_code_output"] = traced_eefk_code.outputs;

    auto traced_spherefk_code = trace_sphere_cc_fk(robot, language, true, false, false);
    data["spherefk_code"] = traced_spherefk_code.code;
    data["spherefk_code_vars"] = traced_spherefk_code.temp_variables;
    data["spherefk_code_output"] = traced_spherefk_code.outputs;

    auto traced_ccfk_code = trace_sphere_cc_fk(robot, language, true, true, false);
    data["ccfk_code"] = traced_ccfk_code.code;
    data["ccfk_code_vars"] = traced_ccfk_code.temp_variables;
    data["ccfk_code_output"] = traced_ccfk_code.outputs;

    auto traced_ccfkee_code = trace_sphere_cc_fk(robot, language, true, true, true);
    data["ccfkee_code"] = traced_ccfkee_code.code;
    data["ccfkee_code_vars"] = traced_ccfkee_code.temp_variables;
    data["ccfkee_code_output"] = traced_ccfkee_code.outputs;

    // Trace mapToConfiguration function
    auto traced_mapconfig_code = trace_map_to_configuration(robot.model, language, bounds);
    data["mapconfig_code"] = traced_mapconfig_code.code;
    data["mapconfig_code_vars"] = traced_mapconfig_code.temp_variables;
    data["mapconfig_code_output"] = traced_mapconfig_code.outputs;
    data["n_u"] = get_randomness_dimension(robot.model);

    // Trace checkBounds function
    auto traced_checkbounds_code = trace_check_bounds(robot.model, language, bounds);
    data["checkbounds_code"] = traced_checkbounds_code.code;
    data["checkbounds_code_vars"] = traced_checkbounds_code.temp_variables;

    // Trace interpolate function
    auto traced_interpolate_code = trace_interpolate(robot.model, language);
    data["interpolate_code"] = traced_interpolate_code.code;
    data["interpolate_code_vars"] = traced_interpolate_code.temp_variables;
    data["interpolate_code_output"] = traced_interpolate_code.outputs;

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

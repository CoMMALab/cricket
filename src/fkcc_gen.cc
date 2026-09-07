#include <cricket/codegen.hh>
#include <cricket/robot_info.hh>

#include <Eigen/Core>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

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
        ("t,output_template", "Output template filename (override configuration file)", cxxopts::value<std::string>())  //
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

    cricket::GenOptions gen_options;
    gen_options.urdf = parent_path / data["urdf"];
    gen_options.srdf = srdf_path;
    gen_options.end_effector = end_effector_name;
    gen_options.template_path = parent_path / data["template"];
    gen_options.language = language;
    gen_options.bounds = bounds;
    gen_options.forward_dynamics = data.value("forward_dynamics", false);
    gen_options.data = data;
    for (const auto &subt : data["subtemplates"])
    {
        gen_options.subtemplates.emplace(
            subt["name"].get<std::string>(), parent_path / subt["template"].get<std::string>());
    }

    const auto generated = cricket::generate_robot_source(gen_options);

    std::string output_template;
    if (result.count("output_template"))
    {
        output_template = result["output_template"].as<std::string>();
    }
    else
    {
        output_template = data["output"];
    }

    std::ofstream generated_file(output_template);
    generated_file << generated.source;
    generated_file.close();

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
    output_file << generated.data.dump();
    output_file.close();

    return 0;
}


#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <inja/inja.hpp>
#include <cxxopts.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>

#include "robot_info.hh"
#include "housekeeping.hh"
#include "tracer_utils.hh"
#include "tsr_constraints.hh"

auto trace_sphere(const SphereInfo &sphere, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &joint_placement = ad_data.oMi[sphere.parent_joint];

    Eigen::Matrix<ADCG, 3, 1> local_translation;
    local_translation[0] = sphere.relative.translation()[0];
    local_translation[1] = sphere.relative.translation()[1];
    local_translation[2] = sphere.relative.translation()[2];

    Eigen::Matrix<ADCG, 3, 1> world_position =
        joint_placement.rotation() * local_translation + joint_placement.translation();

    data[index + 0] = world_position[0];
    data[index + 1] = world_position[1];
    data[index + 2] = world_position[2];
    data[index + 3] = ADCG(sphere.radius);
}

auto trace_frame(std::size_t ee_index, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &oMf = ad_data.oMf[ee_index];

    data[index + 0] = oMf.translation()[0];
    data[index + 1] = oMf.translation()[1];
    data[index + 2] = oMf.translation()[2];

    const auto &R = oMf.rotation();

    // Eigen stores as column major
    data[index + 3] = R(0, 0);
    data[index + 4] = R(1, 0);
    data[index + 5] = R(2, 0);
    data[index + 6] = R(0, 1);
    data[index + 7] = R(1, 1);
    data[index + 8] = R(2, 1);
    data[index + 9] = R(0, 2);
    data[index + 10] = R(1, 2);
    data[index + 11] = R(2, 2);
}

auto trace_sphere_cc_fk(
    const RobotInfo &info,
    bool spheres = true,
    bool bounding_spheres = true,
    bool fk = true) -> Traced
{
    auto nq = info.model.nq;
    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    ADVectorXs ad_q(nq);
    for (auto i = 0U; i < nq; ++i)
    {
        ad_q[i] = ADCG(0.0);
    }

    Independent(ad_q);

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    std::size_t n_spheres_data = (spheres) ? info.spheres.size() * 4 : 0;
    std::size_t n_bounding_spheres_data = (bounding_spheres) ? info.bounding_spheres.size() * 4 : 0;
    std::size_t n_fk_data = (fk) ? 12 * info.end_effector_indexes.size() : 0;
    std::size_t n_out = n_spheres_data + n_bounding_spheres_data + n_fk_data;

    ADVectorXs data(n_out);

    if (spheres)
    {
        for (auto i = 0U; i < info.spheres.size(); ++i)
        {
            const auto &sphere = info.spheres[i];
            trace_sphere(sphere, ad_data, data, sphere.geom_index * 4);
        }
    }

    if (bounding_spheres)
    {
        for (auto i = 0U; i < info.model.frames.size(); ++i)
        {
            auto sphere_it = info.bounding_spheres.find(i);
            if (sphere_it != info.bounding_spheres.end())
            {
                const auto &sphere = sphere_it->second;
                trace_sphere(sphere, ad_data, data, sphere.geom_index * 4 + n_spheres_data);
            }
        }
    }

    if (fk)
    {
        for (size_t i = 0; i < info.end_effector_indexes.size(); i++)
        {
            trace_frame(
                info.end_effector_indexes[i],
                ad_data,
                data,
                n_spheres_data + n_bounding_spheres_data + 12 * i);
        }
    }

    // Create the AD function
    ADFun<CGD> collision_sphere_func(ad_q, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = collision_sphere_func.Forward(0, ind_vars);

    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, result, nameGen);

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), n_out};
}

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

    if (not std::filesystem::exists(json_path))
    {
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
    if (data.contains("end_effectors"))
    {
        for (const auto end_effector_name : data["end_effectors"])
        {
            end_effector_names.push_back(end_effector_name);
        }
    }

    RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_names);

    data.update(robot.json());

    add_to_trace(trace_sphere_cc_fk(robot, false, false, true), "eefk_code", data);
    add_to_trace(trace_sphere_cc_fk(robot, true, false, false), "spherefk_code", data);
    add_to_trace(trace_sphere_cc_fk(robot, true, true, false), "ccfk_code", data);
    add_to_trace(trace_sphere_cc_fk(robot, true, true, true), "ccfkee_code", data);
    add_to_trace(trace_tsr_error_function(robot), "tsr_error_code", data);

    add_to_trace(trace_solve_tsr_function(robot, ProjMethod::InnerLM), "solve_tsr_error_lm_inner_code", data);
    add_to_trace(trace_solve_tsr_function(robot, ProjMethod::OuterLM), "solve_tsr_error_lm_outer_code", data);
    add_to_trace(trace_solve_tsr_function(robot, ProjMethod::GradDesc), "solve_tsr_error_gradient_descent_code", data);

    add_to_trace(trace_and_project_tsr_error_function(robot, ProjMethod::InnerLM), "compute_and_project_tsr_error_lm_inner_code", data);
    add_to_trace(trace_and_project_tsr_error_function(robot, ProjMethod::OuterLM), "compute_and_project_tsr_error_lm_outer_code", data);
    add_to_trace(trace_and_project_tsr_error_function(robot, ProjMethod::GradDesc), "compute_and_project_tsr_error_gradient_descent_code", data);


    if(robot.end_effector_indexes.size() > 1) {
        add_to_trace(trace_tsr_bimanual_error_function(robot), "tsr_bimanual_error_code", data);
        add_to_trace(trace_solve_tsr_function(robot, ProjMethod::InnerLM, true), "solve_relative_tsr_error_lm_inner_code", data);
        add_to_trace(trace_solve_tsr_function(robot, ProjMethod::OuterLM, true), "solve_relative_tsr_error_lm_outer_code", data);
        add_to_trace(trace_solve_tsr_function(robot, ProjMethod::GradDesc, true), "solve_relative_tsr_error_gradient_descent_code", data);
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

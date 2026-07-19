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
#include "iiwa_parameterization_gen.hh"
#include "se3_tracer.hh"

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
    const std::string &language,
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
    std::size_t n_fk_data = (fk) ? 12 : 0;
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
        trace_frame(info.end_effector_index, ad_data, data, n_spheres_data + n_bounding_spheres_data);
    }

    // Create the AD function
    ADFun<CGD> collision_sphere_func(ad_q, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = collision_sphere_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

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

    std::optional<Bounds> bounds;
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
        Bounds b;
        b.lower = Eigen::Vector3d(lower[0], lower[1], lower.size() == 3 ? lower[2] : 0.0);
        b.upper = Eigen::Vector3d(upper[0], upper[1], upper.size() == 3 ? upper[2] : 0.0);
        bounds = b;
    }

    // Expose the sampler's end-effector position bounds to the template as
    // sample_position_lower / sample_position_upper. Defaults to zero when
    // "bounds" is absent from the input JSON.
    std::array<double, 3> sample_position_lower = {0.0, 0.0, 0.0};
    std::array<double, 3> sample_position_upper = {0.0, 0.0, 0.0};
    if (bounds)
    {
        for (int i = 0; i < 3; ++i)
        {
            sample_position_lower[i] = bounds->lower[i];
            sample_position_upper[i] = bounds->upper[i];
        }
    }
    data["sample_position_lower"] = sample_position_lower;
    data["sample_position_upper"] = sample_position_upper;

    RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_name);

    data.update(robot.json());

    auto traced_se3_sampler_code = trace_map_to_se3(robot.model, language, bounds);
    data["se3_sampler_code"] = traced_se3_sampler_code.code;
    data["se3_sampler_code_vars"] = traced_se3_sampler_code.temp_variables;
    data["se3_sampler_code_output"] = traced_se3_sampler_code.outputs;

    std::cout << "Going to interpolate code generation..." << std::endl;
    auto traced_interpolate_code = trace_interpolate(language);
    data["interpolate_code"] = traced_interpolate_code.code;
    data["interpolate_code_vars"] = traced_interpolate_code.temp_variables;
    data["interpolate_code_output"] = traced_interpolate_code.outputs;

    std::cout << "Going to interpolate block code generation..." << std::endl;
    auto interp_block = trace_interpolate_block(language);
    data["interpolate_block_code"] = interp_block.code;
    data["interpolate_block_code_vars"] = interp_block.temp_variables;

    std::cout << "Going to distance code generation..." << std::endl;
    auto dist = trace_SE3_distance(language);
    data["distance_code"] = dist.code;
    data["distance_code_vars"] = dist.temp_variables;


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

    if (data.contains("generate_param_ik") && data["generate_param_ik"].get<bool>())
    {
        // param_ik_code emits two output arrays: `y` (joint angles, sized
        // param_ik_code_output) and `u` (the 4 pre-clip SafeArccos arguments,
        // sized param_ik_num_unclipped), used by parameterized_ik to reject
        // poses that SafeArccos would have silently clipped.
        const size_t param_ik_num_unclipped = 4;
        if (data["name"].contains("Bimanual") == true)
        {
            auto traced_iiwa_param_code = IiwaBimanualParameterizationCG<ADCG>(language, false);
            data["param_ik_code"] = traced_iiwa_param_code.code;
            data["param_ik_code_vars"] = traced_iiwa_param_code.temp_variables;
            data["param_ik_code_output"] = traced_iiwa_param_code.outputs - param_ik_num_unclipped;
            data["param_ik_num_unclipped"] = param_ik_num_unclipped;
        }
        else
        {
            auto traced_iiwa_param_code = IiwaSE3ParameterizationCG<ADCG>(language, false);
            data["param_ik_code"] = traced_iiwa_param_code.code;
            data["param_ik_code_vars"] = traced_iiwa_param_code.temp_variables;
            data["param_ik_code_output"] = traced_iiwa_param_code.outputs - param_ik_num_unclipped;
            data["param_ik_num_unclipped"] = param_ik_num_unclipped;
        }
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

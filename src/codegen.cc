#include <cricket/codegen.hh>
#include <cricket/embedded_templates.hh>

#include "pinocchio_cppadcg.hh"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <fmt/format.h>
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>

#include <sstream>
#include <stdexcept>

#include "lang_cpp.hh"
#include "lang_rust.hh"

namespace cricket
{
    using namespace pinocchio;
    using namespace CppAD;
    using namespace CppAD::cg;

    namespace
    {
        // Typedef for AD types
        using CGD = CG<double>;
        using ADCG = AD<CGD>;

        using ADModel = ModelTpl<ADCG>;
        using ADData = DataTpl<ADCG>;
        using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;

        auto
        trace_sphere(const SphereInfo &sphere, const ADData &ad_data, ADVectorXs &data, std::size_t index)
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
    }  // namespace

    auto trace_sphere_cc_fk(
        const RobotInfo &info,
        const std::string &language,
        bool spheres,
        bool bounding_spheres,
        bool fk) -> Traced
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

    auto generate_robot_source(const GenOptions &opts) -> GenResult
    {
        const bool use_embedded = opts.template_path.empty();
        if (not use_embedded and not std::filesystem::exists(opts.template_path))
        {
            throw std::runtime_error(
                fmt::format(
                    "cricket::generate_robot_source: template_path does not exist: {}",
                    opts.template_path.string()));
        }

        RobotInfo robot(opts.urdf, opts.srdf, opts.end_effector);

        nlohmann::json data = opts.data;
        data.update(robot.json());

        auto eefk = trace_sphere_cc_fk(robot, opts.language, false, false, true);
        data["eefk_code"] = eefk.code;
        data["eefk_code_vars"] = eefk.temp_variables;
        data["eefk_code_output"] = eefk.outputs;

        auto spherefk = trace_sphere_cc_fk(robot, opts.language, true, false, false);
        data["spherefk_code"] = spherefk.code;
        data["spherefk_code_vars"] = spherefk.temp_variables;
        data["spherefk_code_output"] = spherefk.outputs;

        auto ccfk = trace_sphere_cc_fk(robot, opts.language, true, true, false);
        data["ccfk_code"] = ccfk.code;
        data["ccfk_code_vars"] = ccfk.temp_variables;
        data["ccfk_code_output"] = ccfk.outputs;

        auto ccfkee = trace_sphere_cc_fk(robot, opts.language, true, true, true);
        data["ccfkee_code"] = ccfkee.code;
        data["ccfkee_code_vars"] = ccfkee.temp_variables;
        data["ccfkee_code_output"] = ccfkee.outputs;

        inja::Environment env;
        inja::Template main_template;
        if (use_embedded)
        {
            auto ccfk_t = env.parse(std::string(embedded::kCcfkTemplate));
            env.include_template("ccfk", ccfk_t);
            main_template = env.parse(std::string(embedded::kFkTemplate));
        }
        else
        {
            for (const auto &[name, path] : opts.subtemplates)
            {
                if (not std::filesystem::exists(path))
                {
                    throw std::runtime_error(
                        fmt::format(
                            "cricket::generate_robot_source: subtemplate '{}' not found: {}",
                            name,
                            path.string()));
                }
                inja::Template t = env.parse_template(path.string());
                env.include_template(name, t);
            }
            main_template = env.parse_template(opts.template_path.string());
        }

        GenResult result;
        result.source = env.render(main_template, data);
        result.data = std::move(data);

        if (result.data.contains("name"))
        {
            result.robot_name = result.data["name"].get<std::string>();
        }

        if (result.data.contains("n_q"))
        {
            result.dimension = result.data["n_q"].get<std::size_t>();
        }

        if (result.data.contains("n_spheres"))
        {
            result.n_spheres = result.data["n_spheres"].get<std::size_t>();
        }

        return result;
    }
}  // namespace cricket

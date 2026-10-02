#include <cricket/codegen.hh>
#include <cricket/embedded_templates.hh>

#include "codegen/pinocchio_cppadcg.hh"
#include "codegen/lang_cpp.hh"
#include "codegen/lang_rust.hh"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

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
            for (auto i = 0U; i < info.end_effector_indexes.size(); ++i)
            {
                trace_frame(
                    info.end_effector_indexes[i],
                    ad_data,
                    data,
                    n_spheres_data + n_bounding_spheres_data + i * 12);
            }
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

    namespace
    {
        auto edit_distance(std::string_view a, std::string_view b) -> std::size_t
        {
            std::vector<std::size_t> prev(b.size() + 1);
            std::vector<std::size_t> curr(b.size() + 1);
            for (std::size_t j = 0; j <= b.size(); ++j)
            {
                prev[j] = j;
            }
            for (std::size_t i = 1; i <= a.size(); ++i)
            {
                curr[0] = i;
                for (std::size_t j = 1; j <= b.size(); ++j)
                {
                    const std::size_t subst = prev[j - 1] + ((a[i - 1] == b[j - 1]) ? 0 : 1);
                    curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, subst});
                }
                std::swap(prev, curr);
            }
            return prev[b.size()];
        }

        auto check_keys(
            const nlohmann::json &object,
            const std::vector<std::string_view> &allowed,
            const std::string &context,
            std::vector<std::string> &problems) -> void
        {
            for (const auto &[key, _] : object.items())
            {
                if ((not key.empty() and key.front() == '_') or
                    std::find(allowed.begin(), allowed.end(), key) != allowed.end())
                {
                    continue;
                }

                std::string_view best;
                std::size_t best_distance = std::numeric_limits<std::size_t>::max();
                for (const auto &candidate : allowed)
                {
                    const auto d = edit_distance(key, candidate);
                    if (d < best_distance)
                    {
                        best_distance = d;
                        best = candidate;
                    }
                }

                auto problem = fmt::format("unknown key \"{}\" in {}", key, context);
                if (best_distance <= std::max<std::size_t>(2, key.size() / 3))
                {
                    problem += fmt::format(" (did you mean \"{}\"?)", best);
                }
                problems.emplace_back(std::move(problem));
            }
        }
    }  // namespace

    auto validate_recipe(const nlohmann::json &data) -> void
    {
        static const std::vector<std::string_view> top_level = {
            "name",
            "module_name",
            "urdf",
            "srdf",
            "dynamics_urdf",
            "forward_dynamics",
            "end_effector",
            "language",
            "bounds",
            "resolution",
            "template",
            "subtemplates",
            "output",
            "compact_collisions",
            "skip_static_environment_collisions",
            "active_joints",
            "default_configuration",
            "parts",
            "disabled_collisions",
        };
        static const std::vector<std::string_view> bounds_keys = {"lower", "upper"};
        static const std::vector<std::string_view> part_keys = {
            "prefix", "urdf", "srdf", "parent", "xyz", "rpy", "quat"};
        static const std::vector<std::string_view> subtemplate_keys = {"name", "template"};

        std::vector<std::string> problems;
        check_keys(data, top_level, "recipe", problems);

        const auto check_object = [&](const char *key, const std::vector<std::string_view> &allowed)
        {
            if (data.contains(key) and data[key].is_object())
            {
                check_keys(data[key], allowed, fmt::format("\"{}\"", key), problems);
            }
        };
        const auto check_array = [&](const char *key, const std::vector<std::string_view> &allowed)
        {
            if (not data.contains(key) or not data[key].is_array())
            {
                return;
            }
            std::size_t i = 0;
            for (const auto &entry : data[key])
            {
                if (entry.is_object())
                {
                    check_keys(entry, allowed, fmt::format("\"{}\"[{}]", key, i), problems);
                }
                ++i;
            }
        };

        check_object("bounds", bounds_keys);
        check_array("parts", part_keys);
        check_array("subtemplates", subtemplate_keys);

        if (not problems.empty())
        {
            throw std::runtime_error(
                fmt::format("Invalid recipe:\n  {}", fmt::join(problems, "\n  ")));
        }
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

        validate_recipe(opts.data);

        // A "parts" key in the recipe data selects composite assembly; part paths are
        // resolved against the URDF's directory (or as given when no URDF is set).
        const auto composite = CompositeSpec::from_json(opts.data, opts.urdf.parent_path());
        RobotInfo robot =
            composite ?
                RobotInfo(*composite, opts.end_effectors, JointSelection::from_json(opts.data)) :
                RobotInfo(opts.urdf, opts.srdf, opts.end_effectors, JointSelection::from_json(opts.data));

        nlohmann::json data = opts.data;
        const bool compact_collisions = opts.data.value("compact_collisions", false);
        data.update(robot.json(
            opts.bounds, opts.data.value("skip_static_environment_collisions", false)));
        data["compact_collisions"] = compact_collisions;

        // Python/C++ module identifier; must match the registered binding module name.
        if (not data.contains("module_name"))
        {
            std::string module_name = data["name"].get<std::string>();
            for (auto &c : module_name)
            {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            data["module_name"] = module_name;
        }


        auto eefk = trace_sphere_cc_fk(robot, opts.language, false, false, true);
        data["eefk_code"] = eefk.code;
        data["eefk_code_vars"] = eefk.temp_variables;
        data["eefk_code_output"] = eefk.outputs;

        auto eejac = trace_ee_fk_jacobian(robot, opts.language);
        data["eejac_code"] = eejac.code;
        data["eejac_code_vars"] = eejac.temp_variables;
        data["eejac_code_output"] = eejac.outputs;

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

        auto mapconfig = trace_map_to_configuration(robot.model, opts.language, opts.bounds);
        data["mapconfig_code"] = mapconfig.code;
        data["mapconfig_code_vars"] = mapconfig.temp_variables;
        data["mapconfig_code_output"] = mapconfig.outputs;

        auto interp = trace_interpolate(robot.model, opts.language);
        data["interpolate_code"] = interp.code;
        data["interpolate_code_vars"] = interp.temp_variables;

        auto interp_block = trace_interpolate_block(robot.model, opts.language);
        data["interpolate_block_code"] = interp_block.code;
        data["interpolate_block_code_vars"] = interp_block.temp_variables;

        auto dist = trace_distance(robot.model, opts.language);
        data["distance_code"] = dist.code;
        data["distance_code_vars"] = dist.temp_variables;

        auto integration = trace_integrate_configuration(robot.model, opts.language);
        data["integrate_configuration_code"] = integration.code;
        data["integrate_configuration_code_vars"] = integration.temp_variables;
        data["integrate_configuration_code_output"] = integration.outputs;

        if (opts.forward_dynamics)
        {
            pinocchio::Model dynamics_model;
            if (opts.dynamics_urdf)
            {
                pinocchio::urdf::buildModel(*opts.dynamics_urdf, dynamics_model, false, true);
                if (dynamics_model.nq != robot.model.nq || dynamics_model.nv != robot.model.nv)
                {
                    throw std::runtime_error("dynamics URDF dimensions do not match collision URDF");
                }
            }
            else
            {
                dynamics_model = robot.model;
            }
            std::vector<float> effort_lower(dynamics_model.nv);
            std::vector<float> effort_upper(dynamics_model.nv);
            std::vector<float> velocity_lower(dynamics_model.nv);
            std::vector<float> velocity_upper(dynamics_model.nv);
            for (auto i = 0; i < dynamics_model.nv; ++i)
            {
                effort_lower[i] = static_cast<float>(-dynamics_model.effortLimit[i]);
                effort_upper[i] = static_cast<float>(dynamics_model.effortLimit[i]);
                velocity_lower[i] = static_cast<float>(-dynamics_model.velocityLimit[i]);
                velocity_upper[i] = static_cast<float>(dynamics_model.velocityLimit[i]);
            }
            data["effort_lower"] = effort_lower;
            data["effort_upper"] = effort_upper;
            data["velocity_lower"] = velocity_lower;
            data["velocity_upper"] = velocity_upper;
            auto dynamics = trace_forward_dynamics(dynamics_model, opts.language);
            data["forward_dynamics_code"] = dynamics.code;
            data["forward_dynamics_code_vars"] = dynamics.temp_variables;
            data["forward_dynamics_code_output"] = dynamics.outputs;
        }

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

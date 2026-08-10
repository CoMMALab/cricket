#include <cricket/codegen.hh>
#include <cricket/embedded_templates.hh>

#include "codegen/pinocchio_cppadcg.hh"
#include "codegen/lang_cpp.hh"
#include "codegen/lang_rust.hh"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

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

    auto derive_constraint_traces(const RobotInfo &robot, nlohmann::json &data, const std::string &language)
        -> void
    {
        const auto set_trace = [&data](const Traced &traced, const std::string &key)
        {
            data[key + "_code"] = traced.code;
            data[key + "_code_vars"] = traced.temp_variables;
            data[key + "_code_output"] = traced.outputs;
        };

        const bool constraints = data.value("constraints", false);
        data["has_constraints"] = constraints;
        if (constraints)
        {
            set_trace(trace_tsr_error(robot, language), "tsr_error");
            set_trace(
                trace_solve_tsr(robot, language, ProjMethod::InnerLM), "solve_tsr_error_lm_inner");
            set_trace(
                trace_solve_tsr(robot, language, ProjMethod::OuterLM), "solve_tsr_error_lm_outer");
            set_trace(
                trace_solve_tsr(robot, language, ProjMethod::GradDesc),
                "solve_tsr_error_gradient_descent");

            if (robot.end_effector_indexes.size() > 1)
            {
                set_trace(trace_tsr_bimanual_error(robot, language), "tsr_bimanual_error");
                set_trace(
                    trace_solve_tsr(robot, language, ProjMethod::InnerLM, true),
                    "solve_tsr_relative_error_lm_inner");
                set_trace(
                    trace_solve_tsr(robot, language, ProjMethod::OuterLM, true),
                    "solve_tsr_relative_error_lm_outer");
                set_trace(
                    trace_solve_tsr(robot, language, ProjMethod::GradDesc, true),
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
            set_trace(trace_com_jacobian(robot, com_reference_frames, language), "com_jacobian");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::InnerLM, 2),
                "solve_com_error_lm_inner");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::OuterLM, 2),
                "solve_com_error_lm_outer");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::GradDesc, 2),
                "solve_com_error_gradient_descent");
        }

        // Loop-closure distance constraints: "closed_loops" is a list of
        // {"start_frame", "end_frame", "length"} objects.
        const bool has_closed_loops = data.contains("closed_loops");
        data["has_closed_loops"] = has_closed_loops;
        if (has_closed_loops)
        {
            std::vector<ClosedLoop> loops;
            for (const auto &cl : data["closed_loops"])
            {
                loops.push_back(
                    {cl["start_frame"].get<std::string>(),
                     cl["end_frame"].get<std::string>(),
                     cl["length"].get<double>()});
            }

            data["num_closed_loops"] = loops.size();
            set_trace(trace_closed_loop_error(robot, loops, language), "closed_loop_error");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::InnerLM, loops.size()),
                "solve_closed_loop_error_lm_inner");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::OuterLM, loops.size()),
                "solve_closed_loop_error_lm_outer");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::GradDesc, loops.size()),
                "solve_closed_loop_error_gradient_descent");
        }

        // Lead-screw coupling: "lead_screw": true generates the scalar screw invariant h(q)
        // of the first end-effector (axial advance minus pitch-scaled rotation about a
        // reference frame's z-axis) with err_size-1 projection solvers. dh/dq serves as the
        // Pfaffian row of the coupling; the solvers serve its integrable (holonomic)
        // representation.
        const bool has_lead_screw = data.value("lead_screw", false);
        data["has_lead_screw"] = has_lead_screw;
        if (has_lead_screw)
        {
            set_trace(trace_lead_screw_error(robot, language), "lead_screw_error");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::InnerLM, 1),
                "solve_lead_screw_error_lm_inner");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::OuterLM, 1),
                "solve_lead_screw_error_lm_outer");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::GradDesc, 1),
                "solve_lead_screw_error_gradient_descent");
        }

        // Twist Jacobians: "twist": true generates the reference-frame and body-frame twist
        // Jacobians of the first end-effector's offset frame, combined at runtime with
        // constant coefficients into Pfaffian velocity-constraint rows (lead screw,
        // knife-edge, no-slip) without further codegen.
        const bool has_twist = data.value("twist", false);
        data["has_twist"] = has_twist;
        if (has_twist)
        {
            set_trace(trace_twist_jacobians(robot, language), "twist_jacobians");
        }
    }

    // FLASK flat-system (z-robot) sibling: rendered as a nested `Flask` struct inside the
    // geometric robot struct, so a single generated header carries both. The parent robot
    // is always the ambient position-space robot for chart-based constrained planning.
    auto derive_flask_traces(
        const RobotInfo &robot,
        nlohmann::json &data,
        const std::string &language,
        std::string_view flask_template) -> void
    {
        const bool has_flask = data.contains("flask");
        data["has_flask"] = has_flask;
        if (not has_flask)
        {
            return;
        }

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

        // Flask edge validation may need finer sampling than the geometric parent: the
        // z-space cubics bow away from the straight line their endpoints suggest.
        data["flask_resolution"] = fl.contains("resolution") ?
                                       fl["resolution"].get<std::size_t>() :
                                       data["resolution"].get<std::size_t>();

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

        auto flask_interp = trace_flask_interpolate(robot.model, language);
        data["flask_interpolate_code"] = flask_interp.code;
        data["flask_interpolate_code_vars"] = flask_interp.temp_variables;

        auto flask_interp_block = trace_flask_interpolate_block(robot.model, language);
        data["flask_interpolate_block_code"] = flask_interp_block.code;
        data["flask_interpolate_block_code_vars"] = flask_interp_block.temp_variables;

        auto flask_rnea = trace_flask_rnea(robot.model, language);
        data["flask_rnea_code"] = flask_rnea.code;
        data["flask_rnea_code_vars"] = flask_rnea.temp_variables;

        auto flask_rnea_block = trace_flask_rnea_block(robot.model, language);
        data["flask_rnea_block_code"] = flask_rnea_block.code;
        data["flask_rnea_block_code_vars"] = flask_rnea_block.temp_variables;

        auto flask_ke = trace_flask_kinetic_energy(robot.model, language);
        data["flask_kinetic_energy_code"] = flask_ke.code;
        data["flask_kinetic_energy_code_vars"] = flask_ke.temp_variables;

        auto flask_ke_block = trace_flask_kinetic_energy_block(robot.model, language);
        data["flask_kinetic_energy_block_code"] = flask_ke_block.code;
        data["flask_kinetic_energy_block_code_vars"] = flask_ke_block.temp_variables;

        auto flask_eev = trace_flask_eef_velocity(robot, language);
        data["flask_eef_velocity_code"] = flask_eev.code;
        data["flask_eef_velocity_code_vars"] = flask_eev.temp_variables;

        auto flask_eev_block = trace_flask_eef_velocity_block(robot, language);
        data["flask_eef_velocity_block_code"] = flask_eev_block.code;
        data["flask_eef_velocity_block_code_vars"] = flask_eev_block.temp_variables;

        // Pre-render the nested struct so the main template just splices in finished code;
        // a parse-time `{% include %}` would fail for non-flask robots. The flask template
        // uses only inja builtins, so a fresh environment suffices.
        inja::Environment env;
        inja::Template flask_temp = env.parse(
            flask_template.empty() ? std::string(embedded::kFlaskTemplate) :
                                     std::string(flask_template));
        data["flask_struct"] = env.render(flask_temp, data);
    }

    // RBY1 constrained-bimanual parameterized IK: "use_parameterized": true traces the
    // whole-body-relative IK, the dual-hand FK used to derive t_mid_left/t_mid_right, and
    // the sample/distance/interpolate kernels over that parameterized space -- all consumed
    // by fk_template.hh's ParameterizedSpace struct (see rainbow_ik_cg.hh for the math).
    auto derive_parameterized_traces(
        const RobotInfo &robot,
        nlohmann::json &data,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> void
    {
        const bool use_parameterized = data.value("use_parameterized", false);
        data["has_parameterized_space"] = use_parameterized;
        if (not use_parameterized)
        {
            return;
        }

        auto param_ik = trace_rby1_constrained_ik(robot, language);
        data["param_ik_code"] = param_ik.code;
        data["param_ik_code_vars"] = param_ik.temp_variables;
        data["param_ik_code_output"] = param_ik.outputs;

        // RainbowArmParamResult::unclipped is always a Vector3 -- see
        // rainbow_arm_parameterization.hh -- fixed regardless of robot, not something to trace.
        data["param_ik_num_unclipped"] = 3;

        auto param_mid_pose_fk = trace_rby1_mid_pose_fk(robot, language);
        data["param_mid_pose_fk_code"] = param_mid_pose_fk.code;
        data["param_mid_pose_fk_code_vars"] = param_mid_pose_fk.temp_variables;
        data["param_mid_pose_fk_code_output"] = param_mid_pose_fk.outputs;

        // CoM position in the frame of robot base: reuses
        // trace_com_jacobian's error computation with compute_jac=false since
        // ParameterizedSpace::compute_com is a scalar, once-per-problem utility like
        // compute_mid_pose, not a per-lane hot-loop constraint.
        auto param_com = trace_com_jacobian(robot, {"base"}, language, false);
        data["param_com_code"] = param_com.code;
        data["param_com_code_vars"] = param_com.temp_variables;
        data["param_com_code_output"] = param_com.outputs;

        auto param_sample = trace_rby1_constrained_sample(robot.model, language, bounds);
        data["param_sample_code"] = param_sample.code;
        data["param_sample_code_vars"] = param_sample.temp_variables;
        data["param_sample_code_output"] = param_sample.outputs;

        auto param_distance = trace_rby1_constrained_distance(language);
        data["param_distance_code"] = param_distance.code;
        data["param_distance_code_vars"] = param_distance.temp_variables;

        auto param_interpolate = trace_rby1_constrained_interpolate(language);
        data["param_interpolate_code"] = param_interpolate.code;
        data["param_interpolate_code_vars"] = param_interpolate.temp_variables;

        auto param_interpolate_block = trace_rby1_constrained_interpolate_block(language);
        data["param_interpolate_block_code"] = param_interpolate_block.code;
        data["param_interpolate_block_code_vars"] = param_interpolate_block.temp_variables;

        data["param_dimension"] = 19;
        data["param_sample_dimension"] = 17;

        // State layout (see trace_rby1_constrained_sample's header comment above):
        // base(4) + torso(6) + psi_left/right(2) + t_mid pose(7, [x,y,z,qx,qy,qz,qw]) == 19.
        // Non-Euclidean: base_rz is an SO(2) (cos, sin) pair and t_mid's orientation is a
        // genuine SO(3) quaternion block at local offset 12 + 3 == 15.
        data["param_euclidean"] = false;
        data["param_so3_offsets"] = std::vector<std::size_t>{15};
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
            "end_effector",
            "language",
            "bounds",
            "resolution",
            "template",
            "subtemplates",
            "output",
            "flask",
            "constraints",
            "compact_collisions",
            "skip_static_environment_collisions",
            "active_joints",
            "default_configuration",
            "parts",
            "disabled_collisions",
            "com",
            "closed_loops",
            "lead_screw",
            "twist",
            "use_parameterized",
        };
        static const std::vector<std::string_view> bounds_keys = {"lower", "upper"};
        static const std::vector<std::string_view> flask_keys = {
            "rho", "resolution", "velocity_limits", "effort_limits", "template"};
        static const std::vector<std::string_view> com_keys = {"reference_frames"};
        static const std::vector<std::string_view> loop_keys = {"start_frame", "end_frame", "length"};
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
        check_object("flask", flask_keys);
        check_object("com", com_keys);
        check_array("closed_loops", loop_keys);
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

        derive_constraint_traces(robot, data, opts.language);
        derive_flask_traces(robot, data, opts.language);
        derive_parameterized_traces(robot, data, opts.language, opts.bounds);

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

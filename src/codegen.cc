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

    // Per-end-effector collision spheres rigidly attached to each of `info.end_effector_names`
    // (gripper/finger/marker geometry, as opposed to an externally attached object), expressed
    // as a function of a *candidate world-frame pose for that end effector* (translation +
    // rotation matrix, vamp::to_isometry's 12-float layout) instead of the ambient joint
    // configuration -- since these spheres are rigid with their end-effector frame, their
    // world position is just `R * local + t`, with no forwardKinematics needed. Pose is taken
    // as a rotation *matrix* rather than a quaternion so callers that already have one (e.g.
    // ParameterizedSpace::eef_world_poses in fk_template.hh) never need an unsafe-to-trace
    // matrix -> quaternion round trip. Branch-free, so the same "c++"-generated trace is valid
    // whether a caller instantiates it scalar or rake-batched (FloatVector<rake,1>): operator
    // overloading on that type makes the generated arithmetic work unchanged either way.
    // Generic over the number of end effectors -- not tied to any particular robot or
    // parameterization -- so it lives here next to trace_sphere_cc_fk rather than in one of
    // the robot-specific src/parameterization/*.cc TUs; both derive_rby1_parameterized_traces
    // (num_end_effectors == 2) and derive_iiwa_se3_parameterized_traces (num_end_effectors ==
    // 1) below call it.
    //
    // Tape input layout (12 * num_end_effectors): each end effector's world pose (x, y, z,
    // then rotation matrix column-major), in `info.end_effector_names` order.
    //
    // Output: each end effector's spheres (x, y, z, r) each, back to back, in
    // `info.end_effector_names` order, and within an end effector, in the order they appear
    // in `info.spheres`. `EefLocalSpheres::counts` reports how many spheres landed in each end
    // effector's slice, since the flat output size alone doesn't disambiguate the per-eef
    // split.
    auto trace_eef_local_spheres(const RobotInfo &info, const std::string &language) -> EefLocalSpheres
    {
        const auto num_end_effectors = info.end_effector_indexes.size();

        // Per end effector: local sphere offsets + radii relative to the end effector's own
        // frame -- fixed, precomputed in double precision.
        std::vector<std::vector<std::pair<Eigen::Vector3d, float>>> per_eef_spheres(num_end_effectors);
        for (auto k = 0U; k < num_end_effectors; ++k)
        {
            const auto &frame = info.model.frames[info.end_effector_indexes[k]];

            for (const auto &sphere : info.spheres)
            {
                if (sphere.parent_joint != frame.parentJoint)
                {
                    continue;
                }

                Eigen::Vector3d local_offset = frame.placement.inverse().act(sphere.relative.translation());
                per_eef_spheres[k].emplace_back(local_offset, sphere.radius);
                // we need a way of identifying the link the added sphere belongs to to print it
            }
        }

        ADVectorXs ad_pose(12 * num_end_effectors);
        for (auto i = 0U; i < ad_pose.size(); ++i)
        {
            ad_pose[i] = ADCG(0.0);
        }
        Independent(ad_pose);

        std::size_t total_spheres = 0;
        for (const auto &spheres : per_eef_spheres)
        {
            total_spheres += spheres.size();
        }
        ADVectorXs data(total_spheres * 4);

        std::size_t data_offset = 0;
        for (auto k = 0U; k < num_end_effectors; ++k)
        {
            const auto pose_offset = 12 * k;
            Eigen::Matrix<ADCG, 3, 1> translation{
                ad_pose[pose_offset + 0], ad_pose[pose_offset + 1], ad_pose[pose_offset + 2]};

            // Column-major, matching vamp::to_isometry / trace_frame's layout.
            Eigen::Matrix<ADCG, 3, 3> rotation;
            rotation.col(0) << ad_pose[pose_offset + 3], ad_pose[pose_offset + 4], ad_pose[pose_offset + 5];
            rotation.col(1) << ad_pose[pose_offset + 6], ad_pose[pose_offset + 7], ad_pose[pose_offset + 8];
            rotation.col(2) << ad_pose[pose_offset + 9], ad_pose[pose_offset + 10], ad_pose[pose_offset + 11];

            for (const auto &[local_offset, radius] : per_eef_spheres[k])
            {
                Eigen::Matrix<ADCG, 3, 1> local{ADCG(local_offset[0]), ADCG(local_offset[1]), ADCG(local_offset[2])};
                Eigen::Matrix<ADCG, 3, 1> world = rotation * local + translation;

                data[data_offset + 0] = world[0];
                data[data_offset + 1] = world[1];
                data[data_offset + 2] = world[2];
                data[data_offset + 3] = ADCG(radius);
                data_offset += 4;
            }
        }

        ADFun<CGD> eef_local_spheres_func(ad_pose, data);

        CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(static_cast<std::size_t>(ad_pose.size()));
        handler.makeVariables(ind_vars);

        CppAD::vector<CGD> result = eef_local_spheres_func.Forward(0, ind_vars);

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

        std::vector<std::size_t> counts;
        counts.reserve(num_end_effectors);
        for (const auto &spheres : per_eef_spheres)
        {
            counts.push_back(spheres.size());
        }

        return EefLocalSpheres{
            Traced{function_code.str(), handler.getTemporaryVariableCount(), result.size()},
            counts};
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

    // Single-arm SE3+psi task-space parameterized IK ("param_kind": "iiwa_se3"): the
    // analytic per-pose IK plus se3_tracer.hh's generic pose+psi sample/distance/interpolate
    // kernels -- see trace_iiwa_se3_* in codegen.hh for why no robot-specific tracing is
    // needed beyond the IK itself. Populates exactly the same data[] keys
    // derive_rby1_parameterized_traces does for the shared parts of ParameterizedSpace
    // (State/Sample/StateBuffer/StateBlock, sample/distance/interpolate(_block)); the
    // rby1-only keys (param_mid_pose_fk_code, param_eef_world_poses_code,
    // param_eef_spheres_code, param_com_code, n_left/right_eef_spheres) are left unset --
    // fk_template.hh only renders the template blocks that use them under
    // `param_kind == "rby1_bimanual"`.
    auto derive_iiwa_se3_parameterized_traces(
        const RobotInfo &robot,
        nlohmann::json &data,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> void
    {
        auto param_ik = trace_iiwa_se3_ik(robot, language);
        data["param_ik_code"] = param_ik.code;
        data["param_ik_code_vars"] = param_ik.temp_variables;
        data["param_ik_code_output"] = param_ik.outputs;

        // IKParamResult::unclipped (iiwa_parameterization.hh) is always a Vector4.
        data["param_ik_num_unclipped"] = 4;

        auto param_eef_spheres = trace_eef_local_spheres(robot, language);
        data["param_eef_spheres_code"] = param_eef_spheres.traced.code;
        data["param_eef_spheres_code_vars"] = param_eef_spheres.traced.temp_variables;
        data["param_eef_spheres_code_output"] = param_eef_spheres.traced.outputs;
        if (not param_eef_spheres.counts.empty())
        {
            data["n_eef_spheres"] = param_eef_spheres.counts[0];
        }

        auto param_sample = trace_iiwa_se3_sample(robot.model, language, bounds);
        data["param_sample_code"] = param_sample.code;
        data["param_sample_code_vars"] = param_sample.temp_variables;
        data["param_sample_code_output"] = param_sample.outputs;

        auto param_distance = trace_iiwa_se3_distance(language);
        data["param_distance_code"] = param_distance.code;
        data["param_distance_code_vars"] = param_distance.temp_variables;

        auto param_interpolate = trace_iiwa_se3_interpolate(language);
        data["param_interpolate_code"] = param_interpolate.code;
        data["param_interpolate_code_vars"] = param_interpolate.temp_variables;

        auto param_interpolate_block = trace_iiwa_se3_interpolate_block(language);
        data["param_interpolate_block_code"] = param_interpolate_block.code;
        data["param_interpolate_block_code_vars"] = param_interpolate_block.temp_variables;

        // State layout (8): pose(7, [x,y,z,qx,qy,qz,qw]) + psi(1). Sample layout (7): the
        // raw [0,1) values trace_map_to_se3 maps via map_bounded/map_so3_shoemake -- see
        // se3_tracer.hh. Non-Euclidean: the orientation quaternion block sits at offset 3.
        data["param_dimension"] = 8;
        data["param_sample_dimension"] = 7;
        data["param_euclidean"] = false;
        data["param_so3_offsets"] = std::vector<std::size_t>{3};
    }

    // RBY1 constrained-bimanual parameterized IK: "use_parameterized": true traces the
    // whole-body-relative IK, the dual-hand FK used to derive t_mid_left/t_mid_right, and
    // the sample/distance/interpolate kernels over that parameterized space -- all consumed
    // by fk_template.hh's ParameterizedSpace struct (see rainbow_ik_cg.hh for the math).
    auto derive_rby1_parameterized_traces(
        const RobotInfo &robot,
        nlohmann::json &data,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> void
    {
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

        // T_mid -> left/right hand WORLD poses (translation + rotation matrix), the
        // standalone piece of RainbowConstrainedBimanualIkCG that eefs_in_collision needs --
        // see RainbowEefWorldPosesFromMidCG in rainbow_ik_cg.hh.
        auto param_eef_world_poses = trace_rby1_eef_world_poses_from_mid(language);
        data["param_eef_world_poses_code"] = param_eef_world_poses.code;
        data["param_eef_world_poses_code_vars"] = param_eef_world_poses.temp_variables;
        data["param_eef_world_poses_code_output"] = param_eef_world_poses.outputs;

        // Per-end-effector local spheres (gripper/finger geometry) for eefs_in_collision's
        // no-attachment case -- see trace_eef_local_spheres above. Generic
        // over robot.end_effector_names, but eefs_in_collision itself is currently only
        // emitted for the bimanual (num_end_effectors == 2) case, where end_effector_names is
        // [ee_left, ee_right] -- counts[0]/[1] are that pair's sphere counts, in that order.
        auto param_eef_spheres = trace_eef_local_spheres(robot, language);
        data["param_eef_spheres_code"] = param_eef_spheres.traced.code;
        data["param_eef_spheres_code_vars"] = param_eef_spheres.traced.temp_variables;
        data["param_eef_spheres_code_output"] = param_eef_spheres.traced.outputs;
        if (param_eef_spheres.counts.size() >= 2)
        {
            data["n_left_eef_spheres"] = param_eef_spheres.counts[0];
            data["n_right_eef_spheres"] = param_eef_spheres.counts[1];
        }

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

    // Bimanual iiwa leader-follower IK ("param_kind": "iiwa_bimanual"): the leader/left arm's
    // joint angles pass straight through the analytic IK (see IiwaBimanualParameterization in
    // iiwa_parameterization.hh), so State is just q(7) + psi(1) -- fully Euclidean, no task-
    // space pose to sample. This is fk_template.hh's LeaderFollowerSpace, not
    // ParameterizedSpace: unlike derive_rby1_parameterized_traces's t_mid (a sampled
    // task-space pose that both arms are IK'd from), the leader here is sampled directly in
    // its own joint space and only the follower/right arm is expressed relatively, via psi +
    // the FIXED `rel_pose` class member (LeaderFollowerSpace::compute_rel_pose -- a single
    // fixed hand-to-hand offset, not a midpoint, and there's no eef-world-poses-from-
    // mid/eefs_collision_free prefilter for this kind). The leader arm's own sample range
    // (trace_iiwa_bimanual_sample) is read directly off `robot.model`'s q-indices [0:7) -- the
    // bimanual URDF places the "iiwa_left" chain's 7 revolute joints first in kinematic-tree
    // order, matching IiwaBimanualParameterization's q_full.head(7) == q_controlled
    // convention, so those are the leader arm's own joint limits.
    auto derive_iiwa_bimanual_leader_follower_traces(
        const RobotInfo &robot,
        nlohmann::json &data,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> void
    {
        (void)bounds;  // no task-space pose in this State -- nothing here needs Cartesian bounds.

        auto lf_ik = trace_iiwa_bimanual_ik(robot, language);
        data["lf_ik_code"] = lf_ik.code;
        data["lf_ik_code_vars"] = lf_ik.temp_variables;
        data["lf_ik_code_output"] = lf_ik.outputs;

        // IKParamResult::unclipped (iiwa_parameterization.hh) is always a Vector4.
        data["lf_ik_num_unclipped"] = 4;

        auto lf_rel_pose_fk = trace_iiwa_bimanual_rel_pose_fk(robot, language);
        data["lf_rel_pose_fk_code"] = lf_rel_pose_fk.code;
        data["lf_rel_pose_fk_code_vars"] = lf_rel_pose_fk.temp_variables;
        data["lf_rel_pose_fk_code_output"] = lf_rel_pose_fk.outputs;

        auto lf_sample = trace_iiwa_bimanual_sample(robot.model, language);
        data["lf_sample_code"] = lf_sample.code;
        data["lf_sample_code_vars"] = lf_sample.temp_variables;
        data["lf_sample_code_output"] = lf_sample.outputs;

        auto lf_distance = trace_iiwa_bimanual_distance(language);
        data["lf_distance_code"] = lf_distance.code;
        data["lf_distance_code_vars"] = lf_distance.temp_variables;

        auto lf_interpolate = trace_iiwa_bimanual_interpolate(language);
        data["lf_interpolate_code"] = lf_interpolate.code;
        data["lf_interpolate_code_vars"] = lf_interpolate.temp_variables;

        auto lf_interpolate_block = trace_iiwa_bimanual_interpolate_block(language);
        data["lf_interpolate_block_code"] = lf_interpolate_block.code;
        data["lf_interpolate_block_code_vars"] = lf_interpolate_block.temp_variables;

        // State layout (8): q(7, leader/left arm joint angles) + psi(1, follower/right arm's
        // self-motion-manifold parameter). Sample layout (8): the raw [0,1) values
        // trace_iiwa_bimanual_sample maps via map_bounded -- 1:1, no SO3 expansion. Fully
        // Euclidean: no orientation block in State at all (rel_pose is fixed, not sampled).
        data["lf_dimension"] = 8;
        data["lf_sample_dimension"] = 8;
        data["lf_euclidean"] = true;
        data["lf_so3_offsets"] = std::vector<std::size_t>{};
    }

    // Bimanual iiwa "mid-pose" parameterized IK ("param_kind": "iiwa_bimanual", rendered
    // alongside derive_iiwa_bimanual_leader_follower_traces above): populates
    // fk_template.hh's ParameterizedSpace the same way derive_rby1_parameterized_traces does
    // (t_mid_left/t_mid_right, compute_mid_pose, eef_world_poses/eefs_collision_free,
    // sample/distance/interpolate(_block)) but without RBY1's base/torso/COM pieces -- see
    // IiwaBimanualMidParameterizationCG's header comment (iiwa_parameterization_gen.hh) for
    // the IK math and trace_bimanual_mid_sample/_distance/_interpolate(_block)
    // (iiwa_bimanual_tracer.hh) for the State kernels. State layout (9): T_mid pose(7,
    // [x,y,z,qx,qy,qz,qw]) + psi_left(1) + psi_right(1) -- same shape as se3_tracer.hh's
    // pose+psi Space, just two psis instead of one. Non-Euclidean: T_mid's orientation
    // quaternion sits at offset 3 (so3_offsets == {3}, same offset iiwa_se3 uses for its own
    // single pose block).
    auto derive_iiwa_bimanual_mid_parameterized_traces(
        const RobotInfo &robot,
        nlohmann::json &data,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> void
    {
        auto param_ik = trace_iiwa_bimanual_mid_ik(robot, language);
        data["param_ik_code"] = param_ik.code;
        data["param_ik_code_vars"] = param_ik.temp_variables;
        data["param_ik_code_output"] = param_ik.outputs;

        // IKParamResult::unclipped (iiwa_parameterization.hh) is always a Vector4, per arm.
        data["param_ik_num_unclipped"] = 4;

        // trace_iiwa_bimanual_rel_pose_fk's dual-eef FK is exactly what compute_mid_pose
        // needs (leader/left, follower/right world poses) -- reused verbatim, same as
        // derive_iiwa_bimanual_leader_follower_traces's lf_rel_pose_fk_code above.
        auto param_mid_pose_fk = trace_iiwa_bimanual_rel_pose_fk(robot, language);
        data["param_mid_pose_fk_code"] = param_mid_pose_fk.code;
        data["param_mid_pose_fk_code_vars"] = param_mid_pose_fk.temp_variables;
        data["param_mid_pose_fk_code_output"] = param_mid_pose_fk.outputs;

        auto param_eef_world_poses = trace_iiwa_bimanual_eef_world_poses_from_mid(language);
        data["param_eef_world_poses_code"] = param_eef_world_poses.code;
        data["param_eef_world_poses_code_vars"] = param_eef_world_poses.temp_variables;
        data["param_eef_world_poses_code_output"] = param_eef_world_poses.outputs;

        auto param_eef_spheres = trace_eef_local_spheres(robot, language);
        data["param_eef_spheres_code"] = param_eef_spheres.traced.code;
        data["param_eef_spheres_code_vars"] = param_eef_spheres.traced.temp_variables;
        data["param_eef_spheres_code_output"] = param_eef_spheres.traced.outputs;
        if (param_eef_spheres.counts.size() >= 2)
        {
            data["n_left_eef_spheres"] = param_eef_spheres.counts[0];
            data["n_right_eef_spheres"] = param_eef_spheres.counts[1];
        }

        auto param_sample = trace_iiwa_bimanual_mid_sample(language, bounds);
        data["param_sample_code"] = param_sample.code;
        data["param_sample_code_vars"] = param_sample.temp_variables;
        data["param_sample_code_output"] = param_sample.outputs;

        auto param_distance = trace_iiwa_bimanual_mid_distance(language);
        data["param_distance_code"] = param_distance.code;
        data["param_distance_code_vars"] = param_distance.temp_variables;

        auto param_interpolate = trace_iiwa_bimanual_mid_interpolate(language);
        data["param_interpolate_code"] = param_interpolate.code;
        data["param_interpolate_code_vars"] = param_interpolate.temp_variables;

        auto param_interpolate_block = trace_iiwa_bimanual_mid_interpolate_block(language);
        data["param_interpolate_block_code"] = param_interpolate_block.code;
        data["param_interpolate_block_code_vars"] = param_interpolate_block.temp_variables;

        data["param_dimension"] = 9;
        data["param_sample_dimension"] = 8;
        data["param_euclidean"] = false;
        data["param_so3_offsets"] = std::vector<std::size_t>{3};
    }

    // Dispatches to the robot-specific parameterized-IK tracing above when the recipe has
    // "use_parameterized": true, setting the has_parameterized_space/has_leader_follower_space
    // template gates either way. "param_kind" ("rby1_bimanual", the default -- matching every
    // existing recipe that predates this key -- or "iiwa_se3") selects
    // fk_template.hh's ParameterizedSpace (a sampled task-space pose that IK resolves both/the
    // one arm from); "iiwa_bimanual" instead selects LeaderFollowerSpace (the leader arm
    // sampled directly in its own joint space, the follower solved relative to a fixed
    // offset) -- see derive_iiwa_bimanual_leader_follower_traces above for why that doesn't
    // fit ParameterizedSpace's shape. Shared by the offline generator (fkcc_gen) and the JIT
    // path (generate_robot_source) so both accept the same keys.
    auto derive_parameterized_traces(
        const RobotInfo &robot,
        nlohmann::json &data,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> void
    {
        const bool use_parameterized = data.value("use_parameterized", false);
        if (not use_parameterized)
        {
            data["has_parameterized_space"] = false;
            data["has_leader_follower_space"] = false;
            return;
        }

        const std::string param_kind = data.value("param_kind", std::string("rby1_bimanual"));
        data["param_kind"] = param_kind;
        data["joint_limit_margin"] = data.value("joint_limit_margin", 0.0);

        if (param_kind == "iiwa_se3")
        {
            data["has_parameterized_space"] = true;
            data["has_leader_follower_space"] = false;
            derive_iiwa_se3_parameterized_traces(robot, data, language, bounds);
        }
        else if (param_kind == "rby1_bimanual")
        {
            data["has_parameterized_space"] = true;
            data["has_leader_follower_space"] = false;
            derive_rby1_parameterized_traces(robot, data, language, bounds);
        }
        else if (param_kind == "iiwa_bimanual")
        {
            data["has_parameterized_space"] = true;
            data["has_leader_follower_space"] = true;
            derive_iiwa_bimanual_leader_follower_traces(robot, data, language, bounds);
            derive_iiwa_bimanual_mid_parameterized_traces(robot, data, language, bounds);
        }
        else
        {
            throw std::runtime_error(
                fmt::format("derive_parameterized_traces: unknown \"param_kind\" \"{}\"", param_kind));
        }
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
            "param_kind",
            "joint_limit_margin",
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

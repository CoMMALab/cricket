#include <cricket/codegen.hh>
#include <cricket/embedded_templates.hh>

#include "codegen/pinocchio_cppadcg.hh"
#include "codegen/lang_cpp.hh"
#include "codegen/lang_rust.hh"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <fmt/format.h>
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>

#include <cctype>
#include <sstream>
#include <stdexcept>

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

        // Traces a box as [center (3), axis 1 (3), axis 2 (3), axis 3 (3), half-lengths (3)], with the
        // axes being the columns of the box's world rotation. The half-lengths are emitted as outputs
        // (like sphere radii) so every consumer reads the box uniformly from one buffer.
        auto trace_cuboid(const CuboidInfo &cuboid, const ADData &ad_data, ADVectorXs &data, std::size_t index)
        {
            const auto &joint_placement = ad_data.oMi[cuboid.parent_joint];

            Eigen::Matrix<ADCG, 3, 1> local_translation;
            Eigen::Matrix<ADCG, 3, 3> local_rotation;
            for (auto i = 0; i < 3; ++i)
            {
                local_translation[i] = cuboid.relative.translation()[i];
                for (auto j = 0; j < 3; ++j)
                {
                    local_rotation(i, j) = cuboid.relative.rotation()(i, j);
                }
            }

            const Eigen::Matrix<ADCG, 3, 1> world_position =
                joint_placement.rotation() * local_translation + joint_placement.translation();
            const Eigen::Matrix<ADCG, 3, 3> world_rotation = joint_placement.rotation() * local_rotation;

            for (auto i = 0; i < 3; ++i)
            {
                data[index + i] = world_position[i];

                // Column j of the rotation is the box's axis j
                for (auto j = 0; j < 3; ++j)
                {
                    data[index + 3 + 3 * j + i] = world_rotation(i, j);
                }

                data[index + 12 + i] = ADCG(cuboid.half_extents[i]);
            }
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
        bool fk,
        bool cuboids) -> Traced
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
        std::size_t n_cuboids_data = (cuboids) ? info.cuboids.size() * cuboid_stride : 0;
        std::size_t n_fk_data = (fk) ? 12 * info.end_effector_indexes.size() : 0;
        std::size_t n_out = n_spheres_data + n_bounding_spheres_data + n_cuboids_data + n_fk_data;

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

        if (cuboids)
        {
            for (const auto &cuboid : info.cuboids)
            {
                trace_cuboid(
                    cuboid,
                    ad_data,
                    data,
                    n_spheres_data + n_bounding_spheres_data + cuboid.index * cuboid_stride);
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
                    n_spheres_data + n_bounding_spheres_data + n_cuboids_data + i * 12);
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

        // The bounded least-squares projection solver is built from conditionals (SIMD blends), which
        // the Rust backend cannot emit, so it is generated for C++ only.
        const bool bounded_ls = language != "rust";
        data["has_bounded_ls"] = bounded_ls;

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
            if (bounded_ls)
            {
                set_trace(
                    trace_solve_tsr(robot, language, ProjMethod::BoundedLS),
                    "solve_tsr_error_bounded_ls");
            }

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
                if (bounded_ls)
                {
                    set_trace(
                        trace_solve_tsr(robot, language, ProjMethod::BoundedLS, true),
                        "solve_tsr_relative_error_bounded_ls");
                }
            }
        }

        // Straight-line (lerp) curve constraints, for planning along a curve with the curve evaluated at t inside the
        // error kernel and t held fixed in the projection ("lerp_r3_constraint": the first end-effector's position on a
        // polyline in R^3; "lerp_so3r3_constraint": its pose on a polyline of poses in SO(3) x R^3, the rotations
        // slerped, with the weight of the rotation error a runtime input). The curve has cricket::lerp_points points (codegen.hh), a
        // constant of the generated header; the points and their parameters are inputs. Each kernel comes with the
        // projection solvers for its error (3 rows, 6 rows). The segment of t is picked by conditionals, so these are
        // for C++ only.
        const bool lerp_r3_constraint = data.value("lerp_r3_constraint", false);
        const bool lerp_so3r3_constraint = data.value("lerp_so3r3_constraint", false);

        // The same curves with t an unknown of the projection ("lerp_r3_segment_constraint", "lerp_so3r3_segment_constraint"):
        // the end-effector moves along the segment t starts on. The kernel has t as an extra column of the Jacobian and
        // hands over the allowed step of t to the segment's ends; the solvers have the joints and t as unknowns.
        const bool lerp_r3_segment_constraint = data.value("lerp_r3_segment_constraint", false);
        const bool lerp_so3r3_segment_constraint = data.value("lerp_so3r3_segment_constraint", false);
        if ((lerp_r3_constraint or lerp_so3r3_constraint or lerp_r3_segment_constraint or lerp_so3r3_segment_constraint) and
            language == "rust")
        {
            throw std::runtime_error(
                "the lerp constraints need conditionals, which the rust language cannot emit");
        }

        data["lerp_points"] = lerp_points;
        data["has_lerp_r3_constraint"] = lerp_r3_constraint;
        if (lerp_r3_constraint)
        {
            set_trace(trace_lerp_r3_error<lerp_points>(robot, language), "lerp_r3_error");
            set_trace(trace_solve_jacobian(robot, language, ProjMethod::InnerLM, 3), "solve_lerp_r3_error_lm_inner");
            set_trace(trace_solve_jacobian(robot, language, ProjMethod::OuterLM, 3), "solve_lerp_r3_error_lm_outer");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::GradDesc, 3), "solve_lerp_r3_error_gradient_descent");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::BoundedLS, 3), "solve_lerp_r3_error_bounded_ls");
        }

        data["has_lerp_so3r3_constraint"] = lerp_so3r3_constraint;
        if (lerp_so3r3_constraint)
        {
            set_trace(trace_lerp_so3r3_error<lerp_points>(robot, language), "lerp_so3r3_error");
            set_trace(trace_solve_jacobian(robot, language, ProjMethod::InnerLM, 6), "solve_lerp_so3r3_error_lm_inner");
            set_trace(trace_solve_jacobian(robot, language, ProjMethod::OuterLM, 6), "solve_lerp_so3r3_error_lm_outer");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::GradDesc, 6),
                "solve_lerp_so3r3_error_gradient_descent");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::BoundedLS, 6), "solve_lerp_so3r3_error_bounded_ls");
        }

        // The pose curve of the other end-effectors ("lerp_end_effectors": [0, 1, ...], indexes into "end_effector"; 0, the
        // default, is the first and is what lerp_so3r3_error is): the error kernel of end-effector k is lerp_so3r3_error_ek, the
        // solvers are the same (they only see the 6-row error and its Jacobian), and lerp_so3r3_error_for(k, ...) picks one at run time.
        nlohmann::json lerp_extra_end_effectors = nlohmann::json::array();
        if (lerp_so3r3_constraint and data.contains("lerp_end_effectors"))
        {
            for (const auto &entry : data["lerp_end_effectors"])
            {
                const auto k = entry.get<std::size_t>();
                if (k >= robot.end_effector_indexes.size())
                {
                    throw std::runtime_error(fmt::format(
                        "\"lerp_end_effectors\": end-effector {} does not exist (the recipe has {})",
                        k,
                        robot.end_effector_indexes.size()));
                }

                if (k == 0)
                {
                    continue;
                }

                auto robot_k = robot;  // the same robot with end-effector k first, which is the one the kernel traces
                std::swap(robot_k.end_effector_indexes[0], robot_k.end_effector_indexes[k]);
                const auto traced = trace_lerp_so3r3_error<lerp_points>(robot_k, language);
                lerp_extra_end_effectors.push_back(
                    {{"index", k},
                     {"suffix", fmt::format("_ee{}", k)},
                     {"error_code", traced.code},
                     {"error_code_vars", traced.temp_variables},
                     {"error_code_output", traced.outputs}});
            }
        }

        data["lerp_so3r3_extra_ees"] = lerp_extra_end_effectors;

        data["has_lerp_r3_segment_constraint"] = lerp_r3_segment_constraint;
        if (lerp_r3_segment_constraint)
        {
            set_trace(trace_lerp_r3_segment_error<lerp_points>(robot, language), "lerp_r3_segment_error");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::InnerLM, 3, 1),
                "solve_lerp_r3_segment_error_lm_inner");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::OuterLM, 3, 1),
                "solve_lerp_r3_segment_error_lm_outer");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::GradDesc, 3, 1),
                "solve_lerp_r3_segment_error_gradient_descent");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::BoundedLS, 3, 1),
                "solve_lerp_r3_segment_error_bounded_ls");
        }

        data["has_lerp_so3r3_segment_constraint"] = lerp_so3r3_segment_constraint;
        if (lerp_so3r3_segment_constraint)
        {
            set_trace(trace_lerp_so3r3_segment_error<lerp_points>(robot, language), "lerp_so3r3_segment_error");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::InnerLM, 6, 1),
                "solve_lerp_so3r3_segment_error_lm_inner");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::OuterLM, 6, 1),
                "solve_lerp_so3r3_segment_error_lm_outer");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::GradDesc, 6, 1),
                "solve_lerp_so3r3_segment_error_gradient_descent");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::BoundedLS, 6, 1),
                "solve_lerp_so3r3_segment_error_bounded_ls");
        }

        // Center-of-mass kinematics. "com": true is the world-frame CoM. An object selects the frame:
        //   {"frame": "world"}                                 world-frame CoM (the default frame)
        //   {"frame": "feet", "reference_frames": [f1, f2]}    CoM minus the mean position of the reference
        //                                                      frames (e.g. the feet), in world axes
        //   {"frame": "body", "body": "link"}                  CoM in the axes of a body frame, R^T (com - p): a support
        //                                                      polygon fixed to a mobile base, in the base's own frame
        // "reference_frames" is only valid, and required, with "frame": "feet"; "body" with "frame": "body".
        // The support-polygon error consuming these is 2D (xy), hence err_size 2 solvers.
        bool has_com = false;
        std::vector<std::string> com_reference_frames;
        std::string com_body_frame;
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
                const auto frame = cm.value("frame", std::string("world"));
                if (frame == "feet")
                {
                    if (not cm.contains("reference_frames") or cm["reference_frames"].empty())
                    {
                        throw std::runtime_error("\"com\": {\"frame\": \"feet\"} requires non-empty \"reference_frames\"");
                    }

                    com_reference_frames = cm["reference_frames"].get<std::vector<std::string>>();
                }
                else if (frame == "body")
                {
                    if (not cm.contains("body") or not cm["body"].is_string() or cm["body"].get<std::string>().empty())
                    {
                        throw std::runtime_error("\"com\": {\"frame\": \"body\"} requires \"body\", the name of a frame");
                    }

                    if (cm.contains("reference_frames"))
                    {
                        throw std::runtime_error("\"com\": \"reference_frames\" is only valid with \"frame\": \"feet\"");
                    }

                    com_body_frame = cm["body"].get<std::string>();
                }
                else if (frame == "world")
                {
                    if (cm.contains("reference_frames"))
                    {
                        throw std::runtime_error(
                            "\"com\": \"reference_frames\" is only valid with \"frame\": \"feet\" (the default frame is \"world\")");
                    }
                }
                else
                {
                    throw std::runtime_error("\"com\": \"frame\" must be \"world\", \"feet\" or \"body\", got \"" + frame + "\"");
                }
            }
        }

        data["has_com"] = has_com;
        if (has_com)
        {
            data["com_reference_frames"] = com_reference_frames;
            data["com_body_frame"] = com_body_frame;
            set_trace(trace_com_jacobian(robot, com_reference_frames, language, com_body_frame), "com_jacobian");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::InnerLM, 2),
                "solve_com_error_lm_inner");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::OuterLM, 2),
                "solve_com_error_lm_outer");
            set_trace(
                trace_solve_jacobian(robot, language, ProjMethod::GradDesc, 2),
                "solve_com_error_gradient_descent");
            if (bounded_ls)
            {
                set_trace(
                    trace_solve_jacobian(robot, language, ProjMethod::BoundedLS, 2),
                    "solve_com_error_bounded_ls");
            }
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
            if (bounded_ls)
            {
                set_trace(
                    trace_solve_jacobian(robot, language, ProjMethod::BoundedLS, loops.size()),
                    "solve_closed_loop_error_bounded_ls");
            }
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
            if (bounded_ls)
            {
                set_trace(
                    trace_solve_jacobian(robot, language, ProjMethod::BoundedLS, 1),
                    "solve_lead_screw_error_bounded_ls");
            }
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

        RobotInfo robot(opts.urdf, opts.srdf, opts.end_effectors);

        nlohmann::json data = opts.data;
        const bool compact_collisions = opts.data.value("compact_collisions", false);
        if (compact_collisions and not robot.cuboids.empty())
        {
            throw std::runtime_error("compact_collisions is not supported for robots with cuboid collision geometry");
        }

        data.update(robot.json(opts.bounds, opts.data.value("skip_static_environment_collisions", false)));
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
        auto eefk = trace_sphere_cc_fk(robot, opts.language, false, false, true);
        data["eefk_code"] = eefk.code;
        data["eefk_code_vars"] = eefk.temp_variables;
        data["eefk_code_output"] = eefk.outputs;

        auto eejac = trace_ee_fk_jacobian(robot, opts.language);
        data["eejac_code"] = eejac.code;
        data["eejac_code_vars"] = eejac.temp_variables;
        data["eejac_code_output"] = eejac.outputs;

        if (opts.language != "c++" and not robot.cuboids.empty())
        {
            throw std::runtime_error(
                fmt::format("Cuboid collision geometry is only supported for c++, not {}", opts.language));
        }

        auto spherefk = trace_sphere_cc_fk(robot, opts.language, true, false, false);
        data["spherefk_code"] = spherefk.code;
        data["spherefk_code_vars"] = spherefk.temp_variables;
        data["spherefk_code_output"] = spherefk.outputs;

        // Only traced when there are cuboids; robots without them produce exactly the same code as before
        if (not robot.cuboids.empty())
        {
            auto cuboidfk = trace_sphere_cc_fk(robot, opts.language, false, false, false, true);
            data["cuboidfk_code"] = cuboidfk.code;
            data["cuboidfk_code_vars"] = cuboidfk.temp_variables;
            data["cuboidfk_code_output"] = cuboidfk.outputs;
        }

        auto ccfk = trace_sphere_cc_fk(robot, opts.language, true, true, false, true);
        data["ccfk_code"] = ccfk.code;
        data["ccfk_code_vars"] = ccfk.temp_variables;
        data["ccfk_code_output"] = ccfk.outputs;

        auto ccfkee = trace_sphere_cc_fk(robot, opts.language, true, true, true, true);
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

        if (result.data.contains("n_cuboids"))
        {
            result.n_cuboids = result.data["n_cuboids"].get<std::size_t>();
        }

        return result;
    }
}  // namespace cricket

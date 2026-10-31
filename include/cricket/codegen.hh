#pragma once

#include <cricket/robot_info.hh>

#include <pinocchio/multibody/model.hpp>

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cricket
{
    // The number of points of the lerp curves. The kernels are traced for exactly this many (a template argument, so a
    // compile-time constant), and it becomes Robot::lerp_points of the generated header: change it here only.
    inline constexpr std::size_t lerp_points = 32;

    struct Traced
    {
        std::string code;
        std::size_t temp_variables;
        std::size_t outputs;
    };

    auto trace_sphere_cc_fk(
        const RobotInfo &info,
        const std::string &language,
        bool spheres = true,
        bool bounding_spheres = true,
        bool fk = true) -> Traced;

    /// Traces the end-effector pose together with its Jacobian.
    ///
    /// Emits `12 + 6 * nv` outputs: three translation components, then the rotation matrix in
    /// column-major order, then the 6 x nv Jacobian in row-major order.
    /// The first three Jacobian rows map joint velocity to the linear velocity of the end-effector
    /// origin, and the last three map it to angular velocity.
    /// Both are resolved in world axes, not end-effector axes, so the twist the Jacobian acts on is
    /// built from a world-frame position difference and a world-frame rotation difference.
    auto trace_ee_fk_jacobian(const RobotInfo &info, const std::string &language) -> Traced;

    auto trace_map_to_configuration(
        const pinocchio::Model &model,
        const std::string &language,
        const std::optional<Bounds> &bounds = std::nullopt) -> Traced;

    auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced;
    auto trace_interpolate_block(const pinocchio::Model &model, const std::string &language) -> Traced;
    auto trace_distance(const pinocchio::Model &model, const std::string &language) -> Traced;
    auto trace_forward_dynamics(const pinocchio::Model &model, const std::string &language) -> Traced;
    auto trace_integrate_configuration(const pinocchio::Model &model, const std::string &language)
        -> Traced;

    // Least-squares step used to project a configuration onto a constraint manifold.
    enum class ProjMethod
    {
        InnerLM,   // J^T (J J^T + lambda I)^{-1} e: (6 n_eef)^2 factorization
        OuterLM,   // (J^T J + lambda I)^{-1} J^T e: nq^2 factorization
        GradDesc,  // J^T e
        // Inner-form LM with the joint limits as constraints: min 1/2 |J dq + e|^2 + 1/2 lambda |dq|^2
        // s.t. lower - q <= dq <= upper - q, by an unrolled active-set solve (see tracing/bounded_ls.hh).
        // Takes two extra inputs (the allowed step to each limit) and relies on conditionals, so it is
        // generated only for the C++ language (as SIMD blends), never for Rust.
        BoundedLS,
    };

    // TSR (task-space region) error for every end-effector of the robot.
    // Input: q (nq), then per end-effector [rTe (7), wTr (7), lb (6), ub (6)] with transforms
    // as wxyz quaternion + xyz translation. Output: d(err)/dq (6 n_eef x nq, row-major), then
    // the raw error (6 n_eef); bounds are hinged at runtime, not on the tape.
    auto trace_tsr_error(const RobotInfo &info, const std::string &language) -> Traced;

    // Position error of the first end-effector against the straight-line (lerp) curve through N points, with the
    // curve evaluated at the parameter t in the kernel (the segment picked by conditionals, so the language must be
    // C++). Input: q (nq), t, the N points (3 N, x y z per point), their parameters tau (N); see LerpR3InputLayout in
    // tracing/lerp.hh. Output: d(err)/dq (3 x nq, row-major, t is not differentiated), then the raw error (3).
    template <std::size_t N>
    auto trace_lerp_r3_error(const RobotInfo &info, const std::string &language) -> Traced;

    // Pose error of the first end-effector against the SO(3) x R^3 lerp curve through N keyframes (positions lerped,
    // rotations slerped), as the TSR error [R_t^T (p - p_t); log(R_t^T R)] with the rotation rows scaled by a weight
    // input. Input: q (nq), t, weight, the N positions (3 N), the N quaternions (4 N, wxyz), the parameters tau (N);
    // see LerpPoseInputLayout. Output: d(err)/dq (6 x nq, row-major), then the raw error (6).
    template <std::size_t N>
    auto trace_lerp_so3r3_error(const RobotInfo &info, const std::string &language) -> Traced;

    // The same lerp kernels with the curve parameter t an unknown of the projection next to the joints ("segment"): the segment
    // is the one the extra input t_segment is on, and t is the parameter within it, so the end-effector may move along the
    // segment. Input: as the fixed kernels, with t_segment after t (see the Segment layouts in tracing/lerp.hh).
    // Output: d(err)/d(q, t) (m x (nq + 1), row-major, t the last column), then the raw error (m), then the allowed step of
    // t to the start and to the end of the segment (tau_i - t, tau_{i + 1} - t).
    template <std::size_t N>
    auto trace_lerp_r3_segment_error(const RobotInfo &info, const std::string &language) -> Traced;

    template <std::size_t N>
    auto trace_lerp_so3r3_segment_error(const RobotInfo &info, const std::string &language) -> Traced;

    // Relative-pose (bimanual) TSR error between two end-effectors.
    // Input: q (nq), then the reference relative transform lTr (7), lb (6), ub (6).
    // Output: d(err)/dq (6 x nq, row-major), then the raw error (6).
    auto trace_tsr_bimanual_error(
        const RobotInfo &info,
        const std::string &language,
        std::size_t eef1 = 0,
        std::size_t eef2 = 1) -> Traced;

    // Projection step from a TSR error and Jacobian; `relative` selects the 6-row bimanual
    // error instead of the 6 n_eef-row per-end-effector error.
    // Input: J (row-major), then err (and lower - q, upper - q for ProjMethod::BoundedLS).
    // Output: gradient (nq).
    auto trace_solve_tsr(
        const RobotInfo &info,
        const std::string &language,
        ProjMethod method,
        bool relative = false) -> Traced;

    // Projection step from an arbitrary stacked error and Jacobian with err_size rows. The unknowns are the joints and
    // `extra_vars` more (the curve parameter of a segment projection), n = nq + extra_vars in all.
    // Input: J (err_size x n, row-major), then err (err_size); for ProjMethod::BoundedLS also
    // lower - x (n) and upper - x (n), x the joints followed by the extra variables. Output: gradient (n).
    auto trace_solve_jacobian(
        const RobotInfo &info,
        const std::string &language,
        ProjMethod method,
        std::size_t err_size,
        std::size_t extra_vars = 0) -> Traced;

    // Center-of-mass position and Jacobian, optionally expressed relative to the mean
    // position of a set of reference (body) frames, e.g. the feet of a standing humanoid so
    // that a support polygon can be stated in the stance frame.
    // Input: q (nq). Output: d(com)/dq (3 x nq, row-major), then com (3).
    auto trace_com_jacobian(
        const RobotInfo &info,
        const std::vector<std::string> &reference_frames,
        const std::string &language) -> Traced;

    // Loop-closure distance constraint: the distance between two (body) frames must equal a
    // fixed length, e.g. a rigid rod cut from a closed kinematic chain.
    struct ClosedLoop
    {
        std::string start_frame;
        std::string end_frame;
        double length;
    };

    // Input: q (nq). Output: d(err)/dq (n_loops x nq, row-major), then err (n_loops).
    auto trace_closed_loop_error(
        const RobotInfo &info,
        const std::vector<ClosedLoop> &loops,
        const std::string &language) -> Traced;

    // Lead-screw coupling of the first end-effector: axial advance along the reference
    // frame's z-axis locked to rotation about it. h(q) is the conserved quantity of the
    // coupling and dh/dq its Pfaffian row (the velocity form is dh/dq qdot = 0).
    // Input: q (nq), then rTe (7), wTr (7), pitch (1) with transforms as wxyz quaternion +
    // xyz translation. Output: d(h)/dq (1 x nq, row-major), then h (1).
    auto trace_lead_screw_error(const RobotInfo &info, const std::string &language) -> Traced;

    // Twist Jacobians of the first end-effector's offset frame, for constant-coefficient
    // Pfaffian velocity constraints c_ref^T twist_ref + c_loc^T twist_loc = 0 whose rows
    // are combined at runtime in vamp. Input: q (nq), then rTe (7), wTr (7); transforms
    // are wxyz quaternion + xyz translation. Output (12 x nq, row-major): the twist
    // Jacobian [linear; angular] of the offset frame (eef * rTe^-1) expressed in the
    // reference frame wTr's axes, then the same expressed in the offset frame's own
    // (body) axes. Purely geometric (no log map), so rows are smooth for unbounded
    // rotation.
    auto trace_twist_jacobians(const RobotInfo &info, const std::string &language) -> Traced;

    // Derive every JSON-gated constraint kernel (TSR, bimanual TSR, center-of-mass,
    // closed loops, lead screw, twist Jacobians) from the recipe keys already present in
    // `data` ("constraints", "com", "closed_loops", "lead_screw", "twist"), tracing the
    // kernels and setting the has_* template gates. Shared by the offline generator
    // (fkcc_gen) and the JIT path (generate_robot_source) so both accept the same keys.
    auto derive_constraint_traces(const RobotInfo &robot, nlohmann::json &data, const std::string &language)
        -> void;

    // Strict recipe validation: throws if `data` contains keys cricket does not read
    // (typos otherwise fail silently, e.g. a misspelled flag defaulting to false), with a
    // nearest-match suggestion. Keys starting with '_' are ignored as comments. Checks
    // nested blocks (bounds, com, closed_loops, parts, subtemplates) too.
    auto validate_recipe(const nlohmann::json &data) -> void;

    struct GenOptions
    {
        std::filesystem::path urdf;
        std::optional<std::filesystem::path> dynamics_urdf;
        std::optional<std::filesystem::path> srdf;
        std::vector<std::string> end_effectors;
        std::filesystem::path template_path;
        std::map<std::string, std::filesystem::path> subtemplates;
        std::string language = "c++";
        std::optional<Bounds> bounds;
        bool forward_dynamics = false;
        nlohmann::json data;
    };

    struct GenResult
    {
        std::string source;
        nlohmann::json data;
        std::string robot_name;
        std::size_t dimension = 0;
        std::size_t n_spheres = 0;
    };

    auto generate_robot_source(const GenOptions &opts) -> GenResult;
}  // namespace cricket

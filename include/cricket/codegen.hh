#pragma once

#include <cricket/robot_info.hh>

#include <pinocchio/multibody/model.hpp>

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cricket
{
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

    auto trace_map_to_configuration(
        const pinocchio::Model &model,
        const std::string &language,
        const std::optional<Bounds> &bounds = std::nullopt) -> Traced;

    auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced;
    auto trace_interpolate_block(const pinocchio::Model &model, const std::string &language) -> Traced;
    auto trace_distance(const pinocchio::Model &model, const std::string &language) -> Traced;

    // FLASK (r = 2): minimum-acceleration cubic between flat states a = (y0, yd0), b = (yf, ydf)
    // over duration T evaluated at fraction t; outputs 3n rows [y; yd; ydd].
    auto trace_flask_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced;
    auto trace_flask_interpolate_block(const pinocchio::Model &model, const std::string &language)
        -> Traced;

    // Inverse dynamics via RNEA; input x = [q; v; a] (3n rows) plus the end-effector payloads
    // (10 per end-effector), outputs n torques.
    auto trace_flask_rnea(const RobotInfo &info, const std::string &language) -> Traced;
    auto trace_flask_rnea_block(const RobotInfo &info, const std::string &language) -> Traced;

    // Kinetic energy (1/2) v^T M(q) v; input x = [q; v] (2n rows) plus the end-effector
    // payloads (10 per end-effector), 1 output. Requires link masses in the URDF.
    auto trace_flask_kinetic_energy(const RobotInfo &info, const std::string &language) -> Traced;
    auto trace_flask_kinetic_energy_block(const RobotInfo &info, const std::string &language) -> Traced;

    // World-aligned linear velocity of every end-effector origin; input x = [q; v] (2n rows),
    // 3 outputs per end-effector.
    auto trace_flask_eef_velocity(const RobotInfo &info, const std::string &language) -> Traced;
    auto trace_flask_eef_velocity_block(const RobotInfo &info, const std::string &language) -> Traced;

    // Least-squares step used to project a configuration onto a constraint manifold.
    enum class ProjMethod
    {
        InnerLM,   // J^T (J J^T + lambda I)^{-1} e: (6 n_eef)^2 factorization
        OuterLM,   // (J^T J + lambda I)^{-1} J^T e: nq^2 factorization
        GradDesc,  // J^T e
    };

    // TSR (task-space region) error for every end-effector of the robot.
    // Input: q (nq), then per end-effector [rTe (7), wTr (7), lb (6), ub (6)] with transforms
    // as wxyz quaternion + xyz translation. Output: d(err)/dq (6 n_eef x nq, row-major), then
    // the raw error (6 n_eef); bounds are hinged at runtime, not on the tape.
    auto trace_tsr_error(const RobotInfo &info, const std::string &language) -> Traced;

    // Relative-pose (bimanual) TSR error between two end-effectors.
    // Input: q (nq), then the reference relative transform lTr (7), lb (6), ub (6).
    // Output: d(err)/dq (6 x nq, row-major), then the raw error (6).
    auto trace_tsr_bimanual_error(
        const RobotInfo &info,
        const std::string &language,
        std::size_t eef1 = 0,
        std::size_t eef2 = 1) -> Traced;

    // Smooth equality form of the per-end-effector TSR pose error (exact charts): for each
    // end-effector err = [p; vee(R - R^T) / 2] of the offset frame in the reference frame, zero
    // exactly where they coincide. Input: q (nq), qdot (nq), then per end-effector rTe (7), wTr
    // (7). Output: d(err)/dq (6 n_eef x nq, row-major), err (6 n_eef), then the drift
    // d²(err)/dq²[qdot, qdot] (6 n_eef).
    auto trace_tsr_equality(const RobotInfo &info, const std::string &language) -> Traced;

    // Smooth equality form of the relative pose (exact charts): err = [p; vee(R - R^T) / 2] of
    // the relative-pose error transform, zero exactly on the rigid relative pose.
    // Input: q (nq), qdot (nq), lTr (7). Output: d(err)/dq (6 x nq, row-major), err (6), then
    // the drift d²(err)/dq²[qdot, qdot] (6); without `drift`, only J and err (qdot is ignored),
    // for Newton iterations that need no second-order term.
    auto trace_tsr_bimanual_equality(
        const RobotInfo &info,
        const std::string &language,
        std::size_t eef1 = 0,
        std::size_t eef2 = 1,
        bool drift = true) -> Traced;

    // Projection step from a TSR error and Jacobian; `relative` selects the 6-row bimanual
    // error instead of the 6 n_eef-row per-end-effector error.
    // Input: J (row-major), then err. Output: gradient (nq).
    auto trace_solve_tsr(
        const RobotInfo &info,
        const std::string &language,
        ProjMethod method,
        bool relative = false) -> Traced;

    // Projection step from an arbitrary stacked error and Jacobian with err_size rows.
    // Input: J (err_size x nq, row-major), then err (err_size). Output: gradient (nq).
    auto trace_solve_jacobian(
        const RobotInfo &info,
        const std::string &language,
        ProjMethod method,
        std::size_t err_size) -> Traced;

    // Exact-chart normal-space step for err_size equality rows: with M = J N^T (N's rows are
    // normal directions, err_size x nq), solves M w = rhs by unrolled LU without pivoting and
    // returns the joint-space step N^T w. Callers precondition N (N <- N P with P an inverse of
    // M at a nearby point) so M stays near the identity, where pivoting is unnecessary.
    // Input: J (err_size x nq, row-major), rhs (err_size), N (err_size x nq, row-major).
    // Output: step (nq), then the LU pivots (err_size).
    auto trace_solve_exact_normal(
        const RobotInfo &info,
        const std::string &language,
        std::size_t err_size) -> Traced;

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

    // Derive the FLASK flat-system (z-robot) kernels when the recipe has a "flask" block
    // in `data`, setting the has_flask template gate either way: validates rho and the
    // velocity/effort limits, derives the flat-state box, traces the flask kernels, and
    // pre-renders the nested `Flask` struct into data["flask_struct"]. A non-empty
    // `flask_template` overrides the embedded flask template source. Shared by the
    // offline generator (fkcc_gen) and the JIT path (generate_robot_source) so both
    // accept the same keys.
    auto derive_flask_traces(
        const RobotInfo &robot,
        nlohmann::json &data,
        const std::string &language,
        std::string_view flask_template = {}) -> void;

    // Strict recipe validation: throws if `data` contains keys cricket does not read
    // (typos otherwise fail silently, e.g. a misspelled flag defaulting to false), with a
    // nearest-match suggestion. Keys starting with '_' are ignored as comments. Checks
    // nested blocks (bounds, flask, com, closed_loops, parts, subtemplates) too.
    auto validate_recipe(const nlohmann::json &data) -> void;

    struct GenOptions
    {
        std::filesystem::path urdf;
        std::optional<std::filesystem::path> srdf;
        std::vector<std::string> end_effectors;
        std::filesystem::path template_path;
        std::map<std::string, std::filesystem::path> subtemplates;
        std::string language = "c++";
        std::optional<Bounds> bounds;
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

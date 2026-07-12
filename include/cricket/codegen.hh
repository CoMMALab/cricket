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

    // Inverse dynamics via RNEA; input x = [q; v; a] (3n rows), outputs n torques.
    auto trace_flask_rnea(const pinocchio::Model &model, const std::string &language) -> Traced;
    auto trace_flask_rnea_block(const pinocchio::Model &model, const std::string &language) -> Traced;

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
    // the raw un-hinged error (6 n_eef); bounds are hinged at runtime, not on the tape.
    auto trace_tsr_error(const RobotInfo &info, const std::string &language) -> Traced;

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

    struct GenOptions
    {
        std::filesystem::path urdf;
        std::optional<std::filesystem::path> srdf;
        std::optional<std::string> end_effector;
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

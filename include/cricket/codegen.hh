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

    struct GenOptions
    {
        std::filesystem::path urdf;
        std::optional<std::filesystem::path> dynamics_urdf;
        std::optional<std::filesystem::path> srdf;
        std::optional<std::string> end_effector;
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

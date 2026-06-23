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

    // Trace pinocchio's joint-aware mapping from [0,1]^n_u random inputs to a
    // valid configuration vector. SE3/SE2 joints require Cartesian `bounds`.
    auto trace_map_to_configuration(
        const pinocchio::Model &model,
        const std::string &language,
        const std::optional<Bounds> &bounds = std::nullopt) -> Traced;

    // Trace pinocchio's joint-aware configuration interpolation.
    auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced;
    // SIMD-friendly variant: broadcasts the two endpoints over a rake.
    auto trace_interpolate_block(const pinocchio::Model &model, const std::string &language) -> Traced;

    // Trace pinocchio's joint-aware configuration distance.
    auto trace_distance(const pinocchio::Model &model, const std::string &language) -> Traced;

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

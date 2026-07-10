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

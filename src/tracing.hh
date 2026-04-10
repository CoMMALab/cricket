#pragma once

#include "types.hh"
#include "robot_info.hh"

#include <pinocchio/multibody/model.hpp>

#include <optional>
#include <string>

namespace cricket
{

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

auto trace_check_bounds(
    const pinocchio::Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds = std::nullopt) -> Traced;

auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced;

}  // namespace cricket

#pragma once

#include "types.hh"
#include "robot_info.hh"

#include <pinocchio/multibody/model.hpp>

#include <optional>
#include <string>

namespace cricket
{

auto trace_eefk(const RobotInfo &info, const std::string &language) -> Traced;
auto trace_sphere_fk(const RobotInfo &info, const std::string &language) -> Traced;
auto trace_ccfk(const RobotInfo &info, const std::string &language) -> Traced;
auto trace_ccfk_ee(const RobotInfo &info, const std::string &language) -> Traced;
auto trace_map_to_configuration(
    const pinocchio::Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds = std::nullopt) -> Traced;
auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced;
auto trace_interpolate_block(const pinocchio::Model &model, const std::string &language) -> Traced;
auto trace_distance(const pinocchio::Model &model, const std::string &language) -> Traced;

}  // namespace cricket

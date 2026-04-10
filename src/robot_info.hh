#pragma once

#include "types.hh"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/geometry.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace cricket
{

class RobotInfo
{
public:
    RobotInfo(
        const std::filesystem::path &urdf_file,
        const std::optional<std::filesystem::path> &srdf_file,
        const std::optional<std::string> &end_effector);

    auto json() -> nlohmann::json;
    auto dof_to_joint_names() -> std::vector<std::string>;
    auto get_frames_colliding_end_effector() -> std::vector<std::size_t>;
    auto add_mimic_joint(const std::string &name, const std::string &joint, double multiplier, double offset)
        -> void;

    pinocchio::Model model;
    pinocchio::GeometryModel collision_model;
    std::string end_effector_name;
    std::size_t end_effector_index;

    float min_radius{std::numeric_limits<float>::max()};
    float max_radius{std::numeric_limits<float>::min()};
    std::vector<SphereInfo> spheres;
    std::map<std::size_t, SphereInfo> bounding_spheres;
    std::vector<std::size_t> links_with_geometry;
    std::vector<std::vector<std::size_t>> per_link_spheres;
    std::set<std::pair<std::size_t, std::size_t>> allowed_link_pairs;
    std::vector<std::size_t> bounding_sphere_index;

private:
    auto extract_spheres() -> void;
    auto collision_pair_to_frame_pair(const pinocchio::CollisionPair &cp)
        -> std::pair<std::size_t, std::size_t>;
    auto extract_collision_data() -> void;
    auto get_adjacent_frames() -> std::set<std::pair<std::size_t, std::size_t>>;
    auto guess_self_collisions(std::size_t n = 1000000U) -> void;
};

}  // namespace cricket

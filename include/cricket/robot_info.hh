#pragma once

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cricket
{
    struct SphereInfo
    {
        std::size_t geom_index;
        float radius;
        std::size_t parent_joint;
        std::size_t parent_frame;
        pinocchio::SE3 relative;
    };

    auto min_sphere_of_spheres(const std::vector<SphereInfo> &info) -> std::array<float, 4>;

    // Optional explicit Cartesian bounds for FreeFlyer / Planar joints
    struct Bounds
    {
        Eigen::Vector3d lower;
        Eigen::Vector3d upper;
    };

    enum class JointType
    {
        Bounded,            // Revolute/prismatic with limits: nq=1, nu=1
        UnboundedRevolute,  // Continuous revolute (cos,sin): nq=2, nu=1
        SO3,                // Spherical (xyzw quaternion): nq=4, nu=3
        SE3,                // FreeFlyer: nq=7, nu=6
        SE2,                // Planar: nq=4, nu=3
        Unsupported
    };

    struct JointMapping
    {
        JointType type;
        std::size_t joint_id;
        std::size_t idx_q;
        std::size_t idx_u;
        std::size_t nq;
        std::size_t nu;
    };

    class RobotInfo
    {
    public:
        RobotInfo(
            const std::filesystem::path &urdf_file,
            const std::optional<std::filesystem::path> &srdf_file,
            const std::optional<std::string> &end_effector);

        auto json(const std::optional<Bounds> &bounds = std::nullopt) -> nlohmann::json;

        auto dof_to_joint_names() -> std::vector<std::string>;
        auto get_frames_colliding_end_effector() -> std::vector<std::size_t>;
        auto extract_spheres() -> void;
        auto collision_pair_to_frame_pair(const pinocchio::CollisionPair &cp)
            -> std::pair<std::size_t, std::size_t>;
        auto extract_collision_data() -> void;
        auto get_adjacent_frames() -> std::set<std::pair<std::size_t, std::size_t>>;
        auto guess_self_collisions(std::size_t n = 1000000U) -> void;

        pinocchio::Model model;
        pinocchio::GeometryModel collision_model;
        std::string end_effector_name;
        std::size_t end_effector_index;

        float min_radius{std::numeric_limits<float>::max()};
        float max_radius{std::numeric_limits<float>::min()};
        float min_radius_mobile{std::numeric_limits<float>::max()};
        float max_radius_mobile{std::numeric_limits<float>::min()};
        float min_bounding_radius_mobile{std::numeric_limits<float>::max()};
        float max_bounding_radius_mobile{std::numeric_limits<float>::min()};
        std::vector<SphereInfo> spheres;
        std::map<std::size_t, SphereInfo> bounding_spheres;
        std::vector<std::size_t> links_with_geometry;
        std::vector<std::vector<std::size_t>> per_link_spheres;
        std::set<std::pair<std::size_t, std::size_t>> allowed_link_pairs;
        std::vector<std::size_t> bounding_sphere_index;
    };
}  // namespace cricket

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

    // One instance of a robot model within a composite: a URDF loaded under a unique
    // name prefix (every joint/frame/geometry becomes `<prefix>_<name>`) and mounted
    // by a fixed placement on the world or on a frame of an earlier part.
    struct CompositePart
    {
        std::string prefix;
        std::filesystem::path urdf;
        std::optional<std::filesystem::path> srdf;

        // "world" or a prefixed frame name from a part declared earlier.
        std::string parent{"world"};

        // Pose of the part's root in the parent frame.
        pinocchio::SE3 placement{pinocchio::SE3::Identity()};
    };

    // Build one robot from several URDF instances. Within-part self-collision pairs
    // come from each part's SRDF; all cross-part pairs are active unless listed in
    // `disabled_collisions`. Configuration ordering follows part declaration order
    // for world-mounted parts (pinocchio::appendModel ordering otherwise).
    struct CompositeSpec
    {
        std::vector<CompositePart> parts;

        // Prefixed link-name pairs to never collision-check (e.g. mount-adjacent links).
        std::vector<std::pair<std::string, std::string>> disabled_collisions;

        // Reads the "parts" / "disabled_collisions" recipe keys; nullopt if "parts" is
        // absent. Part URDF/SRDF paths are resolved relative to `base_path`.
        static auto from_json(const nlohmann::json &data, const std::filesystem::path &base_path)
            -> std::optional<CompositeSpec>;
    };

    // Plan over a subset of joints: every joint not named in `active_joints` is locked at
    // its value in `default_configuration` (pinocchio::neutral if unnamed) and
    // constant-folded out of the model before spheres or tapes are derived.
    struct JointSelection
    {
        std::vector<std::string> active_joints;

        // Joint name -> configuration values. A single value for a continuous joint is
        // the angle (mapped to cos/sin); otherwise the length must match the joint's nq.
        std::map<std::string, std::vector<double>> default_configuration;

        // Reads the "active_joints" / "default_configuration" recipe keys; nullopt if absent.
        static auto from_json(const nlohmann::json &data) -> std::optional<JointSelection>;
    };

    class RobotInfo
    {
    public:
        RobotInfo(
            const std::filesystem::path &urdf_file,
            const std::optional<std::filesystem::path> &srdf_file,
            const std::optional<std::string> &end_effector);

        RobotInfo(
            const std::filesystem::path &urdf_file,
            const std::optional<std::filesystem::path> &srdf_file,
            const std::vector<std::string> &end_effectors,
            const std::optional<JointSelection> &joint_selection = std::nullopt);

        RobotInfo(
            const CompositeSpec &composite,
            const std::vector<std::string> &end_effectors,
            const std::optional<JointSelection> &joint_selection = std::nullopt);

        // When `skip_static_environment` is set, links with no mobile joint between them
        // and the world (frame parent joint 0) are omitted from the emitted
        // environment-collision tables; self-collision tables are unaffected.
        auto json(
            const std::optional<Bounds> &bounds = std::nullopt,
            bool skip_static_environment = false) -> nlohmann::json;

        auto reduce_model(const JointSelection &selection) -> void;

        auto dof_to_joint_names() -> std::vector<std::string>;
        auto get_frames_colliding_end_effector(std::size_t frame_index) -> std::vector<std::size_t>;
        auto extract_spheres() -> void;
        auto collision_pair_to_frame_pair(const pinocchio::CollisionPair &cp)
            -> std::pair<std::size_t, std::size_t>;
        auto extract_collision_data() -> void;
        auto get_adjacent_frames() -> std::set<std::pair<std::size_t, std::size_t>>;
        auto guess_self_collisions(std::size_t n = 1000000U) -> void;
        auto resolve_end_effectors(const std::vector<std::string> &end_effectors) -> void;

        pinocchio::Model model;
        pinocchio::GeometryModel collision_model;

        // First end-effector, kept for single-EE conveniences (e.g. the emitted
        // `end_effector` name constant).
        std::string end_effector_name;
        std::size_t end_effector_index;

        // All end-effectors, in declaration order (constraint codegen uses every entry).
        std::vector<std::string> end_effector_names;
        std::vector<std::size_t> end_effector_indexes;

        float min_radius{std::numeric_limits<float>::max()};
        float max_radius{std::numeric_limits<float>::min()};
        std::vector<SphereInfo> spheres;
        std::map<std::size_t, SphereInfo> bounding_spheres;
        std::vector<std::size_t> links_with_geometry;
        std::vector<std::vector<std::size_t>> per_link_spheres;
        std::set<std::pair<std::size_t, std::size_t>> allowed_link_pairs;
        std::vector<std::size_t> bounding_sphere_index;
    };
}  // namespace cricket

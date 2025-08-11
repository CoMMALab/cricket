#include "pinocchio_cppadcg.hh"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/collision/collision.hpp>

#include <coal/shape/geometric_shapes.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <CGAL/Min_sphere_of_spheres_d_traits_3.h>

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <inja/inja.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>

#include "lang_gen.hh"

// Joint types
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5

using namespace pinocchio;
using namespace CppAD;
using namespace CppAD::cg;

// Typedef for AD types
using CGD = CG<double>;
using ADCG = AD<CGD>;

using ADModel = ModelTpl<ADCG>;
using ADData = DataTpl<ADCG>;
using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;

struct SphereInfo
{
    std::size_t geom_index;
    float radius;
    std::size_t parent_joint;
    std::size_t parent_frame;
    SE3 relative;
};

struct JointTreeInfo {
    std::vector<std::size_t> parent_joints;
    std::vector<std::size_t> saved_joint_memory_idx;
    std::size_t n_joints_with_multiple_children;
    std::vector<std::size_t> dfs_order;
};

auto min_sphere_of_spheres(const std::vector<SphereInfo> &info) -> std::array<float, 4>
{
    using K = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Traits = CGAL::Min_sphere_of_spheres_d_traits_3<K, double>;
    using Sphere = Traits::Sphere;
    using Point = K::Point_3;
    using MinSphere = CGAL::Min_sphere_of_spheres_d<Traits>;

    std::vector<Sphere> cgal_spheres;
    cgal_spheres.reserve(info.size());

    for (const auto &sphere : info)
    {
        auto pos = sphere.relative.translation();
        cgal_spheres.emplace_back(Point(pos[0], pos[1], pos[2]), sphere.radius);
    }

    MinSphere ms(cgal_spheres.begin(), cgal_spheres.end());
    std::array<float, 4> sphere;
    std::copy(ms.center_cartesian_begin(), ms.center_cartesian_end(), sphere.begin());
    sphere[3] = ms.radius();
    return sphere;
}

struct RobotInfo
{
    RobotInfo(
        const std::filesystem::path &urdf_file,
        const std::optional<std::filesystem::path> &srdf_file,
        const std::string &end_effector)
    {
        if (not std::filesystem::exists(urdf_file))
        {
            throw std::runtime_error(fmt::format("URDF file {} does not exist!", urdf_file.string()));
        }

        pinocchio::urdf::buildModel(urdf_file, model);
        pinocchio::urdf::buildGeom(model, urdf_file, COLLISION, collision_model);

        if (srdf_file and not std::filesystem::exists(*srdf_file))
        {
            throw std::runtime_error(fmt::format("SRDF file () does not exist!", srdf_file->string()));
        }
        else if (not srdf_file)
        {
            fmt::print("No SRDF file provided, guessing collisions!");
            guess_self_collisions();
        }
        else
        {
            collision_model.addAllCollisionPairs();
            pinocchio::srdf::removeCollisionPairs(model, collision_model, *srdf_file);
            extract_collision_data();
        }

        extract_spheres();

        end_effector_name = end_effector;

        if (not model.existFrame(end_effector))
        {
            throw std::runtime_error(fmt::format("Invalid EE name {}", end_effector));
        }

        end_effector_index = model.getFrameId(end_effector);
    }

    auto json() -> nlohmann::json
    {
        const Eigen::VectorXd lower_bound = model.lowerPositionLimit;
        const Eigen::VectorXd upper_bound = model.upperPositionLimit;
        const Eigen::VectorXd bound_range = upper_bound - lower_bound;
        const Eigen::VectorXd bound_descale = bound_range.cwiseInverse();

        nlohmann::json json;
        json["n_q"] = model.nq;
        json["n_spheres"] = spheres.size();
        json["bound_lower"] = std::vector<float>(lower_bound.data(), lower_bound.data() + model.nq);
        json["bound_range"] = std::vector<float>(bound_range.data(), bound_range.data() + model.nq);
        json["bound_descale"] = std::vector<float>(bound_descale.data(), bound_descale.data() + model.nq);
        json["measure"] = bound_range.prod();
        json["end_effector_index"] = end_effector_index;
        json["min_radius"] = min_radius;
        json["max_radius"] = max_radius;
        json["joint_names"] = dof_to_joint_names();
        json["allowed_link_pairs"] = allowed_link_pairs;
        json["per_link_spheres"] = per_link_spheres;
        json["links_with_geometry"] = links_with_geometry;
        json["bounding_sphere_index"] = bounding_sphere_index;
        json["end_effector_collisions"] = get_frames_colliding_end_effector();
        json["per_joint_spheres"] = per_joint_spheres;

        // print parent joint for each joint
        for (auto i = 0U; i < model.joints.size(); ++i) {
            fmt::print("Joint {}: {}\n", i, model.parents[i]);
        }

        // sphere_to_joint
        std::vector<std::size_t> sphere_to_joint(spheres.size(), -1);
        for (auto i = 0U; i < spheres.size(); ++i)
        {
            sphere_to_joint[i] = spheres[i].parent_joint;
        }
        json["sphere_to_joint"] = sphere_to_joint;

        // joint to spheres
        std::vector<std::vector<std::size_t>> joint_to_spheres(model.joints.size());
        for (auto i = 0U; i < spheres.size(); ++i)
        {
            joint_to_spheres[spheres[i].parent_joint].emplace_back(i);
        }
        json["joint_to_spheres"] = joint_to_spheres;
        
        std::vector<int> flattened_joint_to_spheres;
        for (auto i = 0U; i < joint_to_spheres.size(); ++i)
        {
            for (auto j = 0U; j < joint_to_spheres[i].size(); ++j)
            {
                flattened_joint_to_spheres.emplace_back(joint_to_spheres[i][j]);
            }
            flattened_joint_to_spheres.emplace_back(-1);
        }
        json["flattened_joint_to_spheres"] = flattened_joint_to_spheres;

        // joint type for each joint
        std::vector<int> joint_types(model.joints.size());
        for (auto i = 0U; i < model.joints.size(); ++i)
        {
            fmt::print("Joint type {}: {}\n", i, model.joints[i].shortname());
            if (model.joints[i].shortname() == "JointModelRX") {
                joint_types[i] = X_ROT;
            }
            else if (model.joints[i].shortname() == "JointModelRY") {
                joint_types[i] = Y_ROT;
            }
            else if (model.joints[i].shortname() == "JointModelRZ") {
                joint_types[i] = Z_ROT;
            }
            else if (model.joints[i].shortname() == "JointModelPX") {
                joint_types[i] = X_PRISM;
            }
            else if (model.joints[i].shortname() == "JointModelPY") {
                joint_types[i] = Y_PRISM;
            }
            else if (model.joints[i].shortname() == "JointModelPZ") {
                joint_types[i] = Z_PRISM;
            }
            else {
                throw std::runtime_error(fmt::format("Invalid joint type {}", model.joints[i].shortname()));
            }
        }
        json["joint_types"] = joint_types;

        // joint matrices
        std::vector<std::array<std::array<float, 4>, 4>> joint_matrices(model.joints.size());
        for (auto i = 0U; i < model.joints.size(); ++i)
        {
            const auto &joint = model.joints[i];
            const auto &placement = model.jointPlacements[i];
            const auto &matrix = placement.toHomogeneousMatrix();
            for (auto row = 0U; row < 4; ++row)
            {
                for (auto col = 0U; col < 4; ++col)
                {
                    joint_matrices[i][row][col] = matrix(row, col);
                }
            }
        }
        json["joint_matrices"] = joint_matrices;

        std::vector<std::array<int, 2>> collision_pairs;
        for (auto &pair : collision_model.collisionPairs)
        {
            collision_pairs.emplace_back(std::array<int, 2>{static_cast<int>(pair.first), static_cast<int>(pair.second)});
        }
        json["collision_pairs"] = collision_pairs;

        // instead of each pair of spheres to check like (i, j),(i, j+1),...,(i, j+k) we store a range
        // like (i, j, j+k). This is calculated from the collision pairs.
        std::vector<std::array<int, 3>> self_cc_ranges;
        int cur_sphere1 = collision_pairs[0][0];
        int i = 0;
        int range_start = collision_pairs[0][1];
        int range_end = collision_pairs[0][1] - 1;
        while (i < collision_pairs.size()) {
            if (collision_pairs[i][0] == cur_sphere1 && collision_pairs[i][1] == range_end + 1) {
                range_end = collision_pairs[i][1];
                i++;
            }
            else {
                self_cc_ranges.emplace_back(std::array<int, 3>{cur_sphere1, range_start, range_end});
                cur_sphere1 = collision_pairs[i][0];
                range_start = collision_pairs[i][1];
                range_end = collision_pairs[i][1];
                i++;
            }
        }
        json["self_cc_ranges"] = self_cc_ranges;

        auto joint_tree_info = analyze_joint_tree();
        json["joint_parents"] = joint_tree_info.parent_joints;
        json["T_memory_idx"] = joint_tree_info.saved_joint_memory_idx;
        json["dfs_order"] = joint_tree_info.dfs_order;
        json["n_joints_with_multiple_children"] = joint_tree_info.n_joints_with_multiple_children;


        std::vector<std::array<float, 4>> spheres_array;
        for (auto i = 0U; i < spheres.size(); ++i)
        {
            spheres_array.emplace_back(std::array<float, 4>{
                static_cast<float>(spheres[i].relative.translation()[0]),
                static_cast<float>(spheres[i].relative.translation()[1]),
                static_cast<float>(spheres[i].relative.translation()[2]),
                static_cast<float>(spheres[i].radius)});
        }
        json["spheres_array"] = spheres_array;

        std::vector<int> sphere_to_link(spheres.size(), -1);
        for (auto i = 0U; i < spheres.size(); ++i)
        {
            sphere_to_link[i] = spheres[i].parent_frame;
        }
        json["sphere_to_link"] = sphere_to_link;

        std::vector<std::string> link_names;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            link_names.emplace_back(model.frames[i].name);
        }
        json["link_names"] = link_names;

        return json;
    }

    auto dof_to_joint_names() -> std::vector<std::string>
    {
        std::vector<std::size_t> dof_to_joint_id(model.nq);
        for (auto joint_id = 1U; joint_id < model.joints.size(); ++joint_id)
        {
            const auto &joint = model.joints[joint_id];
            auto start_idx = joint.idx_q();
            auto nq = joint.nq();

            for (auto i = 0U; i < nq; ++i)
            {
                dof_to_joint_id[start_idx + i] = joint_id;
            }
        }

        std::vector<std::string> dof_to_joint_name(model.nq);
        for (auto i = 0U; i < model.nq; ++i)
        {
            dof_to_joint_name[i] = model.names[dof_to_joint_id[i]];
        }

        return dof_to_joint_name;
    }

    auto get_frames_colliding_end_effector() -> std::vector<std::size_t>
    {
        std::size_t end_effector_joint = model.frames[end_effector_index].parentJoint;

        std::vector<std::size_t> frames;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            if (model.frames[i].parentJoint == end_effector_joint)
            {
                if (bounding_spheres.find(i) != bounding_spheres.end())
                {
                    frames.emplace_back(i);
                }
            }
        }

        std::set<std::size_t> end_effector_allowed_collisions;
        for (const auto &[first, second] : allowed_link_pairs)
        {
            if (std::find(frames.begin(), frames.end(), first) != frames.end())
            {
                end_effector_allowed_collisions.emplace(second);
            }

            if (std::find(frames.begin(), frames.end(), second) != frames.end())
            {
                end_effector_allowed_collisions.emplace(first);
            }
        }

        return std::vector<std::size_t>(
            end_effector_allowed_collisions.begin(), end_effector_allowed_collisions.end());
    }

    auto print_fixed_transforms() -> void
    {
        fmt::print("_device__ __constant__ float panda_fixed_transforms[] = {{\n");
        
        for (auto i = 0U; i < model.joints.size(); ++i)
        {
            const auto& placement = model.jointPlacements[i];
            const auto& matrix = placement.toHomogeneousMatrix();
            
            fmt::print("        // Joint {}\n", i);
            for (auto row = 0U; row < 4; ++row)
            {
                fmt::print("        ");
                for (auto col = 0U; col < 4; ++col)
                {
                    fmt::print("{:.6f}", matrix(row, col));
                    if (col < 3) fmt::print(", ");
                }
                if (row < 3) fmt::print(",");
                fmt::print("\n");
            }
            if (i < model.joints.size() - 1) fmt::print("\n");
        }
        
        fmt::print("    }};\n");
    }

    auto extract_spheres() -> void
    {
        for (auto i = 0U; i < collision_model.ngeoms; ++i)
        {
            const auto &geom_obj = collision_model.geometryObjects[i];
            const auto &sphere_ptr = std::dynamic_pointer_cast<coal::Sphere>(geom_obj.geometry);
            if (sphere_ptr)
            {
                SphereInfo info;
                info.geom_index = i;
                info.radius = sphere_ptr->radius;
                info.parent_joint = geom_obj.parentJoint;
                info.parent_frame = geom_obj.parentFrame;
                info.relative = geom_obj.placement;

                spheres.emplace_back(info);

                min_radius = std::min(min_radius, info.radius);
                max_radius = std::max(max_radius, info.radius);
                // fmt::print("Sphere {}: {} {} {} {}\n", i, info.relative.translation()[0], info.relative.translation()[1], info.relative.translation()[2], info.radius);
            }
            else
            {
                throw std::runtime_error(
                    fmt::format("Invalid non-sphere geometry in URDF {}", geom_obj.name));
            }
        }

        // extract spheres per joint
        for (auto i = 0U; i < model.joints.size(); ++i) {
            std::vector<std::size_t> sphere_indices;
            for (const auto &info : spheres) {
                if (info.parent_joint == i) {
                    sphere_indices.emplace_back(info.geom_index);
                }
            }

            per_joint_spheres.emplace_back(sphere_indices);
        }

        std::size_t bs = 0;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            std::vector<SphereInfo> link_info;
            std::vector<std::size_t> sphere_indices;
            for (const auto &info : spheres)
            {
                if (info.parent_frame == i)
                {
                    link_info.emplace_back(info);
                    sphere_indices.emplace_back(info.geom_index);
                }
            }

            per_link_spheres.emplace_back(sphere_indices);

            if (not link_info.empty())
            {
                auto sphere = min_sphere_of_spheres(link_info);

                SphereInfo info;
                info.geom_index = bs;
                info.radius = sphere[3];
                info.parent_joint = link_info[0].parent_joint;
                info.relative = SE3::Identity();
                info.relative.translation()[0] = sphere[0];
                info.relative.translation()[1] = sphere[1];
                info.relative.translation()[2] = sphere[2];

                bounding_spheres.emplace(i, info);
                bounding_sphere_index.emplace_back(bs);
                links_with_geometry.emplace_back(i);
                bs++;
            }
            else
            {
                bounding_sphere_index.emplace_back(0);
            }
        }
    }

    auto collision_pair_to_frame_pair(const CollisionPair &cp) -> std::pair<std::size_t, std::size_t>
    {
        const auto &geom1 = collision_model.geometryObjects[cp.first];
        const auto &geom2 = collision_model.geometryObjects[cp.second];

        std::size_t link1_idx = geom1.parentFrame;
        std::size_t link2_idx = geom2.parentFrame;

        return std::make_pair(std::min(link1_idx, link2_idx), std::max(link1_idx, link2_idx));
    }

    auto extract_collision_data() -> void
    {
        for (const auto &cp : collision_model.collisionPairs)
        {
            // fmt::print("Collision pair: {} {}\n", cp.first, cp.second);
            allowed_link_pairs.insert(collision_pair_to_frame_pair(cp));
        }
    }

    auto get_adjacent_frames() -> std::set<std::pair<std::size_t, std::size_t>>
    {
        const auto nf = model.frames.size();
        const auto nj = model.joints.size();

        std::set<std::pair<std::size_t, std::size_t>> adjacents;

        for (auto i = 0U; i < nf; ++i)
        {
            for (auto j = i + 1; j < nf; ++j)
            {
                const auto &frame_i = model.frames[i];
                const auto &frame_j = model.frames[j];

                if (frame_i.parentJoint < nj and frame_j.parentJoint < nj)
                {
                    const auto &joint_i = model.joints[frame_i.parentJoint];
                    const auto &joint_j = model.joints[frame_j.parentJoint];

                    // Check if joints are parent-child related
                    if (model.parents[frame_i.parentJoint] == frame_j.parentJoint or
                        model.parents[frame_j.parentJoint] == frame_i.parentJoint)
                    {
                        adjacents.insert({i, j});
                    }
                }
            }
        }

        return adjacents;
    }

    auto guess_self_collisions(std::size_t n = 1000000U) -> void
    {
        collision_model.addAllCollisionPairs();

        Data data(model);
        GeometryData collision_data(collision_model);

        std::set<std::pair<std::size_t, std::size_t>> always_pairs;

        for (auto j = 0U; j < collision_model.collisionPairs.size(); ++j)
        {
            always_pairs.emplace(collision_pair_to_frame_pair(collision_model.collisionPairs[j]));
        }

        allowed_link_pairs.clear();

        for (auto i = 0U; i < n; ++i)
        {
            auto q = randomConfiguration(model);
            computeCollisions(model, data, collision_model, collision_data, q);

            for (auto j = 0U; j < collision_model.collisionPairs.size(); ++j)
            {
                const auto &cr = collision_data.collisionResults[j];
                auto pair = collision_pair_to_frame_pair(collision_model.collisionPairs[j]);

                if (cr.isCollision())
                {
                    allowed_link_pairs.insert(pair);
                }
                else
                {
                    auto it = always_pairs.find(pair);
                    if (it != always_pairs.end())
                    {
                        always_pairs.erase(it);
                    }
                }
            }
        }

        // Remove all adjacent frames
        auto adjacents = get_adjacent_frames();
        for (const auto &pair : adjacents)
        {
            allowed_link_pairs.erase(pair);
        }

        // Remove all pairs that never collided
        for (const auto &pair : always_pairs)
        {
            allowed_link_pairs.erase(pair);
        }

        // Add remaining potential collisions
        collision_model.removeAllCollisionPairs();
        for (const auto &pair : allowed_link_pairs)
        {
            collision_model.addCollisionPair(CollisionPair(pair.first, pair.second));
        }
    }

    auto analyze_joint_tree() -> JointTreeInfo {
        JointTreeInfo info;
        info.parent_joints.resize(model.joints.size(), -1);
        std::vector<std::vector<std::size_t>> children_joints(model.joints.size());
        for (auto i = 0U; i < model.joints.size(); ++i) {
            info.parent_joints[i] = model.parents[i];
            // Don't add root joint as child of itself to avoid cycles
            if (model.parents[i] != i) {
                children_joints[model.parents[i]].emplace_back(i);
            }
        }
        info.n_joints_with_multiple_children = 0;
        for (auto i = 0U; i < model.joints.size(); ++i) {
            if (children_joints[i].size() > 1) {
                info.n_joints_with_multiple_children++;
            }
        }
        std::cout << "here1" << std::endl;
        info.saved_joint_memory_idx.resize(model.joints.size(), info.n_joints_with_multiple_children);
        std::size_t idx = 0;
        for (auto i = 0U; i < model.joints.size(); ++i) {
            if (children_joints[i].size() > 1) {
                info.saved_joint_memory_idx[i] = idx;
                idx++;
            }
            else {
                info.saved_joint_memory_idx[i] = info.n_joints_with_multiple_children;
            }
        }
        std::cout << "here2" << std::endl;  
        std::function<void(std::size_t, std::vector<std::size_t>&)> dfs = 
            [&](std::size_t joint_idx, std::vector<std::size_t>& order) {
                order.push_back(joint_idx);
                for (auto child : children_joints[joint_idx]) {
                    dfs(child, order);
                }
            };
        info.dfs_order.clear();
        info.dfs_order.reserve(model.joints.size());
        dfs(0, info.dfs_order);

        return info;
    }

    Model model;
    GeometryModel collision_model;
    std::string end_effector_name;
    std::size_t end_effector_index;

    float min_radius{std::numeric_limits<float>::max()};
    float max_radius{std::numeric_limits<float>::min()};
    std::vector<SphereInfo> spheres;
    std::map<std::size_t, SphereInfo> bounding_spheres;
    std::vector<std::size_t> links_with_geometry;
    std::vector<std::vector<std::size_t>> per_link_spheres;
    std::vector<std::vector<std::size_t>> per_joint_spheres;
    std::set<std::pair<std::size_t, std::size_t>> allowed_link_pairs;
    std::vector<std::size_t> bounding_sphere_index;
};

auto trace_sphere(const SphereInfo &sphere, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &joint_placement = ad_data.oMi[sphere.parent_joint];

    Eigen::Matrix<ADCG, 3, 1> local_translation;
    local_translation[0] = sphere.relative.translation()[0];
    local_translation[1] = sphere.relative.translation()[1];
    local_translation[2] = sphere.relative.translation()[2];

    Eigen::Matrix<ADCG, 3, 1> world_position =
        joint_placement.rotation() * local_translation + joint_placement.translation();

    data[index + 0] = world_position[0];
    data[index + 1] = world_position[1];
    data[index + 2] = world_position[2];
    data[index + 3] = ADCG(sphere.radius);
}

auto trace_frame(std::size_t ee_index, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &oMf = ad_data.oMf[ee_index];

    data[index + 0] = oMf.translation()[0];
    data[index + 1] = oMf.translation()[1];
    data[index + 2] = oMf.translation()[2];

    const auto &R = oMf.rotation();

    // Eigen stores as column major
    data[index + 3] = R(0, 0);
    data[index + 4] = R(1, 0);
    data[index + 5] = R(2, 0);
    data[index + 6] = R(0, 1);
    data[index + 7] = R(1, 1);
    data[index + 8] = R(2, 1);
    data[index + 9] = R(0, 2);
    data[index + 10] = R(1, 2);
    data[index + 11] = R(2, 2);
}

struct Traced
{
    std::string code;
    std::size_t temp_variables;
    std::size_t outputs;
};

auto trace_sphere_cc_fk(
    const RobotInfo &info,
    bool spheres = true,
    bool bounding_spheres = true,
    bool fk = true) -> Traced
{
    auto nq = info.model.nq;
    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    ADVectorXs ad_q(nq);
    for (auto i = 0U; i < nq; ++i)
    {
        ad_q[i] = ADCG(0.0);
    }

    Independent(ad_q);

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    std::size_t n_spheres_data = (spheres) ? info.spheres.size() * 4 : 0;
    std::size_t n_bounding_spheres_data = (bounding_spheres) ? info.bounding_spheres.size() * 4 : 0;
    std::size_t n_fk_data = (fk) ? 12 : 0;

    std::size_t n_out = n_spheres_data + n_bounding_spheres_data + n_fk_data;

    ADVectorXs data(n_out);

    if (spheres)
    {
        for (auto i = 0U; i < info.spheres.size(); ++i)
        {
            const auto &sphere = info.spheres[i];
            trace_sphere(sphere, ad_data, data, sphere.geom_index * 4);
        }
    }

    if (bounding_spheres)
    {
        for (auto i = 0U; i < info.model.frames.size(); ++i)
        {
            auto sphere_it = info.bounding_spheres.find(i);
            if (sphere_it != info.bounding_spheres.end())
            {
                const auto &sphere = sphere_it->second;
                trace_sphere(sphere, ad_data, data, sphere.geom_index * 4 + n_spheres_data);
            }
        }
    }

    if (fk)
    {
        trace_frame(info.end_effector_index, ad_data, data, n_spheres_data + n_bounding_spheres_data);
    }

    // Create the AD function
    ADFun<CGD> collision_sphere_func(ad_q, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = collision_sphere_func.Forward(0, ind_vars);

    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, result, nameGen);

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), n_out};
}

int main(int argc, char **argv)
{
    std::filesystem::path json_path(argv[1]);
    auto parent_path = json_path.parent_path();

    if (not std::filesystem::exists(json_path))
    {
        throw std::runtime_error(fmt::format("JSON file {} does not exist!", json_path.string()));
    }

    std::ifstream f(json_path);
    nlohmann::json data = nlohmann::json::parse(f);

    std::optional<std::filesystem::path> srdf_path = {};
    if (data.contains("srdf"))
    {
        srdf_path = parent_path / data["srdf"];
    }

    RobotInfo robot(parent_path / data["urdf"], srdf_path, data["end_effector"]);

    // Print fixed transforms
    robot.print_fixed_transforms();

    data.update(robot.json());

    auto traced_eefk_code = trace_sphere_cc_fk(robot, false, false, true);
    data["eefk_code"] = traced_eefk_code.code;
    data["eefk_code_vars"] = traced_eefk_code.temp_variables;
    data["eefk_code_output"] = traced_eefk_code.outputs;

    auto traced_spherefk_code = trace_sphere_cc_fk(robot, true, false, false);
    data["spherefk_code"] = traced_spherefk_code.code;
    data["spherefk_code_vars"] = traced_spherefk_code.temp_variables;
    data["spherefk_code_output"] = traced_spherefk_code.outputs;

    auto traced_ccfk_code = trace_sphere_cc_fk(robot, true, true, false);
    data["ccfk_code"] = traced_ccfk_code.code;
    data["ccfk_code_vars"] = traced_ccfk_code.temp_variables;
    data["ccfk_code_output"] = traced_ccfk_code.outputs;

    auto traced_ccfkee_code = trace_sphere_cc_fk(robot, true, true, true);
    data["ccfkee_code"] = traced_ccfkee_code.code;
    data["ccfkee_code_vars"] = traced_ccfkee_code.temp_variables;
    data["ccfkee_code_output"] = traced_ccfkee_code.outputs;

    inja::Environment env;

    env.add_callback("capitalize", 1, [](inja::Arguments& args) {
        std::string str = args.at(0)->get<std::string>();
        if (!str.empty())
        {
            str[0] = std::toupper(str[0]);
        }
        return str;
    });

    for (const auto &subt : data["subtemplates"])
    {
        inja::Template temp = env.parse_template(parent_path / subt["template"]);
        env.include_template(subt["name"], temp);
    }

    inja::Template temp = env.parse_template(parent_path / data["template"]);
    env.write(temp, data, data["output"]);

    std::ofstream output_file("output.json");
    output_file << data.dump(4);  // 4 spaces indentation
    output_file.close();

    return 0;
}

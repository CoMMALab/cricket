#include <cricket/robot_info.hh>

#include "joint_utils.hh"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/collision/collision.hpp>

#include <coal/shape/geometric_shapes.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <CGAL/Min_sphere_of_spheres_d_traits_3.h>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cricket
{
    using namespace pinocchio;

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

    RobotInfo::RobotInfo(
        const std::filesystem::path &urdf_file,
        const std::optional<std::filesystem::path> &srdf_file,
        const std::optional<std::string> &end_effector)
    {
        if (not std::filesystem::exists(urdf_file))
        {
            throw std::runtime_error(fmt::format("URDF file {} does not exist!", urdf_file.string()));
        }

        pinocchio::urdf::buildModel(urdf_file, model, false, true);
        pinocchio::urdf::buildGeom(model, urdf_file, COLLISION, collision_model);

        if (srdf_file and not std::filesystem::exists(*srdf_file))
        {
            throw std::runtime_error(fmt::format("SRDF file () does not exist!", srdf_file->string()));
        }
        else if (not srdf_file)
        {
            fmt::print("No SRDF file provided, guessing collisions!\n");
            guess_self_collisions();
        }
        else
        {
            collision_model.addAllCollisionPairs();
            pinocchio::srdf::removeCollisionPairs(model, collision_model, *srdf_file);
            extract_collision_data();
        }

        extract_spheres();

        if (not end_effector)
        {
            end_effector_name = model.frames[model.nframes - 1].name;
            fmt::print("No EE provided, using distal link `{}`.\n", end_effector_name);
        }
        else if (not model.existFrame(*end_effector))
        {
            throw std::runtime_error(fmt::format("Invalid EE name {}", *end_effector));
        }
        else
        {
            end_effector_name = *end_effector;
        }

        end_effector_index = model.getFrameId(end_effector_name);
    }

    namespace
    {
        auto compute_extended_bounds(
            const pinocchio::Model &model,
            const std::optional<Bounds> &bounds)
            -> std::pair<Eigen::VectorXd, Eigen::VectorXd>
        {
            Eigen::VectorXd lower = model.lowerPositionLimit;
            Eigen::VectorXd upper = model.upperPositionLimit;

            if (not bounds)
            {
                return {lower, upper};
            }

            auto [nu, joint_mappings] = classify_joints(model);
            (void)nu;
            for (const auto &jm : joint_mappings)
            {
                if (jm.type == JointType::SE3)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        lower[jm.idx_q + i] = bounds->lower[i];
                        upper[jm.idx_q + i] = bounds->upper[i];
                    }
                }
                else if (jm.type == JointType::SE2)
                {
                    for (int i = 0; i < 2; ++i)
                    {
                        lower[jm.idx_q + i] = bounds->lower[i];
                        upper[jm.idx_q + i] = bounds->upper[i];
                    }
                }
            }

            return {lower, upper};
        }

        auto is_euclidean(const pinocchio::Model &model) -> bool
        {
            auto [nu, joint_mappings] = classify_joints(model);
            (void)nu;
            for (const auto &jm : joint_mappings)
            {
                if (jm.type != JointType::Bounded)
                {
                    return false;
                }
            }
            return true;
        }

        auto joint_type_to_string(JointType type) -> std::string
        {
            switch (type)
            {
                case JointType::Bounded:
                    return "LP";
                case JointType::UnboundedRevolute:
                    return "SO2";
                case JointType::SO3:
                    return "SO3";
                case JointType::SE3:
                    return "SE3";
                case JointType::SE2:
                    return "SE2";
                default:
                    return "Unsupported";
            }
        }

        auto generate_joint_topology(const pinocchio::Model &model) -> nlohmann::json
        {
            auto [nu, joint_mappings] = classify_joints(model);
            (void)nu;
            nlohmann::json topology = nlohmann::json::array();
            for (const auto &jm : joint_mappings)
            {
                nlohmann::json info;
                info["type"] = joint_type_to_string(jm.type);
                info["idx_q"] = jm.idx_q;
                info["nq"] = jm.nq;
                info["nu"] = jm.nu;
                info["idx_u"] = jm.idx_u;
                topology.push_back(info);
            }
            return topology;
        }
    }  // namespace

    auto RobotInfo::json(const std::optional<Bounds> &bounds) -> nlohmann::json
    {
        const auto [lower_bound, upper_bound] = compute_extended_bounds(model, bounds);
        const Eigen::VectorXd bound_range = upper_bound - lower_bound;
        const Eigen::VectorXd bound_descale = bound_range.cwiseInverse();

        float measure = 1.0F;
        for (auto i = 0U; i < bound_range.size(); ++i)
        {
            if (std::isfinite(bound_range[i]))
            {
                measure *= static_cast<float>(bound_range[i]);
            }
        }

        nlohmann::json json;
        json["n_q"] = model.nq;
        json["n_u"] = get_randomness_dimension(model);
        json["n_spheres"] = spheres.size();
        json["bound_lower"] = std::vector<float>(lower_bound.data(), lower_bound.data() + model.nq);
        json["bound_range"] = std::vector<float>(bound_range.data(), bound_range.data() + model.nq);
        json["bound_descale"] = std::vector<float>(bound_descale.data(), bound_descale.data() + model.nq);
        json["lower"] = std::vector<float>(lower_bound.data(), lower_bound.data() + model.nq);
        json["upper"] = std::vector<float>(upper_bound.data(), upper_bound.data() + model.nq);
        json["measure"] = measure;
        json["end_effector"] = end_effector_name;
        json["end_effector_index"] = end_effector_index;
        json["min_radius"] = min_radius;
        json["max_radius"] = max_radius;
        json["joint_names"] = dof_to_joint_names();
        json["allowed_link_pairs"] = allowed_link_pairs;
        json["per_link_spheres"] = per_link_spheres;
        json["links_with_geometry"] = links_with_geometry;
        json["bounding_sphere_index"] = bounding_sphere_index;
        json["end_effector_collisions"] = get_frames_colliding_end_effector();
        json["euclidean"] = is_euclidean(model);
        json["joint_topology"] = generate_joint_topology(model);

        std::vector<std::string> link_names;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            link_names.emplace_back(model.frames[i].name);
        }
        json["link_names"] = link_names;

        return json;
    }

    auto RobotInfo::dof_to_joint_names() -> std::vector<std::string>
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

    auto RobotInfo::get_frames_colliding_end_effector() -> std::vector<std::size_t>
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

    auto RobotInfo::extract_spheres() -> void
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
            }
            else
            {
                throw std::runtime_error(
                    fmt::format("Invalid non-sphere geometry in URDF {}", geom_obj.name));
            }
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

    auto RobotInfo::collision_pair_to_frame_pair(const CollisionPair &cp)
        -> std::pair<std::size_t, std::size_t>
    {
        const auto &geom1 = collision_model.geometryObjects[cp.first];
        const auto &geom2 = collision_model.geometryObjects[cp.second];

        std::size_t link1_idx = geom1.parentFrame;
        std::size_t link2_idx = geom2.parentFrame;

        return std::make_pair(std::min(link1_idx, link2_idx), std::max(link1_idx, link2_idx));
    }

    auto RobotInfo::extract_collision_data() -> void
    {
        for (const auto &cp : collision_model.collisionPairs)
        {
            allowed_link_pairs.insert(collision_pair_to_frame_pair(cp));
        }
    }

    auto RobotInfo::get_adjacent_frames() -> std::set<std::pair<std::size_t, std::size_t>>
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

    auto RobotInfo::guess_self_collisions(std::size_t n) -> void
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
}  // namespace cricket

#include <cricket/robot_info.hh>

#include "joint_utils.hh"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/model.hpp>
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
      : RobotInfo(
            urdf_file,
            srdf_file,
            end_effector ? std::vector<std::string>{*end_effector} : std::vector<std::string>{})
    {
    }

    auto JointSelection::from_json(const nlohmann::json &data) -> std::optional<JointSelection>
    {
        if (not data.contains("active_joints"))
        {
            return std::nullopt;
        }

        JointSelection selection;
        selection.active_joints = data["active_joints"].get<std::vector<std::string>>();

        if (data.contains("default_configuration"))
        {
            for (const auto &[name, value] : data["default_configuration"].items())
            {
                if (value.is_array())
                {
                    selection.default_configuration[name] = value.get<std::vector<double>>();
                }
                else
                {
                    selection.default_configuration[name] = {value.get<double>()};
                }
            }
        }

        return selection;
    }

    auto CompositeSpec::from_json(const nlohmann::json &data, const std::filesystem::path &base_path)
        -> std::optional<CompositeSpec>
    {
        if (not data.contains("parts"))
        {
            return std::nullopt;
        }

        CompositeSpec spec;
        for (const auto &entry : data["parts"])
        {
            CompositePart part;
            part.prefix = entry.at("prefix").get<std::string>();
            part.urdf = base_path / entry.at("urdf").get<std::string>();

            if (entry.contains("srdf"))
            {
                part.srdf = base_path / entry["srdf"].get<std::string>();
            }

            part.parent = entry.value("parent", std::string("world"));

            Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
            if (entry.contains("xyz"))
            {
                const auto v = entry["xyz"].get<std::vector<double>>();
                if (v.size() != 3)
                {
                    throw std::runtime_error(
                        fmt::format("Part `{}`: xyz must have 3 elements!", part.prefix));
                }
                xyz = Eigen::Vector3d(v[0], v[1], v[2]);
            }

            Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
            if (entry.contains("rpy"))
            {
                const auto v = entry["rpy"].get<std::vector<double>>();
                if (v.size() != 3)
                {
                    throw std::runtime_error(
                        fmt::format("Part `{}`: rpy must have 3 elements!", part.prefix));
                }
                // URDF fixed-axis convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
                rotation = (Eigen::AngleAxisd(v[2], Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(v[1], Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(v[0], Eigen::Vector3d::UnitX()))
                               .toRotationMatrix();
            }
            else if (entry.contains("quat"))
            {
                const auto v = entry["quat"].get<std::vector<double>>();
                if (v.size() != 4)
                {
                    throw std::runtime_error(
                        fmt::format("Part `{}`: quat must be [x, y, z, w]!", part.prefix));
                }
                Eigen::Quaterniond q(v[3], v[0], v[1], v[2]);
                q.normalize();
                rotation = q.toRotationMatrix();
            }

            part.placement = pinocchio::SE3(rotation, xyz);
            spec.parts.push_back(std::move(part));
        }

        if (data.contains("disabled_collisions"))
        {
            for (const auto &pair : data["disabled_collisions"])
            {
                if (not pair.is_array() or pair.size() != 2)
                {
                    throw std::runtime_error(
                        "disabled_collisions entries must be [link_a, link_b] pairs!");
                }

                spec.disabled_collisions.emplace_back(
                    pair[0].get<std::string>(), pair[1].get<std::string>());
            }
        }

        return spec;
    }

    RobotInfo::RobotInfo(
        const std::filesystem::path &urdf_file,
        const std::optional<std::filesystem::path> &srdf_file,
        const std::vector<std::string> &end_effectors,
        const std::optional<JointSelection> &joint_selection)
    {
        if (not std::filesystem::exists(urdf_file))
        {
            throw std::runtime_error(fmt::format("URDF file {} does not exist!", urdf_file.string()));
        }

        pinocchio::urdf::buildModel(urdf_file, model, false, true);
        pinocchio::urdf::buildGeom(model, urdf_file, COLLISION, collision_model);

        if (joint_selection and not joint_selection->active_joints.empty())
        {
            reduce_model(*joint_selection);
        }

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
        resolve_end_effectors(end_effectors);
    }

    auto RobotInfo::resolve_end_effectors(const std::vector<std::string> &end_effectors) -> void
    {
        if (end_effectors.empty())
        {
            end_effector_names.push_back(model.frames[model.nframes - 1].name);
            fmt::print("No EE provided, using distal link `{}`.\n", end_effector_names.front());
        }
        else
        {
            for (const auto &name : end_effectors)
            {
                if (not model.existFrame(name, BODY))
                {
                    throw std::runtime_error(fmt::format("Invalid EE name {}", name));
                }

                end_effector_names.push_back(name);
            }
        }

        for (const auto &name : end_effector_names)
        {
            end_effector_indexes.push_back(model.getFrameId(name, BODY));
        }

        end_effector_name = end_effector_names.front();
        end_effector_index = end_effector_indexes.front();
    }

    namespace
    {
        auto prefix_model_names(Model &model, GeometryModel &geom, const std::string &prefix) -> void
        {
            // Index 0 is the universe joint/frame in both containers; appendModel merges
            // it away, so it must keep its canonical name.
            for (auto i = 1; i < model.njoints; ++i)
            {
                model.names[i] = fmt::format("{}_{}", prefix, model.names[i]);
            }

            for (auto i = 1; i < model.nframes; ++i)
            {
                model.frames[i].name = fmt::format("{}_{}", prefix, model.frames[i].name);
            }

            for (auto &g : geom.geometryObjects)
            {
                g.name = fmt::format("{}_{}", prefix, g.name);
            }
        }

        // Applies the part's SRDF to its standalone model and returns the geometry-name
        // pairs it disables (already prefixed), so that after the merge only within-part
        // pairs are removed and every cross-part pair stays active.
        auto record_srdf_disabled_pairs(
            const Model &model,
            GeometryModel &geom,
            const std::filesystem::path &srdf,
            const std::string &prefix) -> std::vector<std::pair<std::string, std::string>>
        {
            geom.addAllCollisionPairs();
            const auto all_pairs = geom.collisionPairs;
            pinocchio::srdf::removeCollisionPairs(model, geom, srdf.string());

            std::set<std::pair<std::size_t, std::size_t>> remaining;
            for (const auto &cp : geom.collisionPairs)
            {
                remaining.emplace(cp.first, cp.second);
            }

            std::vector<std::pair<std::string, std::string>> disabled;
            for (const auto &cp : all_pairs)
            {
                if (remaining.count({cp.first, cp.second}) == 0)
                {
                    disabled.emplace_back(
                        fmt::format("{}_{}", prefix, geom.geometryObjects[cp.first].name),
                        fmt::format("{}_{}", prefix, geom.geometryObjects[cp.second].name));
                }
            }

            geom.removeAllCollisionPairs();
            return disabled;
        }
    }  // namespace

    RobotInfo::RobotInfo(
        const CompositeSpec &composite,
        const std::vector<std::string> &end_effectors,
        const std::optional<JointSelection> &joint_selection)
    {
        if (composite.parts.empty())
        {
            throw std::runtime_error("Composite robot must have at least one part!");
        }

        std::set<std::string> prefixes;
        for (const auto &part : composite.parts)
        {
            if (not prefixes.insert(part.prefix).second)
            {
                throw std::runtime_error(
                    fmt::format("Duplicate part prefix `{}` in composite!", part.prefix));
            }
        }

        std::vector<std::pair<std::string, std::string>> disabled_geom_pairs;
        bool any_srdf = false;

        for (const auto &part : composite.parts)
        {
            if (not std::filesystem::exists(part.urdf))
            {
                throw std::runtime_error(
                    fmt::format("URDF file {} does not exist!", part.urdf.string()));
            }

            Model part_model;
            GeometryModel part_collision;
            pinocchio::urdf::buildModel(part.urdf, part_model, false, true);
            pinocchio::urdf::buildGeom(part_model, part.urdf, COLLISION, part_collision);

            if (part.srdf)
            {
                if (not std::filesystem::exists(*part.srdf))
                {
                    throw std::runtime_error(
                        fmt::format("SRDF file {} does not exist!", part.srdf->string()));
                }

                any_srdf = true;
                const auto disabled =
                    record_srdf_disabled_pairs(part_model, part_collision, *part.srdf, part.prefix);
                disabled_geom_pairs.insert(disabled_geom_pairs.end(), disabled.begin(), disabled.end());
            }

            prefix_model_names(part_model, part_collision, part.prefix);

            FrameIndex attach_frame = 0;
            if (part.parent != "world")
            {
                if (not model.existFrame(part.parent))
                {
                    throw std::runtime_error(
                        fmt::format(
                            "Part `{}`: parent frame `{}` not found (parents must come from "
                            "earlier-declared parts)!",
                            part.prefix,
                            part.parent));
                }

                attach_frame = model.getFrameId(part.parent);
            }

            Model merged;
            GeometryModel merged_collision;
            pinocchio::appendModel(
                model,
                part_model,
                collision_model,
                part_collision,
                attach_frame,
                part.placement,
                merged,
                merged_collision);
            model = merged;
            collision_model = merged_collision;
        }

        if (joint_selection and not joint_selection->active_joints.empty())
        {
            reduce_model(*joint_selection);
        }

        if (not any_srdf and composite.disabled_collisions.empty())
        {
            fmt::print("No SRDF files provided, guessing collisions!\n");
            guess_self_collisions();
        }
        else
        {
            // All pairs active by default so cross-part collisions are checked; remove
            // each part's SRDF-disabled pairs and the recipe's disabled_collisions.
            collision_model.addAllCollisionPairs();

            for (const auto &[name_a, name_b] : disabled_geom_pairs)
            {
                const auto id_a = collision_model.getGeometryId(name_a);
                const auto id_b = collision_model.getGeometryId(name_b);
                collision_model.removeCollisionPair(
                    CollisionPair(std::min(id_a, id_b), std::max(id_a, id_b)));
            }

            for (const auto &[link_a, link_b] : composite.disabled_collisions)
            {
                for (const auto &name : {link_a, link_b})
                {
                    if (not model.existFrame(name, BODY))
                    {
                        throw std::runtime_error(
                            fmt::format("disabled_collisions link `{}` not found!", name));
                    }
                }

                const auto frame_a = static_cast<std::size_t>(model.getFrameId(link_a, BODY));
                const auto frame_b = static_cast<std::size_t>(model.getFrameId(link_b, BODY));
                const auto frame_pair =
                    std::make_pair(std::min(frame_a, frame_b), std::max(frame_a, frame_b));

                std::vector<CollisionPair> to_remove;
                for (const auto &cp : collision_model.collisionPairs)
                {
                    if (collision_pair_to_frame_pair(cp) == frame_pair)
                    {
                        to_remove.push_back(cp);
                    }
                }

                for (const auto &cp : to_remove)
                {
                    collision_model.removeCollisionPair(cp);
                }
            }

            extract_collision_data();
        }

        extract_spheres();
        resolve_end_effectors(end_effectors);
    }

    auto RobotInfo::reduce_model(const JointSelection &selection) -> void
    {
        const std::set<std::string> active(
            selection.active_joints.begin(), selection.active_joints.end());
        for (const auto &name : active)
        {
            if (not model.existJointName(name))
            {
                throw std::runtime_error(fmt::format("Active joint `{}` does not exist!", name));
            }
        }

        std::vector<JointIndex> joints_to_lock;
        for (JointIndex i = 1; i < static_cast<JointIndex>(model.njoints); ++i)
        {
            if (active.count(model.names[i]) == 0)
            {
                joints_to_lock.push_back(i);
            }
        }

        if (joints_to_lock.empty())
        {
            return;
        }

        Eigen::VectorXd q_ref = pinocchio::neutral(model);
        for (const auto &[name, values] : selection.default_configuration)
        {
            if (not model.existJointName(name))
            {
                throw std::runtime_error(
                    fmt::format("Default configuration joint `{}` does not exist!", name));
            }

            // Values for still-active joints are irrelevant to reduction.
            if (active.count(name) != 0)
            {
                continue;
            }

            const auto joint_id = model.getJointId(name);
            const auto idx_q = model.joints[joint_id].idx_q();
            const auto nq = static_cast<std::size_t>(model.joints[joint_id].nq());

            if (values.size() == nq)
            {
                for (std::size_t i = 0; i < nq; ++i)
                {
                    q_ref[idx_q + static_cast<Eigen::Index>(i)] = values[i];
                }
            }
            else if (values.size() == 1 and nq == 2)
            {
                // Continuous joint: a single value is the angle, stored as cos/sin.
                q_ref[idx_q] = std::cos(values[0]);
                q_ref[idx_q + 1] = std::sin(values[0]);
            }
            else
            {
                throw std::runtime_error(
                    fmt::format(
                        "Joint `{}` default configuration has {} values, expected {}!",
                        name,
                        values.size(),
                        nq));
            }
        }

        pinocchio::Model reduced_model;
        pinocchio::GeometryModel reduced_collision;
        pinocchio::buildReducedModel(
            model, collision_model, joints_to_lock, q_ref, reduced_model, reduced_collision);
        model = reduced_model;
        collision_model = reduced_collision;
    }

    namespace
    {
        auto compute_extended_bounds(const pinocchio::Model &model, const std::optional<Bounds> &bounds)
            -> std::pair<Eigen::VectorXd, Eigen::VectorXd>
        {
            Eigen::VectorXd lower = model.lowerPositionLimit;
            Eigen::VectorXd upper = model.upperPositionLimit;

            if (not bounds)
            {
                return {lower, upper};
            }

            auto [_, joint_mappings] = classify_joints(model);
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
            auto [_, joint_mappings] = classify_joints(model);
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
            auto [_, joint_mappings] = classify_joints(model);
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

        auto generate_nn_segments(const pinocchio::Model &model) -> nlohmann::json
        {
            auto [_, joint_mappings] = classify_joints(model);
            nlohmann::json segments = nlohmann::json::array();

            std::size_t lp_start = 0;
            std::size_t lp_size = 0;

            const auto extend_lp = [&](std::size_t idx_q, std::size_t nq)
            {
                if (lp_size == 0)
                {
                    lp_start = idx_q;
                }

                lp_size += nq;
            };

            const auto flush_lp = [&]()
            {
                if (lp_size == 0)
                {
                    return;
                }

                nlohmann::json seg;
                seg["type"] = "LP";
                seg["offset"] = lp_start;
                seg["size"] = lp_size;
                segments.push_back(seg);
                lp_size = 0;
            };

            for (const auto &jm : joint_mappings)
            {
                if (jm.type == JointType::Bounded or jm.type == JointType::UnboundedRevolute or
                    jm.type == JointType::SE2)
                {
                    extend_lp(jm.idx_q, jm.nq);
                }
                else if (jm.type == JointType::SO3)
                {
                    flush_lp();
                    nlohmann::json seg;
                    seg["type"] = "SO3";
                    seg["offset"] = jm.idx_q;
                    segments.push_back(seg);
                }
                else if (jm.type == JointType::SE3)
                {
                    extend_lp(jm.idx_q, 3);
                    flush_lp();
                    nlohmann::json seg;
                    seg["type"] = "SO3";
                    seg["offset"] = jm.idx_q + 3;
                    segments.push_back(seg);
                }
            }
            flush_lp();
            return segments;
        }

        auto generate_so3_offsets(const pinocchio::Model &model) -> nlohmann::json
        {
            auto [_, joint_mappings] = classify_joints(model);
            nlohmann::json offsets = nlohmann::json::array();
            for (const auto &jm : joint_mappings)
            {
                if (jm.type == JointType::SO3)
                {
                    offsets.push_back(jm.idx_q);
                }
                else if (jm.type == JointType::SE3)
                {
                    offsets.push_back(jm.idx_q + 3);
                }
            }
            return offsets;
        }
    }  // namespace

    auto RobotInfo::json(const std::optional<Bounds> &bounds, bool skip_static_environment)
        -> nlohmann::json
    {
        // A frame is static when its parent joint is the universe (0): every joint
        // between it and the world was fixed (or locked out by reduction).
        const auto is_static_link = [&](std::size_t link_index)
        { return model.frames[link_index].parentJoint == 0; };
        const auto [lower_bound, upper_bound] = compute_extended_bounds(model, bounds);
        const Eigen::VectorXd bound_range = upper_bound - lower_bound;
        const Eigen::VectorXd bound_descale = bound_range.cwiseInverse();

        double measure = 1.0;
        for (auto i = 0U; i < bound_range.size(); ++i)
        {
            if (std::isfinite(bound_range[i]))
            {
                measure *= bound_range[i];
            }
        }

        if (not std::isfinite(measure))
        {
            measure = static_cast<double>(std::numeric_limits<float>::max());
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

        const Eigen::VectorXd velocity_limit = model.velocityLimit;
        const Eigen::VectorXd effort_limit = model.effortLimit;
        json["velocity_limits"] =
            std::vector<float>(velocity_limit.data(), velocity_limit.data() + model.nv);
        json["effort_limits"] = std::vector<float>(effort_limit.data(), effort_limit.data() + model.nv);
        json["measure"] = measure;
        json["end_effector"] = end_effector_name;
        json["end_effector_index"] = end_effector_index;
        json["end_effectors"] = end_effector_names;
        json["num_end_effectors"] = end_effector_names.size();
        json["min_radius"] = min_radius;
        json["max_radius"] = max_radius;
        json["joint_names"] = dof_to_joint_names();
        json["allowed_link_pairs"] = allowed_link_pairs;
        json["per_link_spheres"] = per_link_spheres;
        json["links_with_geometry"] = links_with_geometry;
        json["bounding_sphere_index"] = bounding_sphere_index;
        nlohmann::json end_effector_collisions = nlohmann::json::array();
        for (const auto &frame_index : end_effector_indexes)
        {
            end_effector_collisions.push_back(get_frames_colliding_end_effector(frame_index));
        }
        json["end_effector_collisions"] = end_effector_collisions;
        json["euclidean"] = is_euclidean(model);
        json["joint_topology"] = generate_joint_topology(model);
        json["nn_segments"] = generate_nn_segments(model);
        json["so3_offsets"] = generate_so3_offsets(model);

        std::vector<std::string> link_names;
        for (auto i = 0U; i < model.frames.size(); ++i)
        {
            link_names.emplace_back(model.frames[i].name);
        }
        json["link_names"] = link_names;

        // Array indexes (into links_with_geometry, descending — the template's iteration
        // order) of links checked against the environment.
        nlohmann::json environment_links = nlohmann::json::array();
        nlohmann::json env_entries = nlohmann::json::array();
        nlohmann::json env_body_idx = nlohmann::json::array();
        const std::size_t n_links = links_with_geometry.size();
        std::size_t body_offset = 0;
        float env_min_radius = std::numeric_limits<float>::max();
        float env_max_radius = std::numeric_limits<float>::lowest();
        for (std::size_t i = 0; i < n_links; ++i)
        {
            const std::size_t array_index = n_links - 1 - i;
            const std::size_t link_index = links_with_geometry[array_index];
            if (skip_static_environment and is_static_link(link_index))
            {
                continue;
            }
            environment_links.push_back(array_index);
            const auto &body = per_link_spheres[link_index];
            env_entries.push_back({array_index, body_offset, body.size()});
            for (const auto &s : body)
            {
                env_body_idx.push_back(s);
                env_min_radius = std::min(env_min_radius, spheres[s].radius);
                env_max_radius = std::max(env_max_radius, spheres[s].radius);
            }
            body_offset += body.size();
        }
        json["environment_links"] = environment_links;
        json["compact_env_entries"] = env_entries;
        json["compact_env_body_idx"] = env_body_idx;

        // min/max_radius cover the fine spheres queried against the environment, restricted to
        // links that actually get environment checks (mobile links under skip_static_environment).
        // The per-link bounding spheres that gate fine-sphere checks in fkcc are deliberately
        // excluded: CAPT answers queries above max_radius conservatively, so the gate stays sound
        // without inflating the build contract.
        if (not environment_links.empty())
        {
            json["min_radius"] = env_min_radius;
            json["max_radius"] = env_max_radius;
        }

        // Per-sphere skip markers so fkcc_debug's per-sphere environment queries stay
        // consistent with fkcc (skipped spheres get an empty collision list).
        nlohmann::json sphere_env_skip = nlohmann::json::array();
        for (const auto &sphere : spheres)
        {
            sphere_env_skip.push_back(skip_static_environment and sphere.parent_joint == 0);
        }
        json["sphere_env_skip"] = sphere_env_skip;

        nlohmann::json self_entries = nlohmann::json::array();
        nlohmann::json self_pair_a = nlohmann::json::array();
        nlohmann::json self_pair_b = nlohmann::json::array();
        std::size_t pair_offset = 0;
        for (const auto &[link1, link2] : allowed_link_pairs)
        {
            const std::size_t bs1 = bounding_sphere_index[link1];
            const std::size_t bs2 = bounding_sphere_index[link2];
            const auto &spheres1 = per_link_spheres[link1];
            const auto &spheres2 = per_link_spheres[link2];
            const std::size_t count = spheres1.size() * spheres2.size();
            self_entries.push_back({bs1, bs2, pair_offset, count});
            for (const auto &a : spheres1)
            {
                for (const auto &b : spheres2)
                {
                    self_pair_a.push_back(a);
                    self_pair_b.push_back(b);
                }
            }
            pair_offset += count;
        }
        json["compact_self_entries"] = self_entries;
        json["compact_self_pair_a"] = self_pair_a;
        json["compact_self_pair_b"] = self_pair_b;

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

    auto RobotInfo::get_frames_colliding_end_effector(std::size_t frame_index) -> std::vector<std::size_t>
    {
        std::size_t end_effector_joint = model.frames[frame_index].parentJoint;

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

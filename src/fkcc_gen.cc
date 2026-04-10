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
#include <cxxopts.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>

#include "lang_cpp.hh"
#include "lang_rust.hh"

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

// Joint type classification for mapToConfiguration
enum class JointType
{
    Bounded,           // Revolute/Prismatic with limits: nq=1, nu=1
    UnboundedRevolute, // Unbounded revolute (cos,sin): nq=2, nu=1
    SO3,               // Spherical quaternion: nq=4, nu=3
    SE3,               // FreeFlyer: nq=7, nu=6
    SE2,               // Planar: nq=4, nu=3
    Unsupported
};

struct JointMapping
{
    JointType type;
    std::size_t joint_id;
    std::size_t idx_q;  // Start index in configuration vector
    std::size_t idx_u;  // Start index in [0,1] input vector
    std::size_t nq;     // Configuration DOFs
    std::size_t nu;     // Number of [0,1] inputs needed
};

struct Bounds
{
    Eigen::Vector3d lower;
    Eigen::Vector3d upper;
};

auto classify_joint_type(const std::string &shortname, int nq) -> JointType
{
    if (shortname.find("FreeFlyer") != std::string::npos)
    {
        return JointType::SE3;
    }
    if (shortname.find("Planar") != std::string::npos && nq == 4)
    {
        return JointType::SE2;
    }
    if (shortname.find("Spherical") != std::string::npos && nq == 4)
    {
        return JointType::SO3;
    }
    if (shortname.find("Unbounded") != std::string::npos && nq == 2)
    {
        return JointType::UnboundedRevolute;
    }
    if (nq == 1)
    {
        return JointType::Bounded;
    }

    return JointType::Unsupported;
}

// Get number of [0,1] inputs needed for a joint type
auto get_nu_for_type(JointType type) -> std::size_t
{
    switch (type)
    {
        case JointType::Bounded:
            return 1;
        case JointType::UnboundedRevolute:
            return 1;
        case JointType::SO3:
            return 3;
        case JointType::SE3:
            return 6;
        case JointType::SE2:
            return 3;
        default:
            return 0;
    }
}

// Classify all joints in a model, returns (total_nu, joint_mappings)
auto classify_joints(const Model &model) -> std::pair<std::size_t, std::vector<JointMapping>>
{
    std::vector<JointMapping> mappings;
    std::size_t total_nu = 0;

    // Joint 0 is universe/root, skip it
    for (auto joint_id = 1U; joint_id < model.joints.size(); ++joint_id)
    {
        const auto &joint = model.joints[joint_id];
        std::string shortname = joint.shortname();
        auto nq = joint.nq();

        if (nq == 0)
        {
            continue;  // Fixed joint
        }

        JointType type = classify_joint_type(shortname, nq);
        std::size_t nu = get_nu_for_type(type);

        if (type == JointType::Unsupported)
        {
            throw std::runtime_error(
                fmt::format("Unsupported joint type: {} (shortname: {}, nq: {})",
                            model.names[joint_id], shortname, nq));
        }

        JointMapping mapping;
        mapping.type = type;
        mapping.joint_id = joint_id;
        mapping.idx_q = joint.idx_q();
        mapping.idx_u = total_nu;
        mapping.nq = nq;
        mapping.nu = nu;

        mappings.push_back(mapping);
        total_nu += nu;
    }

    return {total_nu, mappings};
}

// Maps [0,1] to bounded range: q = lower + u * (upper - lower)
template <typename Scalar>
auto map_bounded(Scalar u, double lower, double upper) -> Scalar
{
    return Scalar(lower) + u * Scalar(upper - lower);
}

// Maps [0,1] to (cos, sin) for unbounded revolute
template <typename Scalar>
void map_unbounded_revolute(Scalar u, Scalar &cos_out, Scalar &sin_out)
{
    constexpr double two_pi = 2.0 * M_PI;
    Scalar theta = u * Scalar(two_pi);
    cos_out = cos(theta);
    sin_out = sin(theta);
}

// Shoemake's algorithm for uniform quaternion sampling from 3 uniform [0,1] inputs
// Returns quaternion as (x, y, z, w) - Pinocchio convention
template <typename Scalar>
void map_so3_shoemake(Scalar u1, Scalar u2, Scalar u3, Scalar &x, Scalar &y, Scalar &z, Scalar &w)
{
    constexpr double two_pi = 2.0 * M_PI;

    Scalar sqrt1_minus_u1 = sqrt(Scalar(1.0) - u1);
    Scalar sqrt_u1 = sqrt(u1);
    Scalar theta1 = u2 * Scalar(two_pi);
    Scalar theta2 = u3 * Scalar(two_pi);

    // Pinocchio uses (x, y, z, w) quaternion order
    x = sqrt1_minus_u1 * sin(theta1);
    y = sqrt1_minus_u1 * cos(theta1);
    z = sqrt_u1 * sin(theta2);
    w = sqrt_u1 * cos(theta2);
}

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

    auto json() -> nlohmann::json
    {
        const Eigen::VectorXd lower_bound = model.lowerPositionLimit;
        const Eigen::VectorXd upper_bound = model.upperPositionLimit;
        const Eigen::VectorXd bound_range = upper_bound - lower_bound;

        float measure = 1.0;
        for (auto i = 0U; i < bound_range.size(); ++i)
        {
            if (std::isfinite(bound_range[i]))
            {
                measure *= bound_range[i];
            }
        }

        nlohmann::json json;
        json["n_q"] = model.nq;
        json["n_spheres"] = spheres.size();
        json["measure"] = measure;
        json["upper"] = std::vector<float>(upper_bound.data(), upper_bound.data() + model.nq);
        json["lower"] = std::vector<float>(lower_bound.data(), lower_bound.data() + model.nq);;
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

    auto add_mimic_joint(const std::string &name, const std::string &joint, double multiplier, double offset)
        -> void
    {
        Model temp;
        pinocchio::transformJointIntoMimic(
            model, model.getJointId(joint), model.getJointId(name), multiplier, offset, temp);
        model = temp;
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
    const std::string &language,
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

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), n_out};
}

// Trace a function that maps [0,1]^nu inputs to valid robot configurations
auto trace_map_to_configuration(
    const Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds = std::nullopt) -> Traced
{
    auto [nu, joint_mappings] = classify_joints(model);
    auto nq = model.nq;

    // Check if bounds are required
    for (const auto &jm : joint_mappings)
    {
        if (jm.type == JointType::SE3 && !bounds)
        {
            throw std::runtime_error(
                "FreeFlyer joint detected but no bounds provided. "
                "Please specify lower and upper in the configuration.");
        }
        if (jm.type == JointType::SE2 && !bounds)
        {
            throw std::runtime_error(
                "Planar joint detected but no bounds provided. "
                "Please specify lower and upper in the configuration.");
        }
    }

    ADVectorXs ad_u(nu);   // [0,1] inputs
    ADVectorXs ad_q(nq);   // Configuration output

    // Initialize inputs
    for (auto i = 0U; i < nu; ++i)
    {
        ad_u[i] = ADCG(0.0);
    }

    Independent(ad_u);

    // Apply joint-specific mappings
    for (const auto &jm : joint_mappings)
    {
        switch (jm.type)
        {
            case JointType::Bounded:
            {
                // Linear mapping: q = lower + u * (upper - lower)
                double lower = model.lowerPositionLimit[jm.idx_q];
                double upper = model.upperPositionLimit[jm.idx_q];
                ad_q[jm.idx_q] = map_bounded(ad_u[jm.idx_u], lower, upper);
                break;
            }
            case JointType::UnboundedRevolute:
            {
                // Maps [0,1] to (cos, sin)
                ADCG cos_val, sin_val;
                map_unbounded_revolute(ad_u[jm.idx_u], cos_val, sin_val);
                ad_q[jm.idx_q] = cos_val;
                ad_q[jm.idx_q + 1] = sin_val;
                break;
            }
            case JointType::SO3:
            {
                // Shoemake's algorithm for uniform quaternion
                ADCG x, y, z, w;
                map_so3_shoemake(ad_u[jm.idx_u], ad_u[jm.idx_u + 1], ad_u[jm.idx_u + 2], x, y, z, w);
                // Pinocchio quaternion order: (x, y, z, w)
                ad_q[jm.idx_q] = x;
                ad_q[jm.idx_q + 1] = y;
                ad_q[jm.idx_q + 2] = z;
                ad_q[jm.idx_q + 3] = w;
                break;
            }
            case JointType::SE3:
            {
                // FreeFlyer: position (3 inputs) + orientation (3 inputs)
                // Position mapping
                for (int i = 0; i < 3; ++i)
                {
                    double lo = bounds->lower[i];
                    double hi = bounds->upper[i];
                    ad_q[jm.idx_q + i] = map_bounded(ad_u[jm.idx_u + i], lo, hi);
                }
                // Orientation mapping (Shoemake)
                ADCG x, y, z, w;
                map_so3_shoemake(ad_u[jm.idx_u + 3], ad_u[jm.idx_u + 4], ad_u[jm.idx_u + 5], x, y, z, w);
                ad_q[jm.idx_q + 3] = x;
                ad_q[jm.idx_q + 4] = y;
                ad_q[jm.idx_q + 5] = z;
                ad_q[jm.idx_q + 6] = w;
                break;
            }
            case JointType::SE2:
            {
                // Planar: position (2 inputs) + orientation (1 input)
                // Position mapping (x, y)
                for (int i = 0; i < 2; ++i)
                {
                    double lo = bounds->lower[i];
                    double hi = bounds->upper[i];
                    ad_q[jm.idx_q + i] = map_bounded(ad_u[jm.idx_u + i], lo, hi);
                }
                // Orientation mapping (cos(θ), sin(θ))
                ADCG cos_val, sin_val;
                map_unbounded_revolute(ad_u[jm.idx_u + 2], cos_val, sin_val);
                ad_q[jm.idx_q + 2] = cos_val;
                ad_q[jm.idx_q + 3] = sin_val;
                break;
            }
            default:
                throw std::runtime_error("Unsupported joint type in trace_map_to_configuration");
        }
    }

    // Create the AD function
    ADFun<CGD> map_func(ad_u, ad_q);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nu);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = map_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), static_cast<std::size_t>(nq)};
}

// Get the randomness dimension (nu) for a model
auto get_randomness_dimension(const Model &model) -> std::size_t
{
    auto [nu, _] = classify_joints(model);
    return nu;
}

// Trace a function that checks if a configuration is within bounds
// Returns 1.0 if all joints are within bounds, 0.0 otherwise
auto trace_check_bounds(
    const Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds = std::nullopt) -> Traced
{
    auto [nu, joint_mappings] = classify_joints(model);
    auto nq = model.nq;

    // Check if bounds are required
    for (const auto &jm : joint_mappings)
    {
        if (jm.type == JointType::SE3 && !bounds)
        {
            throw std::runtime_error(
                "FreeFlyer joint detected but no bounds provided. "
                "Please specify lower and upper in the configuration.");
        }
        if (jm.type == JointType::SE2 && !bounds)
        {
            throw std::runtime_error(
                "Planar joint detected but no bounds provided. "
                "Please specify lower and upper in the configuration.");
        }
    }

    ADVectorXs ad_q(nq);
    ADVectorXs ad_out(1);

    for (auto i = 0; i < nq; ++i)
    {
        ad_q[i] = ADCG(0.0);
    }

    Independent(ad_q);

    // Start with valid = 1.0, multiply by per-joint validity
    ADCG valid = ADCG(1.0);

    for (const auto &jm : joint_mappings)
    {
        switch (jm.type)
        {
            case JointType::Bounded:
            {
                double lo = model.lowerPositionLimit[jm.idx_q];
                double hi = model.upperPositionLimit[jm.idx_q];
                ADCG q = ad_q[jm.idx_q];
                ADCG above_lo = CppAD::CondExpGe(q, ADCG(lo), ADCG(1.0), ADCG(0.0));
                ADCG below_hi = CppAD::CondExpLe(q, ADCG(hi), ADCG(1.0), ADCG(0.0));
                valid *= above_lo * below_hi;
                break;
            }
            case JointType::UnboundedRevolute:
            case JointType::SO3:
                // Always valid by construction
                break;
            case JointType::SE3:
            {
                for (int i = 0; i < 3; ++i)
                {
                    double lo = bounds->lower[i];
                    double hi = bounds->upper[i];
                    ADCG q = ad_q[jm.idx_q + i];
                    ADCG above_lo = CppAD::CondExpGe(q, ADCG(lo), ADCG(1.0), ADCG(0.0));
                    ADCG below_hi = CppAD::CondExpLe(q, ADCG(hi), ADCG(1.0), ADCG(0.0));
                    valid *= above_lo * below_hi;
                }
                break;
            }
            case JointType::SE2:
            {
                for (int i = 0; i < 2; ++i)
                {
                    double lo = bounds->lower[i];
                    double hi = bounds->upper[i];
                    ADCG q = ad_q[jm.idx_q + i];
                    ADCG above_lo = CppAD::CondExpGe(q, ADCG(lo), ADCG(1.0), ADCG(0.0));
                    ADCG below_hi = CppAD::CondExpLe(q, ADCG(hi), ADCG(1.0), ADCG(0.0));
                    valid *= above_lo * below_hi;
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported joint type in trace_check_bounds");
        }
    }

    ad_out[0] = valid;

    ADFun<CGD> check_func(ad_q, ad_out);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = check_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), 1};
}

// Trace a function that interpolates between two configurations
// Input: q0 (nq), q1 (nq), t (1) -> Output: q_interp (nq)
auto trace_interpolate(const Model &model, const std::string &language) -> Traced
{
    auto nq = model.nq;
    std::size_t n_input = 2 * nq + 1;  // q0, q1, t

    // Cast model to AD scalar type for use with Pinocchio's interpolate
    using ADModel = pinocchio::ModelTpl<ADCG>;
    ADModel ad_model = model.cast<ADCG>();

    ADVectorXs ad_input(n_input);
    ADVectorXs ad_out(nq);

    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }

    Independent(ad_input);

    // Extract q0, q1, t from input
    ADVectorXs ad_q0 = ad_input.head(nq);
    ADVectorXs ad_q1 = ad_input.segment(nq, nq);
    ADCG t = ad_input[2 * nq];

    // Use Pinocchio's interpolate - handles all joint types via Lie group operations
    pinocchio::interpolate(ad_model, ad_q0, ad_q1, t, ad_out);

    ADFun<CGD> interp_func(ad_input, ad_out);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = interp_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), static_cast<std::size_t>(nq)};
}

int main(int argc, char **argv)
{
    cxxopts::Options options(argv[0], "Tracing compiler for forward kinematics and collision checking");

    options.positional_help("[JSON configuration filename]").show_positional_help();

    options.add_options()                                                                       //
        ("f,configuration_file", "JSON configuration filename", cxxopts::value<std::string>())  //
        ("o,output_filename", "Output JSON filename", cxxopts::value<std::string>())            //
        ("t,output_template",
         "Output template filename (override configuration file)",
         cxxopts::value<std::string>())  //
        ("h,help", "Print usage")        //
        ;

    options.parse_positional({"configuration_file"});

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (not result.count("configuration_file"))
    {
        throw std::runtime_error(fmt::format("Must provide configuration file!"));
    }

    std::filesystem::path json_path(result["configuration_file"].as<std::string>());
    auto parent_path = json_path.parent_path();

    if (not std::filesystem::exists(json_path))
    {
        throw std::runtime_error(fmt::format("JSON file {} does not exist!", json_path.string()));
    }

    std::ifstream json_file(json_path);
    nlohmann::json data;

    try
    {
        data = nlohmann::json::parse(json_file);
    }
    catch (std::exception &e)
    {
        throw std::runtime_error(fmt::format("Failed to parse JSON file! Error: \n{}", e.what()));
    }

    std::optional<std::filesystem::path> srdf_path = {};
    if (data.contains("srdf"))
    {
        srdf_path = parent_path / data["srdf"];
    }

    std::optional<std::string> end_effector_name = {};
    if (data.contains("end_effector"))
    {
        end_effector_name = data["end_effector"];
    }

    std::string language = "c++";
    if (data.contains("language"))
    {
        language = data["language"];
    }

    // Parse bounds if provided
    std::optional<Bounds> bounds = std::nullopt;
    if (data.contains("bounds"))
    {
        Bounds b;
        auto &bd = data["bounds"];
        if (!bd.contains("lower") || !bd.contains("upper"))
        {
            throw std::runtime_error(
                "bounds must contain both 'lower' and 'upper' arrays");
        }
        auto lower = bd["lower"].get<std::vector<double>>();
        auto upper = bd["upper"].get<std::vector<double>>();
        if (lower.size() < 2 || lower.size() > 3 || upper.size() < 2 || upper.size() > 3)
        {
            throw std::runtime_error("bounds arrays must have 2 or 3 elements");
        }
        b.lower = Eigen::Vector3d(lower[0], lower[1], lower.size() == 3 ? lower[2] : 0.0);
        b.upper = Eigen::Vector3d(upper[0], upper[1], upper.size() == 3 ? upper[2] : 0.0);
        bounds = b;
    }

    RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_name);

    if (data.contains("mimics"))
    {
        auto mimics = data["mimics"];
        for (const auto &mimic : mimics)
        {
            robot.add_mimic_joint(mimic["name"], mimic["joint"], mimic["multiplier"], mimic["offset"]);
        }
    }

    data.update(robot.json());

    auto traced_eefk_code = trace_sphere_cc_fk(robot, language, false, false, true);
    data["eefk_code"] = traced_eefk_code.code;
    data["eefk_code_vars"] = traced_eefk_code.temp_variables;
    data["eefk_code_output"] = traced_eefk_code.outputs;

    auto traced_spherefk_code = trace_sphere_cc_fk(robot, language, true, false, false);
    data["spherefk_code"] = traced_spherefk_code.code;
    data["spherefk_code_vars"] = traced_spherefk_code.temp_variables;
    data["spherefk_code_output"] = traced_spherefk_code.outputs;

    auto traced_ccfk_code = trace_sphere_cc_fk(robot, language, true, true, false);
    data["ccfk_code"] = traced_ccfk_code.code;
    data["ccfk_code_vars"] = traced_ccfk_code.temp_variables;
    data["ccfk_code_output"] = traced_ccfk_code.outputs;

    auto traced_ccfkee_code = trace_sphere_cc_fk(robot, language, true, true, true);
    data["ccfkee_code"] = traced_ccfkee_code.code;
    data["ccfkee_code_vars"] = traced_ccfkee_code.temp_variables;
    data["ccfkee_code_output"] = traced_ccfkee_code.outputs;

    // Trace mapToConfiguration function
    auto traced_mapconfig_code = trace_map_to_configuration(robot.model, language, bounds);
    data["mapconfig_code"] = traced_mapconfig_code.code;
    data["mapconfig_code_vars"] = traced_mapconfig_code.temp_variables;
    data["mapconfig_code_output"] = traced_mapconfig_code.outputs;
    data["n_u"] = get_randomness_dimension(robot.model);

    // Trace checkBounds function
    auto traced_checkbounds_code = trace_check_bounds(robot.model, language, bounds);
    data["checkbounds_code"] = traced_checkbounds_code.code;
    data["checkbounds_code_vars"] = traced_checkbounds_code.temp_variables;

    // Trace interpolate function
    auto traced_interpolate_code = trace_interpolate(robot.model, language);
    data["interpolate_code"] = traced_interpolate_code.code;
    data["interpolate_code_vars"] = traced_interpolate_code.temp_variables;
    data["interpolate_code_output"] = traced_interpolate_code.outputs;

    inja::Environment env;

    for (const auto &subt : data["subtemplates"])
    {
        inja::Template temp = env.parse_template(parent_path / subt["template"]);
        env.include_template(subt["name"], temp);
    }

    std::string output_template;
    if (result.count("output_template"))
    {
        output_template = result["output_template"].as<std::string>();
    }
    else
    {
        output_template = data["output"];
    }

    inja::Template temp = env.parse_template(parent_path / data["template"]);
    env.write(temp, data, output_template);

    std::string output_filename;
    if (result.count("output_filename"))
    {
        output_filename = result["output_filename"].as<std::string>();
    }
    else
    {
        output_filename = "output.json";
    }

    std::ofstream output_file(output_filename);
    output_file << data.dump();
    output_file.close();

    return 0;
}

// Standalone debugger for the eef-local-spheres rotate+translate math (see
// trace_eef_local_spheres in codegen.cc, and ParameterizedSpace::eef_world_poses /
// eefs_collision_free in fk_template.hh, which is what that trace ends up generating code
// for): loads the iiwa_marker robot directly from its URDF/SRDF (same paths as
// resources/iiwa_marker.json -- no JSON recipe, no inja templating, no codegen pipeline) and
// prints every eef-local sphere's world-frame center + radius at a hardcoded candidate
// end-effector world pose -- `world = R * local_offset + t`, computed here in plain double
// precision rather than through a CppAD/CppADCodeGen trace.
//
// Part 2 cross-checks that math against the real thing: running pinocchio::forwardKinematics
// at an example joint configuration and printing the same eef spheres computed from
// data.oMi[sphere.parent_joint] (see trace_sphere in codegen.cc for the traced equivalent),
// instead of a hand-picked SE3 pose.
#include <cricket/robot_info.hh>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>

#include <fmt/format.h>

#include <Eigen/Geometry>

#include <filesystem>
#include <stdexcept>

int main()
{
    const std::filesystem::path resources_dir = std::filesystem::path(CRICKET_RESOURCES_DIR);
    const std::filesystem::path urdf_path = resources_dir / "iiwa_marker/iiwa_marker_spherized.urdf";
    const std::filesystem::path srdf_path = resources_dir / "iiwa_marker/moveit/config/iiwa.srdf";
    const std::string end_effector_name = "iiwa_link_7";

    cricket::RobotInfo robot(urdf_path, srdf_path, end_effector_name);

    if (robot.end_effector_indexes.empty())
    {
        throw std::runtime_error(fmt::format("end effector '{}' not found on this robot", end_effector_name));
    }

    const auto &frame = robot.model.frames[robot.end_effector_indexes[0]];

    const double x = 0.5383931994438171;
    const double y = -0.2849438190460205;
    const double z = 0.2260778248310089;
    const double qx = 0.0;
    const double qy = -1.0;
    const double qz = 0.0;
    const double qw = 0.0;

    const Eigen::Vector3d translation{x, y, z};
    const Eigen::Matrix3d rotation = Eigen::Quaterniond{qw, qx, qy, qz}.normalized().toRotationMatrix();

    fmt::print(
        "eef '{}' world pose: t=({}, {}, {}) q=({}, {}, {}, {})\n",
        end_effector_name,
        x,
        y,
        z,
        qx,
        qy,
        qz,
        qw);

    std::size_t i = 0;
    for (const auto &sphere : robot.spheres)
    {
        if (sphere.parent_joint != frame.parentJoint)
        {
            continue;
        }

        const Eigen::Vector3d local_offset = frame.placement.inverse().act(sphere.relative.translation());
        const Eigen::Vector3d world = rotation * local_offset + translation;

        fmt::print(
            "  sphere {}: x={} y={} z={} r={}\n", i, world.x(), world.y(), world.z(), sphere.radius);
        ++i;
    }

    // --- Part 2: same spheres, computed by running real forward kinematics at an example
    // joint configuration instead of a hand-picked SE3 pose.
    Eigen::VectorXd q(robot.model.nq);
    q << 0.2228192239999771,
        1.6366102695465088,
        1.6272177696228027,
        1.4590731859207153,
        -1.5115604400634766,
        1.5075032711029053,
        -1.2399876117706299;

    pinocchio::Data data(robot.model);
    pinocchio::forwardKinematics(robot.model, data, q);
    pinocchio::updateFramePlacements(robot.model, data);

    fmt::print("\nq = ({}, {}, {}, {}, {}, {}, {})\n", q[0], q[1], q[2], q[3], q[4], q[5], q[6]);

    const auto &tip_pose = data.oMf[robot.end_effector_indexes[0]];
    const Eigen::Quaterniond tip_quat(tip_pose.rotation());
    fmt::print(
        "iiwa_tip pose: t=({}, {}, {}) q=({}, {}, {}, {})\n",
        tip_pose.translation().x(),
        tip_pose.translation().y(),
        tip_pose.translation().z(),
        tip_quat.x(),
        tip_quat.y(),
        tip_quat.z(),
        tip_quat.w());

    std::size_t j = 0;
    for (const auto &sphere : robot.spheres)
    {
        if (sphere.parent_joint != frame.parentJoint)
        {
            continue;
        }

        const auto &joint_placement = data.oMi[sphere.parent_joint];
        const Eigen::Vector3d world = joint_placement.rotation() * sphere.relative.translation() +
                                       joint_placement.translation();

        fmt::print(
            "  sphere {}: x={} y={} z={} r={}\n", j, world.x(), world.y(), world.z(), sphere.radius);
        ++j;
    }

    return 0;
}

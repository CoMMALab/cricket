// Tests for cuboid (box) collision geometry support: extraction, traced buffer layout, bounding
// spheres, and generated source.

#include <cricket/codegen.hh>
#include <cricket/robot_info.hh>

#include "check.hh"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <string>

namespace
{
    auto contains(const std::string &haystack, const std::string &needle) -> bool
    {
        return haystack.find(needle) != std::string::npos;
    }
}  // namespace

int main()
{
    const auto source_dir = std::filesystem::path(CRICKET_SOURCE_DIR);
    const auto urdf = source_dir / "tests/data/mixed_arm.urdf";
    const auto srdf = source_dir / "tests/data/mixed_arm.srdf";

    cricket::RobotInfo robot(urdf, srdf, std::string("link3"));

    // Extraction: two spheres (base, link2) and three boxes (link1, link2, link3)
    CRICKET_CHECK(robot.spheres.size() == 2);
    CRICKET_CHECK(robot.cuboids.size() == 3);

    // Slots are contiguous per type even though they are interleaved in the URDF
    for (auto i = 0U; i < robot.spheres.size(); ++i)
    {
        CRICKET_CHECK(robot.spheres[i].geom_index == i);
    }
    for (auto i = 0U; i < robot.cuboids.size(); ++i)
    {
        CRICKET_CHECK(robot.cuboids[i].index == i);
    }

    // URDF box sizes are full side lengths, so half-extents are half of them
    const auto &box1 = robot.cuboids[0];
    CRICKET_CHECK(std::abs(box1.half_extents[0] - 0.1F) < 1e-6F);
    CRICKET_CHECK(std::abs(box1.half_extents[1] - 0.05F) < 1e-6F);
    CRICKET_CHECK(std::abs(box1.half_extents[2] - 0.2F) < 1e-6F);

    // All four links have geometry, including link3, which has only a box
    CRICKET_CHECK(robot.links_with_geometry.size() == 4);
    CRICKET_CHECK(robot.bounding_spheres.size() == 4);

    const auto link1 = robot.model.getFrameId("link1");
    const auto link2 = robot.model.getFrameId("link2");
    const auto link3 = robot.model.getFrameId("link3");
    CRICKET_CHECK(robot.per_link_cuboids[link1].size() == 1);
    CRICKET_CHECK(robot.per_link_spheres[link1].empty());
    CRICKET_CHECK(robot.per_link_cuboids[link2].size() == 1);
    CRICKET_CHECK(robot.per_link_spheres[link2].size() == 1);
    CRICKET_CHECK(robot.per_link_cuboids[link3].size() == 1);
    CRICKET_CHECK(robot.per_link_spheres[link3].empty());

    // The bounding sphere of a link with a single box is exactly its circumscribed sphere
    {
        const auto &bs = robot.bounding_spheres.at(link1);
        CRICKET_CHECK(std::abs(bs.radius - box1.bounding_radius()) < 1e-4F);
        CRICKET_CHECK(std::abs(box1.bounding_radius() - std::sqrt(0.1F * 0.1F + 0.05F * 0.05F + 0.2F * 0.2F)) < 1e-6F);
    }

    // A link's bounding sphere must contain its boxes even when it also has a sphere: link2's sphere
    // is at z = 0.3 and its box (half-extents 0.05, 0.05, 0.15) is centered at z = 0.15
    {
        const auto &bs = robot.bounding_spheres.at(link2);
        const auto &box = robot.cuboids[robot.per_link_cuboids[link2][0]];
        const auto center = bs.relative.translation();
        for (auto sx : {-1., 1.})
        {
            for (auto sy : {-1., 1.})
            {
                for (auto sz : {-1., 1.})
                {
                    Eigen::Vector3d corner =
                        box.relative.translation() +
                        box.relative.rotation() *
                            Eigen::Vector3d(sx * box.half_extents[0], sy * box.half_extents[1], sz * box.half_extents[2]);
                    CRICKET_CHECK((corner - center).norm() <= bs.radius + 1e-4);
                }
            }
        }
    }

    // Metadata
    const auto metadata = robot.json();
    CRICKET_CHECK(metadata.at("n_spheres") == 2);
    CRICKET_CHECK(metadata.at("n_cuboids") == 3);
    CRICKET_CHECK(metadata.at("n_bounding_spheres") == 4);
    // Spheres, then bounding spheres, then cuboids
    CRICKET_CHECK(metadata.at("cuboid_buffer_offset") == (2 + 4) * 4);

    // Traced buffer layout: cuboids come after bounding spheres, and omitting them changes nothing before
    const auto sphere_only = cricket::trace_sphere_cc_fk(robot, "c++", true, true, false, false);
    CRICKET_CHECK(sphere_only.outputs == (2 + 4) * 4);
    const auto with_cuboids = cricket::trace_sphere_cc_fk(robot, "c++", true, true, false, true);
    CRICKET_CHECK(with_cuboids.outputs == (2 + 4) * 4 + 3 * cricket::cuboid_stride);
    const auto with_ee = cricket::trace_sphere_cc_fk(robot, "c++", true, true, true, true);
    CRICKET_CHECK(with_ee.outputs == (2 + 4) * 4 + 3 * cricket::cuboid_stride + 12);
    const auto cuboids_only = cricket::trace_sphere_cc_fk(robot, "c++", false, false, false, true);
    CRICKET_CHECK(cuboids_only.outputs == 3 * cricket::cuboid_stride);

    // Generated source
    cricket::GenOptions options;
    options.urdf = urdf;
    options.srdf = srdf;
    options.end_effector = std::string("link3");
    options.language = "c++";
    options.data = nlohmann::json{{"name", "MixedArm"}, {"resolution", 32}};

    const auto generated = cricket::generate_robot_source(options);
    CRICKET_CHECK(generated.n_cuboids == 3);
    CRICKET_CHECK(contains(generated.source, "n_cuboids = 3"));
    CRICKET_CHECK(contains(generated.source, "cuboid_environment_in_collision"));
    CRICKET_CHECK(contains(generated.source, "attachment_cuboid_collision"));
    CRICKET_CHECK(contains(generated.source, "cuboid_cuboid_self_collision") or
                  contains(generated.source, "sphere_cuboid_self_collision"));
    CRICKET_CHECK(contains(generated.source, "void cuboid_fk"));

    // Cuboid collision geometry is only generated for C++
    {
        auto rust_options = options;
        rust_options.language = "rust";
        bool threw = false;
        try
        {
            cricket::generate_robot_source(rust_options);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        CRICKET_CHECK(threw);
    }

    // A robot with only spheres must not pick up any cuboid code
    {
        const auto ur5 = source_dir / "resources/ur5/ur5_spherized_no_offset.urdf";
        const auto ur5_srdf = source_dir / "resources/ur5/ur5.srdf";

        cricket::GenOptions ur5_options;
        ur5_options.urdf = ur5;
        ur5_options.srdf = ur5_srdf;
        ur5_options.end_effector = std::string("robotiq_85_base_link");
        ur5_options.language = "c++";
        ur5_options.data = nlohmann::json{{"name", "UR5"}, {"resolution", 32}};

        const auto sphere_robot = cricket::generate_robot_source(ur5_options);
        CRICKET_CHECK(sphere_robot.n_cuboids == 0);
        CRICKET_CHECK(contains(sphere_robot.source, "n_cuboids = 0"));
        CRICKET_CHECK(not contains(sphere_robot.source, "cuboid_environment_in_collision"));
        CRICKET_CHECK(not contains(sphere_robot.source, "attachment_cuboid_collision"));
        CRICKET_CHECK(not contains(sphere_robot.source, "OrientedBox"));
        CRICKET_CHECK(not contains(sphere_robot.source, "cuboid_fk"));
    }

    return 0;
}

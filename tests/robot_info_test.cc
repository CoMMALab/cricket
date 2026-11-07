#include <cricket/robot_info.hh>

#include "check.hh"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

int main()
{
    const auto source_dir = std::filesystem::path(CRICKET_SOURCE_DIR);
    const auto urdf = source_dir / "resources/ur5/ur5_spherized_1.urdf";
    const auto srdf = source_dir / "resources/ur5/ur5.srdf";
    const auto test_urdf = std::filesystem::temp_directory_path() / "cricket_ur5_robot_info.urdf";

    std::ifstream input(urdf);
    std::string urdf_text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto package_prefix = std::string("package://meshes");
    const auto mesh_prefix = (source_dir / "resources/ur5/meshes").string();
    for (auto offset = urdf_text.find(package_prefix); offset != std::string::npos;
         offset = urdf_text.find(package_prefix, offset + mesh_prefix.size()))
    {
        urdf_text.replace(offset, package_prefix.size(), mesh_prefix);
    }
    const auto relative_prefix = std::string("filename=\"meshes/");
    for (auto offset = urdf_text.find(relative_prefix); offset != std::string::npos;
         offset = urdf_text.find(relative_prefix, offset + mesh_prefix.size()))
    {
        urdf_text.replace(offset, relative_prefix.size(), "filename=\"" + mesh_prefix + "/");
    }
    std::ofstream output(test_urdf);
    output << urdf_text;
    output.close();

    cricket::RobotInfo robot(test_urdf, srdf, std::nullopt);
    const auto metadata = robot.json();

    CRICKET_CHECK(metadata.at("n_q") == robot.model.nq);
    CRICKET_CHECK(metadata.at("n_v") == robot.model.nv);
    CRICKET_CHECK(metadata.at("n_u") == robot.model.nq);
    CRICKET_CHECK(metadata.at("n_q") == 6);
    CRICKET_CHECK(metadata.at("n_v") == 6);

    // Compact collision tables: every link pair's sphere pairs, and every environment link's spheres,
    // appear once in the flat index arrays.
    CRICKET_CHECK(metadata.at("environment_links").size() == metadata.at("links_with_geometry").size());
    CRICKET_CHECK(metadata.at("compact_env_entries").size() == metadata.at("environment_links").size());
    std::size_t env_body_total = 0;
    for (const auto &entry : metadata.at("compact_env_entries"))
    {
        env_body_total += entry.at(2).get<std::size_t>();
    }
    CRICKET_CHECK(env_body_total == metadata.at("compact_env_body_idx").size());

    std::size_t self_pair_total = 0;
    for (const auto &entry : metadata.at("compact_self_entries"))
    {
        self_pair_total += entry.at(3).get<std::size_t>();
    }
    CRICKET_CHECK(self_pair_total == metadata.at("compact_self_pair_a").size());
    CRICKET_CHECK(self_pair_total == metadata.at("compact_self_pair_b").size());
    CRICKET_CHECK(metadata.at("sphere_env_skip").size() == metadata.at("n_spheres").get<std::size_t>());

    // Skipping static links removes exactly the links with no mobile joint to the world.
    const auto skipped = robot.json(std::nullopt, true);
    std::size_t static_links = 0;
    for (const auto &link : metadata.at("links_with_geometry"))
    {
        static_links += (robot.model.frames[link.get<std::size_t>()].parentJoint == 0) ? 1 : 0;
    }
    CRICKET_CHECK(skipped.at("environment_links").size() + static_links == metadata.at("environment_links").size());

    return 0;
}

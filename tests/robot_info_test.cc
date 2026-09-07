#include <cricket/robot_info.hh>

#include <cassert>
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

    assert(metadata.at("n_q") == robot.model.nq);
    assert(metadata.at("n_v") == robot.model.nv);
    assert(metadata.at("n_u") == robot.model.nq);
    assert(metadata.at("n_q") == 6);
    assert(metadata.at("n_v") == 6);

    return 0;
}

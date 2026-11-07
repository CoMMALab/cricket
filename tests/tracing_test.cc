#include <cricket/codegen.hh>

#include <pinocchio/parsers/urdf.hpp>

#include "check.hh"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

int main()
{
    const auto urdf = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/ur5/ur5.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf, model, false, true);

    const auto distance = cricket::trace_distance(model, "rust");
    CRICKET_CHECK(distance.outputs == 1);
    CRICKET_CHECK(distance.temp_variables > 0);
    CRICKET_CHECK(not distance.code.empty());

    const auto interpolate = cricket::trace_interpolate(model, "rust");
    CRICKET_CHECK(interpolate.outputs == static_cast<std::size_t>(model.nq));
    // Every UR5 joint is a bounded revolute joint, so each output is `a + t * (b - a)`.
    // Each subexpression appears once, which leaves nothing to hoist into a temporary.
    CRICKET_CHECK(interpolate.temp_variables == 0);
    CRICKET_CHECK(interpolate.code.find("y[5] = a[5] + t * (b[5] - a[5]);") != std::string::npos);
    CRICKET_CHECK(not interpolate.code.empty());

    const auto dynamics = cricket::trace_forward_dynamics(model, "rust");
    CRICKET_CHECK(dynamics.outputs == static_cast<std::size_t>(model.nv));
    CRICKET_CHECK(dynamics.temp_variables > 0);
    CRICKET_CHECK(not dynamics.code.empty());
    CRICKET_CHECK(dynamics.code.find("Simd::<f32, L>") != std::string::npos);
    CRICKET_CHECK(dynamics.code.find("sin(") != std::string::npos);
    CRICKET_CHECK(dynamics.code.find("cos(") != std::string::npos);

    const auto integration = cricket::trace_integrate_configuration(model, "rust");
    CRICKET_CHECK(integration.outputs == static_cast<std::size_t>(model.nq));
    CRICKET_CHECK(not integration.code.empty());
    CRICKET_CHECK(integration.code.find("x[0] + x[6]") != std::string::npos);

    const auto spherized = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/ur5/ur5_spherized_no_offset.urdf";
    const auto srdf = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/ur5/ur5.srdf";
    cricket::RobotInfo info(spherized, srdf, std::string("robotiq_85_base_link"));

    const auto eejac = cricket::trace_ee_fk_jacobian(info, "rust");
    CRICKET_CHECK(eejac.outputs == static_cast<std::size_t>(12 + 6 * info.model.nv));
    CRICKET_CHECK(eejac.temp_variables > 0);
    CRICKET_CHECK(eejac.code.find("sin(") != std::string::npos);
    CRICKET_CHECK(eejac.code.find("cos(") != std::string::npos);
    // The Rust backend cannot emit conditionals, so a trace containing one would be silently wrong.
    CRICKET_CHECK(eejac.code.find(" ? ") == std::string::npos);

    // Several end-effectors: one 12-value pose per end-effector, in declaration order.
    cricket::RobotInfo two(
        spherized, srdf, std::vector<std::string>{"robotiq_85_base_link", "wrist_3_link"});
    CRICKET_CHECK(two.end_effector_names.size() == 2);
    CRICKET_CHECK(two.end_effector_indexes.size() == 2);
    CRICKET_CHECK(two.end_effector_name == "robotiq_85_base_link");
    CRICKET_CHECK(two.end_effector_index == two.end_effector_indexes.front());
    CRICKET_CHECK(two.end_effector_indexes.front() != two.end_effector_indexes.back());

    const auto fk_one = cricket::trace_sphere_cc_fk(info, "c++", false, false, true);
    CRICKET_CHECK(fk_one.outputs == 12);
    const auto fk_two = cricket::trace_sphere_cc_fk(two, "c++", false, false, true);
    CRICKET_CHECK(fk_two.outputs == 24);

    const auto metadata = two.json();
    CRICKET_CHECK(metadata.at("num_end_effectors") == 2);
    CRICKET_CHECK(metadata.at("end_effectors").size() == 2);
    CRICKET_CHECK(metadata.at("end_effector_collisions").size() == 2);
    CRICKET_CHECK(metadata.at("end_effector") == "robotiq_85_base_link");

    bool rejected = false;
    try
    {
        cricket::RobotInfo bad(spherized, srdf, std::vector<std::string>{"robotiq_85_base_link", "no_such_frame"});
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    CRICKET_CHECK(rejected);

    return 0;
}

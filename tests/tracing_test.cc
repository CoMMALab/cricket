#include <cricket/codegen.hh>

#include <pinocchio/parsers/urdf.hpp>

#include <cassert>
#include <filesystem>
#include <string>

int main()
{
    const auto urdf = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/ur5/ur5.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf, model, false, true);

    const auto distance = cricket::trace_distance(model, "rust");
    assert(distance.outputs == 1);
    assert(distance.temp_variables > 0);
    assert(not distance.code.empty());

    const auto interpolate = cricket::trace_interpolate(model, "rust");
    assert(interpolate.outputs == static_cast<std::size_t>(model.nq));
    assert(interpolate.temp_variables > 0);
    assert(not interpolate.code.empty());

    const auto dynamics = cricket::trace_forward_dynamics(model, "rust");
    assert(dynamics.outputs == static_cast<std::size_t>(model.nv));
    assert(dynamics.temp_variables > 0);
    assert(not dynamics.code.empty());
    assert(dynamics.code.find("Simd::<f32, L>") != std::string::npos);
    assert(dynamics.code.find("sin(") != std::string::npos);
    assert(dynamics.code.find("cos(") != std::string::npos);

    const auto integration = cricket::trace_integrate_configuration(model, "rust");
    assert(integration.outputs == static_cast<std::size_t>(model.nq));
    assert(not integration.code.empty());
    assert(integration.code.find("x[0] + x[6]") != std::string::npos);

    const auto spherized = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/ur5/ur5_spherized_no_offset.urdf";
    const auto srdf = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/ur5/ur5.srdf";
    cricket::RobotInfo info(spherized, srdf, std::string("robotiq_85_base_link"));

    const auto eejac = cricket::trace_ee_fk_jacobian(info, "rust");
    assert(eejac.outputs == static_cast<std::size_t>(12 + 6 * info.model.nv));
    assert(eejac.temp_variables > 0);
    assert(eejac.code.find("sin(") != std::string::npos);
    assert(eejac.code.find("cos(") != std::string::npos);
    // The Rust backend cannot emit conditionals, so a trace containing one would be silently wrong.
    assert(eejac.code.find(" ? ") == std::string::npos);

    return 0;
}

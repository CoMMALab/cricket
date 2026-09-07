#include <cricket/codegen.hh>

#include <pinocchio/parsers/urdf.hpp>

#include <cassert>
#include <filesystem>

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

    const auto integration = cricket::trace_integrate_configuration(model, "rust");
    assert(integration.outputs == static_cast<std::size_t>(model.nq));
    assert(integration.temp_variables > 0);
    assert(not integration.code.empty());

    return 0;
}

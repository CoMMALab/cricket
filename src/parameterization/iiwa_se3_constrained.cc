// Wires the single-arm SE3+psi task-space parameterized IK (iiwa_parameterization_gen.hh,
// se3_tracer.hh) into the main codegen pipeline. Kept in its own translation unit, same
// reasoning as rby1_constrained.cc: codegen.cc's local ADCG/CGD typedefs would collide with
// the ones pulled in here via tracing/internal.hh.
#include <cricket/codegen.hh>

#include "iiwa_parameterization_gen.hh"
#include "se3_tracer.hh"

namespace cricket
{
    auto trace_iiwa_se3_ik(const RobotInfo &info, const std::string &language) -> Traced
    {
        (void)info;
        return IiwaSE3ParameterizationCG<ADCG>(language);
    }

    // Thin forwards to se3_tracer.hh's generic pose+psi Space kernels -- kept under the
    // trace_iiwa_se3_* names codegen.hh declares (and to force emission of these `inline`
    // definitions in this TU) rather than calling trace_map_to_se3 et al. directly from
    // codegen.cc, which can't see them (see this file's header comment).
    auto trace_iiwa_se3_sample(
        const pinocchio::Model &model,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> Traced
    {
        return trace_map_to_se3(model, language, bounds);
    }

    auto trace_iiwa_se3_distance(const std::string &language) -> Traced
    {
        return trace_SE3_distance(language);
    }

    auto trace_iiwa_se3_interpolate(const std::string &language) -> Traced
    {
        return trace_interpolate(language);
    }

    auto trace_iiwa_se3_interpolate_block(const std::string &language) -> Traced
    {
        return trace_interpolate_block(language);
    }
}  // namespace cricket

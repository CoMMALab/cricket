// Wires the single-arm SE3+psi task-space parameterized IK for FR3 (fr3_parameterization_gen.hh,
// se3_tracer.hh) into the main codegen pipeline. Kept in its own translation unit, same
// reasoning as iiwa_se3_constrained.cc: codegen.cc's local ADCG/CGD typedefs would collide with
// the ones pulled in here via tracing/internal.hh.
#include <cricket/codegen.hh>

#include "fr3_parameterization_gen.hh"
#include "se3_tracer.hh"

namespace cricket
{
    auto trace_fr3_se3_ik(const RobotInfo &info, const std::string &language) -> Traced
    {
        (void)info;
        return Fr3SE3ParameterizationCG<ADCG>(language);
    }

    // Thin forwards to se3_tracer.hh's generic pose+psi Space kernels -- kept under the
    // trace_fr3_se3_* names codegen.hh declares (and to force emission of these `inline`
    // definitions in this TU) rather than calling trace_map_to_se3 et al. directly from
    // codegen.cc, which can't see them (see this file's header comment). Identical to
    // trace_iiwa_se3_sample/distance/interpolate(_block) in iiwa_se3_constrained.cc -- FR3's
    // task space is exactly the same 8-dim (pose(7) + free-param(1)) se3_tracer.hh Space --
    // except for the joint index passed below: unlike iiwa's psi (a self-motion-manifold
    // parameter with no direct joint-limit meaning, sampled over its full [0, 2*pi) period),
    // FR3's psi IS joint 7's angle directly (see Fr3SE3Parameterization's header), so it must
    // be sampled from that joint's actual limits rather than the full circle. Index 6 (0-based)
    // is FR3's joint 7, matching Fr3SE3ParameterizationCG's n_q == 7 output/resolve_block's
    // joint-limit loop -- see trace_map_to_se3's header comment for why this is an explicit
    // index rather than derived from `model.nq`.
    auto trace_fr3_se3_sample(
        const pinocchio::Model &model,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> Traced
    {
        return trace_map_to_se3(model, language, bounds, /* psi_dof_index (joint 7) */ std::size_t{6});
    }

    auto trace_fr3_se3_distance(const std::string &language) -> Traced
    {
        return trace_SE3_distance(language);
    }

    auto trace_fr3_se3_interpolate(const std::string &language) -> Traced
    {
        return trace_interpolate(language);
    }

    auto trace_fr3_se3_interpolate_block(const std::string &language) -> Traced
    {
        return trace_interpolate_block(language);
    }
}  // namespace cricket

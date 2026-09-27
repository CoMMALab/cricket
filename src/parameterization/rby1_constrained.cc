// Wires the RBY1 constrained-bimanual parameterized IK (rainbow_ik_cg.hh) into the main
// codegen pipeline. Kept in its own translation unit, separate from codegen.cc: codegen.cc
// declares its own local ADCG/CGD typedefs in an anonymous namespace, which would collide
// with the ones rainbow_ik_cg.hh pulls in from tracing/internal.hh if both were visible in
// the same TU. codegen.cc only ever sees the Traced-returning declarations in
// cricket/codegen.hh; trace_rby1_constrained_sample/_distance/_interpolate/_interpolate_block
// are already fully defined `inline` in rainbow_ik_cg.hh and are pulled in here.
#include <cricket/codegen.hh>

#include "rainbow_ik_cg.hh"

namespace cricket
{
    auto trace_rby1_constrained_ik(const RobotInfo &info, const std::string &language) -> Traced
    {
        return RainbowConstrainedBimanualIkCG<ADCG>(info, language);
    }

    // RainbowMidPoseFkCG is already non-template and already `inline`-defined in
    // rainbow_ik_cg.hh; this thin forward keeps naming consistent with the other
    // trace_rby1_* entry points and is what actually forces its emission in this TU (an
    // unreferenced `inline` definition would otherwise be dropped -- see
    // trace_rby1_constrained_sample/_distance/_interpolate/_interpolate_block above).
    auto trace_rby1_mid_pose_fk(const RobotInfo &info, const std::string &language) -> Traced
    {
        return RainbowMidPoseFkCG(info, language);
    }

    // Same "thin forward to force emission" reasoning as trace_rby1_mid_pose_fk above --
    // RainbowEefWorldPosesFromMidCG is `inline`-defined in rainbow_ik_cg.hh.
    auto trace_rby1_eef_world_poses_from_mid(const std::string &language) -> Traced
    {
        return RainbowEefWorldPosesFromMidCG(language);
    }

    auto trace_rby1_classify_gcp(const RobotInfo &info, const std::string &language) -> Traced
    {
        return RainbowClassifyGcpCG<ADCG>(info, language);
    }

    auto trace_rby1_torso_free_loss_and_jacobian(const RobotInfo &info, const std::string &language) -> Traced
    {
        return RainbowConstrainedBimanualIkCG<ADCG>(info, language, /*compute_gradient=*/true);
    }

    auto trace_rby1_torso_free_solve_gradient_descent(const std::string &language) -> Traced
    {
        return trace_rby1_torso_free_solve(language, ProjMethod::GradDesc);
    }

    auto trace_rby1_torso_free_solve_lm_inner(const std::string &language) -> Traced
    {
        return trace_rby1_torso_free_solve(language, ProjMethod::InnerLM);
    }

    // Independent (unconstrained) bimanual variant: the two hand targets are ordinary,
    // independent tape inputs (RainbowIkCG) rather than derived from a shared mid-pose plus
    // fixed offsets (RainbowConstrainedBimanualIkCG). The step-solve math is unchanged --
    // trace_rby1_torso_free_solve_gradient_descent/_lm_inner above are already generic over
    // any 8-in/8-out "2 losses + their 8-wide Jacobians -> 8-wide step" problem, so this mode
    // reuses those directly and only needs its own loss+Jacobian trace.
    auto trace_rby1_independent_loss_and_jacobian(const RobotInfo &info, const std::string &language) -> Traced
    {
        return RainbowIkCG<ADCG>(info, language, /*compute_gradient=*/true);
    }

}  // namespace cricket

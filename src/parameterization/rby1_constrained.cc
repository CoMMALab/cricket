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
}  // namespace cricket

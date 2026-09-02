// Wires the bimanual iiwa parameterized IK (iiwa_parameterization_gen.hh,
// iiwa_bimanual_tracer.hh) into the main codegen pipeline. Kept in its own translation unit,
// same reasoning as iiwa_se3_constrained.cc: codegen.cc's local ADCG/CGD typedefs would
// collide with the ones pulled in here via tracing/internal.hh.
#include <cricket/codegen.hh>

#include "iiwa_bimanual_tracer.hh"
#include "iiwa_parameterization_gen.hh"

namespace cricket
{
    auto trace_iiwa_bimanual_ik(const RobotInfo &info, const std::string &language) -> Traced
    {
        (void)info;
        return IiwaBimanualParameterizationCG<ADCG>(language);
    }

    auto trace_iiwa_bimanual_sample(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        return trace_bimanual_state_sample(model, language);
    }

    auto trace_iiwa_bimanual_distance(const std::string &language) -> Traced
    {
        return trace_bimanual_state_distance(language);
    }

    auto trace_iiwa_bimanual_interpolate(const std::string &language) -> Traced
    {
        return trace_bimanual_state_interpolate(language);
    }

    auto trace_iiwa_bimanual_interpolate_block(const std::string &language) -> Traced
    {
        return trace_bimanual_state_interpolate_block(language);
    }

    auto trace_iiwa_bimanual_rel_pose_fk(const RobotInfo &info, const std::string &language) -> Traced
    {
        return trace_bimanual_rel_pose_fk(info, language);
    }

    // "iiwa_bimanual" ParameterizedSpace (mid-pose sampling) -- see
    // IiwaBimanualMidParameterizationCG's header comment in iiwa_parameterization_gen.hh.
    auto trace_iiwa_bimanual_mid_ik(const RobotInfo &info, const std::string &language) -> Traced
    {
        (void)info;
        return IiwaBimanualMidParameterizationCG<ADCG>(language);
    }

    auto trace_iiwa_bimanual_mid_sample(const std::string &language, const std::optional<Bounds> &bounds) -> Traced
    {
        return trace_bimanual_mid_sample(language, bounds);
    }

    auto trace_iiwa_bimanual_mid_distance(const std::string &language) -> Traced
    {
        return trace_bimanual_mid_distance(language);
    }

    auto trace_iiwa_bimanual_mid_interpolate(const std::string &language) -> Traced
    {
        return trace_bimanual_mid_interpolate(language);
    }

    auto trace_iiwa_bimanual_mid_interpolate_block(const std::string &language) -> Traced
    {
        return trace_bimanual_mid_interpolate_block(language);
    }

    auto trace_iiwa_bimanual_eef_world_poses_from_mid(const std::string &language) -> Traced
    {
        return trace_bimanual_eef_world_poses_from_mid(language);
    }
}  // namespace cricket

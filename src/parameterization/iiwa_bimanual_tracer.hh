#pragma once

// Sample/distance/interpolate kernels, plus the dual-eef FK used to derive the fixed
// relative-pose offset, for the "iiwa_bimanual" leader-follower parameterized state (see
// IiwaBimanualParameterizationCG in iiwa_parameterization_gen.hh for the IK itself, and
// fk_template.hh's LeaderFollowerSpace for the struct these kernels feed).
//
// State layout (8, fully Euclidean -- no SO3 block): q(7, leader/left arm's own joint
// angles) + psi(1, follower/right arm's self-motion-manifold free parameter). Unlike
// se3_tracer.hh's pose+psi task space (reused by "iiwa_se3", fk_template.hh's
// ParameterizedSpace), the leader arm here is sampled directly in its own joint space --
// there is no IK to invert for it, IiwaBimanualParameterization just copies q straight
// through. The follower/right arm's target is the leader's own FK pose composed with a FIXED
// relative offset (`rel_pose`, LeaderFollowerSpace's class member -- not part of State, the
// equivalent of ParameterizedSpace's rby1_bimanual t_mid_left/t_mid_right) -- see
// trace_bimanual_rel_pose_fk below for how that offset is derived from a reference
// configuration.

#include <cricket/codegen.hh>
#include <cricket/robot_info.hh>

#include "../tracing/internal.hh"
#include "se3_tracer.hh"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace cricket
{
    using namespace pinocchio;
    using namespace CppAD;
    using namespace CppAD::cg;

// Maps 8 raw [0,1) sample values to the 8-dim state: 7 joint angles (uniform within the
// leader/left arm's own position limits, model q-indices [0:7) -- see the header comment on
// derive_iiwa_bimanual_leader_follower_traces in codegen.cc for why that's the leader arm's
// range) and psi (uniform in [0, 2*pi)). Purely Euclidean -- no Shoemake/orientation mapping
// needed, unlike trace_map_to_se3.
inline auto trace_bimanual_state_sample(const pinocchio::Model &model, const std::string &language) -> Traced
{
    const std::size_t n = 8;

    ADVectorXs ad_u(n);
    ADVectorXs ad_state(n);
    for (auto i = 0U; i < n; ++i)
    {
        ad_u[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_u);

    for (auto i = 0U; i < 7; ++i)
    {
        ad_state[i] = map_bounded(ad_u[i], model.lowerPositionLimit[i], model.upperPositionLimit[i]);
    }
    ad_state[7] = map_bounded(ad_u[7], 0.0, 2 * M_PI);

    CppAD::ADFun<CGD> map_func(ad_u, ad_state);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = map_func.Forward(0, ind_vars);

    return Traced{
        generate_code(handler, result, language),
        handler.getTemporaryVariableCount(),
        result.size()};
}

// Plain Euclidean L2 distance over the 8-dim state -- no SE3 log-map needed since neither
// component (joint angles, psi) is an orientation.
inline auto trace_bimanual_state_distance(const std::string &language) -> Traced
{
    const std::size_t n_input = 2 * 8;
    ADVectorXs ad_input(n_input);
    ADVectorXs out(1);
    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    ADVectorXs diff(8);
    for (auto i = 0U; i < 8; ++i)
    {
        diff[i] = ad_input[i] - ad_input[8 + i];
    }
    out[0] = diff.norm();

    CppAD::ADFun<CGD> dist_func(ad_input, out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = dist_func.Forward(0, ind_vars);

    SegmentedVariableNameGenerator<double> nameGen({{"a", 8, true}, {"b", 8, true}});

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        1};
}

inline auto trace_bimanual_state_interpolate_impl(
    const std::string &language,
    std::vector<VarSegment> input_segments,
    std::vector<VarSegment> output_segments) -> Traced
{
    const std::size_t n_input = 2 * 8 + 1;  // a, b, t

    ADVectorXs ad_input(n_input);
    ADVectorXs ad_out(8);

    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_input);

    const ADCG &t = ad_input[16];
    for (auto i = 0U; i < 8; ++i)
    {
        ad_out[i] = (1.0 - t) * ad_input[i] + t * ad_input[8 + i];
    }

    CppAD::ADFun<CGD> interp_func(ad_input, ad_out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = interp_func.Forward(0, ind_vars);

    SegmentedVariableNameGenerator<double> nameGen(
        std::move(input_segments), std::move(output_segments));

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        static_cast<std::size_t>(8)};
}

inline auto trace_bimanual_state_interpolate(const std::string &language) -> Traced
{
    return trace_bimanual_state_interpolate_impl(
        language, {{"a", 8, true}, {"b", 8, true}, {"t", 1, false}}, {});
}

inline auto trace_bimanual_state_interpolate_block(const std::string &language) -> Traced
{
    const std::string lang = (language == "c++") ? "c++_block" : language;
    return trace_bimanual_state_interpolate_impl(
        lang,
        {{"a", 8, true, ".broadcast(", ")"}, {"b", 8, true, ".broadcast(", ")"}, {"t", 1, false}},
        {{"out", 8, true}});
}

// Dual-eef forward kinematics: given the full ambient/whole-body configuration `q`, computes
// both end effectors' world-frame poses (Ambient::end_effector[0] == leader/left,
// Ambient::end_effector[1] == follower/right, matching bimanual_iiwa.json's `end_effector`
// order and IiwaBimanualParameterization's q_full.head(7)/tail(7) split). Feeds
// fk_template.hh's LeaderFollowerSpace::compute_rel_pose, which uses these two poses to derive
// the fixed `rel_pose` class member -- the "iiwa_bimanual" counterpart of RainbowMidPoseFkCG
// (rainbow_ik_cg.hh), generalized to arbitrary end-effector frame names via
// `info.end_effector_indexes` instead of RainbowMidPoseFkCG's hardcoded "ee_left"/"ee_right".
// Output layout (24): [0:12) leader/left eef world pose (translation xyz + rotation matrix,
// column-major), [12:24) follower/right eef world pose, same shape -- vamp::to_isometry's
// expected 12-float input.
inline auto trace_bimanual_rel_pose_fk(const RobotInfo &info, const std::string &language) -> Traced
{
    if (info.end_effector_indexes.size() != 2)
    {
        throw std::runtime_error(
            "trace_bimanual_rel_pose_fk: expected exactly 2 end effectors (master, subordinate)");
    }

    auto nq = info.model.nq;
    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    ADVectorXs ad_q(nq);
    for (auto i = 0U; i < nq; ++i)
    {
        ad_q[i] = ADCG(0.0);
    }
    Independent(ad_q);

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    ADVectorXs data(24);

    auto write_frame = [&](std::size_t frame_id, std::size_t offset) {
        const auto &oMf = ad_data.oMf[frame_id];

        data[offset + 0] = oMf.translation()[0];
        data[offset + 1] = oMf.translation()[1];
        data[offset + 2] = oMf.translation()[2];

        const auto &R = oMf.rotation();

        // Eigen stores as column major
        data[offset + 3] = R(0, 0);
        data[offset + 4] = R(1, 0);
        data[offset + 5] = R(2, 0);
        data[offset + 6] = R(0, 1);
        data[offset + 7] = R(1, 1);
        data[offset + 8] = R(2, 1);
        data[offset + 9] = R(0, 2);
        data[offset + 10] = R(1, 2);
        data[offset + 11] = R(2, 2);
    };

    write_frame(info.end_effector_indexes[0], 0);
    write_frame(info.end_effector_indexes[1], 12);

    ADFun<CGD> rel_pose_fk_func(ad_q, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = rel_pose_fk_func.Forward(0, ind_vars);

    return Traced{
        generate_code(handler, result, language),
        handler.getTemporaryVariableCount(),
        static_cast<std::size_t>(24)};
}

// =====================================================================
// "iiwa_bimanual" ParameterizedSpace support (mid-pose sampling, see
// IiwaBimanualMidParameterizationCG in iiwa_parameterization_gen.hh) --
// the iiwa_bimanual counterpart of rainbow_ik_cg.hh's constrained_bimanual_ik
// section, reusing this file's trace_bimanual_rel_pose_fk above for
// compute_mid_pose's dual-eef FK (it's already generic over 2 end
// effectors) instead of a separate function.
//
// State layout (9, non-Euclidean): T_mid pose(7, [x,y,z,qx,qy,qz,qw]) +
// psi_left(1) + psi_right(1). Unlike the leader-follower State above
// (fully Euclidean, no task-space pose), T_mid is a genuine sampled SE3
// pose -- same shape as se3_tracer.hh's pose+psi Space, just with two
// psis instead of one, so these kernels are that file's trace_map_to_se3/
// trace_SE3_distance/trace_interpolate(_block) extended by one extra
// Euclidean scalar rather than a full rewrite.
// =====================================================================

// Maps 8 raw [0,1) sample values to the 9-dim state: T_mid position (3,
// map_bounded against `bounds`), T_mid orientation (3 raw values,
// map_so3_shoemake -> unit quaternion), psi_left and psi_right (each
// map_bounded(0, 2*pi)).
inline auto trace_bimanual_mid_sample(
    const std::string &language,
    const std::optional<Bounds> &bounds) -> Traced
{
    const std::size_t n_sample = 8;
    const std::size_t n_state = 9;

    ADVectorXs ad_u(n_sample);
    ADVectorXs ad_state(n_state);
    for (auto i = 0U; i < n_sample; ++i)
    {
        ad_u[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_u);

    for (int i = 0; i < 3; ++i)
    {
        ad_state[i] = map_bounded(ad_u[i], bounds->lower[i], bounds->upper[i]);
    }
    ADCG qx, qy, qz, qw;
    map_so3_shoemake(ad_u[3], ad_u[4], ad_u[5], qx, qy, qz, qw);
    ad_state[3] = qx;
    ad_state[4] = qy;
    ad_state[5] = qz;
    ad_state[6] = qw;
    ad_state[7] = map_bounded(ad_u[6], 0.0, 2 * M_PI);  // psi_left
    ad_state[8] = map_bounded(ad_u[7], 0.0, 2 * M_PI);  // psi_right

    CppAD::ADFun<CGD> map_func(ad_u, ad_state);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_sample);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = map_func.Forward(0, ind_vars);

    return Traced{
        generate_code(handler, result, language),
        handler.getTemporaryVariableCount(),
        result.size()};
}

// SE3 log-displacement (6) over T_mid's pose block, plus a plain Euclidean difference for
// psi_left and psi_right -- trace_SE3_distance (se3_tracer.hh) extended by one extra scalar.
inline auto trace_bimanual_mid_distance(const std::string &language) -> Traced
{
    const std::size_t n_input = 2 * 9;
    ADVectorXs ad_input(n_input);
    ADVectorXs out(1);
    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    const auto a = read_transform(ad_input, 0);
    const auto b = read_transform(ad_input, 9);

    SE3Tpl<ADCG> rel = a.inverse() * b;
    auto displacement = se3_displacement(rel);

    ADVectorXs total_displacement(8);
    total_displacement.head(6) = displacement;
    total_displacement[6] = ad_input[7] - ad_input[16];   // psi_left
    total_displacement[7] = ad_input[8] - ad_input[17];   // psi_right

    out[0] = total_displacement.norm();

    CppAD::ADFun<CGD> dist_func(ad_input, out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = dist_func.Forward(0, ind_vars);

    SegmentedVariableNameGenerator<double> nameGen({{"a", 9, true}, {"b", 9, true}});

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        1};
}

// slerp_se3 for T_mid's pose block, plain linear interpolation for psi_left/psi_right --
// trace_interpolate_impl (se3_tracer.hh) extended by one extra scalar. Same _impl + regular/
// block wrapper split as that file's own trace_interpolate/trace_interpolate_block.
inline auto trace_bimanual_mid_interpolate_impl(
    const std::string &language,
    std::vector<VarSegment> input_segments,
    std::vector<VarSegment> output_segments) -> Traced
{
    const std::size_t n_input = 2 * 9 + 1;  // a, b, t

    ADVectorXs ad_input(n_input);
    ADVectorXs ad_out(9);
    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    ADVectorXs ad_a = ad_input.head(7);
    ADVectorXs ad_b = ad_input.segment(9, 7);
    ADCG t = ad_input[2 * 9];

    auto psi_left_a = ad_input[7];
    auto psi_right_a = ad_input[8];
    auto psi_left_b = ad_input[16];
    auto psi_right_b = ad_input[17];

    // slerp_se3 only reads/writes indices [0:7) (translation + quaternion) -- it never
    // touches a hypothetical index 7, so the plain 7-sized pose blocks suffice as-is.
    ADVectorXs pose_out(7);
    slerp_se3(ad_a, ad_b, t, pose_out);

    ad_out.head(7) = pose_out;
    ad_out[7] = (1.0 - t) * psi_left_a + t * psi_left_b;
    ad_out[8] = (1.0 - t) * psi_right_a + t * psi_right_b;

    CppAD::ADFun<CGD> interp_func(ad_input, ad_out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = interp_func.Forward(0, ind_vars);

    SegmentedVariableNameGenerator<double> nameGen(
        std::move(input_segments), std::move(output_segments));

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        static_cast<std::size_t>(9)};
}

inline auto trace_bimanual_mid_interpolate(const std::string &language) -> Traced
{
    return trace_bimanual_mid_interpolate_impl(
        language, {{"a", 9, true}, {"b", 9, true}, {"t", 1, false}}, {});
}

inline auto trace_bimanual_mid_interpolate_block(const std::string &language) -> Traced
{
    const std::string lang = (language == "c++") ? "c++_block" : language;
    return trace_bimanual_mid_interpolate_impl(
        lang,
        {{"a", 9, true, ".broadcast(", ")"}, {"b", 9, true, ".broadcast(", ")"}, {"t", 1, false}},
        {{"out", 9, true}});
}

// Left/right hand WORLD poses derived from a T_mid pose plus the fixed t_mid_left/t_mid_right
// offsets: T_l = T_mid * t_mid_left, T_r = T_mid * t_mid_right -- the same composition
// IiwaBimanualMidParameterizationCG does internally (see that function's header comment)
// before shifting the right hand into its own solver frame, but broken out standalone here
// since ParameterizedSpace::eefs_in_collision (fk_template.hh) only needs the hands' world
// poses, not the arm IK that follows. Identical shape/purpose to
// RainbowEefWorldPosesFromMidCG (rainbow_ik_cg.hh) -- duplicated here (rather than shared)
// to keep this TU independent of rainbow_ik_cg.hh's rainbow-specific includes, same reasoning
// as trace_bimanual_rel_pose_fk above not sharing RainbowMidPoseFkCG.
//
// Tape input layout (21): [0:7) T_mid pose (x, y, z, qx, qy, qz, qw), [7:14) t_mid_left
// (fixed, aliases the `t_mid_left` thread_local member), [14:21) t_mid_right (fixed, aliases
// `t_mid_right`).
//
// Output layout (24): [0:12) left hand world pose (translation xyz + rotation matrix,
// column-major), [12:24) right hand world pose, same shape as trace_bimanual_rel_pose_fk's
// output.
inline auto trace_bimanual_eef_world_poses_from_mid(const std::string &language) -> Traced
{
    const std::size_t num_inp = 7 + 7 + 7;
    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = ADCG(0.0001);
    }
    Independent(ad_inp);

    Eigen::Quaternion<ADCG> t_mid_quat(ad_inp[6], ad_inp[3], ad_inp[4], ad_inp[5]);
    SE3Tpl<ADCG> t_mid_world(t_mid_quat, Eigen::Matrix<ADCG, 3, 1>(ad_inp[0], ad_inp[1], ad_inp[2]));

    Eigen::Quaternion<ADCG> t_mid_left_quat(ad_inp[13], ad_inp[10], ad_inp[11], ad_inp[12]);
    SE3Tpl<ADCG> t_mid_left_offset(t_mid_left_quat, Eigen::Matrix<ADCG, 3, 1>(ad_inp[7], ad_inp[8], ad_inp[9]));

    Eigen::Quaternion<ADCG> t_mid_right_quat(ad_inp[20], ad_inp[17], ad_inp[18], ad_inp[19]);
    SE3Tpl<ADCG> t_mid_right_offset(t_mid_right_quat, Eigen::Matrix<ADCG, 3, 1>(ad_inp[14], ad_inp[15], ad_inp[16]));

    SE3Tpl<ADCG> left_world = t_mid_world * t_mid_left_offset;
    SE3Tpl<ADCG> right_world = t_mid_world * t_mid_right_offset;

    ADVectorXs data(24);

    auto write_pose = [&](const SE3Tpl<ADCG> &pose, std::size_t offset) {
        data[offset + 0] = pose.translation()[0];
        data[offset + 1] = pose.translation()[1];
        data[offset + 2] = pose.translation()[2];

        const auto &R = pose.rotation();
        data[offset + 3] = R(0, 0);
        data[offset + 4] = R(1, 0);
        data[offset + 5] = R(2, 0);
        data[offset + 6] = R(0, 1);
        data[offset + 7] = R(1, 1);
        data[offset + 8] = R(2, 1);
        data[offset + 9] = R(0, 2);
        data[offset + 10] = R(1, 2);
        data[offset + 11] = R(2, 2);
    };

    write_pose(left_world, 0);
    write_pose(right_world, 12);

    ADFun<CGD> eef_world_poses_func(ad_inp, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = eef_world_poses_func.Forward(0, ind_vars);

    return Traced{generate_code(handler, result, language), handler.getTemporaryVariableCount(), result.size()};
}
}  // namespace cricket

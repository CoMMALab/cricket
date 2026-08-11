#pragma once

#include <cricket/codegen.hh>
#include <cricket/robot_info.hh>

#include "../tracing/internal.hh"
#include "rainbow_arm_parameterization.hh"
#include "se3_tracer.hh"

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace cricket
{

// CppAD-CodeGen wrapper that solves both rainbow arms' analytic IK relative
// to the robot's own torso, instead of taking each arm's target directly in
// its solver frame the way RainbowLeftArmParameterizationCG /
// RainbowRightArmParameterizationCG (rainbow_arm_parameterization_gen.hh)
// do. It runs pinocchio forwardKinematics/updateFramePlacements on the
// whole-body model to get `link_torso_5`'s world pose, expresses each arm's
// world-frame goal pose relative to that link, and feeds the result
// straight into RainbowRightArmParameterizationFromPose /
// RainbowLeftArmParameterizationFromPose (the matrix-taking cores added to
// rainbow_arm_parameterization.hh for exactly this: composing SE3 frames
// via pinocchio yields a rotation *matrix*, and round-tripping that through
// a quaternion to satisfy the quaternion-taking wrappers would need a
// trace-sign-dependent case split -- unsafe to tape, see that file's
// header).
//
// This composition is valid because both arm bases are welded to
// `link_torso_5` with an *identity* offset (`torso_left_arm_weld` /
// `torso_right_arm_weld` in rby1_with_holonomic_base_spherized.urdf): each
// arm's own IK solver frame *is* `link_torso_5`, so "goal pose relative to
// link_torso_5" is exactly the `eetrans`/`eerot` the arm parameterizations
// expect -- no extra fixed offset to account for.
//
// Tape input layout (size 32):
//   [0:4)   -- wheeled base (x, y, base_rz_cos, base_rz_sin). `base_rz` is a
//              URDF `continuous` joint, so pinocchio's own configuration
//              vector already wants (cos(theta), sin(theta)) rather than a
//              raw angle -- callers pass that pair directly (see e.g.
//              `sample()` in fk_template.hh, which is responsible for
//              splitting its own single sampled angle into this pair)
//              instead of this function computing cos/sin from a raw theta
//              itself.
//   [4:10)  -- torso_0 .. torso_5 (6 revolute joints)
//   [10:17) -- left arm goal end-effector pose (x, y, z, qx, qy, qz, qw),
//              in world frame
//   [17]    -- left arm free joint angle (j15_free)
//   [18:25) -- right arm goal end-effector pose (x, y, z, qx, qy, qz, qw),
//              in world frame
//   [25]    -- right arm free joint angle (j24_free)
//   [26:29) -- left arm GCP selectors (elbow_sel, shoulder_sel, wrist_sel)
//   [29:32) -- right arm GCP selectors (elbow_sel, shoulder_sel, wrist_sel)
//
// Every other joint in the model (the arms' own joints, head, grippers) is
// irrelevant to `link_torso_5`'s own placement -- it's strictly an ancestor
// quantity in the kinematic tree -- so those configuration slots are simply
// left at zero.
//
// Output layout:
//   "q" (24) -- the combined ambient configuration: base (x, y,
//       base_rz_cos, base_rz_sin -- a pass-through of the input, already in
//       pinocchio's own representation) + torso_0..5 + left arm q (7) +
//       right arm q (7), in that order. This is exactly
//       AmbientConfigurationBlock's layout, so fk_template.hh's
//       parameterized_ik can declare its `q` local directly as the
//       function's return type and hand it straight to {{param_ik_code}}
//       instead of assembling it itself.
//   per arm (left then right): "u" (the 3 pre-clip RainbowAsin/RainbowAcos
//       arguments), "reach_violation" (1), "loss" (1) -- see
//       RainbowArmParamResult in rainbow_arm_parameterization.hh for what
//       each means. Both arms' q already live in the single "q" segment
//       above, not repeated here.
//
// No gradient/Jacobian output here (unlike IiwaSE3ParameterizationCG's
// `compute_gradient` option) -- out of scope for now.
template <typename T>
auto RainbowIkCG(
    const RobotInfo &info,
    const std::string &language
)
{
    using ModelT = pinocchio::ModelTpl<T>;
    using DataT = pinocchio::DataTpl<T>;

    std::cout << "Generating whole-body-relative parameterized IK code for the rainbow arms..." << std::endl;

    const size_t num_inp = 4 + 6 + 7 + 1 + 7 + 1 + 3 + 3; // == 32
    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    const T base_x = ad_inp[0];
    const T base_y = ad_inp[1];
    const T base_rz_cos = ad_inp[2];
    const T base_rz_sin = ad_inp[3];

    const T torso_0 = ad_inp[4];
    const T torso_1 = ad_inp[5];
    const T torso_2 = ad_inp[6];
    const T torso_3 = ad_inp[7];
    const T torso_4 = ad_inp[8];
    const T torso_5 = ad_inp[9];

    const T left_x = ad_inp[10];
    const T left_y = ad_inp[11];
    const T left_z = ad_inp[12];
    const T left_qx = ad_inp[13];
    const T left_qy = ad_inp[14];
    const T left_qz = ad_inp[15];
    const T left_qw = ad_inp[16];
    const T left_j15_free = ad_inp[17];

    const T right_x = ad_inp[18];
    const T right_y = ad_inp[19];
    const T right_z = ad_inp[20];
    const T right_qx = ad_inp[21];
    const T right_qy = ad_inp[22];
    const T right_qz = ad_inp[23];
    const T right_qw = ad_inp[24];
    const T right_j24_free = ad_inp[25];

    const T left_elbow_sel = ad_inp[26];
    const T left_shoulder_sel = ad_inp[27];
    const T left_wrist_sel = ad_inp[28];

    const T right_elbow_sel = ad_inp[29];
    const T right_shoulder_sel = ad_inp[30];
    const T right_wrist_sel = ad_inp[31];

    ModelT ad_model = info.model.cast<T>();
    DataT ad_data(ad_model);

    Eigen::VectorX<T> q_full = Eigen::VectorX<T>::Zero(ad_model.nq);

    // Writes `value` into q_full at the named scalar (nq() == 1) joint's
    // configuration slot. Joint indices/idx_q() are structural (the same
    // for info.model and its T-cast ad_model), so it's safe to query them
    // off the original double model.
    auto set_scalar_joint = [&](const std::string &name, const T &value) {
        if (not info.model.existJointName(name))
        {
            throw std::runtime_error(fmt::format("RainbowIkCG: model has no joint named `{}`", name));
        }
        const auto &joint = info.model.joints[info.model.getJointId(name)];
        if (joint.nq() != 1)
        {
            throw std::runtime_error(
                fmt::format("RainbowIkCG: joint `{}` has nq() == {}, expected 1", name, joint.nq()));
        }
        q_full[joint.idx_q()] = value;
    };

    set_scalar_joint("base_x", base_x);
    set_scalar_joint("base_y", base_y);

    // base_rz is a URDF `continuous` joint -- pinocchio's configuration
    // vector wants (cos(theta), sin(theta)) directly, which is exactly what
    // base_rz_cos/base_rz_sin already are, so this is a plain pass-through.
    if (not info.model.existJointName("base_rz"))
    {
        throw std::runtime_error("RainbowIkCG: model has no joint named `base_rz`");
    }
    {
        const auto &base_rz_joint = info.model.joints[info.model.getJointId("base_rz")];
        if (base_rz_joint.nq() != 2)
        {
            throw std::runtime_error(fmt::format(
                "RainbowIkCG: joint `base_rz` has nq() == {}, expected 2 (continuous)", base_rz_joint.nq()));
        }
        q_full[base_rz_joint.idx_q() + 0] = base_rz_cos;
        q_full[base_rz_joint.idx_q() + 1] = base_rz_sin;
    }

    set_scalar_joint("torso_0", torso_0);
    set_scalar_joint("torso_1", torso_1);
    set_scalar_joint("torso_2", torso_2);
    set_scalar_joint("torso_3", torso_3);
    set_scalar_joint("torso_4", torso_4);
    set_scalar_joint("torso_5", torso_5);

    forwardKinematics(ad_model, ad_data, q_full);
    updateFramePlacements(ad_model, ad_data);

    if (not info.model.existFrame("link_torso_5"))
    {
        throw std::runtime_error("RainbowIkCG: model has no frame named `link_torso_5`");
    }
    const auto torso_frame_id = info.model.getFrameId("link_torso_5");
    const auto &torso_world = ad_data.oMf[torso_frame_id];

    // Each arm's goal pose, world frame -> relative to link_torso_5 (see
    // file header for why that's exactly the arm solver's own frame).
    Eigen::Quaternion<T> left_goal_quat(left_qw, left_qx, left_qy, left_qz);
    pinocchio::SE3Tpl<T> left_goal_world(left_goal_quat, Eigen::Matrix<T, 3, 1>(left_x, left_y, left_z));
    pinocchio::SE3Tpl<T> left_goal_local = torso_world.inverse() * left_goal_world;
    // pinocchio::SE3Tpl<T> left_goal_local = left_goal_world;

    Eigen::Quaternion<T> right_goal_quat(right_qw, right_qx, right_qy, right_qz);
    pinocchio::SE3Tpl<T> right_goal_world(right_goal_quat, Eigen::Matrix<T, 3, 1>(right_x, right_y, right_z));
    pinocchio::SE3Tpl<T> right_goal_local = torso_world.inverse() * right_goal_world;
    // pinocchio::SE3Tpl<T> right_goal_local = right_goal_world;

    auto left_result = RainbowLeftArmParameterizationFromPose<T>(
        left_goal_local.translation(), left_goal_local.rotation(),
        left_j15_free, left_elbow_sel, left_shoulder_sel, left_wrist_sel);

    auto right_result = RainbowRightArmParameterizationFromPose<T>(
        right_goal_local.translation(), right_goal_local.rotation(),
        right_j24_free, right_elbow_sel, right_shoulder_sel, right_wrist_sel);

    const size_t n_base = 4;
    const size_t n_torso = 6;
    const size_t n_arm_q = 7;
    const size_t n_q_combined = n_base + n_torso + n_arm_q + n_arm_q; // == 24

    const size_t n_unclipped = 3; // RainbowArmParamResult::unclipped is always a Vector3
    const size_t n_reach_violation = 1;
    const size_t n_loss = 1;
    const size_t n_out_per_arm_extra = n_unclipped + n_reach_violation + n_loss;
    const size_t n_out = n_q_combined + 2 * n_out_per_arm_extra;

    ADVectorXs data(n_out);

    // q: base (x, y, base_rz_cos, base_rz_sin) + torso_0..5 + left arm q
    // (7) + right arm q (7) -- base/torso are commanded pass-throughs (no IK
    // to solve for them), so they're just the input values, unchanged.
    data[0] = base_x;
    data[1] = base_y;
    data[2] = base_rz_cos;
    data[3] = base_rz_sin;
    data[4] = torso_0;
    data[5] = torso_1;
    data[6] = torso_2;
    data[7] = torso_3;
    data[8] = torso_4;
    data[9] = torso_5;
    for (size_t i = 0; i < n_arm_q; ++i)
    {
        data[n_base + n_torso + i] = left_result.q[i];
    }
    for (size_t i = 0; i < n_arm_q; ++i)
    {
        data[n_base + n_torso + n_arm_q + i] = right_result.q[i];
    }

    size_t offset = n_q_combined;
    for (size_t i = 0; i < n_unclipped; ++i)
    {
        data[offset + i] = left_result.unclipped[i];
    }
    offset += n_unclipped;
    data[offset] = left_result.reach_violation;
    offset += n_reach_violation;
    data[offset] = left_result.loss;
    offset += n_loss;

    for (size_t i = 0; i < n_unclipped; ++i)
    {
        data[offset + i] = right_result.unclipped[i];
    }
    offset += n_unclipped;
    data[offset] = right_result.reach_violation;
    offset += n_reach_violation;
    data[offset] = right_result.loss;
    offset += n_loss;

    ADFun<CGD> rainbow_ik_func(ad_inp, data);
    std::cout << "Created the AD function." << std::endl;
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);

    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = rainbow_ik_func.Forward(0, ind_vars);
    std::cout << "Ran the AD function." << std::endl;

    // Codegen needs the block-oriented language for fk_template.hh-style
    // FloatVector-based parameterized_ik, same remap used elsewhere.
    const std::string lang = (language == "c++") ? "c++_block" : language;

    SegmentedVariableNameGenerator<double> nameGen(
        {{"base", 4, true},
         {"torso", 6, true},
         {"left_pose", 7, true},
         {"left_j15", 1, false},
         {"right_pose", 7, true},
         {"right_j24", 1, false},
         {"left_gcp", 3, true},
         {"right_gcp", 3, true}},
        {{"q", n_q_combined, true},
         {"u_left", n_unclipped, true},
         {"reach_violation_left", n_reach_violation, true},
         {"loss_left", n_loss, true},
         {"u_right", n_unclipped, true},
         {"reach_violation_right", n_reach_violation, true},
         {"loss_right", n_loss, true}});

    std::cout << "Generated the whole-body-relative parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}

// Maps RBY1's Sample (23 raw values in [0, 1)) to its Configuration (26
// values: base(4) + torso(6) + left_pose(8) + right_pose(8) -- see
// RainbowIkCG's header for the exact per-field meaning of the Configuration
// side), mirroring trace_map_to_se3 in se3_tracer.hh but for a mobile base
// plus two arms instead of one fixed-base arm. Reuses that file's
// map_bounded / map_unbounded_revolute / map_so3_shoemake helpers verbatim.
//
// Sample layout (23):
//   [0:2)   -- base (x, y), map_bounded against `bounds` (bounds->lower[0]/
//              [1], bounds->upper[0]/[1]; bounds->*[2], the z bound, is
//              unused -- the wheeled base has no z DOF)
//   [2]     -- base_rz, map_unbounded_revolute -> (cos, sin)
//   [3:9)   -- torso_0 .. torso_5, each map_bounded against that joint's
//              own limits, read off `model`
//   [9:16)  -- left arm: position (3, map_bounded against the same
//              `bounds` used for base x/y -- there's only one Bounds in
//              this pipeline, so base and both arms currently share it),
//              orientation (3 raw values, map_so3_shoemake -> unit
//              quaternion), free joint angle (1, map_bounded(0, 2*pi))
//   [16:23) -- right arm, same shape as left
//
// Configuration layout (26): exactly RainbowIkCG's own pre-GCP tape input
// (see its header comment in this file) -- GCP isn't sampled here, it's
// fixed per fk_template.hh's right_gcp/left_gcp.
inline auto trace_rby1_sample(
    const pinocchio::Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds) -> Traced
{
    const std::size_t n_sample = 23;
    const std::size_t n_config = 26;

    ADVectorXs ad_u(n_sample);
    ADVectorXs ad_out(n_config);

    for (std::size_t i = 0; i < n_sample; ++i)
    {
        ad_u[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_u);

    // base x, y.
    ad_out[0] = map_bounded(ad_u[0], bounds->lower[0], bounds->upper[0]);
    ad_out[1] = map_bounded(ad_u[1], bounds->lower[1], bounds->upper[1]);

    // base_rz -> (cos, sin).
    ADCG base_rz_cos, base_rz_sin;
    map_unbounded_revolute(ad_u[2], base_rz_cos, base_rz_sin);
    ad_out[2] = base_rz_cos;
    ad_out[3] = base_rz_sin;

    // torso_0 .. torso_5, each against its own joint limits.
    auto sample_scalar_joint = [&](const std::string &name, const ADCG &u, ADCG &out) {
        if (not model.existJointName(name))
        {
            throw std::runtime_error(fmt::format("trace_rby1_sample: model has no joint named `{}`", name));
        }
        const auto &joint = model.joints[model.getJointId(name)];
        if (joint.nq() != 1)
        {
            throw std::runtime_error(
                fmt::format("trace_rby1_sample: joint `{}` has nq() == {}, expected 1", name, joint.nq()));
        }
        const double lower = model.lowerPositionLimit[joint.idx_q()];
        const double upper = model.upperPositionLimit[joint.idx_q()];
        out = map_bounded(u, lower, upper);
    };

    sample_scalar_joint("torso_0", ad_u[3], ad_out[4]);
    sample_scalar_joint("torso_1", ad_u[4], ad_out[5]);
    sample_scalar_joint("torso_2", ad_u[5], ad_out[6]);
    sample_scalar_joint("torso_3", ad_u[6], ad_out[7]);
    sample_scalar_joint("torso_4", ad_u[7], ad_out[8]);
    sample_scalar_joint("torso_5", ad_u[8], ad_out[9]);

    // Each arm: position (3, map_bounded), orientation (3 raw values ->
    // map_so3_shoemake -> unit quaternion), free joint angle (1,
    // map_bounded(0, 2*pi)).
    auto sample_arm = [&](std::size_t sample_offset, std::size_t out_offset) {
        for (int i = 0; i < 3; ++i)
        {
            ad_out[out_offset + i] = map_bounded(ad_u[sample_offset + i], bounds->lower[i], bounds->upper[i]);
        }
        ADCG qx, qy, qz, qw;
        map_so3_shoemake(
            ad_u[sample_offset + 3], ad_u[sample_offset + 4], ad_u[sample_offset + 5], qx, qy, qz, qw);
        ad_out[out_offset + 3] = qx;
        ad_out[out_offset + 4] = qy;
        ad_out[out_offset + 5] = qz;
        ad_out[out_offset + 6] = qw;
        ad_out[out_offset + 7] = map_bounded(ad_u[sample_offset + 6], 0.0, 2 * M_PI);
    };

    sample_arm(9, 10);   // left: sample[9:16) -> config[10:18)
    sample_arm(16, 18);  // right: sample[16:23) -> config[18:26)

    CppAD::ADFun<CGD> sample_func(ad_u, ad_out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_sample);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = sample_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), result.size()};
}

// Angular distance (in [0, pi]) between two SO(2) elements given as unit
// (cos, sin) pairs -- the natural metric for base_rz (a URDF `continuous`
// joint represented that way in the parameterized space; see RainbowIkCG's
// header). A plain Euclidean difference of the raw (cos, sin) values isn't
// this: it's a chordal distance, and (more to the point) linearly
// interpolating (cos, sin) directly -- see so2_slerp below -- walks off the
// unit circle entirely partway through, which a distance metric alone can't
// fix. Computed as the angle of the relative rotation z_a^-1 * z_b (treating
// (cos, sin) as a unit complex number z = cos + i*sin): atan2(sin_rel,
// cos_rel) is exactly the shortest signed angle in (-pi, pi] with no
// wraparound ambiguity or domain-clamp branch needed -- so2 doesn't have the
// theta-near-pi axis-recovery singularity so3_log_smooth has to work around.
inline ADCG so2_distance(const ADCG &cos_a, const ADCG &sin_a, const ADCG &cos_b, const ADCG &sin_b)
{
    const ADCG cos_rel = cos_a * cos_b + sin_a * sin_b;
    const ADCG sin_rel = cos_a * sin_b - sin_a * cos_b;
    return abs(atan2(sin_rel, cos_rel));
}

// SO(2) slerp: rotates (cos_a, sin_a) towards (cos_b, sin_b) by fraction `t`
// of the shortest angular path. Computes the same relative rotation z_rel =
// z_a^-1 * z_b as so2_distance, then composes z_a with the t-scaled relative
// rotation z_rel^t = (cos(t * theta_rel), sin(t * theta_rel)). Always
// returns a unit (cos, sin) pair, unlike a plain linear lerp of the raw
// values (which is exactly the bug this replaces).
inline void so2_slerp(
    const ADCG &cos_a, const ADCG &sin_a,
    const ADCG &cos_b, const ADCG &sin_b,
    const ADCG &t,
    ADCG &cos_out, ADCG &sin_out)
{
    const ADCG cos_rel = cos_a * cos_b + sin_a * sin_b;
    const ADCG sin_rel = cos_a * sin_b - sin_a * cos_b;
    const ADCG theta_rel = atan2(sin_rel, cos_rel);
    const ADCG cos_t = cos(t * theta_rel);
    const ADCG sin_t = sin(t * theta_rel);
    cos_out = cos_a * cos_t - sin_a * sin_t;
    sin_out = sin_a * cos_t + cos_a * sin_t;
}

// Distance over RainbowIkCG's own 26-dim parameterized input space -- the
// pre-GCP portion of its 32-dim tape input (see RainbowIkCG's header):
// [base(4), torso(6), left_pose(7)+left_j15(1), right_pose(7)+right_j24(1)].
// base_x/base_y and torso (8 total -- ordinary joint angles) use a plain
// Euclidean difference; base_rz (given as (cos, sin), not a raw angle) uses
// so2_distance instead, collapsing its 2 raw values into the 1 angular
// distance they represent; the remaining two 8-dim blocks (each an SE3 pose
// + free joint angle, exactly se3_tracer.hh's own pose+psi shape) each go
// through the same relative-transform + so3_log_smooth +
// free-angle-difference construction as trace_SE3_distance there. All the
// pieces are concatenated into one displacement vector and a single norm is
// taken over the whole thing, mirroring trace_SE3_distance's own "build one
// displacement vector, norm at the end" pattern.
inline auto trace_rby1_distance(const std::string &language) -> Traced
{
    const std::size_t n_base = 4;   // base_x, base_y, base_rz_cos, base_rz_sin
    const std::size_t n_torso = 6;
    const std::size_t n_se3_block = 8;  // pose(7) + free joint angle(1)
    const std::size_t n_per_side = n_base + n_torso + 2 * n_se3_block;  // == 26
    const std::size_t n_input = 2 * n_per_side;  // a, b

    ADVectorXs ad_input(n_input);
    for (std::size_t i = 0; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    ADVectorXs a = ad_input.head(n_per_side);
    ADVectorXs b = ad_input.segment(n_per_side, n_per_side);

    // base_x, base_y (2, Euclidean) + base_rz (1, so2_distance) + torso (6,
    // Euclidean) + left arm (7) + right arm (7) == 23.
    const std::size_t n_disp_base_torso = 2 + 1 + n_torso;  // == 9
    ADVectorXs displacement(n_disp_base_torso + 2 * 7);  // 9 + 7 + 7 == 23

    displacement.segment(0, 2) = a.segment(0, 2) - b.segment(0, 2);
    displacement[2] = so2_distance(a[2], a[3], b[2], b[3]);
    displacement.segment(3, n_torso) = a.segment(n_base, n_torso) - b.segment(n_base, n_torso);

    // Left arm: SE3 log-displacement (6) + free joint angle difference (1).
    const std::size_t left_offset = n_base + n_torso;  // == 10
    const auto left_a = read_transform(a, left_offset);
    const auto left_b = read_transform(b, left_offset);
    SE3Tpl<ADCG> left_rel = left_a.inverse() * left_b;
    displacement.segment(n_disp_base_torso, 6) = se3_displacement(left_rel);
    displacement[n_disp_base_torso + 6] = a[left_offset + 7] - b[left_offset + 7];

    // Right arm, same construction.
    const std::size_t right_offset = left_offset + n_se3_block;  // == 18
    const auto right_a = read_transform(a, right_offset);
    const auto right_b = read_transform(b, right_offset);
    SE3Tpl<ADCG> right_rel = right_a.inverse() * right_b;
    displacement.segment(n_disp_base_torso + 7, 6) = se3_displacement(right_rel);
    displacement[n_disp_base_torso + 7 + 6] = a[right_offset + 7] - b[right_offset + 7];

    ADVectorXs out(1);
    out[0] = displacement.norm();

    CppAD::ADFun<CGD> dist_func(ad_input, out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = dist_func.Forward(0, ind_vars);

    SegmentedVariableNameGenerator<double> nameGen({{"a", n_per_side, true}, {"b", n_per_side, true}});

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        1};
}

// Interpolation over the same 26-dim parameterized space as
// trace_rby1_distance: plain linear interpolation for base_x/base_y and
// torso (8 total), so2_slerp for base_rz (given as (cos, sin), not a raw
// angle -- a plain linear lerp of those two values would walk off the unit
// circle, exactly the bug so2_slerp exists to avoid), and slerp_se3
// (rotation) + linear (translation, free joint angle) for each of the two
// 8-dim SE3+free-angle blocks -- exactly trace_interpolate_impl's own
// construction, just run twice (once per arm) instead of once. Split into
// _impl (parameterized by the input/output VarSegments, exactly like
// trace_interpolate_impl) plus regular/block wrappers, same reasoning as
// trace_interpolate / trace_interpolate_block: the regular wrapper leaves
// output_segments empty so the generated code uses the default `y[i]`
// naming (matching a plain, non-block caller's own local `y` buffer), while
// the block wrapper broadcasts `a`/`b` (single, not per-lane) across the
// `rake` lanes via `.broadcast(i)` and names the per-lane output `out`
// (matching a block caller's own `out` parameter).
inline auto trace_rby1_interpolate_impl(
    const std::string &language,
    std::vector<VarSegment> input_segments,
    std::vector<VarSegment> output_segments) -> Traced
{
    const std::size_t n_base = 4;   // base_x, base_y, base_rz_cos, base_rz_sin
    const std::size_t n_torso = 6;
    const std::size_t n_se3_block = 8;
    const std::size_t n_per_side = n_base + n_torso + 2 * n_se3_block;  // == 26
    const std::size_t n_input = 2 * n_per_side + 1;  // a, b, t

    ADVectorXs ad_input(n_input);
    for (std::size_t i = 0; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    ADVectorXs a = ad_input.head(n_per_side);
    ADVectorXs b = ad_input.segment(n_per_side, n_per_side);
    ADCG t = ad_input[2 * n_per_side];

    ADVectorXs out(n_per_side);

    // base_x, base_y: plain linear interpolation.
    out.segment(0, 2) = (ADCG(1.0) - t) * a.segment(0, 2) + t * b.segment(0, 2);

    // base_rz: SO(2) slerp.
    ADCG base_rz_cos_out, base_rz_sin_out;
    so2_slerp(a[2], a[3], b[2], b[3], t, base_rz_cos_out, base_rz_sin_out);
    out[2] = base_rz_cos_out;
    out[3] = base_rz_sin_out;

    // torso: plain linear interpolation.
    out.segment(n_base, n_torso) = (ADCG(1.0) - t) * a.segment(n_base, n_torso) + t * b.segment(n_base, n_torso);

    const std::size_t left_offset = n_base + n_torso;  // == 10
    ADVectorXs left_a = a.segment(left_offset, n_se3_block);
    ADVectorXs left_b = b.segment(left_offset, n_se3_block);
    ADVectorXs left_out(n_se3_block);
    slerp_se3(left_a, left_b, t, left_out);
    left_out[7] = (ADCG(1.0) - t) * left_a[7] + t * left_b[7];
    out.segment(left_offset, n_se3_block) = left_out;

    const std::size_t right_offset = left_offset + n_se3_block;  // == 18
    ADVectorXs right_a = a.segment(right_offset, n_se3_block);
    ADVectorXs right_b = b.segment(right_offset, n_se3_block);
    ADVectorXs right_out(n_se3_block);
    slerp_se3(right_a, right_b, t, right_out);
    right_out[7] = (ADCG(1.0) - t) * right_a[7] + t * right_b[7];
    out.segment(right_offset, n_se3_block) = right_out;

    CppAD::ADFun<CGD> interp_func(ad_input, out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = interp_func.Forward(0, ind_vars);

    SegmentedVariableNameGenerator<double> nameGen(
        std::move(input_segments), std::move(output_segments));

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        n_per_side};
}

inline auto trace_rby1_interpolate(const std::string &language) -> Traced
{
    const std::size_t n_per_side = 4 + 6 + 2 * 8;  // == 26
    return trace_rby1_interpolate_impl(
        language, {{"a", n_per_side, true}, {"b", n_per_side, true}, {"t", 1, false}}, {});
}

inline auto trace_rby1_interpolate_block(const std::string &language) -> Traced
{
    const std::size_t n_per_side = 4 + 6 + 2 * 8;  // == 26
    const std::string lang = (language == "c++") ? "c++_block" : language;
    return trace_rby1_interpolate_impl(
        lang,
        {{"a", n_per_side, true, ".broadcast(", ")"}, {"b", n_per_side, true, ".broadcast(", ")"}, {"t", 1, false}},
        {{"out", n_per_side, true}});
}

// =====================================================================
// constrained_bimanual_ik support
//
// Instead of sampling each hand's world-frame pose directly (as
// RainbowIkCG's "q" input does via left_pose/right_pose), this mode
// samples a single mid-frame pose T_mid plus one free joint angle per
// arm, and derives each hand's goal pose as T_l = T_mid * T_mid_left,
// T_r = T_mid * T_mid_right, where T_mid_left/T_mid_right are fixed
// (per-planning-problem) SE3 offsets -- not sampled, and not known at
// codegen time. They're computed at runtime by fk_template.hh's
// compute_mid_pose(q) (fed by RainbowMidPoseFkCG below) from a
// reference whole-body configuration, and stored in fk_template.hh's
// own `t_mid_left`/`t_mid_right` thread_local statics, which
// RainbowConstrainedBimanualIkCG's generated code reads directly by
// name -- exactly the same "fixed tape input named to alias an outer
// class member" trick RainbowIkCG already uses for `left_gcp`/
// `right_gcp` (see that function's header comment): the segment is
// still a genuine CppAD tape independent (SegmentedVariableNameGenerator
// is a pure code-emission layer over one flat tape, see
// lang_name_gen.hh), but naming it `t_mid_left` instead of `x[N]` makes
// the generated C++ text -- spliced into a method body of the same
// struct that declares `t_mid_left` -- resolve to that runtime-mutable
// member instead of a tape-input array slot.
// =====================================================================

// Dual-hand forward kinematics: given the full ambient/whole-body
// configuration `q`, computes both hands' world-frame poses (ee_left,
// ee_right). Feeds fk_template.hh's compute_mid_pose, which uses these
// two poses to derive T_mid_left/T_mid_right (see file-header comment
// above). Not templated on T -- runs at ADCG directly, same style as
// fkcc_gen.cc's trace_sphere_cc_fk (its trace_frame helper is
// duplicated inline here rather than shared, to avoid touching that
// working call site).
//
// Output layout (24): [0:12) ee_left world pose (translation xyz +
// rotation matrix, column-major), [12:24) ee_right world pose, same
// shape -- exactly vamp::to_isometry's expected 12-float input, so
// compute_mid_pose can call it directly on each half.
inline auto RainbowMidPoseFkCG(const RobotInfo &info, const std::string &language) -> Traced
{
    std::cout << "Generating dual-hand FK code for compute_mid_pose..." << std::endl;

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

    if (not info.model.existFrame("ee_left"))
    {
        throw std::runtime_error("RainbowMidPoseFkCG: model has no frame named `ee_left`");
    }
    if (not info.model.existFrame("ee_right"))
    {
        throw std::runtime_error("RainbowMidPoseFkCG: model has no frame named `ee_right`");
    }
    const auto left_id = info.model.getFrameId("ee_left");
    const auto right_id = info.model.getFrameId("ee_right");

    ADVectorXs data(24);

    // Writes a 12-value world pose (translation xyz + rotation matrix,
    // column-major) for `frame_id` into `data[offset:offset+12)` --
    // duplicate of fkcc_gen.cc's trace_frame body.
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

    write_frame(left_id, 0);
    write_frame(right_id, 12);

    ADFun<CGD> mid_pose_fk_func(ad_q, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = mid_pose_fk_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), 24};
}

// Left/right end-effector WORLD poses derived from a T_mid pose plus the fixed
// t_mid_left/t_mid_right offsets: T_l = T_mid * t_mid_left, T_r = T_mid * t_mid_right -- the
// same composition RainbowConstrainedBimanualIkCG does internally (see that function's
// header comment) before converting to link_torso_5-relative frame for arm IK, but broken
// out standalone here since eefs_in_collision (fk_template.hh) only needs the hands' world
// poses, not the arm IK that follows. Branch-free (pure SE3 composition), so -- like
// RainbowMidPoseFkCG -- a single "c++"-generated trace is valid whether fk_template.hh
// instantiates it scalar or rake-batched (FloatVector<rake,1>); operator overloading on that
// type makes the generated arithmetic work unchanged either way, no "c++_block" variant
// needed. Output is translation + rotation matrix (vamp::to_isometry's 12-float layout)
// rather than a quaternion, both because that's what to_isometry wants and to avoid ever
// needing to convert a rotation matrix back to a quaternion in taped code (unsafe to trace,
// same reasoning as the header comment above RainbowConstrainedBimanualIkCG) -- this is also
// why RainbowEefLocalSpheresFkCG below takes its per-eef pose as matrix, not quaternion: it
// consumes this function's output directly.
//
// Tape input layout (21): [0:7) T_mid pose (x, y, z, qx, qy, qz, qw), [7:14) t_mid_left
// (fixed, aliases the `t_mid_left` thread_local member -- not actually driven by a caller,
// see RainbowConstrainedBimanualIkCG's header comment for this convention), [14:21)
// t_mid_right (fixed, aliases `t_mid_right`).
//
// Output layout (24): [0:12) left hand world pose (translation xyz + rotation matrix,
// column-major), [12:24) right hand world pose, same shape as RainbowMidPoseFkCG's output.
inline auto RainbowEefWorldPosesFromMidCG(const std::string &language) -> Traced
{
    std::cout << "Generating T_mid -> hand world pose code for eefs_in_collision..." << std::endl;

    const std::size_t num_inp = 7 + 7 + 7;
    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = ADCG(0.0001);
    }
    Independent(ad_inp);

    Eigen::Quaternion<ADCG> t_mid_quat(ad_inp[6], ad_inp[3], ad_inp[4], ad_inp[5]);
    pinocchio::SE3Tpl<ADCG> t_mid_world(t_mid_quat, Eigen::Matrix<ADCG, 3, 1>(ad_inp[0], ad_inp[1], ad_inp[2]));

    Eigen::Quaternion<ADCG> t_mid_left_quat(ad_inp[13], ad_inp[10], ad_inp[11], ad_inp[12]);
    pinocchio::SE3Tpl<ADCG> t_mid_left_offset(
        t_mid_left_quat, Eigen::Matrix<ADCG, 3, 1>(ad_inp[7], ad_inp[8], ad_inp[9]));

    Eigen::Quaternion<ADCG> t_mid_right_quat(ad_inp[20], ad_inp[17], ad_inp[18], ad_inp[19]);
    pinocchio::SE3Tpl<ADCG> t_mid_right_offset(
        t_mid_right_quat, Eigen::Matrix<ADCG, 3, 1>(ad_inp[14], ad_inp[15], ad_inp[16]));

    pinocchio::SE3Tpl<ADCG> left_world = t_mid_world * t_mid_left_offset;
    pinocchio::SE3Tpl<ADCG> right_world = t_mid_world * t_mid_right_offset;

    ADVectorXs data(24);

    // Duplicate of RainbowMidPoseFkCG's write_frame, operating on an SE3Tpl instead of a
    // pinocchio frame placement.
    auto write_pose = [&](const pinocchio::SE3Tpl<ADCG> &pose, std::size_t offset) {
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

// Per-end-effector collision spheres rigidly attached to each of
// `info.end_effector_names` (e.g. gripper/finger geometry, as opposed to an
// externally attached object), expressed as a function of a *candidate
// world-frame pose for that end effector* (translation + rotation matrix,
// vamp::to_isometry's 12-float layout) instead of the ambient joint
// configuration -- since these spheres are rigid with their end-effector
// frame, their world position is just `R * local + t`, with no
// forwardKinematics needed. Pose is taken as a rotation *matrix* rather
// than a quaternion so this consumes RainbowEefWorldPosesFromMidCG's own
// matrix-form output directly, with no matrix->quaternion round trip
// through taped code -- see that function's header comment for why that
// round trip is unsafe to tape. Also branch-free, like RainbowMidPoseFkCG,
// so the same "c++"-generated trace is valid whether fk_template.hh
// instantiates it scalar or rake-batched (FloatVector<rake,1>): operator
// overloading on that type makes the generated arithmetic work unchanged
// either way. Generic over the number of end effectors (not hardcoded to
// the bimanual ee_left/ee_right pair), so it works unchanged if a robot
// config declares 1, 2, or more. Lets ParameterizedSpace::eefs_in_collision
// (fk_template.hh) reject a sampled T_mid pose whose hands would already
// be in collision, without solving arm IK for the rest of the body first.
//
// `local_offset` per sphere is precomputed here (double precision, not
// taped) as `frame.placement.inverse().act(sphere.relative.translation())`
// -- i.e. the sphere's fixed offset from the URDF sphere's own parent
// joint (`sphere.relative`), re-expressed relative to the end effector's
// own *frame* instead, since that's the pose eef_world_poses() hands us
// (see fk_template.hh's ParameterizedSpace::eef_world_poses), not the
// frame's parent joint's pose. Sphere orientation is irrelevant (spheres
// are symmetric), so only the translation is carried through.
//
// Tape input layout (12 * num_end_effectors): each end effector's world
// pose (x, y, z, then rotation matrix column-major), in
// `info.end_effector_names` order.
//
// Output: each end effector's spheres (x, y, z, r) each, back to back, in
// `info.end_effector_names` order, and within an end effector, in the
// order they appear in `info.spheres`. `EefLocalSpheres::counts` reports
// how many spheres landed in each end effector's slice.
inline auto RainbowEefLocalSpheresFkCG(const RobotInfo &info, const std::string &language) -> EefLocalSpheres
{
    std::cout << "Generating per-end-effector local-sphere FK code for eefs_in_collision..." << std::endl;

    const auto num_end_effectors = info.end_effector_indexes.size();

    // Per end effector: local sphere offsets + radii relative to the end effector's own
    // frame -- fixed, precomputed in double precision.
    std::vector<std::vector<std::pair<Eigen::Vector3d, float>>> per_eef_spheres(num_end_effectors);
    for (auto k = 0U; k < num_end_effectors; ++k)
    {
        const auto &frame = info.model.frames[info.end_effector_indexes[k]];

        for (const auto &sphere : info.spheres)
        {
            if (sphere.parent_joint != frame.parentJoint)
            {
                continue;
            }

            Eigen::Vector3d local_offset = frame.placement.inverse().act(sphere.relative.translation());
            per_eef_spheres[k].emplace_back(local_offset, sphere.radius);
        }
    }

    ADVectorXs ad_pose(12 * num_end_effectors);
    for (auto i = 0U; i < ad_pose.size(); ++i)
    {
        ad_pose[i] = ADCG(0.0);
    }
    Independent(ad_pose);

    std::size_t total_spheres = 0;
    for (const auto &spheres : per_eef_spheres)
    {
        total_spheres += spheres.size();
    }
    ADVectorXs data(total_spheres * 4);

    std::size_t data_offset = 0;
    for (auto k = 0U; k < num_end_effectors; ++k)
    {
        const auto pose_offset = 12 * k;
        Eigen::Matrix<ADCG, 3, 1> translation{
            ad_pose[pose_offset + 0], ad_pose[pose_offset + 1], ad_pose[pose_offset + 2]};

        // Column-major, matching vamp::to_isometry / trace_frame's layout.
        Eigen::Matrix<ADCG, 3, 3> rotation;
        rotation.col(0) << ad_pose[pose_offset + 3], ad_pose[pose_offset + 4], ad_pose[pose_offset + 5];
        rotation.col(1) << ad_pose[pose_offset + 6], ad_pose[pose_offset + 7], ad_pose[pose_offset + 8];
        rotation.col(2) << ad_pose[pose_offset + 9], ad_pose[pose_offset + 10], ad_pose[pose_offset + 11];

        for (const auto &[local_offset, radius] : per_eef_spheres[k])
        {
            Eigen::Matrix<ADCG, 3, 1> local{ADCG(local_offset[0]), ADCG(local_offset[1]), ADCG(local_offset[2])};
            Eigen::Matrix<ADCG, 3, 1> world = rotation * local + translation;

            data[data_offset + 0] = world[0];
            data[data_offset + 1] = world[1];
            data[data_offset + 2] = world[2];
            data[data_offset + 3] = ADCG(radius);
            data_offset += 4;
        }
    }

    ADFun<CGD> eef_local_spheres_func(ad_pose, data);

    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(static_cast<std::size_t>(ad_pose.size()));
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = eef_local_spheres_func.Forward(0, ind_vars);

    std::vector<std::size_t> counts;
    counts.reserve(num_end_effectors);
    for (const auto &spheres : per_eef_spheres)
    {
        counts.push_back(spheres.size());
    }

    return EefLocalSpheres{
        Traced{generate_code(handler, result, language), handler.getTemporaryVariableCount(), result.size()},
        counts};
}

// Whole-body-relative parameterized IK for the constrained_bimanual_ik
// mode: the counterpart of RainbowIkCG above, except the two hands'
// goal poses aren't tape inputs directly -- they're derived from a
// single mid-frame pose T_mid plus the fixed T_mid_left/T_mid_right
// offsets (see file-header comment above): T_l = T_mid * T_mid_left,
// T_r = T_mid * T_mid_right, both composed in WORLD frame *before*
// being expressed relative to link_torso_5 -- so this composition is
// valid regardless of what frame T_mid itself is sampled in, the same
// way RainbowIkCG's left_pose/right_pose -> link_torso_5-relative step
// doesn't care how the caller arrived at those world-frame goals.
//
// Tape input layout (size 39):
//   [0:4)   wheeled base (x, y, base_rz_cos, base_rz_sin) -- same as
//           RainbowIkCG.
//   [4:10)  torso_0 .. torso_5 -- same as RainbowIkCG.
//   [10:17) T_mid pose (x, y, z, qx, qy, qz, qw), WORLD frame.
//   [17]    psi_left  -- left arm's free joint angle (this mode's
//           analogue of RainbowIkCG's left_j15_free).
//   [18]    psi_right -- right arm's free joint angle (right_j24_free).
//   [19:26) t_mid_left pose (x, y, z, qx, qy, qz, qw) -- FIXED, not
//           actually driven by a caller of the generated function;
//           named to alias fk_template.hh's `t_mid_left` thread_local
//           static member (see file-header comment).
//   [26:33) t_mid_right pose (x, y, z, qx, qy, qz, qw) -- FIXED, aliases
//           `t_mid_right`.
//   [33:36) left arm GCP selectors (elbow_sel, shoulder_sel, wrist_sel)
//           -- FIXED, aliases the *same* `left_gcp` member RainbowIkCG's
//           generated code already reads (shared between both modes).
//   [36:39) right arm GCP selectors -- FIXED, aliases `right_gcp`.
//
// Output layout: identical shape to RainbowIkCG's -- "q" (24: base +
// torso + left arm q(7) + right arm q(7)) followed by, per arm (left
// then right), "u"(3), "reach_violation"(1), "loss"(1). See
// RainbowArmParamResult in rainbow_arm_parameterization.hh for what
// each of those means.
template <typename T>
auto RainbowConstrainedBimanualIkCG(
    const RobotInfo &info,
    const std::string &language
)
{
    using ModelT = pinocchio::ModelTpl<T>;
    using DataT = pinocchio::DataTpl<T>;

    std::cout << "Generating constrained-bimanual (mid-pose-relative) parameterized IK code for the rainbow arms..."
               << std::endl;

    const size_t num_inp = 4 + 6 + 7 + 1 + 1 + 7 + 7 + 3 + 3;  // == 39
    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    const T base_x = ad_inp[0];
    const T base_y = ad_inp[1];
    const T base_rz_cos = ad_inp[2];
    const T base_rz_sin = ad_inp[3];

    const T torso_0 = ad_inp[4];
    const T torso_1 = ad_inp[5];
    const T torso_2 = ad_inp[6];
    const T torso_3 = ad_inp[7];
    const T torso_4 = ad_inp[8];
    const T torso_5 = ad_inp[9];

    const T t_mid_x = ad_inp[10];
    const T t_mid_y = ad_inp[11];
    const T t_mid_z = ad_inp[12];
    const T t_mid_qx = ad_inp[13];
    const T t_mid_qy = ad_inp[14];
    const T t_mid_qz = ad_inp[15];
    const T t_mid_qw = ad_inp[16];

    const T psi_left = ad_inp[17];
    const T psi_right = ad_inp[18];

    const T t_mid_left_x = ad_inp[19];
    const T t_mid_left_y = ad_inp[20];
    const T t_mid_left_z = ad_inp[21];
    const T t_mid_left_qx = ad_inp[22];
    const T t_mid_left_qy = ad_inp[23];
    const T t_mid_left_qz = ad_inp[24];
    const T t_mid_left_qw = ad_inp[25];

    const T t_mid_right_x = ad_inp[26];
    const T t_mid_right_y = ad_inp[27];
    const T t_mid_right_z = ad_inp[28];
    const T t_mid_right_qx = ad_inp[29];
    const T t_mid_right_qy = ad_inp[30];
    const T t_mid_right_qz = ad_inp[31];
    const T t_mid_right_qw = ad_inp[32];

    const T left_elbow_sel = ad_inp[33];
    const T left_shoulder_sel = ad_inp[34];
    const T left_wrist_sel = ad_inp[35];

    const T right_elbow_sel = ad_inp[36];
    const T right_shoulder_sel = ad_inp[37];
    const T right_wrist_sel = ad_inp[38];

    ModelT ad_model = info.model.cast<T>();
    DataT ad_data(ad_model);

    Eigen::VectorX<T> q_full = Eigen::VectorX<T>::Zero(ad_model.nq);

    // Same lambda as RainbowIkCG's -- writes `value` into q_full at the
    // named scalar (nq() == 1) joint's configuration slot.
    auto set_scalar_joint = [&](const std::string &name, const T &value) {
        if (not info.model.existJointName(name))
        {
            throw std::runtime_error(
                fmt::format("RainbowConstrainedBimanualIkCG: model has no joint named `{}`", name));
        }
        const auto &joint = info.model.joints[info.model.getJointId(name)];
        if (joint.nq() != 1)
        {
            throw std::runtime_error(fmt::format(
                "RainbowConstrainedBimanualIkCG: joint `{}` has nq() == {}, expected 1", name, joint.nq()));
        }
        q_full[joint.idx_q()] = value;
    };

    set_scalar_joint("base_x", base_x);
    set_scalar_joint("base_y", base_y);

    if (not info.model.existJointName("base_rz"))
    {
        throw std::runtime_error("RainbowConstrainedBimanualIkCG: model has no joint named `base_rz`");
    }
    {
        const auto &base_rz_joint = info.model.joints[info.model.getJointId("base_rz")];
        if (base_rz_joint.nq() != 2)
        {
            throw std::runtime_error(fmt::format(
                "RainbowConstrainedBimanualIkCG: joint `base_rz` has nq() == {}, expected 2 (continuous)",
                base_rz_joint.nq()));
        }
        q_full[base_rz_joint.idx_q() + 0] = base_rz_cos;
        q_full[base_rz_joint.idx_q() + 1] = base_rz_sin;
    }

    set_scalar_joint("torso_0", torso_0);
    set_scalar_joint("torso_1", torso_1);
    set_scalar_joint("torso_2", torso_2);
    set_scalar_joint("torso_3", torso_3);
    set_scalar_joint("torso_4", torso_4);
    set_scalar_joint("torso_5", torso_5);

    forwardKinematics(ad_model, ad_data, q_full);
    updateFramePlacements(ad_model, ad_data);

    if (not info.model.existFrame("link_torso_5"))
    {
        throw std::runtime_error("RainbowConstrainedBimanualIkCG: model has no frame named `link_torso_5`");
    }
    const auto torso_frame_id = info.model.getFrameId("link_torso_5");
    const auto &torso_world = ad_data.oMf[torso_frame_id];

    // T_mid in WORLD frame, and the two fixed offsets from it to each
    // hand -- see this function's header comment for why composing
    // these in WORLD frame (before the torso-relative transform below)
    // is valid.
    Eigen::Quaternion<T> t_mid_quat(t_mid_qw, t_mid_qx, t_mid_qy, t_mid_qz);
    pinocchio::SE3Tpl<T> t_mid_world(t_mid_quat, Eigen::Matrix<T, 3, 1>(t_mid_x, t_mid_y, t_mid_z));

    Eigen::Quaternion<T> t_mid_left_quat(t_mid_left_qw, t_mid_left_qx, t_mid_left_qy, t_mid_left_qz);
    pinocchio::SE3Tpl<T> t_mid_left_offset(
        t_mid_left_quat, Eigen::Matrix<T, 3, 1>(t_mid_left_x, t_mid_left_y, t_mid_left_z));

    Eigen::Quaternion<T> t_mid_right_quat(t_mid_right_qw, t_mid_right_qx, t_mid_right_qy, t_mid_right_qz);
    pinocchio::SE3Tpl<T> t_mid_right_offset(
        t_mid_right_quat, Eigen::Matrix<T, 3, 1>(t_mid_right_x, t_mid_right_y, t_mid_right_z));

    pinocchio::SE3Tpl<T> left_goal_world = t_mid_world * t_mid_left_offset;
    pinocchio::SE3Tpl<T> right_goal_world = t_mid_world * t_mid_right_offset;

    pinocchio::SE3Tpl<T> left_goal_local = torso_world.inverse() * left_goal_world;
    pinocchio::SE3Tpl<T> right_goal_local = torso_world.inverse() * right_goal_world;

    auto left_result = RainbowLeftArmParameterizationFromPose<T>(
        left_goal_local.translation(), left_goal_local.rotation(),
        psi_left, left_elbow_sel, left_shoulder_sel, left_wrist_sel);

    auto right_result = RainbowRightArmParameterizationFromPose<T>(
        right_goal_local.translation(), right_goal_local.rotation(),
        psi_right, right_elbow_sel, right_shoulder_sel, right_wrist_sel);

    const size_t n_base = 4;
    const size_t n_torso = 6;
    const size_t n_arm_q = 7;
    const size_t n_q_combined = n_base + n_torso + n_arm_q + n_arm_q;  // == 24

    const size_t n_unclipped = 3;
    const size_t n_reach_violation = 1;
    const size_t n_loss = 1;
    const size_t n_out_per_arm_extra = n_unclipped + n_reach_violation + n_loss;
    const size_t n_out = n_q_combined + 2 * n_out_per_arm_extra;

    ADVectorXs data(n_out);

    data[0] = base_x;
    data[1] = base_y;
    data[2] = base_rz_cos;
    data[3] = base_rz_sin;
    data[4] = torso_0;
    data[5] = torso_1;
    data[6] = torso_2;
    data[7] = torso_3;
    data[8] = torso_4;
    data[9] = torso_5;
    for (size_t i = 0; i < n_arm_q; ++i)
    {
        data[n_base + n_torso + i] = left_result.q[i];
    }
    for (size_t i = 0; i < n_arm_q; ++i)
    {
        data[n_base + n_torso + n_arm_q + i] = right_result.q[i];
    }

    size_t offset = n_q_combined;
    for (size_t i = 0; i < n_unclipped; ++i)
    {
        data[offset + i] = left_result.unclipped[i];
    }
    offset += n_unclipped;
    data[offset] = left_result.reach_violation;
    offset += n_reach_violation;
    data[offset] = left_result.loss;
    offset += n_loss;

    for (size_t i = 0; i < n_unclipped; ++i)
    {
        data[offset + i] = right_result.unclipped[i];
    }
    offset += n_unclipped;
    data[offset] = right_result.reach_violation;
    offset += n_reach_violation;
    data[offset] = right_result.loss;
    offset += n_loss;

    ADFun<CGD> constrained_ik_func(ad_inp, data);
    std::cout << "Created the AD function." << std::endl;
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);

    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = constrained_ik_func.Forward(0, ind_vars);
    std::cout << "Ran the AD function." << std::endl;

    const std::string lang = (language == "c++") ? "c++_block" : language;

    SegmentedVariableNameGenerator<double> nameGen(
        {{"base", 4, true},
         {"torso", 6, true},
         {"t_mid_pose", 7, true},
         {"psi_left", 1, false},
         {"psi_right", 1, false},
         {"t_mid_left", 7, true},
         {"t_mid_right", 7, true},
         {"left_gcp", 3, true},
         {"right_gcp", 3, true}},
        {{"q", n_q_combined, true},
         {"u_left", n_unclipped, true},
         {"reach_violation_left", n_reach_violation, true},
         {"loss_left", n_loss, true},
         {"u_right", n_unclipped, true},
         {"reach_violation_right", n_reach_violation, true},
         {"loss_right", n_loss, true}});

    std::cout << "Generated the constrained-bimanual parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}

// Maps constrained_bimanual_ik's Sample (17 raw values in [0, 1)) to its
// Configuration (19 values), mirroring trace_rby1_sample but with a
// single mid-pose block instead of two arm-pose blocks, and with
// psi_left/psi_right as independent trailing scalars rather than being
// bundled into an arm-pose block.
//
// Sample layout (17):
//   [0:2)   base (x, y), map_bounded against `bounds`
//   [2]     base_rz, map_unbounded_revolute -> (cos, sin)
//   [3:9)   torso_0 .. torso_5, each map_bounded against that joint's
//           own limits
//   [9:12)  t_mid position (3, map_bounded against the same `bounds`
//           used for base x/y)
//   [12:15) t_mid orientation (3 raw values, map_so3_shoemake -> unit
//           quaternion)
//   [15]    psi_left,  map_bounded(0, 2*pi)
//   [16]    psi_right, map_bounded(0, 2*pi)
//
// Configuration layout (19, note the reordering relative to
// RainbowConstrainedBimanualIkCG's own tape-input layout -- psi_left/
// psi_right come *before* the mid-pose here so that fk_template.hh's
// nn_key_constrained can use a zero-copy contiguous NNFloatArray<12>
// for everything but the quaternion, matching nn_key's own layout):
//   [0:4)   base (x, y, base_rz_cos, base_rz_sin)
//   [4:10)  torso_0 .. torso_5
//   [10]    psi_left
//   [11]    psi_right
//   [12:19) t_mid pose (x, y, z, qx, qy, qz, qw)
//
// Deliberately not `inline`: this is one of the cricket/codegen.hh-declared entry points,
// defined only by src/parameterization/rby1_constrained.cc (the sole TU that includes this
// header) and called only from other TUs (codegen.cc, fkcc_gen.cc) that see just the
// declaration -- an unreferenced `inline` definition is dead code to the compiler and gets
// dropped from the object file entirely, which is exactly what produced "undefined
// reference" link errors here before. Internal helpers this function itself calls (e.g.
// trace_rby1_constrained_interpolate_impl below, read_transform/se3_displacement in
// se3_tracer.hh) are fine to keep `inline`, since being called from here makes them genuinely
// used within this TU.
auto trace_rby1_constrained_sample(
    const pinocchio::Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds) -> Traced
{
    const std::size_t n_sample = 17;
    const std::size_t n_config = 19;

    ADVectorXs ad_u(n_sample);
    ADVectorXs ad_out(n_config);

    for (std::size_t i = 0; i < n_sample; ++i)
    {
        ad_u[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_u);

    // base x, y.
    ad_out[0] = map_bounded(ad_u[0], bounds->lower[0], bounds->upper[0]);
    ad_out[1] = map_bounded(ad_u[1], bounds->lower[1], bounds->upper[1]);

    // base_rz -> (cos, sin).
    ADCG base_rz_cos, base_rz_sin;
    map_unbounded_revolute(ad_u[2], base_rz_cos, base_rz_sin);
    ad_out[2] = base_rz_cos;
    ad_out[3] = base_rz_sin;

    // torso_0 .. torso_5, each against its own joint limits.
    auto sample_scalar_joint = [&](const std::string &name, const ADCG &u, ADCG &out) {
        if (not model.existJointName(name))
        {
            throw std::runtime_error(fmt::format("trace_rby1_constrained_sample: model has no joint named `{}`", name));
        }
        const auto &joint = model.joints[model.getJointId(name)];
        if (joint.nq() != 1)
        {
            throw std::runtime_error(fmt::format(
                "trace_rby1_constrained_sample: joint `{}` has nq() == {}, expected 1", name, joint.nq()));
        }
        const double lower = model.lowerPositionLimit[joint.idx_q()];
        const double upper = model.upperPositionLimit[joint.idx_q()];
        out = map_bounded(u, lower, upper);
    };

    sample_scalar_joint("torso_0", ad_u[3], ad_out[4]);
    sample_scalar_joint("torso_1", ad_u[4], ad_out[5]);
    sample_scalar_joint("torso_2", ad_u[5], ad_out[6]);
    sample_scalar_joint("torso_3", ad_u[6], ad_out[7]);
    sample_scalar_joint("torso_4", ad_u[7], ad_out[8]);
    sample_scalar_joint("torso_5", ad_u[8], ad_out[9]);

    // psi_left, psi_right.
    ad_out[10] = map_bounded(ad_u[15], 0.0, 2 * M_PI);
    ad_out[11] = map_bounded(ad_u[16], 0.0, 2 * M_PI);

    // t_mid: position (3, map_bounded), orientation (3 raw values ->
    // map_so3_shoemake -> unit quaternion).
    for (int i = 0; i < 3; ++i)
    {
        ad_out[12 + i] = map_bounded(ad_u[9 + i], bounds->lower[i], bounds->upper[i]);
    }
    ADCG qx, qy, qz, qw;
    map_so3_shoemake(ad_u[12], ad_u[13], ad_u[14], qx, qy, qz, qw);
    ad_out[15] = qx;
    ad_out[16] = qy;
    ad_out[17] = qz;
    ad_out[18] = qw;

    CppAD::ADFun<CGD> sample_func(ad_u, ad_out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_sample);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = sample_func.Forward(0, ind_vars);

    LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), result.size()};
}

// Distance over constrained_bimanual_ik's own 19-dim Configuration (see
// trace_rby1_constrained_sample's header for the exact field layout).
// base_x/base_y and torso (8 total) use a plain Euclidean difference;
// base_rz uses so2_distance; psi_left/psi_right (2) use a plain
// Euclidean difference; the mid-pose block goes through the same
// relative-transform + so3_log_smooth construction as
// trace_rby1_distance's own arm-pose blocks. Mirrors that function's
// "build one displacement vector, norm at the end" pattern.
auto trace_rby1_constrained_distance(const std::string &language) -> Traced
{
    const std::size_t n_base = 4;
    const std::size_t n_torso = 6;
    const std::size_t n_psi = 2;
    const std::size_t n_pose = 7;
    const std::size_t n_per_side = n_base + n_torso + n_psi + n_pose;  // == 19
    const std::size_t n_input = 2 * n_per_side;

    ADVectorXs ad_input(n_input);
    for (std::size_t i = 0; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    ADVectorXs a = ad_input.head(n_per_side);
    ADVectorXs b = ad_input.segment(n_per_side, n_per_side);

    // base_x, base_y (2, Euclidean) + base_rz (1, so2_distance) + torso
    // (6, Euclidean) + psi_left, psi_right (2, Euclidean) + mid-pose (6,
    // se3_displacement) == 17.
    const std::size_t n_disp_base_torso_psi = 2 + 1 + n_torso + n_psi;  // == 11
    ADVectorXs displacement(n_disp_base_torso_psi + 6);  // == 17

    displacement.segment(0, 2) = a.segment(0, 2) - b.segment(0, 2);
    displacement[2] = so2_distance(a[2], a[3], b[2], b[3]);
    displacement.segment(3, n_torso) = a.segment(n_base, n_torso) - b.segment(n_base, n_torso);
    displacement.segment(3 + n_torso, n_psi) =
        a.segment(n_base + n_torso, n_psi) - b.segment(n_base + n_torso, n_psi);

    const std::size_t pose_offset = n_base + n_torso + n_psi;  // == 12
    const auto pose_a = read_transform(a, pose_offset);
    const auto pose_b = read_transform(b, pose_offset);
    SE3Tpl<ADCG> pose_rel = pose_a.inverse() * pose_b;
    displacement.segment(n_disp_base_torso_psi, 6) = se3_displacement(pose_rel);

    ADVectorXs out(1);
    out[0] = displacement.norm();

    CppAD::ADFun<CGD> dist_func(ad_input, out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = dist_func.Forward(0, ind_vars);

    SegmentedVariableNameGenerator<double> nameGen({{"a", n_per_side, true}, {"b", n_per_side, true}});

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        1};
}

// Interpolation over the same 19-dim Configuration as
// trace_rby1_constrained_distance: plain linear interpolation for
// base_x/base_y, torso, and psi_left/psi_right (10 total), so2_slerp
// for base_rz, and slerp_se3 for the mid-pose block -- mirrors
// trace_rby1_interpolate_impl's construction. Split into _impl (shared
// between the regular and block variants) + regular/block wrappers,
// same reasoning as trace_rby1_interpolate_impl.
inline auto trace_rby1_constrained_interpolate_impl(
    const std::string &language,
    std::vector<VarSegment> input_segments,
    std::vector<VarSegment> output_segments) -> Traced
{
    const std::size_t n_base = 4;
    const std::size_t n_torso = 6;
    const std::size_t n_psi = 2;
    const std::size_t n_pose = 7;
    const std::size_t n_per_side = n_base + n_torso + n_psi + n_pose;  // == 19
    const std::size_t n_input = 2 * n_per_side + 1;  // a, b, t

    ADVectorXs ad_input(n_input);
    for (std::size_t i = 0; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    ADVectorXs a = ad_input.head(n_per_side);
    ADVectorXs b = ad_input.segment(n_per_side, n_per_side);
    ADCG t = ad_input[2 * n_per_side];

    ADVectorXs out(n_per_side);

    // base_x, base_y: plain linear interpolation.
    out.segment(0, 2) = (ADCG(1.0) - t) * a.segment(0, 2) + t * b.segment(0, 2);

    // base_rz: SO(2) slerp.
    ADCG base_rz_cos_out, base_rz_sin_out;
    so2_slerp(a[2], a[3], b[2], b[3], t, base_rz_cos_out, base_rz_sin_out);
    out[2] = base_rz_cos_out;
    out[3] = base_rz_sin_out;

    // torso: plain linear interpolation.
    out.segment(n_base, n_torso) = (ADCG(1.0) - t) * a.segment(n_base, n_torso) + t * b.segment(n_base, n_torso);

    // psi_left, psi_right: plain linear interpolation.
    out.segment(n_base + n_torso, n_psi) =
        (ADCG(1.0) - t) * a.segment(n_base + n_torso, n_psi) + t * b.segment(n_base + n_torso, n_psi);

    // mid-pose: SE3 slerp.
    const std::size_t pose_offset = n_base + n_torso + n_psi;  // == 12
    ADVectorXs pose_a = a.segment(pose_offset, n_pose);
    ADVectorXs pose_b = b.segment(pose_offset, n_pose);
    ADVectorXs pose_out(n_pose);
    slerp_se3(pose_a, pose_b, t, pose_out);
    out.segment(pose_offset, n_pose) = pose_out;

    CppAD::ADFun<CGD> interp_func(ad_input, out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = interp_func.Forward(0, ind_vars);

    SegmentedVariableNameGenerator<double> nameGen(
        std::move(input_segments), std::move(output_segments));

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        n_per_side};
}

auto trace_rby1_constrained_interpolate(const std::string &language) -> Traced
{
    const std::size_t n_per_side = 4 + 6 + 2 + 7;  // == 19
    return trace_rby1_constrained_interpolate_impl(
        language, {{"a", n_per_side, true}, {"b", n_per_side, true}, {"t", 1, false}}, {});
}

auto trace_rby1_constrained_interpolate_block(const std::string &language) -> Traced
{
    const std::size_t n_per_side = 4 + 6 + 2 + 7;  // == 19
    const std::string lang = (language == "c++") ? "c++_block" : language;
    return trace_rby1_constrained_interpolate_impl(
        lang,
        {{"a", n_per_side, true, ".broadcast(", ")"}, {"b", n_per_side, true, ".broadcast(", ")"}, {"t", 1, false}},
        {{"out", n_per_side, true}});
}
}  // namespace cricket
#pragma once

#include <cricket/codegen.hh>

#include "../tracing/internal.hh"
#include "iiwa_parameterization.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cricket
{
    using namespace CppAD;
    using namespace CppAD::cg;

template <typename T>
auto IiwaBimanualParameterizationCG(
    const std::string &language,
    bool compute_gradient = false
)
{

    std::cout << "Generating parameterized IK code for iiwa..." << std::endl;
    const size_t num_inp = 8 + 1 + 1 + 1 + 7;

    ADVectorXs ad_inp(num_inp);  // 3 4x4 matrices
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    auto ik_result = IiwaBimanualParameterization<T>(ad_inp);

    // Output layout: first n_q joint angles ("y"), followed by the 4 pre-clip
    // SafeArccos arguments ("u"). A |u[i]| > 1 means the pose/psi/GC
    // combination has no valid IK solution on this branch -- SafeArccos would
    // have silently clipped it -- so callers must reject rather than trust y.
    const size_t n_q = 14;
    const size_t n_unclipped = 4;
    const size_t n_out = n_q + n_unclipped;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_q; ++i)
    {
        data[i] = ik_result.q[i];
    }
    for (int i = 0; i < n_unclipped; ++i)
    {
        data[n_q + i] = ik_result.unclipped[i];
    }

    std::cout << "Copied to data." << std::endl;
    ADFun<CGD> iiwa_param_func(ad_inp, data);
    std::cout << "Created the AD function." << std::endl;
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);

    handler.makeVariables(ind_vars);


    CppAD::vector<CGD> result = iiwa_param_func.Forward(0, ind_vars);
    std::cout << "Ran the AD function." << std::endl;


    if (compute_gradient)
    {
      // this is the full jacobian
      CppAD::vector<CGD> jac_e_q = iiwa_param_func.Jacobian(ind_vars);
    //   CppAD::vector<CGD> jac_e_q(n_out * nq);  // this is jacobian with respect to joint configs only.

    //   for (auto i = 0U; i < n_out; i++)
    //       for (auto j = 0U; j < nq; j++)
    //           jac_e_q[i * nq + j] = jac[i * num_inp + j];

      std::move(jac_e_q.begin(), jac_e_q.end(), std::back_inserter(result));
    }

    // Codegen needs the block-oriented language for fk_template.hh's
    // FloatVector-based parameterized_ik, same remap used by the SE3 path.
    const std::string lang = (language == "c++") ? "c++_block" : language;

    SegmentedVariableNameGenerator<double> nameGen(
        {{"q", 7, true}, {"psi", 1, false}, {"smm", 3, true}, {"rel_pose", 7, true}},
        {{"y", n_q, true}, {"u", n_unclipped, true}});

    std::cout << "Generated the parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}


// CppAD-CodeGen wrapper around IiwaSE3Parameterization -- the single-arm
// counterpart of RainbowIkCG (rainbow_ik_cg.hh), used by fk_template.hh's
// ParameterizedSpace for the `param_kind == "iiwa_se3"` case. Tape input
// layout (size 11): [0:7) end-effector pose (x, y, z, qx, qy, qz, qw) in the
// robot's own base frame, [7] psi (self-motion-manifold free parameter),
// [8:11) GC2/GC4/GC6 (shoulder/elbow/wrist configuration selectors, expected
// +-1). Output: "y" (7 joint angles), "u" (4 pre-clip SafeArccos arguments --
// see IKParamResult in iiwa_parameterization.hh for what a value outside
// [-1, 1] means and why callers must check it themselves rather than trust
// `y`). `loss` (see IKParamResult) is intentionally not part of the CG
// output here -- ParameterizedSpace::resolve_block only needs the hard
// accept/reject `u` gives it, not a smooth optimization target.
template <typename T>
auto IiwaSE3ParameterizationCG(
    const std::string &language,
    bool compute_gradient = false
)
{

    std::cout << "Generating task parameterized IK code for iiwa..." << std::endl;
    const size_t num_inp = 7 + 1 + 1 + 1 + 1; // 6 for the pose, 1 for psi, 3 for GC2, GC4, GC6

    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    auto ik_result = IiwaSE3Parameterization<T>(ad_inp);

    // Output layout: first n_q joint angles ("y"), followed by the 4 pre-clip
    // SafeArccos arguments ("u"). A |u[i]| > 1 means the pose/psi/GC
    // combination has no valid IK solution on this branch -- SafeArccos would
    // have silently clipped it -- so callers must reject rather than trust y.
    const size_t n_q = 7;
    const size_t n_unclipped = 4;
    const size_t n_out = n_q + n_unclipped;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_q; ++i)
    {
        data[i] = ik_result.q[i];
    }
    for (int i = 0; i < n_unclipped; ++i)
    {
        data[n_q + i] = ik_result.unclipped[i];
    }

    ADFun<CGD> iiwa_param_func(ad_inp, data);
    std::cout << "Created the AD function." << std::endl;
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);

    handler.makeVariables(ind_vars);


    CppAD::vector<CGD> result = iiwa_param_func.Forward(0, ind_vars);
    std::cout << "Ran the AD function." << std::endl;


    if (compute_gradient)
    {
      // this is the full jacobian
      CppAD::vector<CGD> jac_e_q = iiwa_param_func.Jacobian(ind_vars);
    //   CppAD::vector<CGD> jac_e_q(n_out * nq);  // this is jacobian with respect to joint configs only.

    //   for (auto i = 0U; i < n_out; i++)
    //       for (auto j = 0U; j < nq; j++)
    //           jac_e_q[i * nq + j] = jac[i * num_inp + j];

      std::move(jac_e_q.begin(), jac_e_q.end(), std::back_inserter(result));
    }

    // `pose` already varies per-lane (it's the block itself), so it's
    // indexed normally like any other block array. GC2/GC4/GC6 (self-motion-
    // manifold selectors) are fixed for the whole planning problem, so
    // instead of being read from the input they're read off the `smm` class
    // member directly: this segment occupies the same 3 tape positions that
    // used to be GC2/GC4/GC6, but is named "smm" so the generated code emits
    // `smm[0]`, `smm[1]`, `smm[2]`. Output is split into `y[i]` (joint
    // angles, matching the `y` variable already declared in
    // parameterized_ik) and `u[i]` (pre-clip SafeArccos arguments, matching
    // the `u` variable declared there for the joint-limit-style rejection
    // check).
    const std::string lang = (language == "c++") ? "c++_block" : language;

    SegmentedVariableNameGenerator<double> nameGen(
        {{"pose", 7, true},
         {"psi", 1, false},
         {"smm", 3, true}},
        {{"y", n_q, true}, {"u", n_unclipped, true}});

    std::cout << "Generated the parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}

// Bimanual iiwa "mid-pose" parameterized IK, ParameterizedSpace's iiwa_bimanual counterpart of
// RainbowConstrainedBimanualIkCG (rainbow_ik_cg.hh): instead of IiwaBimanualParameterizationCG's
// leader-follower split above (LeaderFollowerSpace), this samples a single mid-frame pose T_mid
// (defined in the left arm's own base/DH frame -- the same frame IiwaBimanualParameterization
// treats as the reference) plus one free joint angle per arm, and derives each hand's goal pose
// as T_l = T_mid * t_mid_left, T_r = T_mid * t_mid_right, where t_mid_left/t_mid_right are fixed
// per-planning-problem SE3 offsets (fk_template.hh's ParameterizedSpace::t_mid_left/
// t_mid_right -- the same class members rby1_bimanual already uses, computed at runtime by
// compute_mid_pose from a dual-eef FK -- see trace_iiwa_bimanual_rel_pose_fk, reused for this
// too). T_l feeds IiwaSE3Parameterization directly (already in the left arm's own solver
// frame); T_r additionally needs IiwaBimanualParameterization's same fixed (0, -0.765, 0)
// base_translation to shift a left-frame pose into the right arm's own solver frame (pure
// translation, no rotation -- see that function's header for why).
//
// Orientations are composed via quaternion Hamilton product, not by extracting a quaternion
// from a composed rotation matrix -- see rainbow_ik_cg.hh's header comment on
// RainbowConstrainedBimanualIkCG for why the latter is unsafe to trace (branches on trace
// sign). Translations use the rotation matrix built directly from T_mid's own input quaternion
// (the safe, forward-only direction), same construction IiwaBimanualParameterization already
// uses for its `rel_pose` offset.
//
// Tape input layout (29): [0:7) T_mid pose (x,y,z,qx,qy,qz,qw), left arm's own base/DH frame.
// [7] psi_left, [8] psi_right. [9:16) t_mid_left (fixed, aliases
// ParameterizedSpace::t_mid_left). [16:23) t_mid_right (fixed, aliases
// ParameterizedSpace::t_mid_right). [23:26) left arm GC selectors (GC2, GC4, GC6). [26:29)
// right arm GC selectors.
//
// Output layout (22): "q" (14: left arm q(7) + right arm q(7)), "u_left" (4, pre-clip
// SafeArccos arguments -- see IKParamResult), "u_right" (4).
template <typename T>
auto IiwaBimanualMidParameterizationCG(
    const std::string &language
)
{
    std::cout << "Generating mid-pose parameterized IK code for bimanual iiwa..." << std::endl;

    const size_t num_inp = 7 + 1 + 1 + 7 + 7 + 3 + 3;  // == 29
    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    const T t_mid_x = ad_inp[0];
    const T t_mid_y = ad_inp[1];
    const T t_mid_z = ad_inp[2];
    const T t_mid_qx = ad_inp[3];
    const T t_mid_qy = ad_inp[4];
    const T t_mid_qz = ad_inp[5];
    const T t_mid_qw = ad_inp[6];

    const T psi_left = ad_inp[7];
    const T psi_right = ad_inp[8];

    const T t_mid_left_x = ad_inp[9];
    const T t_mid_left_y = ad_inp[10];
    const T t_mid_left_z = ad_inp[11];
    const T t_mid_left_qx = ad_inp[12];
    const T t_mid_left_qy = ad_inp[13];
    const T t_mid_left_qz = ad_inp[14];
    const T t_mid_left_qw = ad_inp[15];

    const T t_mid_right_x = ad_inp[16];
    const T t_mid_right_y = ad_inp[17];
    const T t_mid_right_z = ad_inp[18];
    const T t_mid_right_qx = ad_inp[19];
    const T t_mid_right_qy = ad_inp[20];
    const T t_mid_right_qz = ad_inp[21];
    const T t_mid_right_qw = ad_inp[22];

    const T left_gc2 = ad_inp[23];
    const T left_gc4 = ad_inp[24];
    const T left_gc6 = ad_inp[25];

    const T right_gc2 = ad_inp[26];
    const T right_gc4 = ad_inp[27];
    const T right_gc6 = ad_inp[28];

    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);

    auto normalize_quat = [&](const T &x, const T &y, const T &z, const T &w, T &ox, T &oy, T &oz, T &ow) {
        const T n = sqrt(x * x + y * y + z * z + w * w + static_cast<T>(1e-12));
        ox = x / n;
        oy = y / n;
        oz = z / n;
        ow = w / n;
    };

    T mqx, mqy, mqz, mqw;
    normalize_quat(t_mid_qx, t_mid_qy, t_mid_qz, t_mid_qw, mqx, mqy, mqz, mqw);
    T lqx, lqy, lqz, lqw;
    normalize_quat(t_mid_left_qx, t_mid_left_qy, t_mid_left_qz, t_mid_left_qw, lqx, lqy, lqz, lqw);
    T rqx, rqy, rqz, rqw;
    normalize_quat(t_mid_right_qx, t_mid_right_qy, t_mid_right_qz, t_mid_right_qw, rqx, rqy, rqz, rqw);

    // Rotation matrix from T_mid's own (normalized) quaternion -- forward-only direction, safe
    // to trace (same construction as IiwaBimanualParameterization's R_rel).
    Eigen::Matrix3<T> R_mid;
    R_mid << one - two * (mqy * mqy + mqz * mqz), two * (mqx * mqy - mqw * mqz),       two * (mqx * mqz + mqw * mqy),
             two * (mqx * mqy + mqw * mqz),       one - two * (mqx * mqx + mqz * mqz), two * (mqy * mqz - mqw * mqx),
             two * (mqx * mqz - mqw * mqy),       two * (mqy * mqz + mqw * mqx),       one - two * (mqx * mqx + mqy * mqy);

    // Hamilton product q_mid * q_offset -- branch-free, safe to trace (unlike extracting a
    // quaternion back out of a composed rotation matrix).
    auto quat_multiply = [&](
        const T &aw, const T &ax, const T &ay, const T &az,
        const T &bw, const T &bx, const T &by, const T &bz,
        T &ow, T &ox, T &oy, T &oz) {
        ow = aw * bw - ax * bx - ay * by - az * bz;
        ox = aw * bx + ax * bw + ay * bz - az * by;
        oy = aw * by - ax * bz + ay * bw + az * bx;
        oz = aw * bz + ax * by - ay * bx + az * bw;
    };

    Eigen::Matrix<T, 3, 1> t_mid_left_vec(t_mid_left_x, t_mid_left_y, t_mid_left_z);
    Eigen::Matrix<T, 3, 1> rotated_left = R_mid * t_mid_left_vec;
    const T left_x = t_mid_x + rotated_left(0);
    const T left_y = t_mid_y + rotated_left(1);
    const T left_z = t_mid_z + rotated_left(2);
    T left_qw, left_qx, left_qy, left_qz;
    quat_multiply(mqw, mqx, mqy, mqz, lqw, lqx, lqy, lqz, left_qw, left_qx, left_qy, left_qz);

    Eigen::Matrix<T, 3, 1> t_mid_right_vec(t_mid_right_x, t_mid_right_y, t_mid_right_z);
    Eigen::Matrix<T, 3, 1> rotated_right = R_mid * t_mid_right_vec;
    // Right hand's goal, still expressed in the left arm's own base/DH frame -- shifted into
    // the right arm's own solver frame below via the same fixed translation-only offset
    // IiwaBimanualParameterization uses.
    const Eigen::Vector3d base_translation(0, -0.765, 0);
    const T right_x = t_mid_x + rotated_right(0) + static_cast<T>(base_translation.x());
    const T right_y = t_mid_y + rotated_right(1) + static_cast<T>(base_translation.y());
    const T right_z = t_mid_z + rotated_right(2) + static_cast<T>(base_translation.z());
    T right_qw, right_qx, right_qy, right_qz;
    quat_multiply(mqw, mqx, mqy, mqz, rqw, rqx, rqy, rqz, right_qw, right_qx, right_qy, right_qz);

    Eigen::Matrix<T, 11, 1> left_arm_inp;
    left_arm_inp << left_x, left_y, left_z, left_qx, left_qy, left_qz, left_qw, psi_left, left_gc2, left_gc4,
        left_gc6;

    Eigen::Matrix<T, 11, 1> right_arm_inp;
    right_arm_inp << right_x, right_y, right_z, right_qx, right_qy, right_qz, right_qw, psi_right, right_gc2,
        right_gc4, right_gc6;

    auto left_result = IiwaSE3Parameterization<T>(left_arm_inp);
    auto right_result = IiwaSE3Parameterization<T>(right_arm_inp);

    const size_t n_arm_q = 7;
    const size_t n_unclipped = 4;
    const size_t n_out = 2 * n_arm_q + 2 * n_unclipped;  // == 22
    ADVectorXs data(n_out);
    for (size_t i = 0; i < n_arm_q; ++i)
    {
        data[i] = left_result.q[i];
    }
    for (size_t i = 0; i < n_arm_q; ++i)
    {
        data[n_arm_q + i] = right_result.q[i];
    }
    for (size_t i = 0; i < n_unclipped; ++i)
    {
        data[2 * n_arm_q + i] = left_result.unclipped[i];
    }
    for (size_t i = 0; i < n_unclipped; ++i)
    {
        data[2 * n_arm_q + n_unclipped + i] = right_result.unclipped[i];
    }

    ADFun<CGD> iiwa_mid_func(ad_inp, data);
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = iiwa_mid_func.Forward(0, ind_vars);

    const std::string lang = (language == "c++") ? "c++_block" : language;

    SegmentedVariableNameGenerator<double> nameGen(
        {{"t_mid", 7, true},
         {"psi_left", 1, false},
         {"psi_right", 1, false},
         {"t_mid_left", 7, true},
         {"t_mid_right", 7, true},
         {"left_gc", 3, true},
         {"right_gc", 3, true}},
        {{"q", 2 * n_arm_q, true}, {"u_left", n_unclipped, true}, {"u_right", n_unclipped, true}});

    std::cout << "Generated the mid-pose parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}
}  // namespace cricket

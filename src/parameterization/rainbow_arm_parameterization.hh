#pragma once

// Templated-on-T port of the branchless left-arm analytic IK in
// `rainbow_left_arm_ik_split_nobranch.cpp` (namespace
// `rby1_left_arm_split_nobranch`), written in the same style as
// `iiwa_parameterization.hh` so it can be taped with
// T = CppAD::AD<CppAD::cg::CG<double>> by `rainbow_arm_parameterization_gen.hh`
// (see that file's `RainbowLeftArmParameterizationCG`).
//
// Differences from `rainbow_left_arm_ik_split_nobranch.cpp`, all required to
// make the function a single differentiable expression instead of 8 separate
// compile-time-selected functions:
//
//   1. `IkReal` (double-only) is replaced by a template parameter `T`, and
//      every call that file made through `std::`-qualified names
//      (`std::sin`, `std::atan2`, `std::fabs`, ...) is made unqualified, so
//      that when T = CppAD::AD<CppAD::cg::CG<double>> argument-dependent
//      lookup finds CppAD's overloads instead of (or in addition to) the
//      `double` ones declared below -- exactly the convention
//      `iiwa_parameterization.hh` already uses for `cos`/`sin`/`sqrt`/
//      `atan2`/`acos`.
//   2. The file-local `CondExpLt`/`CondExpLe`/`CondExpGt`/`CondExpGe`/
//      `CondExpEq` double-only stand-ins are kept (ADL picks CppAD's real
//      `AD<Base>` overloads for T = ADCG the same way `iiwa_parameterization
//      .hh`'s `CondExpGe`/`CondExpLe` do), but `CondExpGe`/`CondExpLe`
//      themselves are *not* redeclared here -- this file includes
//      `iiwa_parameterization.hh` and reuses its copies (and its
//      `ScalarClip`/`SafeArccos`) to avoid a duplicate-definition clash if
//      both headers ever land in the same translation unit. `HingeSqPenalty`
//      is defined locally below instead (not reused from
//      `iiwa_parameterization.hh`, which doesn't currently have it) so this
//      file doesn't depend on that one carrying it.
//   3. `ElbowBranch` / `ShoulderBranch` were compile-time `int` non-type
//      template parameters in the split file (resolved outside the tape, one
//      function per GCP combination). Here they become ordinary `T` tape
//      inputs (`elbow_sel`, `shoulder_sel`, expected to hold exactly 0 or 1)
//      selected with `CondExpEq`, so a single traced function covers all 4
//      elbow/shoulder combinations -- mirroring how `IiwaSE3Parameterization`
//      takes GC2/GC4/GC6 as runtime AD inputs rather than template
//      parameters. `WristSign` collapses to a plain `T wrist_sel` (expected
//      +-1) multiplying the two places it appeared as a sign choice, since
//      unlike the elbow/shoulder branches it was already a pure sign flip
//      (no `CondExp` needed).
//
// Joint order in the output `q` (size 7): [j13, j14, j15(free), j16, j17,
// j18, j19], matching `rainbow_left_arm_ik_split_nobranch.cpp` / `sol[0..6]`
// in `src/rby1_analytic_ik.py:gcp_of_arm`.

#include "iiwa_parameterization.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

#define RIK2PI ((double)6.28318530717959)
#define RIKPI ((double)3.14159265358979)
#define RIKPI_2 ((double)1.57079632679490)

// --- double-only CondExp/misc stand-ins --------------------------------
// See file header point 2: only the three not already declared by
// `iiwa_parameterization.hh` (CondExpGe/CondExpLe) live here.
inline double CondExpLt(double left, double right, double true_val, double false_val)
{
    return (left < right) ? true_val : false_val;
}
inline double CondExpGt(double left, double right, double true_val, double false_val)
{
    return (left > right) ? true_val : false_val;
}
inline double CondExpEq(double left, double right, double true_val, double false_val)
{
    return (left == right) ? true_val : false_val;
}

// Clamp to [-1, 1] via CondExp, matching `ClampToUnit` in
// `rainbow_left_arm_ik_split.cpp`: asin(-1) == -PI/2, asin(1) == PI/2,
// acos(-1) == PI, acos(1) == 0 are exactly the saturated values the
// original `if`-guarded IKasin/IKacos returned.
template <typename T>
T RainbowClampToUnit(const T &f)
{
    T lo = CondExpLt(f, static_cast<T>(-1.0), static_cast<T>(-1.0), f);
    return CondExpGt(lo, static_cast<T>(1.0), static_cast<T>(1.0), lo);
}

template <typename T>
T RainbowAsin(const T &f) { return asin(RainbowClampToUnit(f)); }

template <typename T>
T RainbowAcos(const T &f) { return acos(RainbowClampToUnit(f)); }

// Clamp negative-under-roundoff discriminants to 0 before sqrt, matching
// `IKsqrt`'s `f <= 0.0 ? 0.0 : sqrt(f)`.
template <typename T>
T RainbowSqrt(const T &f)
{
    return sqrt(CondExpLe(f, static_cast<T>(0.0), static_cast<T>(0.0), f));
}

// Reciprocal that never literally divides by zero on either CondExp branch:
// substitute 1.0 for the denominator *before* dividing, then select between
// the real reciprocal and a sentinel (matches `SafeRecip`).
template <typename T>
T RainbowSafeRecip(const T &f)
{
    T f_safe = CondExpEq(f, static_cast<T>(0.0), static_cast<T>(1.0), f);
    T recip = static_cast<T>(1.0) / f_safe;
    return CondExpEq(f, static_cast<T>(0.0), static_cast<T>(1.0e30), recip);
}

// Wrap to (-PI, PI], branchless, matching `Wrap`.
template <typename T>
T RainbowWrap(const T &j)
{
    T hi = CondExpGt(j, static_cast<T>(RIKPI), j - static_cast<T>(RIK2PI), j);
    return CondExpLt(hi, static_cast<T>(-RIKPI), hi + static_cast<T>(RIK2PI), hi);
}

// max(f, 0), branchless, matching `Relu`.
template <typename T>
T RainbowRelu(const T &f)
{
    return CondExpGt(f, static_cast<T>(0.0), f, static_cast<T>(0.0));
}

// Smooth (C1) exterior penalty for a RainbowAsin/RainbowAcos argument
// falling outside [-clip, clip]: zero inside the band, grows quadratically
// outside, with zero value and zero derivative exactly at the boundary.
// Used to build a differentiable `loss` for optimizing the free joint angle
// against the self-motion-manifold constraints, in place of (or alongside)
// the hard reject that RainbowAsin/RainbowAcos's silent clipping otherwise
// requires. Local copy of the same helper iiwa_parameterization.hh used to
// have under this name, built from RainbowRelu above instead of re-deriving
// the CondExp pair by hand.
template <typename T, typename U>
T HingeSqPenalty(const T &val, U clip)
{
    T over = RainbowRelu(val - static_cast<T>(clip));
    T under = RainbowRelu(static_cast<T>(-clip) - val);
    return over * over + under * under;
}

// Fixed affine change of frame from the caller's end-effector frame into the
// solver's internal frame. Identical across all branches; verbatim port of
// `rainbow_left_arm_ik_split.cpp`'s `TransformToSolverFrame`
// (itself a verbatim port of `rainbow_left_arm_ik.cpp:465-477`).
template <typename T>
void RainbowTransformToSolverFrame(
    const Eigen::Matrix<T, 3, 1> &eetrans,
    const Eigen::Matrix3<T> &eerot,
    T &px, T &py, T &pz,
    T &r00, T &r01, T &r02,
    T &r10, T &r11, T &r12,
    T &r20, T &r21, T &r22)
{
    const T in_r00 = eerot(0, 0), in_r01 = eerot(0, 1), in_r02 = eerot(0, 2);
    const T in_r10 = eerot(1, 0), in_r11 = eerot(1, 1), in_r12 = eerot(1, 2);
    const T in_r20 = eerot(2, 0), in_r21 = eerot(2, 1), in_r22 = eerot(2, 2);
    const T in_px = eetrans(0), in_py = eetrans(1), in_pz = eetrans(2);

    r00 = in_r00;
    r01 = in_r01;
    r02 = in_r02;
    px = (in_px + (static_cast<T>(0.1548) * in_r02));
    r10 = ((static_cast<T>(0.342021230003561) * in_r10)) + ((static_cast<T>(-0.93969222526679) * in_r20));
    r11 = ((static_cast<T>(0.342021230003561) * in_r11)) + ((static_cast<T>(-0.93969222526679) * in_r21));
    r12 = ((static_cast<T>(0.342021230003561) * in_r12)) + ((static_cast<T>(-0.93969222526679) * in_r22));
    py = static_cast<T>(-2.72185494444282e-7) + (static_cast<T>(-0.145464356471299) * in_r22) +
         (static_cast<T>(-0.93969222526679) * in_pz) + (static_cast<T>(0.0529448864045513) * in_r12) +
         (static_cast<T>(0.342021230003561) * in_py);
    r20 = ((static_cast<T>(0.93969222526679) * in_r10)) + ((static_cast<T>(0.342021230003561) * in_r20));
    r21 = ((static_cast<T>(0.93969222526679) * in_r11)) + ((static_cast<T>(0.342021230003561) * in_r21));
    r22 = ((static_cast<T>(0.93969222526679) * in_r12)) + ((static_cast<T>(0.342021230003561) * in_r22));
    pz = static_cast<T>(-0.234119109418322) + (static_cast<T>(0.145464356471299) * in_r12) +
         (static_cast<T>(0.93969222526679) * in_py) + (static_cast<T>(0.0529448864045513) * in_r22) +
         (static_cast<T>(0.342021230003561) * in_pz);
}

// Result of RainbowLeftArmParameterization / RainbowRightArmParameterization:
// the 7 joint angles `q` (left: [j13, j14, j15, j16, j17, j18, j19]; right:
// [j22, j23, j24, j25, j26, j27, j28]), the 3 pre-clip arguments that were
// fed to RainbowAsin/RainbowAcos (in call order: elbow's asin, shoulder's
// asin, wrist's acos), `reach_violation` (a smooth, always-nonnegative
// measure of how far any of those three arguments overshot [-1, 1] -- zero
// iff `q` is an exact, unclamped IK solution on this GCP branch), and
// `loss`, a smooth penalty (see HingeSqPenalty above)
// over the same three arguments for gradient-based search over the free
// joint angle, this arm's analogue of iiwa's `psi`.
template <typename T>
struct RainbowArmParamResult
{
    Eigen::VectorX<T> q;
    Eigen::Vector3<T> unclipped;
    T reach_violation{};
    T loss{};
};

// Forward declarations so RainbowLeftArmParameterization / RainbowRight
// ArmParameterization below can call their *FromPose core (defined further
// down in this file, after each wrapper) -- ordinary (non-ADL) unqualified
// lookup for a dependent call needs to see at least a declaration at the
// point the calling template is defined, not just before it's instantiated;
// the `auto` return type itself only gets resolved once the real definition
// is parsed later in this same header, before any instantiation can occur.
template <typename T>
auto RainbowLeftArmParameterizationFromPose(
    const Eigen::Matrix<T, 3, 1> &eetrans,
    const Eigen::Matrix3<T> &eerot,
    const T &j15_free,
    const T &elbow_sel,
    const T &shoulder_sel,
    const T &wrist_sel
);

template <typename T>
auto RainbowRightArmParameterizationFromPose(
    const Eigen::Matrix<T, 3, 1> &eetrans,
    const Eigen::Matrix3<T> &eerot,
    const T &j24_free,
    const T &elbow_sel,
    const T &shoulder_sel,
    const T &wrist_sel
);

// Branch-generalized, fully differentiable analytic IK for the rainbow
// left arm, combining all 8 GCP branches of
// `rainbow_left_arm_ik_split.cpp` into one traceable function.
//
// `ad_inp` layout (size 11):
//   [0:3)  -- end-effector translation (x, y, z)
//   [3:7)  -- end-effector orientation quaternion (qx, qy, qz, qw)
//   [7]    -- j15_free: the free (self-motion) joint angle, this arm's
//             analogue of iiwa's `psi`
//   [8]    -- elbow_sel: 0 or 1, selects `ElbowBranch` (E+ / E-)
//   [9]    -- shoulder_sel: 0 or 1, selects `ShoulderBranch` (S+ / S-)
//   [10]   -- wrist_sel: +1 or -1, selects `WristSign` (W+ / W-)
//
// Split into a thin quaternion-unpacking wrapper (this function) and
// RainbowLeftArmParameterizationFromPose (the actual arithmetic, taking
// `eetrans`/`eerot` directly) so callers that already have a rotation
// *matrix* -- e.g. rainbow_ik_cg.hh, which gets one out of composing SE3
// frames via pinocchio's forwardKinematics -- don't have to round-trip it
// through a quaternion first. That round trip isn't just wasted work: going
// matrix -> quaternion (unlike the quaternion -> matrix formula used below,
// which is a single algebraic expression) is normally done via a
// trace-sign-dependent case split, which is exactly the kind of data-
// dependent branch this whole file exists to avoid inside a taped
// expression.
template <typename T, typename InputVector>
auto RainbowLeftArmParameterization(
    InputVector &ad_inp
)
{
    const T x = ad_inp[0];
    const T y = ad_inp[1];
    const T z = ad_inp[2];
    const T qx = ad_inp[3];
    const T qy = ad_inp[4];
    const T qz = ad_inp[5];
    const T qw = ad_inp[6];
    const T j15_free = ad_inp[7];
    const T elbow_sel = ad_inp[8];
    const T shoulder_sel = ad_inp[9];
    const T wrist_sel = ad_inp[10];

    // End-effector rotation matrix from the quaternion (qx, qy, qz, qw),
    // same formula as IiwaSE3Parameterization.
    Eigen::Matrix3<T> eerot;
    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);
    eerot << one - two * (qy * qy + qz * qz), two * (qx * qy - qw * qz),       two * (qx * qz + qw * qy),
             two * (qx * qy + qw * qz),       one - two * (qx * qx + qz * qz), two * (qy * qz - qw * qx),
             two * (qx * qz - qw * qy),       two * (qy * qz + qw * qx),       one - two * (qx * qx + qy * qy);
    Eigen::Matrix<T, 3, 1> eetrans(x, y, z);

    return RainbowLeftArmParameterizationFromPose<T>(eetrans, eerot, j15_free, elbow_sel, shoulder_sel, wrist_sel);
}

// Core of RainbowLeftArmParameterization, taking the end-effector pose as
// a translation + rotation matrix directly instead of a quaternion. See
// RainbowLeftArmParameterization's header comment above for why this split
// exists.
template <typename T>
auto RainbowLeftArmParameterizationFromPose(
    const Eigen::Matrix<T, 3, 1> &eetrans,
    const Eigen::Matrix3<T> &eerot,
    const T &j15_free,
    const T &elbow_sel,
    const T &shoulder_sel,
    const T &wrist_sel
)
{
    const T clip = static_cast<T>(1.0 - 1e-4);

    T px, py, pz;
    T r00, r01, r02, r10, r11, r12, r20, r21, r22;
    RainbowTransformToSolverFrame<T>(eetrans, eerot, px, py, pz, r00, r01, r02, r10, r11, r12, r20, r21, r22);

    T j15 = j15_free;
    T cj15 = cos(j15_free);
    T sj15 = sin(j15_free);
    T pp = (px * px) + (py * py) + (pz * pz);

    // --- j16 (elbow), verbatim arithmetic from
    //     rainbow_left_arm_ik.cpp:498-521. The reachability guard is
    //     dropped -- RainbowAsin saturates instead of failing -- and its
    //     pre-clip overshoot feeds reach_violation/loss below instead.
    T asin16_arg = (static_cast<T>(-1.00275505726673) + (static_cast<T>(6.98132097739206) * pp));
    T x121 = RainbowAsin(asin16_arg);
    T j16 = CondExpEq(
        elbow_sel, static_cast<T>(0.0),
        static_cast<T>(-1.80315340003661) + x121,
        static_cast<T>(1.33843925355319) - x121);
    j16 = RainbowWrap(j16);
    T cj16 = cos(j16), sj16 = sin(j16);

    // --- j13 (shoulder), verbatim arithmetic from the generic leaf at
    //     rainbow_left_arm_ik.cpp:3102-3138. `atan2`'s validity guard and
    //     the (always-false) dead-code guard are dropped; the reciprocal is
    //     RainbowSafeRecip; the asin reachability guard folds into
    //     reach_violation/loss, as for j16 above.
    T x1157 = (static_cast<T>(0.031) * cj15);
    T x1158 = atan2(-px, -py);
    T x1161 = RainbowSafeRecip(RainbowSqrt((px * px) + (py * py)));
    T asin13_arg =
        x1161 * (x1157 + ((static_cast<T>(-1.0) * cj16 * x1157)) + (static_cast<T>(-0.256) * cj15 * sj16));
    T x1159 = RainbowAsin(asin13_arg);
    T j13 = CondExpEq(
        shoulder_sel, static_cast<T>(0.0),
        (static_cast<T>(-1.0) * x1158) + (static_cast<T>(-1.0) * x1159),
        static_cast<T>(3.14159265358979) + x1159 + (static_cast<T>(-1.0) * x1158));
    j13 = RainbowWrap(j13);
    T cj13 = cos(j13), sj13 = sin(j13);

    // --- j14, verbatim arithmetic from the generic leaf at
    //     rainbow_left_arm_ik.cpp:4210-4247, including the branchless
    //     IKsign/SafeRecip-based pivot already used in
    //     `rainbow_left_arm_ik_split.cpp`'s no-branch variant.
    T x1298 = px * px;
    T x1299 = cj13 * cj13;
    T x1300 = py * py;
    T x1301 = (static_cast<T>(0.0943978594809829) * cj15);
    T x1302 = (cj13 * py);
    T x1303 = (cj15 * sj16);
    T x1304 = (px * sj13);
    T x1305 = (py * pz);
    T x1306 = (static_cast<T>(0.93969222526679) * sj15);
    T x1307 = (static_cast<T>(0.259355054173634) * cj15);
    T x1308 = (static_cast<T>(1.0) * cj15);
    T x1309 = (static_cast<T>(0.342021230003561) * sj13 * sj15);
    T x1310 = (cj13 * x1298);
    T x1311 = (px * py * sj15);
    T x1312 = (static_cast<T>(0.0875574348809117) * cj15 * cj16);
    T x1313 = (cj13 * x1300);
    T x1314 = (cj15 * x1298);
    T x1315 = (cj13 * px * pz);
    T x1316 = (static_cast<T>(0.240561209668298) * cj15 * cj16);
    T x1317 = atan2(
        ((((static_cast<T>(-1.0) * sj13 * x1306 * x1310)) +
          (static_cast<T>(-0.342021230003561) * sj15 * x1315) +
          (static_cast<T>(-0.0106026581301104) * x1303 * x1304) +
          (sj13 * x1306 * x1313) +
          (static_cast<T>(1.87938445053358) * x1299 * x1311) +
          (static_cast<T>(0.0106026581301104) * x1302 * x1303) +
          (static_cast<T>(-1.0) * pz * x1316) + (static_cast<T>(-1.0) * x1305 * x1309) +
          (x1304 * x1312) + (static_cast<T>(-1.0) * x1301 * x1302) +
          (static_cast<T>(-1.0) * x1302 * x1312) + (static_cast<T>(-1.0) * pz * x1307) +
          (x1301 * x1304) + (static_cast<T>(0.0291304589832705) * pz * x1303) +
          (static_cast<T>(-1.0) * px * py * x1306))),
        (((pz * x1312) + (static_cast<T>(0.0291304589832705) * x1302 * x1303) +
          (static_cast<T>(-0.0106026581301104) * pz * x1303) +
          (static_cast<T>(-1.0) * x1306 * x1315) + (pz * x1301) +
          (x1309 * x1310) +
          (static_cast<T>(-0.0291304589832705) * x1303 * x1304) +
          (x1304 * x1307) + (static_cast<T>(-1.0) * sj13 * x1305 * x1306) +
          (x1304 * x1316) +
          (static_cast<T>(-0.684042460007122) * x1299 * x1311) +
          (static_cast<T>(0.342021230003561) * x1311) +
          (static_cast<T>(-1.0) * x1302 * x1316) + (static_cast<T>(-1.0) * x1302 * x1307) +
          (static_cast<T>(-1.0) * x1309 * x1313))));
    T j14sign_arg =
        (static_cast<T>(-1.0) * x1308 * (pz * pz)) + (static_cast<T>(-1.0) * x1299 * x1300 * x1308) +
        (static_cast<T>(-1.0) * x1298 * x1308) + (x1299 * x1314) +
        (static_cast<T>(2.0) * cj15 * x1302 * x1304);
    // RainbowSign(f): +1 if f > 0, -1 if f < 0, 0 at f == 0 (only ever fed
    // into RainbowSafeRecip, so the f == 0 case never actually divides).
    T j14sign = CondExpGt(
        j14sign_arg, static_cast<T>(0.0), static_cast<T>(1.0),
        CondExpLt(j14sign_arg, static_cast<T>(0.0), static_cast<T>(-1.0), static_cast<T>(0.0)));
    T x1318 = RainbowSafeRecip(j14sign);
    T j14 = (static_cast<T>(-1.5707963267949) + x1317 + (static_cast<T>(1.5707963267949) * x1318));
    j14 = RainbowWrap(j14);
    T cj14 = cos(j14), sj14 = sin(j14);

    // --- wrist-relative rotation, verbatim from `rotationfunction0`
    //     (rainbow_left_arm_ik.cpp:4324-4364); no branches in the original.
    T x189 = (cj15 * sj14);
    T x190 = (cj13 * cj15);
    T x191 = (static_cast<T>(1.0) * sj14);
    T x192 = (cj15 * sj13);
    T x193 = (static_cast<T>(0.93969222526679) * sj15);
    T x194 = (static_cast<T>(0.342021230003561) * sj15);
    T x195 = (static_cast<T>(1.0) * sj13);
    T x196 = (cj13 * sj15);
    T x197 = (cj14 * cj15);
    T x198 = (cj16 * x190);
    T x199 = (sj16 * x190);
    T x200 = (sj16 * x192);
    T x201 = (static_cast<T>(-1.0) * sj13);
    T x202 = ((cj16 * x193)) + (static_cast<T>(-0.342021230003561) * sj16);
    T x203 = ((static_cast<T>(-0.93969222526679) * cj16)) + (sj16 * x194);
    T x204 = ((cj16 * x194)) + (static_cast<T>(0.93969222526679) * sj16);
    T x205 = ((static_cast<T>(-0.93969222526679) * x189)) + (static_cast<T>(0.342021230003561) * x197);
    T x206 = ((static_cast<T>(0.342021230003561) * cj16)) + (sj16 * x193);
    T x207 = (cj14 * x203);
    T x208 = (cj13 * x205);
    T x209 = ((static_cast<T>(0.93969222526679) * x197)) + (static_cast<T>(0.342021230003561) * x189);
    T x210 = (x208 + (sj15 * x201));
    T x211 = ((static_cast<T>(-1.0) * x195 * x205)) + (static_cast<T>(-1.0) * x196);
    T x212 = ((static_cast<T>(-1.0) * x191 * x202)) + (cj14 * x204);
    T x213 = ((cj14 * x202)) + (sj14 * x204);
    T x214 = ((static_cast<T>(-1.0) * x191 * x206)) + x207;
    T x215 = ((cj14 * x206)) + (sj14 * x203);
    T x216 = (x207 + (static_cast<T>(-1.0) * sj14 * x206));
    T x217 = ((cj13 * x212)) + (cj16 * x192);
    T x218 = (x198 + (x201 * x212));
    T x219 = ((cj13 * x216)) + x200;
    T x220 = (x199 + (x201 * x216));
    T new_r00 = ((r00 * ((x198 + (static_cast<T>(-1.0) * x195 * x212))))) +
                ((r10 * x217)) + ((r20 * x213));
    T new_r01 = ((r21 * x213)) + ((r11 * x217)) + ((r01 * x218));
    T new_r02 = ((r12 * x217)) + ((r02 * x218)) + ((r22 * x213));
    T new_r10 = ((r20 * x209)) + ((r00 * x211)) + ((r10 * x210));
    T new_r11 = ((r01 * (((x201 * x205)) + (static_cast<T>(-1.0) * x196)))) +
                ((r11 * ((x208 + (static_cast<T>(-1.0) * sj15 * x195))))) +
                ((r21 * x209));
    T new_r12 = ((r22 * x209)) + ((r12 * x210)) + ((r02 * x211));
    T new_r20 = ((r10 * (((cj13 * x214)) + x200))) +
                ((r00 * ((x199 + (x201 * x214))))) + ((r20 * x215));
    T new_r21 = ((r21 * x215)) + ((r01 * x220)) + ((r11 * x219));
    T new_r22 = ((r12 * x219)) + ((r22 * x215)) + ((r02 * x220));

    // --- j18 (wrist), verbatim arithmetic from
    //     rainbow_left_arm_ik.cpp:4369-4399; reachability guard dropped
    //     (RainbowAcos saturates), folded into reach_violation/loss below.
    //     `wrist_sel` (+-1) replaces the compile-time `WristSign` ternary --
    //     it was already a pure sign flip, so no CondExp is needed.
    T cj18 = new_r22;
    T j18_pos = RainbowAcos(cj18);
    T j18 = wrist_sel * j18_pos;
    // sj18 is computed further below (see the j17/j19 singularity
    // tie-break), since the branchless CondExp form needs it as a runtime
    // magnitude check rather than a compile-time sign hoist.

    // --- j17, verbatim arithmetic from the generic leaf at
    //     rainbow_left_arm_ik.cpp:10784-10802.
    T x615 = atan2(new_r12, new_r02);
    T j17_generic = (static_cast<T>(-1.5707963267949) + (static_cast<T>(1.5707963267949) * wrist_sel) + x615);

    // --- j19, verbatim arithmetic from the generic leaf at
    //     rainbow_left_arm_ik.cpp:29633 (reached via brace-depth match of
    //     the `else` of the outermost near-zero `j19eval` check -- same
    //     navigation method used for the right arm's j26/j28). This is the
    //     branch this file's other joints were extracted from; it replaces
    //     an earlier transcription (`x730*x732*x733*(-new_r01*sj18 +
    //     new_r20*sj17)` fed through atan2) that was numerically correct
    //     too (rotation-matrix orthogonality after j17/j18 are fixed makes
    //     several closed forms for j19 agree) but was the wrong branch
    //     textually. `IKsign(sj18)` hoisted to `wrist_sel`, same reasoning
    //     as j17 above; `atan2` validity guard dropped.
    T x730 = atan2(new_r21, static_cast<T>(-1.0) * new_r20);
    T j19_generic = (static_cast<T>(-1.5707963267949) + x730 + (static_cast<T>(1.5707963267949) * wrist_sel));

    // --- wrist-singularity tie-break (sj18 == 0, i.e. j18 == 0 or +-PI),
    //     ported from `rainbow_left_arm_ik_nosplit.cpp`'s j17/j19 (mirrors
    //     `rainbow_left_arm_ik_split_nobranch.cpp`'s fix). At this point
    //     new_r02/new_r12/new_r20/new_r21 all vanish and j17/j19 become
    //     individually indeterminate, so x615/x730 above degrade into
    //     atan2(~0, ~0) noise. `rainbow_left_arm_ik.cpp`'s degenerate
    //     branch (search "j17mul = -1.0" / "j17mul = 1.0" near its
    //     j18-singularity evalcond checks) breaks the tie by fixing
    //     j19 = 0 and computing j17 = atan2(-new_r01, sign(cj18)*new_r00),
    //     independent of `wrist_sel` (both roots collapse to the same
    //     physical point here). `sj18 = wrist_sel * sin(j18_pos)` since
    //     `j18 = wrist_sel * j18_pos` and `wrist_sel` is +-1.
    T sj18 = wrist_sel * sin(j18_pos);
    T j17_singular = atan2(
        static_cast<T>(-1.0) * new_r01,
        CondExpGe(cj18, static_cast<T>(0.0), new_r00, static_cast<T>(-1.0) * new_r00));
    T j19_singular = static_cast<T>(0.0);

    T j17 = CondExpLt(fabs(sj18), static_cast<T>(1.0e-6), j17_singular, j17_generic);
    j17 = RainbowWrap(j17);
    T j19 = CondExpLt(fabs(sj18), static_cast<T>(1.0e-6), j19_singular, j19_generic);
    j19 = RainbowWrap(j19);

    Eigen::VectorX<T> q(7);
    q(0) = j13;
    q(1) = j14;
    q(2) = j15;
    q(3) = j16;
    q(4) = j17;
    q(5) = j18;
    q(6) = j19;

    Eigen::Vector3<T> unclipped(asin16_arg, asin13_arg, cj18);

    // Continuous, branch-free reachability residual: sum of how far each of
    // the three domain-clamped arguments overshot [-1, 1] before clamping.
    // Zero iff all three clamps were inactive, i.e. iff q is the exact
    // (unclamped) IK solution for this GCP branch -- see file header.
    T reach_violation =
        RainbowRelu(asin16_arg - static_cast<T>(1.0)) + RainbowRelu(static_cast<T>(-1.0) - asin16_arg) +
        RainbowRelu(asin13_arg - static_cast<T>(1.0)) + RainbowRelu(static_cast<T>(-1.0) - asin13_arg) +
        RainbowRelu(cj18 - static_cast<T>(1.0)) + RainbowRelu(static_cast<T>(-1.0) - cj18);

    // Smooth penalty (see HingeSqPenalty above) over
    // the same three arguments, for gradient-based search over j15_free.
    T loss = HingeSqPenalty(asin16_arg, clip) + HingeSqPenalty(asin13_arg, clip) + HingeSqPenalty(cj18, clip);

    return RainbowArmParamResult<T>{q, unclipped, reach_violation, loss};
}

// Fixed affine change of frame from the caller's end-effector frame into
// the solver's internal frame, right-arm variant. Verbatim port of
// `rainbow_right_arm_ik_split_nobranch.cpp`'s `TransformToSolverFrame`
// (itself `rainbow_right_arm_ik.cpp:715-745`) -- note the sign flips
// throughout relative to the left arm's version (e.g. the
// `0.342021230003561` coefficient's sign, and which of `py`'s/`pz`'s
// constant terms are negated).
template <typename T>
void RainbowRightTransformToSolverFrame(
    const Eigen::Matrix<T, 3, 1> &eetrans,
    const Eigen::Matrix3<T> &eerot,
    T &px, T &py, T &pz,
    T &r00, T &r01, T &r02,
    T &r10, T &r11, T &r12,
    T &r20, T &r21, T &r22)
{
    const T in_r00 = eerot(0, 0), in_r01 = eerot(0, 1), in_r02 = eerot(0, 2);
    const T in_r10 = eerot(1, 0), in_r11 = eerot(1, 1), in_r12 = eerot(1, 2);
    const T in_r20 = eerot(2, 0), in_r21 = eerot(2, 1), in_r22 = eerot(2, 2);
    const T in_px = eetrans(0), in_py = eetrans(1), in_pz = eetrans(2);

    r00 = in_r00;
    r01 = in_r01;
    r02 = in_r02;
    px = (in_px + (static_cast<T>(0.1548) * in_r02));
    r10 = ((static_cast<T>(-0.342021230003561) * in_r10)) + ((static_cast<T>(-0.93969222526679) * in_r20));
    r11 = ((static_cast<T>(-0.342021230003561) * in_r11)) + ((static_cast<T>(-0.93969222526679) * in_r21));
    r12 = ((static_cast<T>(-0.342021230003561) * in_r12)) + ((static_cast<T>(-0.93969222526679) * in_r22));
    py = static_cast<T>(-2.72185494444282e-7) + (static_cast<T>(-0.145464356471299) * in_r22) +
         (static_cast<T>(-0.93969222526679) * in_pz) + (static_cast<T>(-0.342021230003561) * in_py) +
         (static_cast<T>(-0.0529448864045513) * in_r12);
    r20 = ((static_cast<T>(0.93969222526679) * in_r10)) + ((static_cast<T>(-0.342021230003561) * in_r20));
    r21 = ((static_cast<T>(0.93969222526679) * in_r11)) + ((static_cast<T>(-0.342021230003561) * in_r21));
    r22 = ((static_cast<T>(0.93969222526679) * in_r12)) + ((static_cast<T>(-0.342021230003561) * in_r22));
    pz = static_cast<T>(0.234119109418322) + (static_cast<T>(0.145464356471299) * in_r12) +
         (static_cast<T>(0.93969222526679) * in_py) + (static_cast<T>(-0.342021230003561) * in_pz) +
         (static_cast<T>(-0.0529448864045513) * in_r22);
}

// Branch-generalized, fully differentiable analytic IK for the rainbow
// right arm, combining all 8 GCP branches of
// `rainbow_right_arm_ik_split_nobranch.cpp` into one traceable function --
// the right-arm counterpart of RainbowLeftArmParameterization above (see
// its header for how the compile-time `ElbowBranch`/`ShoulderBranch`/
// `WristSign` template parameters became runtime `T` selectors).
//
// `ad_inp` layout (size 11), same shape as RainbowLeftArmParameterization:
//   [0:3)  -- end-effector translation (x, y, z)
//   [3:7)  -- end-effector orientation quaternion (qx, qy, qz, qw)
//   [7]    -- j24_free: the free (self-motion) joint angle, this arm's
//             analogue of iiwa's `psi`
//   [8]    -- elbow_sel: 0 or 1, selects `ElbowBranch` (E+ / E-)
//   [9]    -- shoulder_sel: 0 or 1, selects `ShoulderBranch` (S+ / S-)
//   [10]   -- wrist_sel: +1 or -1, selects `WristSign` (W+ / W-)
//
// Joint order in the output `q` (size 7): [j22, j23, j24(free), j25, j26,
// j27, j28], matching `rainbow_right_arm_ik_split_nobranch.cpp` /
// `sol[0..6]` in `src/rby1_analytic_ik.py:gcp_of_arm`.
//
// Split into a thin quaternion-unpacking wrapper (this function) and
// RainbowRightArmParameterizationFromPose (the actual arithmetic) for the
// same reason as RainbowLeftArmParameterization above.
template <typename T, typename InputVector>
auto RainbowRightArmParameterization(
    InputVector &ad_inp
)
{
    const T x = ad_inp[0];
    const T y = ad_inp[1];
    const T z = ad_inp[2];
    const T qx = ad_inp[3];
    const T qy = ad_inp[4];
    const T qz = ad_inp[5];
    const T qw = ad_inp[6];
    const T j24_free = ad_inp[7];
    const T elbow_sel = ad_inp[8];
    const T shoulder_sel = ad_inp[9];
    const T wrist_sel = ad_inp[10];

    // End-effector rotation matrix from the quaternion (qx, qy, qz, qw),
    // same formula as IiwaSE3Parameterization / RainbowLeftArmParameterization.
    Eigen::Matrix3<T> eerot;
    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);
    eerot << one - two * (qy * qy + qz * qz), two * (qx * qy - qw * qz),       two * (qx * qz + qw * qy),
             two * (qx * qy + qw * qz),       one - two * (qx * qx + qz * qz), two * (qy * qz - qw * qx),
             two * (qx * qz - qw * qy),       two * (qy * qz + qw * qx),       one - two * (qx * qx + qy * qy);
    Eigen::Matrix<T, 3, 1> eetrans(x, y, z);

    return RainbowRightArmParameterizationFromPose<T>(eetrans, eerot, j24_free, elbow_sel, shoulder_sel, wrist_sel);
}

// Core of RainbowRightArmParameterization, taking the end-effector pose as
// a translation + rotation matrix directly instead of a quaternion. See
// RainbowLeftArmParameterization's header comment for why this split
// exists.
template <typename T>
auto RainbowRightArmParameterizationFromPose(
    const Eigen::Matrix<T, 3, 1> &eetrans,
    const Eigen::Matrix3<T> &eerot,
    const T &j24_free,
    const T &elbow_sel,
    const T &shoulder_sel,
    const T &wrist_sel
)
{
    const T clip = static_cast<T>(1.0 - 1e-4);

    T px, py, pz;
    T r00, r01, r02, r10, r11, r12, r20, r21, r22;
    RainbowRightTransformToSolverFrame<T>(eetrans, eerot, px, py, pz, r00, r01, r02, r10, r11, r12, r20, r21, r22);

    T j24 = j24_free;
    T cj24 = cos(j24_free);
    T sj24 = sin(j24_free);
    T pp = (px * px) + (py * py) + (pz * pz);

    // --- j25 (elbow), verbatim arithmetic from
    //     rainbow_right_arm_ik.cpp:766-776; reachability guard dropped --
    //     RainbowAsin saturates instead, folded into reach_violation/loss
    //     below. `elbow_sel` (0/1) replaces the compile-time `ElbowBranch`
    //     ternary via CondExpEq, same treatment as j16 in the left arm.
    T asin25_arg = (static_cast<T>(-1.00275505726673) + (static_cast<T>(6.98132097739206) * pp));
    T x124 = RainbowAsin(asin25_arg);
    T j25 = CondExpEq(
        elbow_sel, static_cast<T>(0.0),
        static_cast<T>(-1.80315340003661) + x124,
        static_cast<T>(1.33843925355319) - x124);
    j25 = RainbowWrap(j25);
    T cj25 = cos(j25), sj25 = sin(j25);

    // --- j22 (shoulder), verbatim arithmetic from the generic leaf at
    //     rainbow_right_arm_ik.cpp:5434-5462. `atan2`'s validity guard,
    //     the dead-code guard, and the asin reachability guard are dropped
    //     as in the left arm's j13; reciprocal is RainbowSafeRecip;
    //     `shoulder_sel` (0/1) replaces `ShoulderBranch` via CondExpEq.
    T x1108 = (static_cast<T>(0.031) * cj24);
    T x1109 = atan2(-px, -py);
    T x1112 = RainbowSafeRecip(RainbowSqrt((px * px) + (py * py)));
    T asin22_arg =
        x1112 * ((static_cast<T>(-0.256) * cj24 * sj25) + x1108 + (static_cast<T>(-1.0) * cj25 * x1108));
    T x1110 = RainbowAsin(asin22_arg);
    T j22 = CondExpEq(
        shoulder_sel, static_cast<T>(0.0),
        (static_cast<T>(-1.0) * x1110) + (static_cast<T>(-1.0) * x1109),
        static_cast<T>(3.14159265358979) + x1110 + (static_cast<T>(-1.0) * x1109));
    j22 = RainbowWrap(j22);
    T cj22 = cos(j22), sj22 = sin(j22);

    // --- j23, verbatim arithmetic from the generic leaf at
    //     rainbow_right_arm_ik.cpp:7129-7194. Not a GCP axis: the
    //     data-dependent pivot, made branch-free via RainbowSign/
    //     RainbowSafeRecip, same treatment as j14 in the left arm.
    T x1240 = cj22 * cj22;
    T x1241 = px * px;
    T x1242 = py * py;
    T x1243 = (static_cast<T>(0.0875574348809117) * cj25);
    T x1244 = (cj22 * py);
    T x1245 = (static_cast<T>(0.259355054173634) * cj24);
    T x1246 = (static_cast<T>(0.0106026581301104) * sj25);
    T x1247 = (static_cast<T>(1.0) * cj24);
    T x1248 = (static_cast<T>(0.0943978594809829) * cj24);
    T x1249 = (px * pz);
    T x1250 = (pz * sj22);
    T x1251 = (cj24 * pz);
    T x1252 = (static_cast<T>(0.240561209668298) * cj25);
    T x1253 = (cj24 * px * sj22);
    T x1254 = (static_cast<T>(0.342021230003561) * cj22 * sj24);
    T x1255 = (sj22 * x1241);
    T x1256 = (static_cast<T>(0.0291304589832705) * cj24 * sj25);
    T x1257 = (static_cast<T>(0.93969222526679) * cj22 * sj24);
    T x1258 = (sj22 * x1242);
    T x1259 = (static_cast<T>(0.93969222526679) * py * sj24);
    T x1260 = (px * py * sj24);
    T x1261_signarg =
        ((cj24 * x1240 * x1241) + (static_cast<T>(2.0) * x1244 * x1253) +
         (static_cast<T>(-1.0) * x1240 * x1242 * x1247) + (static_cast<T>(-1.0) * x1247 * (pz * pz)) +
         (static_cast<T>(-1.0) * x1241 * x1247));
    T x1261_sign = CondExpGt(
        x1261_signarg, static_cast<T>(0.0), static_cast<T>(1.0),
        CondExpLt(x1261_signarg, static_cast<T>(0.0), static_cast<T>(-1.0), static_cast<T>(0.0)));
    T x1261 = RainbowSafeRecip(x1261_sign);
    T x1262 = atan2(
        (((static_cast<T>(-1.0) * px * sj22 * x1248) + (x1246 * x1253) +
          (static_cast<T>(0.342021230003561) * py * sj24 * x1250) +
          (static_cast<T>(-1.0) * pz * x1245) + (static_cast<T>(-1.0) * x1255 * x1257) +
          (static_cast<T>(-1.0) * px * x1259) + (static_cast<T>(-1.0) * cj24 * x1244 * x1246) +
          (x1244 * x1248) + (static_cast<T>(-1.0) * x1251 * x1252) +
          (x1257 * x1258) + (static_cast<T>(1.87938445053358) * x1240 * x1260) +
          (x1249 * x1254) + (static_cast<T>(-1.0) * x1243 * x1253) +
          (static_cast<T>(0.0291304589832705) * sj25 * x1251) +
          (cj24 * x1243 * x1244))),
        (((static_cast<T>(-1.0) * x1249 * x1257) + (x1246 * x1251) +
          (static_cast<T>(-1.0) * cj24 * x1244 * x1252) + (static_cast<T>(-1.0) * x1244 * x1245) +
          (static_cast<T>(0.684042460007122) * x1240 * x1260) +
          (static_cast<T>(-1.0) * pz * x1248) + (x1252 * x1253) +
          (static_cast<T>(-1.0) * x1254 * x1255) + (x1254 * x1258) +
          (static_cast<T>(-1.0) * x1250 * x1259) +
          (static_cast<T>(-0.342021230003561) * x1260) +
          (static_cast<T>(-0.0291304589832705) * sj25 * x1253) +
          (px * sj22 * x1245) + (static_cast<T>(-1.0) * x1243 * x1251) +
          (x1244 * x1256))));
    T j23 = (static_cast<T>(-1.5707963267949) + (static_cast<T>(1.5707963267949) * x1261) + x1262);
    j23 = RainbowWrap(j23);
    T cj23 = cos(j23), sj23 = sin(j23);

    // --- wrist-relative rotation, verbatim from `rotationfunction0`
    //     (rainbow_right_arm_ik.cpp:7341-7388); no branches in the original.
    T x191 = (static_cast<T>(0.342021230003561) * sj25);
    T x192 = (cj22 * cj24);
    T x193 = (static_cast<T>(1.0) * sj22);
    T x194 = (static_cast<T>(0.93969222526679) * cj24);
    T x195 = (cj22 * sj24);
    T x196 = (static_cast<T>(0.93969222526679) * cj25);
    T x197 = (static_cast<T>(0.342021230003561) * cj25);
    T x198 = (cj24 * sj22);
    T x199 = (static_cast<T>(0.93969222526679) * sj25);
    T x200 = (static_cast<T>(1.0) * sj23);
    T x201 = (static_cast<T>(0.342021230003561) * cj24);
    T x202 = (cj25 * x192);
    T x203 = (cj25 * x198);
    T x204 = (sj25 * x192);
    T x205 = (static_cast<T>(-1.0) * sj22);
    T x206 = (x199 + (static_cast<T>(-1.0) * sj24 * x197));
    T x207 = ((sj24 * x199)) + (static_cast<T>(-1.0) * x197);
    T x208 = ((sj24 * x196)) + x191;
    T x209 = (cj23 * x206);
    T x210 = ((cj23 * x194)) + (static_cast<T>(-1.0) * sj23 * x201);
    T x211 = (static_cast<T>(-1.0) * x196) + (static_cast<T>(-1.0) * sj24 * x191);
    T x212 = (static_cast<T>(-1.0) * sj23 * x194) + (static_cast<T>(-1.0) * cj23 * x201);
    T x213 = (cj22 * x212);
    T x214 = (sj22 * x212);
    T x215 = (static_cast<T>(-1.0) * sj24 * x193) + x213;
    T x216 = ((sj23 * x206)) + ((cj23 * x208));
    T x217 = (static_cast<T>(-1.0) * x200 * x208) + x209;
    T x218 = (static_cast<T>(-1.0) * x200 * x207) + ((cj23 * x211));
    T x219 = ((cj23 * x207)) + ((sj23 * x211));
    T x220 = (static_cast<T>(-1.0) * sj23 * x208) + x209;
    T x221 = (static_cast<T>(-1.0) * x195) + (x205 * x212);
    T x222 = (x203 + (cj22 * x217));
    T x223 = (x204 + (static_cast<T>(-1.0) * x193 * x218));
    T x224 = ((sj25 * x198)) + ((cj22 * x218));
    T x225 = ((x205 * x217)) + x202;
    T new_r00 = ((r10 * ((cj22 * x220) + x203))) +
                ((r00 * ((x205 * x220) + x202))) + ((r20 * x216));
    T new_r01 = ((r21 * x216)) + ((r01 * x225)) + ((r11 * x222));
    T new_r02 = ((r12 * x222)) + ((r22 * x216)) + ((r02 * x225));
    T new_r10 = ((r00 * x221)) + ((r10 * x215)) + ((r20 * x210));
    T new_r11 = ((r21 * x210)) +
                ((r01 * ((static_cast<T>(-1.0) * x195) + (static_cast<T>(-1.0) * x193 * x212)))) +
                ((r11 * (x213 + (sj24 * x205))));
    T new_r12 = ((r12 * x215)) + ((r22 * x210)) + ((r02 * x221));
    T new_r20 = ((r00 * x223)) + ((r10 * x224)) + ((r20 * x219));
    T new_r21 = ((r21 * x219)) + ((r01 * x223)) + ((r11 * x224));
    T new_r22 = ((r12 * x224)) + ((r22 * x219)) +
                ((r02 * ((x205 * x218) + x204)));

    // --- j27 (wrist), verbatim arithmetic from
    //     rainbow_right_arm_ik.cpp:7393-7401; reachability guard dropped
    //     (RainbowAcos saturates), folded into reach_violation/loss below.
    //     `wrist_sel` (+-1) replaces `WristSign` -- pure sign flip, same as
    //     j18 in the left arm.
    T cj27 = new_r22;
    T j27_pos = RainbowAcos(cj27);
    T j27 = wrist_sel * j27_pos;
    // sj27 is computed further below (see the j26/j28 singularity
    // tie-break), since the branchless CondExp form needs it as a runtime
    // magnitude check rather than a compile-time sign hoist.

    // --- j26, verbatim arithmetic from the generic leaf at
    //     rainbow_right_arm_ik.cpp:22952-22963. `IKsign(sj27)` hoisted to
    //     `wrist_sel`, same reasoning as j17 in the left arm.
    T x596 = atan2(new_r12, new_r02);
    T j26_generic = (static_cast<T>(-1.5707963267949) + x596 + (static_cast<T>(1.5707963267949) * wrist_sel));

    // --- j28, verbatim arithmetic from the generic leaf at
    //     rainbow_right_arm_ik.cpp:29565-29577 -- the *simple* form (same
    //     shape as j26 above); see rainbow_right_arm_ik_split_nobranch.cpp's
    //     header for why this differs from the left arm's j19 formula.
    //     `IKsign(sj27)` hoisted to `wrist_sel` as for j26.
    T x732 = atan2(new_r21, static_cast<T>(-1.0) * new_r20);
    T j28_generic = (static_cast<T>(-1.5707963267949) + x732 + (static_cast<T>(1.5707963267949) * wrist_sel));

    // --- wrist-singularity tie-break (sj27 == 0, i.e. j27 == 0 or +-PI),
    //     ported from `rainbow_left_arm_ik_nosplit.cpp`'s j26/j28
    //     (mirroring `rainbow_left_arm_ik_split_nobranch.cpp`'s j17/j19
    //     fix). At this point new_r02/new_r12/new_r20/new_r21 all vanish
    //     and j26/j28 become individually indeterminate, so x596/x732 above
    //     degrade into atan2(~0, ~0) noise. `rainbow_right_arm_ik.cpp`'s
    //     degenerate branch (search "j26mul = -1.0" / "j26mul = 1.0" near
    //     its j27-singularity evalcond checks) breaks the tie by fixing
    //     j28 = 0 and computing j26 = atan2(-new_r01, sign(cj27)*new_r00),
    //     independent of `wrist_sel` (both roots collapse to the same
    //     physical point here). `sj27 = wrist_sel * sin(j27_pos)` since
    //     `j27 = wrist_sel * j27_pos` and `wrist_sel` is +-1.
    T sj27 = wrist_sel * sin(j27_pos);
    T j26_singular = atan2(
        static_cast<T>(-1.0) * new_r01,
        CondExpGe(cj27, static_cast<T>(0.0), new_r00, static_cast<T>(-1.0) * new_r00));
    T j28_singular = static_cast<T>(0.0);

    T j26 = CondExpLt(fabs(sj27), static_cast<T>(1.0e-6), j26_singular, j26_generic);
    j26 = RainbowWrap(j26);
    T j28 = CondExpLt(fabs(sj27), static_cast<T>(1.0e-6), j28_singular, j28_generic);
    j28 = RainbowWrap(j28);

    Eigen::VectorX<T> q(7);
    q(0) = j22;
    q(1) = j23;
    q(2) = j24;
    q(3) = j25;
    q(4) = j26;
    q(5) = j27;
    q(6) = j28;

    Eigen::Vector3<T> unclipped(asin25_arg, asin22_arg, cj27);

    // Continuous, branch-free reachability residual: sum of how far each of
    // the three domain-clamped arguments (j25's asin, j22's asin, j27's
    // acos) overshot [-1, 1] before clamping -- see
    // RainbowLeftArmParameterization's reach_violation above.
    T reach_violation =
        RainbowRelu(asin25_arg - static_cast<T>(1.0)) + RainbowRelu(static_cast<T>(-1.0) - asin25_arg) +
        RainbowRelu(asin22_arg - static_cast<T>(1.0)) + RainbowRelu(static_cast<T>(-1.0) - asin22_arg) +
        RainbowRelu(cj27 - static_cast<T>(1.0)) + RainbowRelu(static_cast<T>(-1.0) - cj27);

    // Smooth penalty over the same three arguments, for gradient-based
    // search over j24_free.
    T loss = HingeSqPenalty(asin25_arg, clip) + HingeSqPenalty(asin22_arg, clip) + HingeSqPenalty(cj27, clip);

    return RainbowArmParamResult<T>{q, unclipped, reach_violation, loss};
}

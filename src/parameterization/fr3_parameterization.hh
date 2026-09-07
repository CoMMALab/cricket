// panda_ik_split_nobranch.cpp
//
// Branchless variant of `panda_ik_split.cpp`: same 4 (case6 x case1)
// entry points, same straight-line generic-region trace lifted from
// `franka_IK_EE_CC` (panda_ik.hh:222-441), but every *runtime*
// `if`/`else`/early-`return false` remaining in that file (domain
// clamps on the three law-of-cosines/asin trig calls, the seven
// joint-limit range checks, the q6 2*PI wraparound, the wrist-point
// singularity formula switch, and the Case1_1 sign-dependent PI
// pivot) has been rewritten as a `CondExp` expression, following the
// exact pattern `../rby1_ik/rainbow_left_arm_ik_split_nobranch.cpp`
// uses for RBY1 (see that file's header for the full CppAD-migration
// rationale this one reuses without repeating).
//
// The two compile-time branch selectors (`Case6_0`, `Case1_1`) are
// deliberately *not* touched here: exactly as in the RBY1 nobranch
// file, they stay plain `?:` on the template's non-type `bool`
// parameters, resolved entirely at compile time (one function per
// branch, never appearing as data-dependent control flow on the
// tape) -- see `panda_ik_split.cpp`'s header for why there are 4 of
// these branches, not 8.
//
// What THIS file additionally removes, beyond what
// `panda_ik_split.cpp` already resolved at compile time:
//
//   1. Domain clamps on the trig-inverse calls whose arguments can
//      leave [-1, 1] when the target is out of reach for this branch
//      (`theta246`'s and `theta462`'s law-of-cosines `acos`, `Theta6`'s
//      `asin`, and the wrist-elbow `acos` for q2 in the non-singular
//      case1 formula). As in the RBY1 file, `IKasin`/`IKacos` clamp
//      their argument to [-1, 1] instead of the original's explicit
//      `if (...) return q_NAN;` guards -- the triangle-inequality
//      check ahead of `theta246` (panda_ik_split.cpp:133) is subsumed
//      by its own acos clamp, since a valid triangle existing for
//      sides (L24, L46, L26) is exactly the condition under which
//      *all three* of that triangle's law-of-cosines ratios lie in
//      [-1, 1] simultaneously; `theta462` therefore gets the same
//      belt-and-suspenders clamp+residual treatment even though the
//      original never checked it explicitly. The overshoot of each
//      pre-clamp argument is accumulated into `reach_violation` (see
//      below) instead of aborting.
//   2. The seven joint-limit range checks (`q_min[i]`/`q_max[i]` for
//      i = 0..6) that `franka_IK_EE_CC` enforces as early `return
//      q_NAN`s. Unlike RBY1's ikfast-generated solver -- which only
//      guards raw kinematic-domain validity and knows nothing about
//      the physical joint limits -- Panda's closed form bakes the
//      actual Franka joint limits into its control flow, so there are
//      seven range checks here with no RBY1-file analogue. Each
//      becomes two `Relu`-based one-sided residual terms (`q < q_min`
//      and `q > q_max`) folded into the same `reach_violation`
//      accumulator, so one scalar still answers "did this branch's
//      output satisfy every guard" for a caller building a CppAD
//      tape.
//   3. The data-dependent `q6` 2*PI wraparound (`if (q6 <= q_min) q6
//      += 2*PI; else if (q6 >= q_max) q6 -= 2*PI;`,
//      panda_ik_split.cpp:162-165) -- a genuine runtime branch on the
//      *value* of q6, not a compile-time branch selector -- rewritten
//      via nested `CondExp` (`WrapToRange` below), mirroring `Wrap()`
//      in the RBY1 file except into Panda's asymmetric
//      [q_min[5], q_max[5]] window instead of (-PI, PI].
//   4. The wrist-point singularity fallback (`|V2P.z| / L2P > 0.999`,
//      panda_ik_split.cpp:181): both the singular (`q1 = q_actual_0,
//      q2 = 0`) and non-singular (`q1 = atan2(...), q2 = acos(...)`)
//      formulas are now always evaluated, and `CondExp` selects
//      between the two complete results -- the same "bucket 2, but on
//      a whole alternate formula rather than a scalar pivot" pattern
//      the RBY1 file uses for its own wrist-singularity j17/j19
//      tie-break.
//   5. The Case1_1 sign-dependent PI pivot nested inside the
//      non-singular formula (`if (q1 < 0) q1 += PI; else q1 -= PI;`,
//      panda_ik_split.cpp:190-193) -- data-dependent on the *sign of a
//      computed value*, not on Case1_1 itself (Case1_1 only chooses
//      *whether* the pivot applies at all, which stays a compile-time
//      `?:`) -- via `CondExpLt`.
//   6. Every vector normalization (`x_6 /= x_6.norm()`, the `Y_6`/
//      `Z_6`/`y_3` unit vectors) and every division by a
//      data-dependent (not fixed-robot-constant) quantity
//      (`theta462`'s `1/L26`, `LP6`'s `1/sin(thetaP)`, the wrist-pole
//      ratio's `1/L2P`) now goes through `SafeRecip`, matching the
//      RBY1 file's SS3.2 rationale: none of these denominators is
//      expected to be zero for a valid pose, but a literal `1/0`
//      computed unconditionally -- as all of these are; they sit in
//      the single straight-line trace every branch shares, not on an
//      untaken `CondExp` arm -- would poison the entire result with
//      `inf`/`NaN` even in ordinary `double` evaluation, and
//      definitely poisons an AD tape's gradient. `SafeRecip`'s
//      arbitrary sentinel reciprocal at an exact zero is *not* folded
//      into `reach_violation`: like the RBY1 file's coordinate
//      singularities, these are measure-zero configurations orthogonal
//      to the continuous reachability question `reach_violation`
//      answers. Fixed robot-constant denominators (e.g. `2*L24*L46`)
//      are left as plain division, exactly as the RBY1 file leaves
//      its own fixed-constant divisions alone.
//
// Consequently this file's functions always return `true` -- exactly
// as in the RBY1 nobranch file, there is no remaining runtime
// condition that can make them fail. Whether the returned `q_out` is
// the *actual, in-limits* IK solution for this branch must now be
// judged from `reach_violation`:
//
//   `reach_violation == 0`  => every domain clamp was inactive and
//                              every joint stayed in [q_min, q_max];
//                              `q_out` is the exact solution for this
//                              branch.
//   `reach_violation  > 0`  => at least one clamp saturated or one
//                              joint limit was exceeded; `q_out` is
//                              the nearest representable point on
//                              this branch's formula, not a valid
//                              solution, and the residual grows
//                              continuously with how far outside the
//                              branch's domain the target was.
//
// Joint ordering matches `panda_ik_split.cpp` / `panda_ik.hh`:
// q_out[0..6] = [q1, q2, q3, q4, q5, q6, q7], with q7 the free
// parameter supplied by the caller (never solved for).
//
// This is a mechanical CondExp port only, same scope as the RBY1
// nobranch file: no attempt is made here to retemplate on
// `CppAD::AD<double>` or on Eigen's AD scalar support -- `IkReal` is
// `double` and the local `CondExp*`/`SafeRecip`/etc. family below is
// a double-only stand-in with CppAD's exact signature, ready to be
// deleted in favor of `#include <cppad/cppad.hpp>` once `IkReal`
// (and the `Eigen::Matrix<IkReal, ...>` aliases below) are
// retemplated.

#include <array>
#include <cmath>
#include "Eigen/Dense"

namespace panda_ik_split_nobranch {

typedef double IkReal;

// --- double-only CondExp family --------------------------------------
// Identical stand-in to `rainbow_left_arm_ik_split_nobranch.cpp`'s;
// delete this block (and switch `IkReal` to `CppAD::AD<double>`) to
// run this same file through CppAD -- every call site below already
// uses exactly CppAD's own `CondExpXx(left, right, trueCase,
// falseCase)` signature.
inline IkReal CondExpLt(IkReal left, IkReal right, IkReal trueCase, IkReal falseCase) {
  return (left < right) ? trueCase : falseCase;
}
inline IkReal CondExpLe(IkReal left, IkReal right, IkReal trueCase, IkReal falseCase) {
  return (left <= right) ? trueCase : falseCase;
}
inline IkReal CondExpGt(IkReal left, IkReal right, IkReal trueCase, IkReal falseCase) {
  return (left > right) ? trueCase : falseCase;
}
inline IkReal CondExpGe(IkReal left, IkReal right, IkReal trueCase, IkReal falseCase) {
  return (left >= right) ? trueCase : falseCase;
}
inline IkReal CondExpEq(IkReal left, IkReal right, IkReal trueCase, IkReal falseCase) {
  return (left == right) ? trueCase : falseCase;
}

inline IkReal IKabs(IkReal f) { return std::fabs(f); }

/// Clamp to [-1, 1] via CondExp (RBY1 file, plan SS3.1 bucket 1).
/// `IKasin`/`IKacos` composed with this reproduce the original's
/// guarded asin/acos exactly: saturating to the boundary value
/// instead of failing.
inline IkReal ClampToUnit(IkReal f) {
  return CondExpLt(f, IkReal(-1.0), IkReal(-1.0),
         CondExpGt(f, IkReal(1.0), IkReal(1.0), f));
}
inline IkReal IKasin(IkReal f) { return std::asin(ClampToUnit(f)); }
inline IkReal IKacos(IkReal f) { return std::acos(ClampToUnit(f)); }

/// Clamp negative-under-roundoff arguments to 0 before sqrt (plan
/// SS3.1 bucket 1); every sqrt argument below is a sum of squares so
/// this only ever guards roundoff, never a real domain violation.
inline IkReal IKsqrt(IkReal f) {
  return std::sqrt(CondExpLe(f, IkReal(0.0), IkReal(0.0), f));
}

/// Reciprocal that never literally divides by zero (file header point
/// 6, plan SS3.2): substitute 1.0 for the denominator *before*
/// dividing, then select between the real reciprocal and an arbitrary
/// large sentinel.
inline IkReal SafeRecip(IkReal f) {
  IkReal f_safe = CondExpEq(f, IkReal(0.0), IkReal(1.0), f);
  IkReal recip = IkReal(1.0) / f_safe;
  return CondExpEq(f, IkReal(0.0), IkReal(1.0e30), recip);
}

/// max(f, 0), branchless. Used to turn each domain-clamp's raw
/// pre-clamp argument, and each joint's raw over/under-shoot past its
/// limit, into a one-sided, always-nonnegative `reach_violation`
/// contribution.
inline IkReal Relu(IkReal f) { return CondExpGt(f, IkReal(0.0), f, IkReal(0.0)); }

/// Wrap `v` into [lo, hi] by adding/subtracting one `period`,
/// branchless; replaces `if (v <= lo) v += period; else if (v >= hi)
/// v -= period;` (panda_ik.hh:358-361). Only ever called with a
/// `period` large enough that a single wrap suffices, exactly as in
/// the original -- the nesting order (add-branch checked first)
/// matches the original's if/else-if so the two guards stay mutually
/// exclusive.
inline IkReal WrapToRange(IkReal v, IkReal lo, IkReal hi, IkReal period) {
  return CondExpLe(v, lo, v + period,
         CondExpGe(v, hi, v - period, v));
}

typedef Eigen::Matrix<IkReal, 3, 1> Vec3;
typedef Eigen::Matrix<IkReal, 3, 3> Mat3;

inline IkReal Norm3(const Vec3& v) {
  return IKsqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/// `v` scaled by the SafeRecip of its own norm (file header point 6):
/// every `.normalized()` / `v /= v.norm()` in panda_ik.hh becomes
/// this.
inline Vec3 SafeNormalize(const Vec3& v) {
  return v * SafeRecip(Norm3(v));
}

// -----------------------------------------------------------------------
// The 4 branches. Naming/semantics identical to `panda_ik_split.cpp`
// (see that file's header for the case6/case1 derivation and why
// there are 4, not 8, branches here).
// -----------------------------------------------------------------------

template <bool Case6_0, bool Case1_1>
inline bool SolveArmBranch(const std::array<IkReal, 16>& O_T_EE_array,
                            IkReal q7, IkReal q_actual_0,
                            std::array<IkReal, 7>& q_out,
                            IkReal& reach_violation) {
  Eigen::Map<const Eigen::Matrix<IkReal, 4, 4>> O_T_EE(O_T_EE_array.data());

  const IkReal d1 = 0.3330;
  const IkReal d3 = 0.3160;
  const IkReal d5 = 0.3840;
  const IkReal d7e = 0.2104;
  const IkReal a4 = 0.0825;
  const IkReal a7 = 0.0880;

  const IkReal LL24 = 0.10666225;     // a4^2 + d3^2
  const IkReal LL46 = 0.15426225;     // a4^2 + d5^2
  const IkReal L24 = 0.326591870689;  // sqrt(LL24)
  const IkReal L46 = 0.392762332715;  // sqrt(LL46)

  const IkReal thetaH46 = 1.35916951803;   // atan(d5/a4)
  const IkReal theta342 = 1.31542071191;   // atan(d3/a4)
  const IkReal theta46H = 0.211626808766;  // acot(d5/a4)

  const std::array<IkReal, 7> q_min = {
      {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
  const std::array<IkReal, 7> q_max = {
      {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};

  // free parameter -- domain guard dropped (file header point 2); q7
  // is always passed through and its own violation folded into
  // reach_violation below.
  q_out[6] = q7;

  // compute p_6 -- verbatim, no branches in the original either.
  Mat3 R_EE = O_T_EE.topLeftCorner<3, 3>();
  Vec3 z_EE = O_T_EE.block<3, 1>(0, 2);
  Vec3 p_EE = O_T_EE.block<3, 1>(0, 3);
  Vec3 p_7 = p_EE - d7e * z_EE;

  Vec3 x_EE_6;
  x_EE_6 << std::cos(q7 - M_PI_4), -std::sin(q7 - M_PI_4), IkReal(0.0);
  Vec3 x_6 = SafeNormalize(Vec3(R_EE * x_EE_6));  // file header point 6
  Vec3 p_6 = p_7 - a7 * x_6;

  // compute q4 (single deterministic value, no branch on it at all --
  // triangle-inequality guard dropped, file header point 1: its
  // violation is exactly `theta246_arg` leaving [-1, 1], captured in
  // reach_violation below).
  Vec3 p_2;
  p_2 << IkReal(0.0), IkReal(0.0), d1;
  Vec3 V26 = p_6 - p_2;

  IkReal LL26 = V26[0] * V26[0] + V26[1] * V26[1] + V26[2] * V26[2];
  IkReal L26 = IKsqrt(LL26);

  IkReal theta246_arg = (LL24 + LL46 - LL26) / (2.0 * L24 * L46);  // fixed nonzero denom
  IkReal theta246 = IKacos(theta246_arg);
  q_out[3] = theta246 + thetaH46 + theta342 - 2.0 * M_PI;

  // compute q6 (case6 branch selected at compile time)
  IkReal theta462_arg = (LL26 + LL46 - LL24) * SafeRecip(2.0 * L26 * L46);
  IkReal theta462 = IKacos(theta462_arg);
  IkReal theta26H = theta46H + theta462;
  IkReal D26 = -L26 * std::cos(theta26H);

  Vec3 Z_6_raw = z_EE.cross(x_6);
  Vec3 Y_6_raw = Z_6_raw.cross(x_6);
  Mat3 R_6;
  R_6.col(0) = x_6;
  R_6.col(1) = SafeNormalize(Y_6_raw);
  R_6.col(2) = SafeNormalize(Z_6_raw);
  Vec3 V_6_62 = R_6.transpose() * (-V26);

  IkReal Phi6 = std::atan2(V_6_62[1], V_6_62[0]);
  IkReal theta6_denom = IKsqrt(V_6_62[0] * V_6_62[0] + V_6_62[1] * V_6_62[1]);
  IkReal theta6_arg = D26 * SafeRecip(theta6_denom);
  IkReal Theta6 = IKasin(theta6_arg);

  IkReal q6_raw = Case6_0 ? (M_PI - Theta6 - Phi6) : (Theta6 - Phi6);
  q_out[5] = WrapToRange(q6_raw, q_min[5], q_max[5], 2.0 * M_PI);

  // compute q1 & q2 (case1 branch selected at compile time)
  IkReal thetaP26 = 3.0 * M_PI_2 - theta462 - theta246 - theta342;
  IkReal thetaP = M_PI - thetaP26 - theta26H;
  IkReal LP6 = L26 * std::sin(thetaP26) * SafeRecip(std::sin(thetaP));

  Vec3 z_6_5;
  z_6_5 << std::sin(q_out[5]), std::cos(q_out[5]), IkReal(0.0);
  Vec3 z_5 = R_6 * z_6_5;
  Vec3 V2P = p_6 - LP6 * z_5 - p_2;

  IkReal L2P = Norm3(V2P);
  IkReal pole_arg = V2P[2] * SafeRecip(L2P);  // == V2P[2] / L2P

  // singular fallback formula (wrist point aligned with shoulder axis)
  IkReal q0_singular = q_actual_0;
  IkReal q1_singular = IkReal(0.0);

  // non-singular formula, Case1_1-adjusted at compile time; the sign
  // pivot inside the adjustment is data-dependent (file header point
  // 5), so it alone needs a real CondExp regardless of which case1
  // branch this is.
  IkReal q0_case0 = std::atan2(V2P[1], V2P[0]);
  IkReal q1_case0 = IKacos(pole_arg);
  IkReal q0_case1 =
      CondExpLt(q0_case0, IkReal(0.0), q0_case0 + M_PI, q0_case0 - M_PI);
  IkReal q1_case1 = -q1_case0;
  IkReal q0_nonsingular = Case1_1 ? q0_case1 : q0_case0;
  IkReal q1_nonsingular = Case1_1 ? q1_case1 : q1_case0;

  // singularity select (file header point 4): data-dependent, so a
  // real CondExp, chosen the same way regardless of Case1_1 (the
  // fallback ignores which case1 branch this is, exactly as in
  // franka_IK_EE / franka_IK_EE_CC).
  IkReal pole_indicator = IKabs(pole_arg);  // == fabs(V2P[2] / L2P)
  q_out[0] = CondExpGt(pole_indicator, IkReal(0.999), q0_singular, q0_nonsingular);
  q_out[1] = CondExpGt(pole_indicator, IkReal(0.999), q1_singular, q1_nonsingular);

  // compute q3 (single deterministic value given q1, q2) -- uses the
  // same V2P regardless of which q0/q1 formula was selected above,
  // exactly as the original (z_3/y_3/x_3 never depended on the
  // singularity branch, only on V2P itself).
  Vec3 z_3 = SafeNormalize(V2P);
  Vec3 Y_3 = -V26.cross(V2P);
  Vec3 y_3 = SafeNormalize(Y_3);
  Vec3 x_3 = y_3.cross(z_3);

  IkReal c1 = std::cos(q_out[0]);
  IkReal s1 = std::sin(q_out[0]);
  Mat3 R_1;
  R_1 << c1, -s1, IkReal(0.0), s1, c1, IkReal(0.0), IkReal(0.0), IkReal(0.0),
      IkReal(1.0);
  IkReal c2 = std::cos(q_out[1]);
  IkReal s2 = std::sin(q_out[1]);
  Mat3 R_1_2;
  R_1_2 << c2, -s2, IkReal(0.0), IkReal(0.0), IkReal(0.0), IkReal(1.0), -s2,
      -c2, IkReal(0.0);
  Mat3 R_2 = R_1 * R_1_2;
  Vec3 x_2_3 = R_2.transpose() * x_3;
  q_out[2] = std::atan2(x_2_3[2], x_2_3[0]);

  // compute q5 (single deterministic value given the rest)
  Vec3 VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5;
  IkReal c6 = std::cos(q_out[5]);
  IkReal s6 = std::sin(q_out[5]);
  Mat3 R_5_6;
  R_5_6 << c6, -s6, IkReal(0.0), IkReal(0.0), IkReal(0.0), IkReal(-1.0), s6,
      c6, IkReal(0.0);
  Mat3 R_5 = R_6 * R_5_6.transpose();
  Vec3 V_5_H4 = R_5.transpose() * VH4;

  q_out[4] = -std::atan2(V_5_H4[1], V_5_H4[0]);

  // Continuous, branch-free reachability + joint-limit residual (file
  // header points 1-2): sum of how far each domain-clamped argument
  // overshot [-1, 1] before clamping, plus how far each of the 7
  // joints landed outside its physical range. Zero iff q_out is the
  // exact, in-limits solution for this branch. The `for` below has a
  // fixed, IkReal-independent trip count (7): it is ordinary loop
  // unrolling, not data-dependent control flow, so it introduces no
  // branch on the tape.
  reach_violation =
      Relu(theta246_arg - IkReal(1.0)) + Relu(IkReal(-1.0) - theta246_arg) +
      Relu(theta462_arg - IkReal(1.0)) + Relu(IkReal(-1.0) - theta462_arg) +
      Relu(theta6_arg - IkReal(1.0)) + Relu(IkReal(-1.0) - theta6_arg) +
      Relu(pole_arg - IkReal(1.0)) + Relu(IkReal(-1.0) - pole_arg);
  for (int i = 0; i < 7; ++i) {
    reach_violation += Relu(q_min[i] - q_out[i]) + Relu(q_out[i] - q_max[i]);
  }

  return true;
}

// --- The 4 concrete branch entry points --------------------------------
//
// `reach_violation` (see file header) is always written; `>0` means
// the returned `q_out` is a clamped and/or out-of-limits, non-exact
// solution for this branch.

bool SolveArm_Case6_0_Case1_0(const std::array<IkReal, 16>& O_T_EE_array,
                               IkReal q7, IkReal q_actual_0,
                               std::array<IkReal, 7>& q_out,
                               IkReal& reach_violation) {
  return SolveArmBranch<true, false>(O_T_EE_array, q7, q_actual_0, q_out,
                                      reach_violation);
}

bool SolveArm_Case6_0_Case1_1(const std::array<IkReal, 16>& O_T_EE_array,
                               IkReal q7, IkReal q_actual_0,
                               std::array<IkReal, 7>& q_out,
                               IkReal& reach_violation) {
  return SolveArmBranch<true, true>(O_T_EE_array, q7, q_actual_0, q_out,
                                     reach_violation);
}

bool SolveArm_Case6_1_Case1_0(const std::array<IkReal, 16>& O_T_EE_array,
                               IkReal q7, IkReal q_actual_0,
                               std::array<IkReal, 7>& q_out,
                               IkReal& reach_violation) {
  return SolveArmBranch<false, false>(O_T_EE_array, q7, q_actual_0, q_out,
                                       reach_violation);
}

bool SolveArm_Case6_1_Case1_1(const std::array<IkReal, 16>& O_T_EE_array,
                               IkReal q7, IkReal q_actual_0,
                               std::array<IkReal, 7>& q_out,
                               IkReal& reach_violation) {
  return SolveArmBranch<false, true>(O_T_EE_array, q7, q_actual_0, q_out,
                                      reach_violation);
}

// ---------------------------------------------------------------------------
// Templated, branch-generalized variant for CppAD tracing -- the FR3/Panda
// counterpart of IiwaSE3Parameterization (iiwa_parameterization.hh). This is
// exactly SolveArmBranch<Case6_0, Case1_1> above, mechanically retemplated on
// `T` per the file header's closing note ("ready to be deleted in favor of
// ... once IkReal ... [is] retemplated"), with its two compile-time bool
// selectors folded into two continuous tape inputs (`case6_sel`,
// `case1_sel`, each expected 0 or 1) chosen via CondExp -- exactly the
// pattern the file already uses for every genuinely *data*-dependent branch
// (the wrist-point singularity fallback, the Case1_1 sign pivot). Unlike
// IiwaSE3Parameterization's GC2/GC4/GC6 (independent sign flips folded via
// plain multiplication), Case6_0/Case1_1 each pick between two entire
// alternate formulas rather than a sign, so they need a real CondExp select
// -- the same shape as this file's own singularity-fallback select just
// below.
//
// The `template <typename T> ... (const T &)` overloads of
// ClampToUnit/IKasin/IKacos/IKsqrt/SafeRecip/Relu/WrapToRange/IKabs below
// sit alongside (never replace) the double-only versions above: a `double`
// call site still binds to the non-template exact match, while `T =
// CppAD::AD<CGD>` (or any other AD scalar) only has the template to match,
// resolving CondExpLt/CondExpGt/CondExpEq/CondExpGe/CondExpLe and the
// unqualified sqrt/sin/cos/atan2/asin/acos/fabs calls below via ADL into
// CppAD's own overloads at instantiation time -- the same trick
// iiwa_parameterization.hh's ScalarClip/SafeArccos already rely on.
//
// `case6_sel`/`case1_sel` are expected in {0, 1} (0 selects Case6_0/
// Case1_0's formula, 1 selects Case6_1/Case1_1's) -- remap a {-1, +1}
// convention (as IiwaSE3Parameterization's GC2/GC4/GC6 use) via
// `0.5 * (gc + 1.0)` before calling this.
//
// `q_actual_0` is accepted but deliberately UNUSED: FR3's closed form only
// has two real branch axes (case6_sel, case1_sel), one fewer than iiwa_se3's
// GC2/GC4/GC6, but fk_template.hh's ParameterizedSpace wants one uniformly
// 3-wide `smm` selector triple across every `param_kind` (set_smm/
// set_smm_lanes, single array-of-3 API) -- so this parameter exists purely
// to keep FR3's tape/smm layout 3-wide like iiwa_se3's, occupying the slot
// that would otherwise disambiguate the (measure-zero) wrist-point-singular
// case (`|V2P.z / L2P| > 0.999`); that case's q1 is instead pinned to a
// fixed 0.0 below rather than reading this parameter. Pass any value here --
// it has no effect on `q_out`.
//
// `reach_violation` means exactly what the file header says: `== 0` iff
// every domain clamp was inactive and every joint stayed within
// [q_min, q_max] (`q_out` is the exact, in-limits solution for this
// branch/pose/q7); `> 0` means `q_out` is not a valid solution and grows
// continuously with how far outside the branch's domain the target was --
// callers must reject on `reach_violation > 0` rather than trusting
// `q_out`, exactly as IKParamResult::unclipped must be checked rather than
// trusted.
template <typename T>
inline T ClampToUnit(const T &f)
{
  return CondExpLt(f, T(-1.0), T(-1.0), CondExpGt(f, T(1.0), T(1.0), f));
}
template <typename T>
inline T IKasin(const T &f) { return asin(ClampToUnit(f)); }
template <typename T>
inline T IKacos(const T &f) { return acos(ClampToUnit(f)); }
template <typename T>
inline T IKsqrt(const T &f) { return sqrt(CondExpLe(f, T(0.0), T(0.0), f)); }
template <typename T>
inline T SafeRecip(const T &f)
{
  T f_safe = CondExpEq(f, T(0.0), T(1.0), f);
  T recip = T(1.0) / f_safe;
  return CondExpEq(f, T(0.0), T(1.0e30), recip);
}
template <typename T>
inline T Relu(const T &f) { return CondExpGt(f, T(0.0), f, T(0.0)); }
template <typename T>
inline T WrapToRange(const T &v, const T &lo, const T &hi, const T &period)
{
  return CondExpLe(v, lo, v + period, CondExpGe(v, hi, v - period, v));
}
template <typename T>
inline T IKabs(const T &f) { return fabs(f); }

template <typename T>
inline bool SolveArmBranchTaped(
    const std::array<T, 16> &O_T_EE_array,
    const T &q7,
    const T &case6_sel,
    const T &case1_sel,
    const T & /* q_actual_0 -- deliberately unused, see header comment above */,
    std::array<T, 7> &q_out,
    T &reach_violation)
{
  using Vec3T = Eigen::Matrix<T, 3, 1>;
  using Mat3T = Eigen::Matrix<T, 3, 3>;
  Eigen::Map<const Eigen::Matrix<T, 4, 4>> O_T_EE(O_T_EE_array.data());

  const T d1 = T(0.3330);
  const T d3 = T(0.3160);
  const T d5 = T(0.3840);
  const T d7e = T(0.2104);
  const T a4 = T(0.0825);
  const T a7 = T(0.0880);

  const T LL24 = T(0.10666225);     // a4^2 + d3^2
  const T LL46 = T(0.15426225);     // a4^2 + d5^2
  const T L24 = T(0.326591870689);  // sqrt(LL24)
  const T L46 = T(0.392762332715);  // sqrt(LL46)

  const T thetaH46 = T(1.35916951803);   // atan(d5/a4)
  const T theta342 = T(1.31542071191);   // atan(d3/a4)
  const T theta46H = T(0.211626808766);  // acot(d5/a4)

  const std::array<T, 7> q_min = {
      {T(-2.8973), T(-1.7628), T(-2.8973), T(-3.0718), T(-2.8973), T(-0.0175), T(-2.8973)}};
  const std::array<T, 7> q_max = {
      {T(2.8973), T(1.7628), T(2.8973), T(-0.0698), T(2.8973), T(3.7525), T(2.8973)}};

  // free parameter -- q7 is FR3's analogue of iiwa's psi, always passed
  // through; its own violation folds into reach_violation below like every
  // other joint.
  q_out[6] = q7;

  // compute p_6 -- verbatim, no branches in the original either.
  Mat3T R_EE = O_T_EE.template topLeftCorner<3, 3>();
  Vec3T z_EE = O_T_EE.template block<3, 1>(0, 2);
  Vec3T p_EE = O_T_EE.template block<3, 1>(0, 3);
  Vec3T p_7 = p_EE - d7e * z_EE;

  Vec3T x_EE_6;
  x_EE_6 << cos(q7 - T(M_PI_4)), -sin(q7 - T(M_PI_4)), T(0.0);
  Vec3T x_6_raw = R_EE * x_EE_6;
  Vec3T x_6 = x_6_raw * SafeRecip(IKsqrt(x_6_raw[0] * x_6_raw[0] + x_6_raw[1] * x_6_raw[1] + x_6_raw[2] * x_6_raw[2]));
  Vec3T p_6 = p_7 - a7 * x_6;

  // compute q4 (single deterministic value, no branch on it at all --
  // triangle-inequality guard dropped, file header point 1: its violation
  // is exactly `theta246_arg` leaving [-1, 1], captured in
  // reach_violation below).
  Vec3T p_2;
  p_2 << T(0.0), T(0.0), d1;
  Vec3T V26 = p_6 - p_2;

  T LL26 = V26[0] * V26[0] + V26[1] * V26[1] + V26[2] * V26[2];
  T L26 = IKsqrt(LL26);

  T theta246_arg = (LL24 + LL46 - LL26) / (T(2.0) * L24 * L46);  // fixed nonzero denom
  T theta246 = IKacos(theta246_arg);
  q_out[3] = theta246 + thetaH46 + theta342 - T(2.0 * M_PI);

  // compute q6 (Case6_0 above, folded into a data-dependent CondExp select)
  T theta462_arg = (LL26 + LL46 - LL24) * SafeRecip(T(2.0) * L26 * L46);
  T theta462 = IKacos(theta462_arg);
  T theta26H = theta46H + theta462;
  T D26 = -L26 * cos(theta26H);

  Vec3T Z_6_raw = z_EE.cross(x_6);
  Vec3T Y_6_raw = Z_6_raw.cross(x_6);
  Mat3T R_6;
  R_6.col(0) = x_6;
  R_6.col(1) = Y_6_raw * SafeRecip(IKsqrt(Y_6_raw[0] * Y_6_raw[0] + Y_6_raw[1] * Y_6_raw[1] + Y_6_raw[2] * Y_6_raw[2]));
  R_6.col(2) = Z_6_raw * SafeRecip(IKsqrt(Z_6_raw[0] * Z_6_raw[0] + Z_6_raw[1] * Z_6_raw[1] + Z_6_raw[2] * Z_6_raw[2]));
  Vec3T V_6_62 = R_6.transpose() * (-V26);

  T Phi6 = atan2(V_6_62[1], V_6_62[0]);
  T theta6_denom = IKsqrt(V_6_62[0] * V_6_62[0] + V_6_62[1] * V_6_62[1]);
  T theta6_arg = D26 * SafeRecip(theta6_denom);
  T Theta6 = IKasin(theta6_arg);

  T q6_case0 = T(M_PI) - Theta6 - Phi6;  // Case6_0
  T q6_case1 = Theta6 - Phi6;            // Case6_1
  T q6_raw = CondExpGt(case6_sel, T(0.5), q6_case1, q6_case0);
  q_out[5] = WrapToRange(q6_raw, q_min[5], q_max[5], T(2.0 * M_PI));

  // compute q1 & q2 (Case1_1 above, folded into a data-dependent CondExp
  // select)
  T thetaP26 = T(3.0 * M_PI_2) - theta462 - theta246 - theta342;
  T thetaP = T(M_PI) - thetaP26 - theta26H;
  T LP6 = L26 * sin(thetaP26) * SafeRecip(sin(thetaP));

  Vec3T z_6_5;
  z_6_5 << sin(q_out[5]), cos(q_out[5]), T(0.0);
  Vec3T z_5 = R_6 * z_6_5;
  Vec3T V2P = p_6 - LP6 * z_5 - p_2;

  T L2P = IKsqrt(V2P[0] * V2P[0] + V2P[1] * V2P[1] + V2P[2] * V2P[2]);
  T pole_arg = V2P[2] * SafeRecip(L2P);  // == V2P[2] / L2P

  // singular fallback formula (wrist point aligned with shoulder axis).
  // The original franka_IK_EE_CC substitutes the arm's actual current
  // joint-1 angle here; that would be a third genuinely independent input
  // this closed form needs, which we deliberately don't carry (see this
  // function's header) -- pin to a fixed 0.0 instead. This only matters in
  // the measure-zero pole_indicator > 0.999 case, and any fixed value works
  // equally well as a "no better information available" fallback.
  T q0_singular = T(0.0);
  T q1_singular = T(0.0);

  // non-singular formula; the sign pivot inside the Case1_1 adjustment is
  // data-dependent regardless of which case1 branch this is, so it was
  // already a real CondExp in the original file.
  T q0_case0 = atan2(V2P[1], V2P[0]);
  T q1_case0 = IKacos(pole_arg);
  T q0_case1 = CondExpLt(q0_case0, T(0.0), q0_case0 + T(M_PI), q0_case0 - T(M_PI));
  T q1_case1 = -q1_case0;
  T q0_nonsingular = CondExpGt(case1_sel, T(0.5), q0_case1, q0_case0);
  T q1_nonsingular = CondExpGt(case1_sel, T(0.5), q1_case1, q1_case0);

  // singularity select: data-dependent, chosen the same way regardless of
  // case1_sel (the fallback ignores which case1 branch this is, exactly as
  // in franka_IK_EE / franka_IK_EE_CC).
  T pole_indicator = IKabs(pole_arg);  // == fabs(V2P[2] / L2P)
  q_out[0] = CondExpGt(pole_indicator, T(0.999), q0_singular, q0_nonsingular);
  q_out[1] = CondExpGt(pole_indicator, T(0.999), q1_singular, q1_nonsingular);

  // compute q3 (single deterministic value given q1, q2) -- uses the same
  // V2P regardless of which q0/q1 formula was selected above, exactly as
  // the original.
  Vec3T z_3 = V2P * SafeRecip(L2P);
  Vec3T Y_3 = -V26.cross(V2P);
  Vec3T y_3 = Y_3 * SafeRecip(IKsqrt(Y_3[0] * Y_3[0] + Y_3[1] * Y_3[1] + Y_3[2] * Y_3[2]));
  Vec3T x_3 = y_3.cross(z_3);

  T c1 = cos(q_out[0]);
  T s1 = sin(q_out[0]);
  Mat3T R_1;
  R_1 << c1, -s1, T(0.0), s1, c1, T(0.0), T(0.0), T(0.0), T(1.0);
  T c2 = cos(q_out[1]);
  T s2 = sin(q_out[1]);
  Mat3T R_1_2;
  R_1_2 << c2, -s2, T(0.0), T(0.0), T(0.0), T(1.0), -s2, -c2, T(0.0);
  Mat3T R_2 = R_1 * R_1_2;
  Vec3T x_2_3 = R_2.transpose() * x_3;
  q_out[2] = atan2(x_2_3[2], x_2_3[0]);

  // compute q5 (single deterministic value given the rest)
  Vec3T VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5;
  T c6 = cos(q_out[5]);
  T s6 = sin(q_out[5]);
  Mat3T R_5_6;
  R_5_6 << c6, -s6, T(0.0), T(0.0), T(0.0), T(-1.0), s6, c6, T(0.0);
  Mat3T R_5 = R_6 * R_5_6.transpose();
  Vec3T V_5_H4 = R_5.transpose() * VH4;

  q_out[4] = -atan2(V_5_H4[1], V_5_H4[0]);

  // Continuous, branch-free reachability + joint-limit residual (file
  // header points 1-2): sum of how far each domain-clamped argument
  // overshot [-1, 1] before clamping, plus how far each of the 7 joints
  // landed outside its physical range. Zero iff q_out is the exact,
  // in-limits solution for this branch. The `for` below has a fixed,
  // T-independent trip count (7): ordinary loop unrolling, not
  // data-dependent control flow, so it introduces no branch on the tape.
  reach_violation =
      Relu(theta246_arg - T(1.0)) + Relu(T(-1.0) - theta246_arg) +
      Relu(theta462_arg - T(1.0)) + Relu(T(-1.0) - theta462_arg) +
      Relu(theta6_arg - T(1.0)) + Relu(T(-1.0) - theta6_arg) +
      Relu(pole_arg - T(1.0)) + Relu(T(-1.0) - pole_arg);
  for (int i = 0; i < 7; ++i)
  {
    reach_violation = reach_violation + Relu(q_min[i] - q_out[i]) + Relu(q_out[i] - q_max[i]);
  }

  return true;
}

}  // namespace panda_ik_split_nobranch

// Result of Fr3SE3Parameterization: joint configuration `q` (7) and a single
// continuous `reach_violation` (panda_ik_split_nobranch::SolveArmBranchTaped's
// output, folding all of that file's domain-clamp + joint-limit residuals
// into one scalar) -- the FR3/Panda counterpart of IKParamResult
// (iiwa_parameterization.hh), which instead exposes 4 raw pre-clip
// SafeArccos arguments (this arm's closed form has no analogous per-joint
// arccos to expose individually -- see SolveArmBranchTaped's header).
// `reach_violation == 0` means `q` is the exact, in-limits IK solution for
// this (pose, q7, case6_sel, case1_sel) branch; `> 0` means it is not, and
// callers must check this field themselves rather than trust `q`.
template <typename T>
struct Fr3IKParamResult
{
    Eigen::VectorX<T> q;
    T reach_violation{};
};

// CppAD-traceable single-arm SE3+psi task-space IK for the FR3/Panda arm --
// the FR3 counterpart of IiwaSE3Parameterization (iiwa_parameterization.hh).
// `ad_inp` layout (size 11): [0:7) end-effector pose (x, y, z, qx, qy, qz,
// qw) in the robot's own base frame, [7] `psi` (named to match
// IiwaSE3Parameterization's tape layout; here it IS joint 7's angle
// directly rather than a shoulder-elbow-wrist decomposition parameter --
// unpacked below into a local `q7` since that's what it actually is), [8]
// case6_sel, [9] case1_sel (SolveArmBranchTaped's two real branch
// selectors, each expected in {0, 1}), [10] unused -- kept only so this
// arm's tape/smm layout stays 3-wide like IiwaSE3Parameterization's
// GC2/GC4/GC6, even though FR3's closed form only has two independent
// branch axes (see SolveArmBranchTaped's header).
template <typename T, typename InputVector>
auto Fr3SE3Parameterization(InputVector &ad_inp)
{
    const T x = ad_inp[0];
    const T y = ad_inp[1];
    const T z = ad_inp[2];
    const T qx = ad_inp[3];
    const T qy = ad_inp[4];
    const T qz = ad_inp[5];
    const T qw = ad_inp[6];
    const T q7 = ad_inp[7];
    const T case6_sel = ad_inp[8];
    const T case1_sel = ad_inp[9];
    const T unused_smm2 = ad_inp[10];  // see this function's header comment

    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);

    // Same quaternion -> rotation-matrix formula as IiwaSE3Parameterization.
    Eigen::Matrix3<T> R;
    R << one - two * (qy * qy + qz * qz), two * (qx * qy - qw * qz),       two * (qx * qz + qw * qy),
         two * (qx * qy + qw * qz),       one - two * (qx * qx + qz * qz), two * (qy * qz - qw * qx),
         two * (qx * qz - qw * qy),       two * (qy * qz + qw * qx),       one - two * (qx * qx + qy * qy);

    // Column-major 4x4, matching Eigen::Map<const Eigen::Matrix<T, 4, 4>>'s
    // default (column-major) storage order as read by SolveArmBranchTaped.
    std::array<T, 16> O_T_EE_array{};
    O_T_EE_array[0] = R(0, 0); O_T_EE_array[1] = R(1, 0); O_T_EE_array[2] = R(2, 0); O_T_EE_array[3] = T(0.0);
    O_T_EE_array[4] = R(0, 1); O_T_EE_array[5] = R(1, 1); O_T_EE_array[6] = R(2, 1); O_T_EE_array[7] = T(0.0);
    O_T_EE_array[8] = R(0, 2); O_T_EE_array[9] = R(1, 2); O_T_EE_array[10] = R(2, 2); O_T_EE_array[11] = T(0.0);
    O_T_EE_array[12] = x; O_T_EE_array[13] = y; O_T_EE_array[14] = z; O_T_EE_array[15] = T(1.0);

    std::array<T, 7> q_out{};
    T reach_violation{};
    panda_ik_split_nobranch::SolveArmBranchTaped<T>(
        O_T_EE_array, q7, case6_sel, case1_sel, unused_smm2, q_out, reach_violation);

    Eigen::VectorX<T> q(7);
    for (int i = 0; i < 7; ++i)
    {
        q[i] = q_out[i];
    }

    return Fr3IKParamResult<T>{q, reach_violation};
}

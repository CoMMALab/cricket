#pragma once

#include <vector>

#include "housekeeping.hh"
#include "robot_info.hh"
#include "tracer_utils.hh"
#include "lang_name_gen.hh"
#include "internal.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>




template <typename Scalar>
auto map_bounded(Scalar u, double lower, double upper) -> Scalar
{
    return Scalar(lower) + u * Scalar(upper - lower);
}

template <typename Scalar>
void map_unbounded_revolute(Scalar u, Scalar &cos_out, Scalar &sin_out)
{
    constexpr double two_pi = 2.0 * M_PI;
    Scalar theta = u * Scalar(two_pi);
    cos_out = cos(theta);
    sin_out = sin(theta);
}

template <typename Scalar>
void map_so3_shoemake(Scalar u1, Scalar u2, Scalar u3, Scalar &x, Scalar &y, Scalar &z, Scalar &w)
{
    constexpr double two_pi = 2.0 * M_PI;

    Scalar sqrt1_minus_u1 = sqrt(Scalar(1.0) - u1);
    Scalar sqrt_u1 = sqrt(u1);
    Scalar theta1 = u2 * Scalar(two_pi);
    Scalar theta2 = u3 * Scalar(two_pi);

    x = sqrt1_minus_u1 * sin(theta1);
    y = sqrt1_minus_u1 * cos(theta1);
    z = sqrt_u1 * sin(theta2);
    w = sqrt_u1 * cos(theta2);
}


namespace se3_detail
{
    constexpr double EPS = 1e-7;
}

template <typename Scalar>
inline auto sinc_smooth(Scalar x) -> Scalar
{
    const Scalar x2 = x * x;
    const Scalar taylor = 1.0 - x2 / 6.0 * (1.0 - x2 / 20.0);

    // Guard the divisor itself, not just the blend weight: at x == 0 this is a
    // literal 0/0 -> NaN, and since both branches of a CondExp-free blend are
    // always evaluated, `w * NaN` is still NaN (not 0) and poisons the result
    // even though w == 0 there.
    const Scalar safe_x = CondExpLt(x2, Scalar(se3_detail::EPS), Scalar(1.0), x);
    const Scalar full = sin(safe_x) / safe_x;
    const Scalar w = x2 / (x2 + se3_detail::EPS);

    return (1.0 - w) * taylor + w * full;
}

// Recovers axis * angle from R when theta is near pi, where sin(theta) ~ 0
// makes `vee` (and thus the generic/taylor branches below) collapse to ~0
// and lose the axis direction entirely. Uses the identity
// R_ii = cos(theta) + (1 - cos(theta)) n_i^2 and, for i != j,
// R_ij + R_ji = 2 (1 - cos(theta)) n_i n_j, both of which are well-conditioned
// near theta = pi (unlike the vee-based extraction).
template <typename Matrix, typename Scalar>
inline auto so3_log_pi_axis(const Matrix &R, Scalar cos_theta, Scalar theta) -> Matrix
{
    const Scalar one_minus_c = Scalar(1.0) - cos_theta;
    const Scalar safe_one_minus_c = CondExpLt(one_minus_c, Scalar(se3_detail::EPS), Scalar(1.0), one_minus_c);

    Scalar n2_0 = (R(0, 0) - cos_theta) / safe_one_minus_c;
    Scalar n2_1 = (R(1, 1) - cos_theta) / safe_one_minus_c;
    Scalar n2_2 = (R(2, 2) - cos_theta) / safe_one_minus_c;
    // Clamp tiny negative numerical noise.
    n2_0 = CondExpLt(n2_0, Scalar(0.0), Scalar(0.0), n2_0);
    n2_1 = CondExpLt(n2_1, Scalar(0.0), Scalar(0.0), n2_1);
    n2_2 = CondExpLt(n2_2, Scalar(0.0), Scalar(0.0), n2_2);

    // Pick the largest-magnitude axis component as pivot so the divisions
    // below never see a near-zero denominator.
    const Scalar pivot_is_0 = CondExpGe(n2_0, n2_1, CondExpGe(n2_0, n2_2, Scalar(1.0), Scalar(0.0)), Scalar(0.0));
    const Scalar pivot_is_1 = CondExpGe(n2_1, n2_0, CondExpGe(n2_1, n2_2, Scalar(1.0), Scalar(0.0)), Scalar(0.0));
    const Scalar pivot_is_2 = Scalar(1.0) - pivot_is_0 - pivot_is_1;

    const Scalar n2_pivot = pivot_is_0 * n2_0 + pivot_is_1 * n2_1 + pivot_is_2 * n2_2;
    const Scalar n_pivot = sqrt(n2_pivot);
    const Scalar safe_n_pivot = CondExpLt(n_pivot, Scalar(se3_detail::EPS), Scalar(1.0), n_pivot);

    const Scalar sym_01 = (R(0, 1) + R(1, 0)) / (Scalar(2.0) * safe_one_minus_c);
    const Scalar sym_02 = (R(0, 2) + R(2, 0)) / (Scalar(2.0) * safe_one_minus_c);
    const Scalar sym_12 = (R(1, 2) + R(2, 1)) / (Scalar(2.0) * safe_one_minus_c);

    const Scalar n_0 = pivot_is_0 * sqrt(n2_0)
                      + pivot_is_1 * (sym_01 / safe_n_pivot)
                      + pivot_is_2 * (sym_02 / safe_n_pivot);
    const Scalar n_1 = pivot_is_1 * sqrt(n2_1)
                      + pivot_is_0 * (sym_01 / safe_n_pivot)
                      + pivot_is_2 * (sym_12 / safe_n_pivot);
    const Scalar n_2 = pivot_is_2 * sqrt(n2_2)
                      + pivot_is_0 * (sym_02 / safe_n_pivot)
                      + pivot_is_1 * (sym_12 / safe_n_pivot);

    Matrix pi_axis = Matrix::Zero(3, 1);
    pi_axis(0, 0) = theta * n_0;
    pi_axis(1, 0) = theta * n_1;
    pi_axis(2, 0) = theta * n_2;
    return pi_axis;
}

template <typename Matrix, typename Scalar>
inline auto so3_log_smooth(const Matrix &R) -> Matrix
{
    // Clamp for acos domain safety (trace can drift slightly outside [-1, 1]
    // due to floating point).
    const Scalar cos_theta_raw = (R.trace() - 1) / 2;
    const Scalar cos_theta = CondExpGt(cos_theta_raw, Scalar(1.0), Scalar(1.0),
        CondExpLt(cos_theta_raw, Scalar(-1.0), Scalar(-1.0), cos_theta_raw));
    const Scalar theta = acos(cos_theta);

    Matrix vee = Matrix::Zero(3, 1);
    vee(0, 0) = R(2, 1) - R(1, 2);
    vee(1, 0) = R(0, 2) - R(2, 0);
    vee(2, 0) = R(1, 0) - R(0, 1);

    // Near theta == 0: log(R) ~ 0.5 * vee.
    const auto taylor = 0.5 * vee;

    // Generic branch: factor = theta / (2 sin(theta)). sinc_smooth is itself
    // NaN-safe at theta == 0, so this never produces a literal 0/0.
    const Scalar factor = 0.5 / (sinc_smooth(theta) + se3_detail::EPS);
    const auto generic = vee * factor;

    // Near theta == pi: vee collapses to ~0 and loses the axis direction, so
    // recover axis * angle from the symmetric part of R instead.
    const auto pi_axis = so3_log_pi_axis<Matrix, Scalar>(R, cos_theta, theta);

    // Two independent singularities live at opposite ends of cos_theta, so
    // blend each in with its own weight rather than reusing a single
    // "near singular" indicator for both (that was the source of the theta ==
    // pi bug: it picked the theta == 0 fallback, which is wrong at pi).
    const Scalar d0 = 1.0 - cos_theta;    // -> 0 only as theta -> 0
    const Scalar d_pi = 1.0 + cos_theta;  // -> 0 only as theta -> pi
    const Scalar w0 = se3_detail::EPS / (d0 + se3_detail::EPS);
    const Scalar w_pi = se3_detail::EPS / (d_pi + se3_detail::EPS);
    const Scalar w_generic = 1.0 - w0 - w_pi;

    return w0 * taylor + w_pi * pi_axis + w_generic * generic;
}

auto trace_map_to_se3(
    const pinocchio::Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds) -> Traced
{

    ADVectorXs ad_u(7);
    ADVectorXs ad_se3(8);

    for (auto i = 0U; i < 7; ++i)
    {
        ad_u[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_u);

    for (int i = 0; i < 3; ++i)
    {
        ad_se3[i] =
            map_bounded(ad_u[i], bounds->lower[i], bounds->upper[i]);
    }
    ADCG x, y, z, w;
    map_so3_shoemake(
        ad_u[3], ad_u[4], ad_u[5], x, y, z, w);
    ad_se3[3] = x;
    ad_se3[4] = y;
    ad_se3[5] = z;
    ad_se3[6] = w;
    ad_se3[7] = map_bounded(ad_u[6], 0.0, 2 * M_PI); // psi is bounded between 0 and 2*pi

    CppAD::ADFun<CGD> map_func(ad_u, ad_se3);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(7);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = map_func.Forward(0, ind_vars);

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

    std::cout << "Sampled." << std::endl;
    return Traced{function_code.str(), handler.getTemporaryVariableCount(), result.size()};    
}

template <typename Scalar>
void slerp_se3(
    const ADVectorXs &a,
    const ADVectorXs &b,
    Scalar t,
    ADVectorXs &out)

{
    // Interpolate translation linearly
    out.template head<3>() = (Scalar(1.0) - t) * a.template head<3>() + t * b.template head<3>();

    // Interpolate rotation using a branchless SLERP. Eigen::Quaternion::slerp
    // uses `if` statements on the dot product (shortest-path sign flip, and
    // the near-parallel fallback to lerp) which requires the operand to be a
    // compile-time parameter under CppADCodeGen. Here the operands are tape
    // variables, so those branches are replaced with CondExp selects, which
    // CodeGen can trace and emit as data-dependent (but branchless) code.
    Eigen::Quaternion<Scalar> qa(a[6], a[3], a[4], a[5]); // w, x, y, z
    Eigen::Quaternion<Scalar> qb(b[6], b[3], b[4], b[5]); // w, x, y, z

    Scalar d = qa.dot(qb);

    // Flip qb so we always interpolate along the shorter arc.
    Scalar sign = CondExpLt(d, Scalar(0.0), Scalar(-1.0), Scalar(1.0));
    Eigen::Vector4<Scalar> qb_coeffs = qb.coeffs() * sign;
    Scalar d_abs = d * sign;

    // Clamp for acos domain safety.
    Scalar d_clamped = CondExpGt(d_abs, Scalar(1.0), Scalar(1.0), d_abs);

    Scalar theta = acos(d_clamped);
    Scalar sin_theta = sin(theta);

    // Guard against sin_theta ~ 0 (near-parallel quaternions): fall back to
    // linear scales there instead of dividing by (near) zero.
    Scalar eps = Scalar(1e-6);
    Scalar safe_sin_theta = CondExpLt(sin_theta, eps, Scalar(1.0), sin_theta);

    Scalar scale0_slerp = sin((Scalar(1.0) - t) * theta) / safe_sin_theta;
    Scalar scale1_slerp = sin(t * theta) / safe_sin_theta;

    Scalar scale0 = CondExpLt(sin_theta, eps, Scalar(1.0) - t, scale0_slerp);
    Scalar scale1 = CondExpLt(sin_theta, eps, t, scale1_slerp);

    Eigen::Vector4<Scalar> interp_coeffs = scale0 * qa.coeffs() + scale1 * qb_coeffs;

    out[3] = interp_coeffs[0]; // x
    out[4] = interp_coeffs[1]; // y
    out[5] = interp_coeffs[2]; // z
    out[6] = interp_coeffs[3]; // w
}

auto trace_interpolate_impl(
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

    ADVectorXs ad_a = ad_input.head(7); // first 7 elements are a
    ADVectorXs ad_b = ad_input.segment(8, 7); // elements 8-14 are b
    ADCG t = ad_input[2 * 8];

    auto psi_a = ad_input[7];
    auto psi_b = ad_input[15];

    slerp_se3(ad_a, ad_b, t, ad_out);
    ad_out[7] = (1.0 - t) * psi_a + t * psi_b; // linear interpolation of psi

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
        static_cast<std::size_t>(7)};
}

auto trace_interpolate(const std::string &language) -> Traced
{
    return trace_interpolate_impl(
        language, {{"a", 8, true}, {"b", 8, true}, {"t", 1, false}}, {});
}

    auto trace_interpolate_block(const std::string &language) -> Traced
{
    const std::string lang = (language == "c++") ? "c++_block" : language;
    return trace_interpolate_impl(
        lang,
        {{"a", 8, true, ".broadcast(", ")"}, {"b", 8, true, ".broadcast(", ")"}, {"t", 1, false}},
        {{"out", 8, true}});
}

auto read_transform(const ADVectorXs &inp, std::size_t offset) -> SE3Tpl<ADCG>
{
    Eigen::Quaternion<ADCG> rotation(
        inp[offset + 6], inp[offset + 3], inp[offset + 4], inp[offset + 5]);

    Eigen::Vector3<ADCG> translation;
    for (auto i = 0U; i < 3; ++i)
    {
        translation[i] = inp[offset + i];
    }

    return SE3Tpl<ADCG>(rotation, translation);
}

auto se3_displacement(const SE3Tpl<ADCG> &transform) -> ADVectorXs
{
    ADVectorXs displacement(6);
    displacement << transform.translation_impl(),
        so3_log_smooth<ADMatrixXs, ADCG>(transform.rotation_impl());
    return displacement;
}

auto trace_SE3_distance(const std::string &language) -> Traced {

    const std::size_t n_input = 2 * 8;  // a, b, t
    ADVectorXs ad_input(n_input);
    ADVectorXs out(1);
    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    const auto a = read_transform(ad_input, 0);
    const auto b = read_transform(ad_input, 8);

    SE3Tpl<ADCG> R_rel = a.inverse() * b;

    // get axis angle
    // Eigen::AngleAxis<ADCG> angle_axis(R_rel.rotation());
    // ADCG rot_dist = angle_axis.angle();

    // ADCG trans_dist = R_rel.translation().norm();

    auto displacement = se3_displacement(R_rel);

    ADVectorXs total_displacement(7);
    total_displacement.head(6) = displacement;
    total_displacement[6] = ad_input[7] - ad_input[15];

    out[0] = total_displacement.norm();

    CppAD::ADFun<CGD> dist_func(ad_input, out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = dist_func.Forward(0, ind_vars);

    // const auto nq_size = static_cast<std::size_t>(nq);
    SegmentedVariableNameGenerator<double> nameGen({{"a", 8, true}, {"b", 8, true}});

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        1};    

}
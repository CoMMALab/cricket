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
    const Scalar full = sin(x) / x;
    const Scalar w = x2 / (x2 + se3_detail::EPS);

    return (1.0 - w) * taylor + w * full;
}

template <typename Matrix, typename Scalar>
inline auto so3_log_smooth(const Matrix &R) -> Matrix
{
    const Scalar cos_theta = (R.trace() - 1) / 2;
    const Scalar sin_theta2 = 1.0 - cos_theta * cos_theta;
    const Scalar w = sin_theta2 / (sin_theta2 + se3_detail::EPS);

    Matrix vee = Matrix::Zero(3, 1);
    vee(0, 0) = R(2, 1) - R(1, 2);
    vee(1, 0) = R(0, 2) - R(2, 0);
    vee(2, 0) = R(1, 0) - R(0, 1);

    const Scalar theta = acos(cos_theta);

    const auto taylor = 0.5 * vee;

    // factor = theta / (2 sin(theta))
    const Scalar factor = 0.5 / (sinc_smooth(theta) + se3_detail::EPS);
    const auto full = vee * factor;

    return (1.0 - w) * taylor + w * full;
}

auto trace_map_to_se3(
    const pinocchio::Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds) -> Traced
{

    ADVectorXs ad_u(6);
    ADVectorXs ad_se3(7);

    for (auto i = 0U; i < 6; ++i)
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

    CppAD::ADFun<CGD> map_func(ad_u, ad_se3);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(6);
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
    const std::size_t n_input = 2 * 7 + 1;  // a, b, t


    ADVectorXs ad_input(n_input);
    ADVectorXs ad_out(7);

    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_input);

    ADVectorXs ad_a = ad_input.head(7);
    ADVectorXs ad_b = ad_input.segment(7, 7);
    ADCG t = ad_input[2 * 7];

    slerp_se3(ad_a, ad_b, t, ad_out);

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
        language, {{"a", 7, true}, {"b", 7, true}, {"t", 1, false}}, {});
}

    auto trace_interpolate_block(const std::string &language) -> Traced
{
    const std::string lang = (language == "c++") ? "c++_block" : language;
    return trace_interpolate_impl(
        lang,
        {{"a", 7, true, ".broadcast(", ")"}, {"b", 7, true, ".broadcast(", ")"}, {"t", 1, false}},
        {{"out", 7, true}});
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

    const std::size_t n_input = 2 * 7;  // a, b, t
    ADVectorXs ad_input(n_input);
    ADVectorXs out(1);
    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }
    CppAD::Independent(ad_input);

    const auto a = read_transform(ad_input, 0);
    const auto b = read_transform(ad_input, 7);

    SE3Tpl<ADCG> R_rel = a.inverse() * b;

    // get axis angle
    // Eigen::AngleAxis<ADCG> angle_axis(R_rel.rotation());
    // ADCG rot_dist = angle_axis.angle();

    // ADCG trans_dist = R_rel.translation().norm();

    auto displacement = se3_displacement(R_rel);

    out[0] = displacement.norm();

    CppAD::ADFun<CGD> dist_func(ad_input, out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = dist_func.Forward(0, ind_vars);

    // const auto nq_size = static_cast<std::size_t>(nq);
    SegmentedVariableNameGenerator<double> nameGen({{"a", 7, true}, {"b", 7, true}});

    return Traced{
        generate_code(handler, result, language, nameGen),
        handler.getTemporaryVariableCount(),
        1};    

}
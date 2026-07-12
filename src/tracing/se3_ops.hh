#pragma once

namespace cricket
{
    // Smooth blends between Taylor expansions and the exact formulas keep these functions
    // differentiable through theta = 0 when traced by CppAD (no conditionals on the tape).
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
}  // namespace cricket

#pragma once

namespace cricket
{
    // Smooth blends between Taylor expansions and the exact formulas keep these functions
    // differentiable through theta = 0 when traced by CppAD (no conditionals on the tape).
    namespace se3_detail
    {
        constexpr double EPS = 1e-7;
        constexpr double SIN_FLOOR = 1e-12;  // added to sin^2(theta) before the square root
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

    // theta comes from atan2, which CppAD expands into CondExp operations: the traced code contains
    // conditionals and must be generated with the block language (SIMD blends), never Rust.
    template <typename Matrix, typename Scalar>
    inline auto so3_log_smooth(const Matrix &R) -> Matrix
    {
        const Scalar cos_theta = (R.trace() - 1) / 2;

        // vee(R) = sin(theta) * axis. Note |vee|^2 / 4 = sin^2(theta).
        Matrix vee = Matrix::Zero(3, 1);
        vee(0, 0) = R(2, 1) - R(1, 2);
        vee(1, 0) = R(0, 2) - R(2, 0);
        vee(2, 0) = R(1, 0) - R(0, 1);

        // theta from atan2(sin, cos) instead of acos(cos): acos has an infinite derivative at
        // cos = 1 (and is NaN if rounding pushes cos above 1), which poisons the Jacobian as the
        // error approaches zero. The sqrt floor keeps sin > 0, so atan2 and sqrt stay differentiable.
        const Scalar sin_theta = sqrt(vee.squaredNorm() / 4.0 + se3_detail::SIN_FLOOR);
        const Scalar theta = atan2(sin_theta, cos_theta);

        // log(R) = theta / (2 sin(theta)) * vee, which tends to vee / 2 as theta -> 0
        return vee * (0.5 * theta / sin_theta);
    }
}  // namespace cricket

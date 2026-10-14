#pragma once

#include <cmath>  // before cholesky.hh: its unqualified sqrt must see the float/double overloads

#include "cholesky.hh"

#include <cppad/cppad.hpp>

#include <cstddef>

namespace cricket
{
    namespace bounded_ls_detail
    {
        constexpr double bound_tolerance = 1e-6;  // distance to a bound / gradient sign slack
        constexpr double tiny = 1e-9;             // keeps the ratio-test divisions finite

        // Branch-free selects: `name(a, b, if_true, if_false)` is `a op b ? if_true : if_false`.
        // On the CppAD tape they become CondExp operations (SIMD blends in the generated code); on plain
        // doubles they are the ordinary ternary, so the same solver runs unchanged in tests.
#define CRICKET_BOUNDED_LS_SELECT(name, op, cond_exp)                                                       \
    inline auto name(double a, double b, double if_true, double if_false)->double                           \
    {                                                                                                       \
        return (a op b) ? if_true : if_false;                                                               \
    }                                                                                                       \
    template <class Base>                                                                                   \
    inline auto name(                                                                                       \
        const CppAD::AD<Base> &a,                                                                           \
        const CppAD::AD<Base> &b,                                                                           \
        const CppAD::AD<Base> &if_true,                                                                     \
        const CppAD::AD<Base> &if_false)                                                                    \
        ->CppAD::AD<Base>                                                                                   \
    {                                                                                                       \
        return CppAD::cond_exp(a, b, if_true, if_false);                                                    \
    }

        CRICKET_BOUNDED_LS_SELECT(select_lt, <, CondExpLt)
        CRICKET_BOUNDED_LS_SELECT(select_le, <=, CondExpLe)
        CRICKET_BOUNDED_LS_SELECT(select_gt, >, CondExpGt)
        CRICKET_BOUNDED_LS_SELECT(select_ge, >=, CondExpGe)
        CRICKET_BOUNDED_LS_SELECT(select_eq, ==, CondExpEq)
#undef CRICKET_BOUNDED_LS_SELECT
    }  // namespace bounded_ls_detail

    // Bounded least-squares projection step, without data-dependent control flow:
    //
    //     min_x  1/2 |J x + err|^2 + 1/2 damping |x|^2     s.t.  lower <= x <= upper
    //
    // `lower`/`upper` are the allowed *step* (joint limit minus the current configuration), so
    // lower <= 0 <= upper and x = 0 is feasible. Returns the gradient -x, to be applied like the other
    // projection methods as q <- q - alpha * gradient.
    //
    // This is a primal active-set method (Stark & Parker's BVLS; Nocedal & Wright ch. 16) unrolled to
    // a fixed number of iterations, every branch a select. State per joint: 0 free, -1 fixed at its
    // lower bound, +1 fixed at its upper bound. An iteration
    //   1. solves the damped least squares over the free joints with the fixed joints held at their
    //      bounds (the m x m inner form, so fixed joints are just zeroed columns of J),
    //   2. if that overshoots a bound, moves only up to the first one hit and fixes that joint there,
    //   3. otherwise checks the KKT sign of every fixed joint's gradient and frees the worst violator.
    // Once the KKT conditions hold an iteration changes nothing, so lanes that finish early just idle;
    // running out of iterations still returns a feasible step with a lower cost than x = 0.
    template <typename Matrix, typename Vector, typename Scalar>
    auto bounded_ls_step(
        const Matrix &J,
        const Vector &err,
        const Vector &lower_in,
        const Vector &upper_in,
        double damping,
        std::size_t iterations) -> Vector
    {
        using namespace bounded_ls_detail;

        const Eigen::Index m = J.rows();
        const Eigen::Index n = J.cols();

        const Scalar zero(0.0), one(1.0), minus_one(-1.0), half(0.5);
        const Scalar tol(bound_tolerance), tiny_value(tiny), lambda(damping);

        Vector lower(n), upper(n), x(n), state(n);
        for (Eigen::Index i = 0; i < n; ++i)
        {
            // rounding may leave the configuration a hair outside its limit; keep 0 feasible
            lower[i] = select_gt(lower_in[i], zero, zero, lower_in[i]);
            upper[i] = select_lt(upper_in[i], zero, zero, upper_in[i]);

            // a joint with no room to move starts fixed at that bound
            state[i] = select_ge(lower[i], Scalar(-tol), minus_one, select_le(upper[i], tol, one, zero));
            x[i] = zero;
        }

        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            Vector free_flag(n), fixed_part(n);
            for (Eigen::Index i = 0; i < n; ++i)
            {
                free_flag[i] = select_eq(state[i], zero, one, zero);
                fixed_part[i] = x[i] * (one - free_flag[i]);
            }

            // residual left to the free joints once the fixed ones are held at their bounds
            const Vector residual = err + J * fixed_part;

            Matrix J_free(m, n);
            for (Eigen::Index i = 0; i < n; ++i)
            {
                J_free.col(i) = J.col(i) * free_flag[i];
            }

            Matrix A = J_free * J_free.transpose();
            for (Eigen::Index i = 0; i < m; ++i)
            {
                A(i, i) = A(i, i) + lambda;
            }

            const Matrix factor = cholesky_factor<Matrix, Scalar>(A);
            const Vector y = cholesky_solve<Matrix, Vector, Scalar>(factor, residual);
            const Vector pull = J_free.transpose() * y;  // J_F^T y, zero on the fixed joints

            // unconstrained minimiser over the free joints
            Vector target(n);
            for (Eigen::Index i = 0; i < n; ++i)
            {
                target[i] = free_flag[i] * (zero - pull[i]) + (one - free_flag[i]) * x[i];
            }

            // ratio test: how far along x -> target before the first bound is hit
            Vector limit(n), below(n), above(n);
            Scalar alpha = one;
            for (Eigen::Index i = 0; i < n; ++i)
            {
                below[i] = select_lt(target[i], Scalar(lower[i] - tol), free_flag[i], zero);
                above[i] = select_gt(target[i], Scalar(upper[i] + tol), free_flag[i], zero);

                const Scalar down = x[i] - target[i];
                const Scalar up = target[i] - x[i];
                const Scalar safe_down = select_gt(down, tiny_value, down, tiny_value);
                const Scalar safe_up = select_gt(up, tiny_value, up, tiny_value);

                const Scalar to_lower = (x[i] - lower[i]) / safe_down;
                const Scalar to_upper = (upper[i] - x[i]) / safe_up;

                Scalar fraction = select_gt(below[i], half, to_lower, select_gt(above[i], half, to_upper, one));
                fraction = select_lt(fraction, zero, zero, fraction);
                limit[i] = fraction;
                alpha = select_lt(fraction, alpha, fraction, alpha);
            }

            // move, and fix the joint(s) that blocked the move at their bound
            for (Eigen::Index i = 0; i < n; ++i)
            {
                const Scalar moved = x[i] + alpha * (target[i] - x[i]);
                const Scalar blocking = select_le(limit[i], alpha, one, zero);
                const Scalar blocked_lower = blocking * below[i];
                const Scalar blocked_upper = blocking * above[i];

                x[i] = select_gt(
                    blocked_lower, half, lower[i], select_gt(blocked_upper, half, upper[i], moved));
                state[i] = select_gt(
                    blocked_lower, half, minus_one, select_gt(blocked_upper, half, one, state[i]));
            }

            // KKT check, only when this iteration was not blocked: the gradient of a joint fixed at its
            // lower bound must be >= 0 and at its upper bound <= 0; free the worst violator
            const Scalar feasible = select_ge(alpha, one, one, zero);

            const Vector error_at_x = J * x + err;
            Vector gradient = J.transpose() * error_at_x;
            Vector violation(n);
            Scalar worst = tol;
            for (Eigen::Index i = 0; i < n; ++i)
            {
                gradient[i] = gradient[i] + lambda * x[i];
                violation[i] = state[i] * gradient[i];
                worst = select_gt(violation[i], worst, violation[i], worst);
            }

            for (Eigen::Index i = 0; i < n; ++i)
            {
                const Scalar release = select_gt(violation[i], tol, select_ge(violation[i], worst, one, zero), zero);
                state[i] = select_gt(release * feasible, half, zero, state[i]);
            }
        }

        Vector gradient(n);
        for (Eigen::Index i = 0; i < n; ++i)
        {
            gradient[i] = zero - x[i];
        }

        return gradient;
    }
}  // namespace cricket

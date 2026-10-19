#pragma once

#include <cstddef>

namespace cricket
{
    // Hand-rolled Cholesky factorization and triangular solves: Eigen's .llt() cannot be
    // traced by CppAD, so these unrolled loops are used instead when building the tape.
    template <typename Matrix, typename Scalar>
    auto cholesky_factor(const Matrix &input) -> Matrix
    {
        const std::size_t n = input.rows();
        Matrix result(n, n);
        result.setZero();

        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t k = 0; k < i; ++k)
            {
                Scalar value = input(i, k);
                for (std::size_t j = 0; j < k; ++j)
                {
                    value -= result(i, j) * result(k, j);
                }

                result(i, k) = value / result(k, k);
            }

            Scalar value = input(i, i);
            for (std::size_t j = 0; j < i; ++j)
            {
                value -= result(i, j) * result(i, j);
            }

            result(i, i) = sqrt(value);
        }

        return result;
    }

    template <typename Matrix, typename Vector, typename Scalar>
    auto lower_solve(const Matrix &A, const Vector &b) -> Vector
    {
        const std::size_t n = b.rows();
        Vector result(n);
        result.setZero();

        for (std::size_t i = 0; i < n; ++i)
        {
            Scalar value = b(i);
            for (std::size_t j = 0; j < i; ++j)
            {
                value -= A(i, j) * result(j);
            }

            result(i) = value / A(i, i);
        }

        return result;
    }

    template <typename Matrix, typename Vector, typename Scalar>
    auto upper_solve(const Matrix &A, const Vector &b) -> Vector
    {
        const std::size_t n = b.rows();
        Vector result(n);
        result.setZero();

        for (auto i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i)
        {
            Scalar value = b(i);
            for (std::size_t j = i + 1; j < n; ++j)
            {
                value -= A(i, j) * result(j);
            }

            result(i) = value / A(i, i);
        }

        return result;
    }

    template <typename Matrix, typename Vector, typename Scalar>
    auto cholesky_solve(const Matrix &A, const Vector &b) -> Vector
    {
        const auto y = lower_solve<Matrix, Vector, Scalar>(A, b);
        return upper_solve<Matrix, Vector, Scalar>(A.transpose(), y);
    }
}  // namespace cricket

#pragma once

#include <cstddef>

namespace cricket
{
    // Unrolled LU solve without pivoting (Doolittle) for tracing: solves A x = b and reports the
    // pivots (U's diagonal) so the caller can reject near-singular systems. Without pivoting the
    // factorization is stable only for systems near a well-conditioned diagonal -- e.g. a Newton
    // system preconditioned by an inverse at a nearby point.
    template <typename Matrix, typename Vector>
    auto lu_solve_nopivot(Matrix A, Vector b, Vector &pivots) -> Vector
    {
        const std::size_t n = A.rows();
        for (std::size_t k = 0; k < n; ++k)
        {
            for (std::size_t i = k + 1; i < n; ++i)
            {
                A(i, k) = A(i, k) / A(k, k);
                for (std::size_t j = k + 1; j < n; ++j)
                {
                    A(i, j) = A(i, j) - A(i, k) * A(k, j);
                }
            }
        }

        for (std::size_t i = 1; i < n; ++i)
        {
            for (std::size_t j = 0; j < i; ++j)
            {
                b[i] = b[i] - A(i, j) * b[j];
            }
        }

        for (std::size_t i = n; i-- > 0;)
        {
            for (std::size_t j = i + 1; j < n; ++j)
            {
                b[i] = b[i] - A(i, j) * b[j];
            }

            b[i] = b[i] / A(i, i);
            pivots[i] = A(i, i);
        }

        return b;
    }
}  // namespace cricket

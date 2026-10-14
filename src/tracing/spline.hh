#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <vector>

namespace cricket
{
    // Values of the (degree + 1) B-spline basis functions that are non-zero on one knot span, by the
    // Cox-de Boor recursion (Piegl & Tiller, "The NURBS Book", BasisFuns).
    //
    // `knot` points at the span's start knot U[i], so that U[i] <= t < U[i + 1]; the recursion reads
    // U[i - degree + 1] .. U[i + degree]. On return basis[j] is N_{i - degree + j, degree}(t); the values
    // are non-negative and sum to one.
    //
    // The loops have fixed bounds and nothing branches on t, so this also instantiates with a CppAD scalar
    // and traces to straight-line code. The caller chooses the span.
    template <typename Scalar, int degree>
    inline void bspline_basis(const Scalar *knot, const Scalar &t, Scalar (&basis)[degree + 1])
    {
        Scalar left[degree + 1];
        Scalar right[degree + 1];

        basis[0] = Scalar(1.0);
        for (int j = 1; j <= degree; ++j)
        {
            left[j] = t - knot[1 - j];
            right[j] = knot[j] - t;

            Scalar saved = Scalar(0.0);
            for (int r = 0; r < j; ++r)
            {
                const Scalar term = basis[r] / (right[r + 1] + left[j - r]);
                basis[r] = saved + right[r + 1] * term;
                saved = left[j - r] * term;
            }

            basis[j] = saved;
        }
    }

    // Cubic B-spline curve in R^3 on t in [0, 1].
    //
    // n control points need n + 4 knots and give n - 3 cubic pieces; for n = 20 that is 24 knots and 17
    // pieces. With a clamped knot vector (the first four knots are 0, the last four are 1) the curve starts
    // at the first control point and ends at the last.
    struct CubicBSpline
    {
        static constexpr int degree = 3;

        std::vector<Eigen::Vector3d> control_points;
        std::vector<double> knots;

        auto size() const -> std::size_t
        {
            return control_points.size();
        }

        auto segments() const -> std::size_t
        {
            return control_points.size() - degree;
        }
    };
}  // namespace cricket

#pragma once

#include "spline.hh"

#include <cppad/cppad.hpp>

#include <Eigen/Dense>

#include <array>
#include <cstddef>

// The spline constraint's curve evaluation, written for any scalar type: the traced kernel instantiates it
// with the CppAD code-generation scalar, and the tests with double and CppAD::AD<double>.
namespace cricket
{
    // The spline part of the traced inputs: the curve parameter, and the data of a spline with N control
    // points (its control points and knots)
    template <typename Scalar, std::size_t N>
    struct SplineInputs
    {
        Scalar t;
        std::array<Eigen::Matrix<Scalar, 3, 1>, SplineInputLayout<N>::control_points> control_points;
        std::array<Scalar, SplineInputLayout<N>::knots> knots;
    };

    // Read the spline inputs off an input vector laid out as SplineInputLayout<N> says; `nq` is the number
    // of joints in front of them. `inp` is anything indexable (a CppAD vector, an Eigen vector, ...).
    template <typename Scalar, std::size_t N, typename Inputs>
    auto read_spline(const Inputs &inp, std::size_t nq) -> SplineInputs<Scalar, N>
    {
        using Layout = SplineInputLayout<N>;

        SplineInputs<Scalar, N> spline;
        spline.t = inp[Layout::t_idx(nq)];

        for (std::size_t point = 0; point < Layout::control_points; ++point)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                spline.control_points[point][axis] = inp[Layout::control_point_idx(nq, point, axis)];
            }
        }

        for (std::size_t i = 0; i < Layout::knots; ++i)
        {
            spline.knots[i] = inp[Layout::knot_idx(nq, i)];
        }

        return spline;
    }

    // Point on the spline at the parameter t.
    //
    // The span containing t is chosen by conditionals: for each span a CondExp picks that span's window (the
    // four control points that influence it, and the six knots bspline_basis reads) when t lies before the
    // span's end, and otherwise keeps the choice made for the later spans. Spans are visited from the last
    // to the first, so the result is the first span whose end is after t, and t = 1 stays in the last span.
    // The selected values do not depend on t, so d(point)/dt flows only through the basis. bspline_basis
    // then runs once, not once per span.
    template <typename Scalar, std::size_t N>
    auto spline_point(const SplineInputs<Scalar, N> &spline) -> Eigen::Matrix<Scalar, 3, 1>
    {
        constexpr std::size_t degree = CubicBSpline::degree;
        constexpr std::size_t window_knots = 2 * degree;  // U[i - 2] .. U[i + 3] for span i

        std::array<Eigen::Matrix<Scalar, 3, 1>, degree + 1> window_points;
        std::array<Scalar, window_knots> window;

        // start from the last span, i = N - 1
        for (std::size_t j = 0; j <= degree; ++j)
        {
            window_points[j] = spline.control_points[N - 1 - degree + j];
        }

        for (std::size_t j = 0; j < window_knots; ++j)
        {
            window[j] = spline.knots[N - 1 - (degree - 1) + j];
        }

        for (std::size_t span = N - 2; span >= degree; --span)
        {
            const Scalar &span_end = spline.knots[span + 1];

            for (std::size_t j = 0; j <= degree; ++j)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    window_points[j][axis] = CppAD::CondExpLt(
                        spline.t, span_end, spline.control_points[span - degree + j][axis], window_points[j][axis]);
                }
            }

            for (std::size_t j = 0; j < window_knots; ++j)
            {
                window[j] = CppAD::CondExpLt(spline.t, span_end, spline.knots[span - (degree - 1) + j], window[j]);
            }
        }

        // window[degree - 1] is the span's start knot U[i]; bspline_basis reads from U[i - 2] to U[i + 3]
        Scalar basis[degree + 1];
        bspline_basis<Scalar, degree>(&window[degree - 1], spline.t, basis);

        Eigen::Matrix<Scalar, 3, 1> point = basis[0] * window_points[0];
        for (std::size_t j = 1; j <= degree; ++j)
        {
            point += basis[j] * window_points[j];
        }

        return point;
    }
}  // namespace cricket

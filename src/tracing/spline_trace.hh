#pragma once

#include "spline.hh"

#include <cppad/cppad.hpp>

#include <Eigen/Dense>

#include <array>
#include <cmath>
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

    // The part of the spline that matters at one parameter: the four control points that influence the span containing
    // t, and the six knots U[i - 2] .. U[i + 3] that bspline_basis reads for span i.
    template <typename Scalar>
    struct SplineWindow
    {
        std::array<Eigen::Matrix<Scalar, 3, 1>, CubicBSpline::degree + 1> points;
        std::array<Scalar, 2 * CubicBSpline::degree> knots;
    };

    // Pick the window of the span containing the traced parameter t.
    //
    // The span is chosen by conditionals: for each span a CondExp picks that span's window when t lies before the
    // span's end, and otherwise keeps the choice made for the later spans. Spans are visited from the last to the
    // first, so the result is the first span whose end is after t, and t = 1 stays in the last span. The selected
    // values do not depend on t, so derivatives with respect to t flow only through the basis.
    template <typename Scalar, std::size_t N>
    auto select_window(const SplineInputs<Scalar, N> &spline) -> SplineWindow<Scalar>
    {
        constexpr std::size_t degree = CubicBSpline::degree;
        constexpr std::size_t window_knots = 2 * degree;

        SplineWindow<Scalar> window;

        // start from the last span, i = N - 1
        for (std::size_t j = 0; j <= degree; ++j)
        {
            window.points[j] = spline.control_points[N - 1 - degree + j];
        }

        for (std::size_t j = 0; j < window_knots; ++j)
        {
            window.knots[j] = spline.knots[N - 1 - (degree - 1) + j];
        }

        for (std::size_t span = N - 2; span >= degree; --span)
        {
            const Scalar &span_end = spline.knots[span + 1];

            for (std::size_t j = 0; j <= degree; ++j)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    window.points[j][axis] = CppAD::CondExpLt(
                        spline.t, span_end, spline.control_points[span - degree + j][axis], window.points[j][axis]);
                }
            }

            for (std::size_t j = 0; j < window_knots; ++j)
            {
                window.knots[j] =
                    CppAD::CondExpLt(spline.t, span_end, spline.knots[span - (degree - 1) + j], window.knots[j]);
            }
        }

        return window;
    }

    // Point on the spline at the parameter t: the four basis values of t's span weight the window's control points.
    // bspline_basis runs once, not once per span.
    template <typename Scalar, std::size_t N>
    auto spline_point(const SplineInputs<Scalar, N> &spline) -> Eigen::Matrix<Scalar, 3, 1>
    {
        constexpr std::size_t degree = CubicBSpline::degree;

        const auto window = select_window(spline);

        // window.knots[degree - 1] is the span's start knot U[i]; bspline_basis reads from U[i - 2] to U[i + 3]
        Scalar basis[degree + 1];
        bspline_basis<Scalar, degree>(&window.knots[degree - 1], spline.t, basis);

        Eigen::Matrix<Scalar, 3, 1> point = basis[0] * window.points[0];
        for (std::size_t j = 1; j <= degree; ++j)
        {
            point += basis[j] * window.points[j];
        }

        return point;
    }

    template <typename Scalar>
    struct PointAndVelocity
    {
        Eigen::Matrix<Scalar, 3, 1> point;
        Eigen::Matrix<Scalar, 3, 1> velocity;  // dP/dt
    };

    // Point and dP/dt at the parameter t, from one window selection. The velocity is the hodograph: a degree-2 B-spline
    // on the same knots with control points Q_j = 3 (P_{j+1} - P_j) / (U_{j+1} - U_{j-2}) in window terms, which for
    // the window's knots w[0..5] (= U[i - 2] .. U[i + 3]) is Q_j = 3 (p[j+1] - p[j]) / (w[j+3] - w[j]), j = 0, 1, 2,
    // weighted by the three degree-2 basis values of t's span. The divisions are by knot spans of the window, which
    // are positive for a valid (strictly increasing interior) knot vector.
    template <typename Scalar, std::size_t N>
    auto spline_point_and_velocity(const SplineInputs<Scalar, N> &spline) -> PointAndVelocity<Scalar>
    {
        constexpr std::size_t degree = CubicBSpline::degree;

        const auto window = select_window(spline);

        Scalar basis[degree + 1];
        bspline_basis<Scalar, degree>(&window.knots[degree - 1], spline.t, basis);

        PointAndVelocity<Scalar> result;
        result.point = basis[0] * window.points[0];
        for (std::size_t j = 1; j <= degree; ++j)
        {
            result.point += basis[j] * window.points[j];
        }

        // degree - 1 = 2: three basis functions N_{i-2,2} .. N_{i,2}, which pair with Q_0 .. Q_2
        Scalar basis_velocity[degree];
        bspline_basis<Scalar, degree - 1>(&window.knots[degree - 1], spline.t, basis_velocity);

        const auto hodograph_point = [&window](std::size_t j) -> Eigen::Matrix<Scalar, 3, 1>
        {
            return Scalar(3.0) * (window.points[j + 1] - window.points[j]) /
                   (window.knots[j + degree] - window.knots[j]);
        };

        result.velocity = basis_velocity[0] * hodograph_point(0);
        for (std::size_t j = 1; j < degree; ++j)
        {
            result.velocity += basis_velocity[j] * hodograph_point(j);
        }

        return result;
    }

    // Seed for the curve parameter of the point of the curve nearest to some position x, shared by every kernel that
    // needs one. Inputs: the squared distances from x to the N fitted waypoints, and the parameters at which the curve
    // passes through them.
    //
    // It takes the nearest waypoint k and its nearer neighbour nb in the fitted sequence (a neighbour in the sequence, so
    // a curve that loops back cannot pair k with a far-away waypoint), and interpolates between their parameters,
    // weighted by the distances:
    //
    //   t = t_k + w (t_nb - t_k),    w = d_k / (d_k + d_nb)        (d = the distances, not squared)
    //
    // At waypoint k this gives t_k, at a point equidistant from the two it gives the midpoint, and for a point on the
    // segment between the two waypoints the distances are proportional to the position along it, so w is the linear
    // fraction exactly. Off the segment it is an approximation. The seed is only a starting guess for a refinement.
    //
    // All of it is selects (CondExp on the tape, plain ternaries on doubles), so it traces to straight-line code; the
    // winner's four values are carried through the scan and the two square roots are taken once at the end.
    template <typename Scalar, std::size_t N>
    auto weighted_waypoint_seed(const std::array<Scalar, N> &distance2, const std::array<Scalar, N> &parameters) -> Scalar
    {
        static_assert(N >= 2, "a seed needs at least two waypoints");

        // the running winner: its squared distance and parameter, and its nearer neighbour's
        Scalar best_distance2 = distance2[0];
        Scalar best_parameter = parameters[0];
        Scalar neighbour_distance2 = distance2[1];  // waypoint 0 has only one neighbour
        Scalar neighbour_parameter = parameters[1];

        for (std::size_t point = 1; point < N; ++point)
        {
            // this waypoint's nearer neighbour
            Scalar candidate_distance2 = distance2[point - 1];
            Scalar candidate_parameter = parameters[point - 1];
            if (point + 1 < N)
            {
                candidate_distance2 =
                    CppAD::CondExpLt(distance2[point + 1], distance2[point - 1], distance2[point + 1], candidate_distance2);
                candidate_parameter =
                    CppAD::CondExpLt(distance2[point + 1], distance2[point - 1], parameters[point + 1], candidate_parameter);
            }

            // take this waypoint if it is nearer than the winner so far: every update compares against the previous best
            neighbour_distance2 =
                CppAD::CondExpLt(distance2[point], best_distance2, candidate_distance2, neighbour_distance2);
            neighbour_parameter =
                CppAD::CondExpLt(distance2[point], best_distance2, candidate_parameter, neighbour_parameter);
            best_parameter = CppAD::CondExpLt(distance2[point], best_distance2, parameters[point], best_parameter);
            best_distance2 = CppAD::CondExpLt(distance2[point], best_distance2, distance2[point], best_distance2);
        }

        using std::sqrt;
        const Scalar distance = sqrt(best_distance2);
        const Scalar neighbour_distance = sqrt(neighbour_distance2);

        // the small term keeps the weight defined if both waypoints coincide with the position
        return best_parameter + (neighbour_parameter - best_parameter) * distance /
                                    (distance + neighbour_distance + Scalar(1e-12));
    }
}  // namespace cricket

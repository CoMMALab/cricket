#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
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

        // t at which fit_cubic_bspline() made the curve pass through each input point (empty if the spline
        // was built by hand)
        std::vector<double> parameters;

        auto size() const -> std::size_t
        {
            return control_points.size();
        }

        auto segments() const -> std::size_t
        {
            return control_points.size() - degree;
        }

        // Knot span i with knots[i] <= t < knots[i + 1], in [degree, size() - 1]. Only these spans have
        // length (the clamped end knots are repeated), and t = 1 belongs to the last one.
        auto find_span(double t) const -> std::size_t
        {
            const std::size_t n = control_points.size();
            if (t >= knots[n])
            {
                return n - 1;
            }

            if (t <= knots[degree])
            {
                return degree;
            }

            // first knot greater than t, among knots[degree] .. knots[n]; the span is the one before it
            const auto upper = std::upper_bound(knots.begin() + degree, knots.begin() + n + 1, t);
            return static_cast<std::size_t>(upper - knots.begin()) - 1;
        }

        // Point on the curve at t (clamped to [0, 1]): the four basis values of t's span weight the four
        // control points that influence that span.
        auto value(double t) const -> Eigen::Vector3d
        {
            t = std::clamp(t, 0.0, 1.0);
            const auto span = find_span(t);

            double basis[degree + 1];
            bspline_basis<double, degree>(&knots[span], t, basis);

            Eigen::Vector3d point = Eigen::Vector3d::Zero();
            for (int j = 0; j <= degree; ++j)
            {
                point += basis[j] * control_points[span - degree + j];
            }

            return point;
        }

        // dP/dt. The derivative of a degree-3 B-spline is a degree-2 B-spline on the same knots whose control
        // points are Q_m = 3 (P_{m+1} - P_m) / (U_{m+4} - U_{m+1}) (the hodograph).
        auto derivative(double t) const -> Eigen::Vector3d
        {
            t = std::clamp(t, 0.0, 1.0);
            const auto span = find_span(t);

            // degree - 1 = 2: three basis functions, N_{span-2,2} .. N_{span,2}
            double basis[degree];
            bspline_basis<double, degree - 1>(&knots[span], t, basis);

            Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
            for (int j = 0; j < degree; ++j)
            {
                const std::size_t m = span - degree + j;  // Q_m pairs with the basis function N_{m+1,2}
                const Eigen::Vector3d q = static_cast<double>(degree) * (control_points[m + 1] - control_points[m]) /
                                          (knots[m + degree + 1] - knots[m + 1]);
                velocity += basis[j] * q;
            }

            return velocity;
        }
    };

    // Order of the inputs of the traced spline constraint, for a spline with N control points (N is fixed when
    // the kernel is traced). `nq` is the robot's number of joints, known at trace time too.
    //
    //   [ q (nq) | t (1) | control points (3 N, point-major: x y z of point 0, then point 1, ...) | knots (N + 4) ]
    //
    // `q` is indexed directly (joint j is input j). `q` and `t` are the variables the projection moves; the control points and knots are data that the
    // runtime fills in per spline. The vamp-side input struct has to index its data in this same order.
    template <std::size_t N>
    struct SplineInputLayout
    {
        static_assert(N >= CubicBSpline::degree + 1, "a cubic B-spline needs at least four control points");

        static constexpr std::size_t control_points = N;
        static constexpr std::size_t knots = N + CubicBSpline::degree + 1;
        static constexpr std::size_t pieces = N - CubicBSpline::degree;

        // t, then the control point coordinates, then the knots
        static constexpr std::size_t spline_inputs = 1 + 3 * control_points + knots;

        static constexpr auto t_idx(std::size_t nq) -> std::size_t
        {
            return nq;
        }

        static constexpr auto control_point_idx(std::size_t nq, std::size_t point, std::size_t axis) -> std::size_t
        {
            return nq + 1 + 3 * point + axis;
        }

        static constexpr auto knot_idx(std::size_t nq, std::size_t index) -> std::size_t
        {
            return nq + 1 + 3 * control_points + index;
        }

        static constexpr auto total(std::size_t nq) -> std::size_t
        {
            return nq + spline_inputs;
        }
    };

    // Cubic B-spline that passes through `points` (at least four, R^3), by global interpolation in three
    // stages:
    //   1. parameters: point k gets t_k = (distance along the polyline up to k) / (total length), so t_0 = 0
    //      and t_{n-1} = 1 and uneven spacing is respected;
    //   2. knots: four 0s, then the average of `degree` consecutive t_k for each interior knot, then four 1s;
    //   3. control points: solve  sum_j N_j(t_k) P_j = x_k  for k = 0 .. n-1. Each row has at most four
    //      non-zeros (the basis of t_k's span); it is solved densely, which is instant for n around 20.
    //
    // The points must not repeat consecutively: a zero chord would give two equal t_k and a singular system.
    inline auto fit_cubic_bspline(const std::vector<Eigen::Vector3d> &points) -> CubicBSpline
    {
        constexpr int degree = CubicBSpline::degree;
        const std::size_t n = points.size();
        if (n < static_cast<std::size_t>(degree) + 1)
        {
            throw std::invalid_argument("a cubic B-spline needs at least four points");
        }

        CubicBSpline spline;

        // find_span() takes the number of control points from control_points.size(), so size it up front
        spline.control_points.assign(n, Eigen::Vector3d::Zero());

        // 1. parameters
        spline.parameters.assign(n, 0.0);
        double total = 0.0;
        for (std::size_t k = 1; k < n; ++k)
        {
            total += (points[k] - points[k - 1]).norm();
        }

        double travelled = 0.0;
        for (std::size_t k = 1; k < n; ++k)
        {
            travelled += (points[k] - points[k - 1]).norm();
            spline.parameters[k] = travelled / total;
        }

        spline.parameters[n - 1] = 1.0;

        // 2. knots: n + 4 in total; knots[0..3] = 0 and knots[n..n+3] = 1 already, the n - 4 interior ones are
        // knots[4 .. n-1], the average of parameters[j .. j+2] for j = 1 .. n-4
        spline.knots.assign(n + degree + 1, 0.0);
        for (std::size_t j = 1; j + degree < n; ++j)
        {
            spline.knots[j + degree] = (spline.parameters[j] + spline.parameters[j + 1] + spline.parameters[j + 2]) / 3.0;
        }

        for (std::size_t i = n; i < n + degree + 1; ++i)
        {
            spline.knots[i] = 1.0;
        }

        // 3. control points
        Eigen::MatrixXd system = Eigen::MatrixXd::Zero(n, n);
        Eigen::MatrixXd rhs(n, 3);
        for (std::size_t k = 0; k < n; ++k)
        {
            const double t = spline.parameters[k];
            const auto span = spline.find_span(t);

            double basis[degree + 1];
            bspline_basis<double, degree>(&spline.knots[span], t, basis);
            for (int j = 0; j <= degree; ++j)
            {
                system(k, span - degree + j) = basis[j];
            }

            rhs.row(k) = points[k].transpose();
        }

        const Eigen::MatrixXd solution = system.partialPivLu().solve(rhs);
        for (std::size_t k = 0; k < n; ++k)
        {
            spline.control_points[k] = solution.row(k).transpose();
        }

        return spline;
    }
}  // namespace cricket

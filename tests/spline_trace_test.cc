// Checks the curve evaluation of the traced spline constraint (src/tracing/spline_trace.hh) against the host
// spline (src/tracing/spline.hh): the value, the derivative with respect to t, and the C2 behaviour across
// knots. The evaluation is generic over the scalar type, so it runs here with double and with CppAD::AD<double>
// (the tape is recorded once, at one t, and replayed at others: that is what exercises the span selection).
//
// Uses CHECK instead of assert: the project builds in Release by default, which removes asserts.

#include "tracing/spline_trace.hh"

#include <pinocchio/autodiff/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

static int failures = 0;

#define CHECK(condition)                                                                                     \
    do                                                                                                       \
    {                                                                                                        \
        if (not(condition))                                                                                  \
        {                                                                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                        \
            ++failures;                                                                                      \
        }                                                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tolerance)                                                                          \
    do                                                                                                       \
    {                                                                                                        \
        const double check_a = (a);                                                                          \
        const double check_b = (b);                                                                          \
        if (not(std::abs(check_a - check_b) <= (tolerance)))                                                 \
        {                                                                                                    \
            std::fprintf(                                                                                    \
                stderr,                                                                                      \
                "FAIL %s:%d: %s = %.12g vs %s = %.12g\n",                                                    \
                __FILE__,                                                                                    \
                __LINE__,                                                                                    \
                #a,                                                                                          \
                check_a,                                                                                     \
                #b,                                                                                          \
                check_b);                                                                                    \
            ++failures;                                                                                      \
        }                                                                                                    \
    } while (0)

using AD = CppAD::AD<double>;

static constexpr std::size_t N = 20;
static constexpr std::size_t nq = 2;  // joints in front of the spline inputs; unused by the curve
using Layout = cricket::SplineInputLayout<N>;

// The flat input vector for a given curve parameter
static auto make_inputs(const cricket::CubicBSpline &spline, double t) -> std::vector<double>
{
    std::vector<double> x(Layout::total(nq), 0.0);
    x[0] = 0.3;
    x[1] = -0.2;
    x[Layout::t_idx(nq)] = t;
    for (std::size_t p = 0; p < N; ++p)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            x[Layout::control_point_idx(nq, p, axis)] = spline.control_points[p][axis];
        }
    }

    for (std::size_t i = 0; i < Layout::knots; ++i)
    {
        x[Layout::knot_idx(nq, i)] = spline.knots[i];
    }

    return x;
}

int main()
{
    // N points on a helix with uneven spacing
    std::vector<Eigen::Vector3d> points;
    for (std::size_t i = 0; i < N; ++i)
    {
        const double s = static_cast<double>(i) / static_cast<double>(N - 1);
        const double angle = 6.28318530718 * std::pow(s, 1.3);
        points.emplace_back(0.5 * std::cos(angle) + 0.6, 0.5 * std::sin(angle), 0.2 + 0.4 * s);
    }

    const auto spline = cricket::fit_cubic_bspline(points);

    // Parameters to evaluate at: a fine grid, the ends, every knot exactly, and just either side of every knot
    std::vector<double> ts;
    for (int s = 0; s <= 997; ++s)
    {
        ts.push_back(s / 997.0);
    }

    ts.push_back(0.0);
    ts.push_back(1.0);
    for (std::size_t i = 3; i <= N; ++i)
    {
        const double knot = spline.knots[i];
        ts.push_back(knot);
        if (knot > 1e-6 and knot < 1.0 - 1e-6)
        {
            ts.push_back(knot - 1e-9);
            ts.push_back(knot + 1e-9);
        }
    }

    // 1. double: the span selection and the sum agree with CubicBSpline::value
    for (const double t : ts)
    {
        const auto x = make_inputs(spline, t);
        const auto in = cricket::read_spline<double, N>(x, nq);
        const Eigen::Vector3d point = cricket::spline_point<double, N>(in);
        CHECK_NEAR((point - spline.value(t)).norm(), 0.0, 1e-10);
    }

    // 2. AD<double>: record the tape once at t = 0.5, replay it at every t
    std::vector<AD> ax(Layout::total(nq));
    {
        const auto x0 = make_inputs(spline, 0.5);
        for (std::size_t i = 0; i < ax.size(); ++i)
        {
            ax[i] = x0[i];
        }
    }

    CppAD::Independent(ax);
    const auto ad_in = cricket::read_spline<AD, N>(ax, nq);
    const Eigen::Matrix<AD, 3, 1> ad_point = cricket::spline_point<AD, N>(ad_in);
    std::vector<AD> ay(3);
    for (std::size_t i = 0; i < 3; ++i)
    {
        ay[i] = ad_point[i];
    }

    CppAD::ADFun<double> f(ax, ay);

    const std::size_t n = Layout::total(nq);
    for (const double t : ts)
    {
        const auto x = make_inputs(spline, t);
        const std::vector<double> y = f.Forward(0, x);
        const std::vector<double> jacobian = f.Jacobian(x);  // row-major, 3 x n

        // value, and d(point)/dt against the host derivative (the hodograph)
        const Eigen::Vector3d expected_value = spline.value(t);
        const Eigen::Vector3d expected_velocity = spline.derivative(t);
        for (std::size_t i = 0; i < 3; ++i)
        {
            CHECK_NEAR(y[i], expected_value[i], 1e-10);
            CHECK_NEAR(jacobian[i * n + Layout::t_idx(nq)], expected_velocity[i], 1e-7 * (1.0 + expected_velocity.norm()));
        }

        // The point is a weighted sum of control points whose weights add up to one: d(x)/d(P_j,x) summed over j
        // is 1, and the point does not depend on q at all
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            double weights = 0.0;
            for (std::size_t p = 0; p < N; ++p)
            {
                weights += jacobian[axis * n + Layout::control_point_idx(nq, p, axis)];
            }

            CHECK_NEAR(weights, 1.0, 1e-10);
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                CHECK(jacobian[axis * n + joint] == 0.0);
            }
        }
    }

    // 3. C2 across the interior knots: the replayed tape gives the same point and velocity just before and after
    for (std::size_t i = 4; i < N; ++i)
    {
        const double knot = spline.knots[i];
        const double e = 1e-7;

        const auto x_left = make_inputs(spline, knot - e);
        const auto x_right = make_inputs(spline, knot + e);
        const auto y_left = f.Forward(0, x_left);
        const auto y_right = f.Forward(0, x_right);
        const auto jacobian_left = f.Jacobian(x_left);
        const auto jacobian_right = f.Jacobian(x_right);

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            CHECK_NEAR(y_left[axis], y_right[axis], 1e-5);
            CHECK_NEAR(
                jacobian_left[axis * n + Layout::t_idx(nq)],
                jacobian_right[axis * n + Layout::t_idx(nq)],
                1e-4);
        }
    }

    if (failures != 0)
    {
        std::fprintf(stderr, "spline_trace_test: %d check(s) failed\n", failures);
        return 1;
    }

    std::printf("spline_trace_test: all checks passed\n");
    return 0;
}

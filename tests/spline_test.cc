// Checks for the cubic B-spline helper (src/tracing/spline.hh), built up one step at a time.
//
// Uses CHECK instead of assert: the project builds in Release by default, which removes asserts.

#include "tracing/spline.hh"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <iostream>

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

static constexpr std::size_t k = 20;  // number of points

int main()
{
    // Step 1: the layout for k control points
    {
        cricket::CubicBSpline spline;
        spline.control_points.assign(k, Eigen::Vector3d::Zero());
        spline.knots.assign(k + 4, 0.0);

        CHECK(spline.size() == 20);
        CHECK(spline.knots.size() == 24);
        CHECK(spline.segments() == 17);
    }

    // Step 2: basis functions. Uniform knots 0, 1, ..., 9; span i = 4 is [4, 5), and `&knots[4]` is its start.
    {
        std::vector<double> knots;
        for (int i = 0; i < 10; ++i)
        {
            knots.push_back(i);
        }

        // At the start of the span the uniform cubic basis is (1, 4, 1, 0) / 6
        double basis[4];
        cricket::bspline_basis<double, 3>(&knots[4], 4.0, basis);
        CHECK_NEAR(basis[0], 1.0 / 6.0, 1e-12);
        CHECK_NEAR(basis[1], 4.0 / 6.0, 1e-12);
        CHECK_NEAR(basis[2], 1.0 / 6.0, 1e-12);
        CHECK_NEAR(basis[3], 0.0, 1e-12);

        // Halfway: (1 - u)^3 / 6, (3u^3 - 6u^2 + 4) / 6, (-3u^3 + 3u^2 + 3u + 1) / 6, u^3 / 6 at u = 0.5
        cricket::bspline_basis<double, 3>(&knots[4], 4.5, basis);
        CHECK_NEAR(basis[0], 0.125 / 6.0, 1e-12);
        CHECK_NEAR(basis[1], 2.875 / 6.0, 1e-12);
        CHECK_NEAR(basis[2], 2.875 / 6.0, 1e-12);
        CHECK_NEAR(basis[3], 0.125 / 6.0, 1e-12);

        // Anywhere in the span: non-negative, sum to one
        for (int s = 0; s < 100; ++s)
        {
            const double t = 4.0 + s / 100.0;
            cricket::bspline_basis<double, 3>(&knots[4], t, basis);
            CHECK_NEAR(basis[0] + basis[1] + basis[2] + basis[3], 1.0, 1e-12);
            for (int j = 0; j < 4; ++j)
            {
                CHECK(basis[j] >= 0.0);
            }
        }
    }

    // Step 3: find_span and value. Six control points, clamped knots 0 0 0 0 1/3 2/3 1 1 1 1: three pieces.
    {
        cricket::CubicBSpline spline;
        spline.knots = {0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0};
        spline.control_points = {
            Eigen::Vector3d(0.0, 0.0, 0.0),
            Eigen::Vector3d(1.0, 2.0, 0.0),
            Eigen::Vector3d(2.0, -1.0, 1.0),
            Eigen::Vector3d(3.0, 1.0, 2.0),
            Eigen::Vector3d(4.0, 0.0, 1.0),
            Eigen::Vector3d(5.0, 3.0, 0.0)};

        // Spans: [0, 1/3) is span 3, [1/3, 2/3) is 4, [2/3, 1) is 5; t = 1 belongs to the last
        CHECK(spline.find_span(0.0) == 3);
        CHECK(spline.find_span(0.2) == 3);
        CHECK(spline.find_span(1.0 / 3.0) == 4);
        CHECK(spline.find_span(0.5) == 4);
        CHECK(spline.find_span(0.7) == 5);
        CHECK(spline.find_span(1.0) == 5);

        // Clamped ends: the curve starts and ends at the first and last control point
        CHECK_NEAR((spline.value(0.0) - spline.control_points.front()).norm(), 0.0, 1e-12);
        CHECK_NEAR((spline.value(1.0) - spline.control_points.back()).norm(), 0.0, 1e-12);

        // t outside [0, 1] is clamped
        CHECK_NEAR((spline.value(-0.5) - spline.value(0.0)).norm(), 0.0, 1e-12);
        CHECK_NEAR((spline.value(1.5) - spline.value(1.0)).norm(), 0.0, 1e-12);

        // Linear precision: control points on the line (g, 2g, 0), where g are the knot averages
        // (U[i+1] + U[i+2] + U[i+3]) / 3 (Greville abscissae), reproduce that line exactly as (t, 2t, 0)
        cricket::CubicBSpline line;
        line.knots = spline.knots;
        const double greville[6] = {0.0, 1.0 / 9.0, 1.0 / 3.0, 2.0 / 3.0, 8.0 / 9.0, 1.0};
        for (int i = 0; i < 6; ++i)
        {
            line.control_points.emplace_back(greville[i], 2.0 * greville[i], 0.0);
        }

        for (int s = 0; s <= 100; ++s)
        {
            const double t = s / 100.0;
            CHECK_NEAR((line.value(t) - Eigen::Vector3d(t, 2.0 * t, 0.0)).norm(), 0.0, 1e-12);
        }

        // Every point of the curve stays within the bounding box of the control points (convex hull)
        for (int s = 0; s <= 100; ++s)
        {
            const Eigen::Vector3d point = spline.value(s / 100.0);
            CHECK(point.x() >= -1e-12 and point.x() <= 5.0 + 1e-12);
            CHECK(point.y() >= -1.0 - 1e-12 and point.y() <= 3.0 + 1e-12);
            CHECK(point.z() >= -1e-12 and point.z() <= 2.0 + 1e-12);
        }
    }

    // Step 4: fit a spline through k points on a helix with uneven spacing
    {
        std::vector<Eigen::Vector3d> points;
        for (std::size_t i = 0; i < k; ++i)
        {
            const double s = static_cast<double>(i) / static_cast<double>(k - 1);
            const double angle = 6.28318530718 * std::pow(s, 1.3);
            points.emplace_back(0.5 * std::cos(angle) + 0.6, 0.5 * std::sin(angle), 0.2 + 0.4 * s);
        }

        const auto spline = cricket::fit_cubic_bspline(points);

        // Layout: k control points, k + 4 knots, k - 3 pieces
        CHECK(spline.size() == k);
        CHECK(spline.knots.size() == k + 4);
        CHECK(spline.segments() == k - 3);
        CHECK(spline.parameters.size() == k);

        // Parameters run from 0 to 1, strictly increasing, and are not uniform (the spacing is uneven)
        CHECK(spline.parameters.front() == 0.0);
        CHECK(spline.parameters.back() == 1.0);
        for (std::size_t i = 1; i < k; ++i)
        {
            CHECK(spline.parameters[i] > spline.parameters[i - 1]);
        }

        CHECK(std::abs(spline.parameters[1] - 1.0 / static_cast<double>(k - 1)) > 1e-3);

        // Clamped knots: four 0s, strictly increasing interior, four 1s
        for (int i = 0; i < 4; ++i)
        {
            CHECK(spline.knots[i] == 0.0);
            CHECK(spline.knots[k + i] == 1.0);
        }

        for (std::size_t i = 3; i < k; ++i)
        {
            CHECK(spline.knots[i] < spline.knots[i + 1]);
        }

        // The curve passes through every input point, at that point's parameter
        for (std::size_t i = 0; i < k; ++i)
        {
            CHECK_NEAR((spline.value(spline.parameters[i]) - points[i]).norm(), 0.0, 1e-9);
        }

        // The end control points are the first and last point (clamped), the others are not the waypoints
        CHECK_NEAR((spline.control_points.front() - points.front()).norm(), 0.0, 1e-12);
        CHECK_NEAR((spline.control_points.back() - points.back()).norm(), 0.0, 1e-12);
        CHECK((spline.control_points[10] - points[10]).norm() > 1e-3);

        // Fewer than four points cannot define a cubic
        bool threw = false;
        try
        {
            cricket::fit_cubic_bspline(std::vector<Eigen::Vector3d>(points.begin(), points.begin() + 3));
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }

        CHECK(threw);
    }

    std::cout << "spline_test: " << failures << " check(s) failed\n";

    // Trace input layout for N = 20 control points and a 7-joint robot:
    // q at 0..6, t at 7, control points at 8..67, knots at 68..91, 92 inputs in all
    {
        using Layout = cricket::SplineInputLayout<20>;
        constexpr std::size_t nq = 7;

        static_assert(Layout::control_points == 20);
        static_assert(Layout::knots == 24);
        static_assert(Layout::pieces == 17);
        static_assert(Layout::spline_inputs == 85);

        CHECK(Layout::t_idx(nq) == 7);
        CHECK(Layout::control_point_idx(nq, 0, 0) == 8);
        CHECK(Layout::control_point_idx(nq, 0, 2) == 10);
        CHECK(Layout::control_point_idx(nq, 1, 0) == 11);
        CHECK(Layout::control_point_idx(nq, 19, 2) == 67);
        CHECK(Layout::knot_idx(nq, 0) == 68);
        CHECK(Layout::knot_idx(nq, 23) == 91);
        CHECK(Layout::total(nq) == 92);

        // Every input has exactly one slot: the indices of q, t, control points and knots tile 0 .. total - 1
        std::vector<int> seen(Layout::total(nq), 0);
        for (std::size_t j = 0; j < nq; ++j)
        {
            ++seen[j];
        }

        ++seen[Layout::t_idx(nq)];
        for (std::size_t p = 0; p < Layout::control_points; ++p)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                ++seen[Layout::control_point_idx(nq, p, axis)];
            }
        }

        for (std::size_t i = 0; i < Layout::knots; ++i)
        {
            ++seen[Layout::knot_idx(nq, i)];
        }

        for (const auto count : seen)
        {
            CHECK(count == 1);
        }
    }

    if (failures != 0)
    {
        std::fprintf(stderr, "spline_test: %d check(s) failed\n", failures);
        return 1;
    }

    std::printf("spline_test: all checks passed\n");
    return 0;
}

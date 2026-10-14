// Checks for the cubic B-spline helper (src/tracing/spline.hh), built up one step at a time.
//
// Uses CHECK instead of assert: the project builds in Release by default, which removes asserts.

#include "tracing/spline.hh"

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

    if (failures != 0)
    {
        std::fprintf(stderr, "spline_test: %d check(s) failed\n", failures);
        return 1;
    }

    std::printf("spline_test: all checks passed\n");
    return 0;
}

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

    if (failures != 0)
    {
        std::fprintf(stderr, "spline_test: %d check(s) failed\n", failures);
        return 1;
    }

    std::printf("spline_test: all checks passed\n");
    return 0;
}

#pragma once

#include <cppad/cppad.hpp>

#include <Eigen/Dense>

#include <array>
#include <cstddef>

// The Catmull-Rom curve of the fixed-parameter constraint, written for any scalar type: the traced kernel
// instantiates it with the CppAD code-generation scalar, and tests can use double.
//
// The curve passes through N points P_0 .. P_{N-1} at the parameters tau_0 < ... < tau_{N-1} (tau_0 = 0, tau_{N-1} = 1
// for a curve on t in [0, 1]); between P_i and P_{i+1} it is the cubic through P_{i-1} .. P_{i+2}, evaluated by the
// Barry-Goldman pyramid, which takes any knot spacing (uniform, chordal, centripetal). The ends need a neighbour
// outside the curve, so the data is extended by one phantom point on each side: the runtime hands the kernel the
// N + 2 extended points E_k = P_{k - 1} and their extended knots T_k = tau_{k - 1} (E_0, E_{N + 1} and T_0, T_{N + 1}
// the phantoms), which keeps the kernel independent of how they were made.
namespace cricket
{
    // Order of the inputs of the traced Catmull-Rom kernel, for a curve through N points (N is fixed when the kernel
    // is traced). `nq` is the robot's number of joints.
    //
    //   [ q (nq) | t (1) | extended points (3 (N + 2), point-major: x y z of E_0, then E_1, ...) | extended knots (N + 2) ]
    //
    // The parameter t is an input but never differentiated: the kernel is for projections that hold t fixed.
    template <std::size_t N>
    struct CatmullRomInputLayout
    {
        static_assert(N >= 2, "a Catmull-Rom curve needs at least two points");

        static constexpr std::size_t points = N + 2;
        static constexpr std::size_t knots = N + 2;
        static constexpr std::size_t segments = N - 1;

        static constexpr std::size_t curve_inputs = 1 + 3 * points + knots;

        static constexpr auto t_idx(std::size_t nq) -> std::size_t
        {
            return nq;
        }

        static constexpr auto point_idx(std::size_t nq, std::size_t point, std::size_t axis) -> std::size_t
        {
            return nq + 1 + 3 * point + axis;
        }

        static constexpr auto knot_idx(std::size_t nq, std::size_t index) -> std::size_t
        {
            return nq + 1 + 3 * points + index;
        }

        static constexpr auto total(std::size_t nq) -> std::size_t
        {
            return nq + curve_inputs;
        }
    };

    template <typename Scalar, std::size_t N>
    struct CatmullRomInputs
    {
        Scalar t;
        std::array<Eigen::Matrix<Scalar, 3, 1>, CatmullRomInputLayout<N>::points> points;
        std::array<Scalar, CatmullRomInputLayout<N>::knots> knots;
    };

    // Read the curve inputs off an input vector laid out as CatmullRomInputLayout<N> says; `nq` is the number of
    // joints in front of them.
    template <typename Scalar, std::size_t N, typename Inputs>
    auto read_catmull_rom(const Inputs &inp, std::size_t nq) -> CatmullRomInputs<Scalar, N>
    {
        using Layout = CatmullRomInputLayout<N>;

        CatmullRomInputs<Scalar, N> curve;
        curve.t = inp[Layout::t_idx(nq)];

        for (std::size_t point = 0; point < Layout::points; ++point)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                curve.points[point][axis] = inp[Layout::point_idx(nq, point, axis)];
            }
        }

        for (std::size_t i = 0; i < Layout::knots; ++i)
        {
            curve.knots[i] = inp[Layout::knot_idx(nq, i)];
        }

        return curve;
    }

    // The four points and knots that matter at one parameter: for the segment between P_i and P_{i + 1}, the extended
    // points E_i .. E_{i + 3} (= P_{i - 1} .. P_{i + 2}) and knots T_i .. T_{i + 3}; the segment covers
    // t in [T_{i + 1}, T_{i + 2}].
    template <typename Scalar>
    struct CatmullRomWindow
    {
        std::array<Eigen::Matrix<Scalar, 3, 1>, 4> points;
        std::array<Scalar, 4> knots;
    };

    // Pick the window of the segment containing the traced parameter t, by conditionals as for the B-spline: for each
    // segment a CondExp picks that segment's window when t lies before the segment's end T_{i + 2}, and otherwise keeps
    // the choice made for the later segments. Segments are visited from the last to the first, so the result is the
    // first segment whose end is after t, and t = 1 stays in the last one. The selected values do not depend on t.
    template <typename Scalar, std::size_t N>
    auto select_catmull_rom_window(const CatmullRomInputs<Scalar, N> &curve) -> CatmullRomWindow<Scalar>
    {
        CatmullRomWindow<Scalar> window;

        // start from the last segment, i = N - 2
        for (std::size_t j = 0; j < 4; ++j)
        {
            window.points[j] = curve.points[N - 2 + j];
            window.knots[j] = curve.knots[N - 2 + j];
        }

        // segments N - 3 down to 0
        for (std::size_t k = 1; k + 1 < N; ++k)
        {
            const std::size_t segment = N - 2 - k;
            const Scalar &segment_end = curve.knots[segment + 2];

            for (std::size_t j = 0; j < 4; ++j)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    window.points[j][axis] =
                        CppAD::CondExpLt(curve.t, segment_end, curve.points[segment + j][axis], window.points[j][axis]);
                }

                window.knots[j] = CppAD::CondExpLt(curve.t, segment_end, curve.knots[segment + j], window.knots[j]);
            }
        }

        return window;
    }

    // Point on the curve at the parameter t (Barry-Goldman): linear blends of neighbouring points, then of those, then
    // of those. Divides by differences of the window's knots, which are positive for strictly increasing knots.
    template <typename Scalar, std::size_t N>
    auto catmull_rom_point(const CatmullRomInputs<Scalar, N> &curve) -> Eigen::Matrix<Scalar, 3, 1>
    {
        using Vector = Eigen::Matrix<Scalar, 3, 1>;

        const auto window = select_catmull_rom_window(curve);
        const auto &P = window.points;
        const auto &T = window.knots;
        const Scalar &t = curve.t;

        // the point at t on the line through a (at ta) and b (at tb)
        const auto blend = [&t](const Vector &a, const Vector &b, const Scalar &ta, const Scalar &tb) -> Vector
        { return ((tb - t) * a + (t - ta) * b) / (tb - ta); };

        const Vector a1 = blend(P[0], P[1], T[0], T[1]);
        const Vector a2 = blend(P[1], P[2], T[1], T[2]);
        const Vector a3 = blend(P[2], P[3], T[2], T[3]);

        const Vector b1 = blend(a1, a2, T[0], T[2]);
        const Vector b2 = blend(a2, a3, T[1], T[3]);

        return blend(b1, b2, T[1], T[2]);
    }
}  // namespace cricket

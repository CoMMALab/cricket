#pragma once

#include "se3_ops.hh"

#include <cppad/cppad.hpp>

#include <Eigen/Dense>

#include <array>
#include <cstddef>

// The straight-line (lerp) curves of the fixed-parameter constraints, written for any scalar type: the traced kernels
// instantiate them with the CppAD code-generation scalar, and tests can use double.
//
// The curve passes through N keyframes at the parameters tau_0 < ... < tau_{N-1} (tau_0 = 0, tau_{N-1} = 1), the
// parameters handed over by the runtime (they are the normalised chord lengths of the keyframes, so they depend on the
// rotation weight of the pose curve, which the runtime knows). Between keyframe i and i + 1, with
// s = (t - tau_i) / (tau_{i + 1} - tau_i):
//
//   R^3          P(t) = p_i + s (p_{i + 1} - p_i)
//   SO(3) x R^3  the same position, and the rotation R(t) = R_i exp(s log(R_i^T R_{i + 1})), the slerp
//
// The segment is picked by conditionals (SIMD blends in the block language), so every lane of a vector takes its own.
namespace cricket
{
    // Order of the inputs of the traced R^3 kernel, for a curve through N points (N is fixed when the kernel is traced).
    // `nq` is the robot's number of joints.
    //
    //   [ q (nq) | t (1) | points (3 N, x y z of point 0, then point 1, ...) | taus (N) ]
    //
    // The parameter t is an input but never differentiated: the kernel is for projections that hold t fixed.
    template <std::size_t N>
    struct LerpR3InputLayout
    {
        static_assert(N >= 2, "a lerp curve needs at least two points");

        static constexpr std::size_t curve_inputs = 1 + 3 * N + N;

        static constexpr auto t_idx(std::size_t nq) -> std::size_t
        {
            return nq;
        }

        static constexpr auto point_idx(std::size_t nq, std::size_t point, std::size_t axis) -> std::size_t
        {
            return nq + 1 + 3 * point + axis;
        }

        static constexpr auto tau_idx(std::size_t nq, std::size_t index) -> std::size_t
        {
            return nq + 1 + 3 * N + index;
        }

        static constexpr auto total(std::size_t nq) -> std::size_t
        {
            return nq + curve_inputs;
        }
    };

    // Order of the inputs of the traced SO(3) x R^3 kernel:
    //
    //   [ q (nq) | t (1) | weight (1) | positions (3 N) | quaternions (4 N, w x y z of keyframe 0, then 1, ...) | taus (N) ]
    //
    // The weight (m / rad) scales the rotation part of the error against the translation part. t and the curve data are
    // inputs but never differentiated.
    template <std::size_t N>
    struct LerpPoseInputLayout
    {
        static_assert(N >= 2, "a lerp curve needs at least two keyframes");

        static constexpr std::size_t curve_inputs = 2 + 3 * N + 4 * N + N;

        static constexpr auto t_idx(std::size_t nq) -> std::size_t
        {
            return nq;
        }

        static constexpr auto weight_idx(std::size_t nq) -> std::size_t
        {
            return nq + 1;
        }

        static constexpr auto position_idx(std::size_t nq, std::size_t keyframe, std::size_t axis) -> std::size_t
        {
            return nq + 2 + 3 * keyframe + axis;
        }

        static constexpr auto quaternion_idx(std::size_t nq, std::size_t keyframe, std::size_t component) -> std::size_t
        {
            return nq + 2 + 3 * N + 4 * keyframe + component;
        }

        static constexpr auto tau_idx(std::size_t nq, std::size_t index) -> std::size_t
        {
            return nq + 2 + 7 * N + index;
        }

        static constexpr auto total(std::size_t nq) -> std::size_t
        {
            return nq + curve_inputs;
        }
    };

    // The value at(segment) of the segment that contains t, by conditionals: from the last segment to the first, each
    // CondExp takes that segment's value when t is before the segment's end tau_{segment + 1} and otherwise keeps the
    // choice made for the later segments. The result is the first segment whose end is after t, and t = 1 stays in the
    // last one. The values `at` returns do not depend on t.
    template <typename Scalar, std::size_t N, typename At>
    auto select_segment(const Scalar &t, const std::array<Scalar, N> &taus, At at) -> Scalar
    {
        Scalar selected = at(N - 2);
        for (std::size_t k = 1; k + 1 < N; ++k)
        {
            const std::size_t segment = N - 2 - k;
            selected = CppAD::CondExpLt(t, taus[segment + 1], at(segment), selected);
        }

        return selected;
    }

    // Point of the R^3 curve at t
    template <typename Scalar, std::size_t N>
    auto lerp_r3_point(
        const Scalar &t,
        const std::array<Eigen::Matrix<Scalar, 3, 1>, N> &points,
        const std::array<Scalar, N> &taus) -> Eigen::Matrix<Scalar, 3, 1>
    {
        const Scalar start = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return taus[i]; });
        const Scalar end = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return taus[i + 1]; });
        const Scalar s = (t - start) / (end - start);

        Eigen::Matrix<Scalar, 3, 1> point;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const Scalar from = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return points[i][axis]; });
            const Scalar to = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return points[i + 1][axis]; });
            point[axis] = from + s * (to - from);
        }

        return point;
    }

    template <typename Scalar>
    struct LerpPose
    {
        Eigen::Matrix<Scalar, 3, 1> position;
        Eigen::Matrix<Scalar, 3, 3> rotation;
    };

    // Rotation matrix of a unit quaternion given as w x y z
    template <typename Scalar>
    auto quaternion_matrix(const std::array<Scalar, 4> &q) -> Eigen::Matrix<Scalar, 3, 3>
    {
        return Eigen::Quaternion<Scalar>(q[0], q[1], q[2], q[3]).toRotationMatrix();
    }

    // Pose of the SO(3) x R^3 curve at t: the position is lerped and the rotation slerped, R_i exp(s log(R_i^T
    // R_{i + 1})), from the relative rotation (so the short way round, and no division by sin(theta)). The exponential
    // is Rodrigues' formula with the smooth sinc of se3_ops.hh, and the rotation angle carries a small floor, so the
    // derivative is finite through a zero relative rotation.
    template <typename Scalar, std::size_t N>
    auto lerp_pose(
        const Scalar &t,
        const std::array<Eigen::Matrix<Scalar, 3, 1>, N> &positions,
        const std::array<std::array<Scalar, 4>, N> &quaternions,
        const std::array<Scalar, N> &taus) -> LerpPose<Scalar>
    {
        using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
        using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
        using DynamicMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

        const Scalar start = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return taus[i]; });
        const Scalar end = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return taus[i + 1]; });
        const Scalar s = (t - start) / (end - start);

        Vector3 position;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const Scalar from = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return positions[i][axis]; });
            const Scalar to = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return positions[i + 1][axis]; });
            position[axis] = from + s * (to - from);
        }

        std::array<Scalar, 4> from_quaternion;
        std::array<Scalar, 4> to_quaternion;
        for (std::size_t c = 0; c < 4; ++c)
        {
            from_quaternion[c] = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return quaternions[i][c]; });
            to_quaternion[c] = select_segment<Scalar, N>(t, taus, [&](std::size_t i) { return quaternions[i + 1][c]; });
        }

        const Matrix3 from_rotation = quaternion_matrix<Scalar>(from_quaternion);
        const Matrix3 to_rotation = quaternion_matrix<Scalar>(to_quaternion);

        const DynamicMatrix relative = from_rotation.transpose() * to_rotation;
        const DynamicMatrix log = so3_log_smooth<DynamicMatrix, Scalar>(relative);

        const Vector3 rotation_vector = s * Vector3(log(0, 0), log(1, 0), log(2, 0));
        const Scalar theta = sqrt(rotation_vector.squaredNorm() + se3_detail::SIN_FLOOR);
        const Scalar sinc = sinc_smooth<Scalar>(theta);
        const Scalar half_sinc = sinc_smooth<Scalar>(0.5 * theta);

        Matrix3 hat;
        hat << Scalar(0.0), -rotation_vector[2], rotation_vector[1],  //
            rotation_vector[2], Scalar(0.0), -rotation_vector[0],     //
            -rotation_vector[1], rotation_vector[0], Scalar(0.0);

        // exp(v) = I + sin(theta) / theta [v] + (1 - cos(theta)) / theta^2 [v]^2, the last factor being sinc(theta / 2)^2 / 2
        const Matrix3 exponential = Matrix3::Identity() + sinc * hat + (0.5 * half_sinc * half_sinc) * (hat * hat);

        return {position, from_rotation * exponential};
    }
}  // namespace cricket

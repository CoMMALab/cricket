#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>


// define a macro constant for ang
#define ang ((180.0 - 2.0 * 68.0) * M_PI / 180.0)

inline double CondExpGe(double x, double y, double true_val, double false_val)
{
    return (x >= y) ? true_val : false_val;
}
inline double CondExpLe(double x, double y, double true_val, double false_val)
{
    return (x <= y) ? true_val : false_val;
}


template <typename T>
Eigen::Matrix4<T> ComputeDHMatrix(const T &ti, T ai, T di)
{
    T ct = cos(ti);
    T st = sin(ti);
    T ca = cos(ai);
    T sa = sin(ai);

    Eigen::Matrix4<T> mat;
    mat << ct, -st * ca, st * sa, static_cast<T>(0), st, ct * ca, -ct * sa, static_cast<T>(0), static_cast<T>(0), sa, ca, di, static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(1);
    return mat;
}

template <typename T>
Eigen::Matrix3<T> CrossProductMatrix(const Eigen::Matrix<T, 3, 1> &a)
{
    Eigen::Matrix3<T> A;
    A << (T)0, -a(2), a(1), a(2), (T)0, -a(0), -a(1), a(0), (T)0;
    return A;
}

template <typename T, typename U>
T ScalarClip(const T &val, U a, U b)
{
    // return std::max(static_cast<T>(a), std::min(static_cast<T>(b), val));
    // replace the min max with CondGe and CondLe to make it differentiable
    T clipped_val = CondExpGe(val, static_cast<T>(a), val, static_cast<T>(a));
    clipped_val = CondExpLe(clipped_val, static_cast<T>(b), clipped_val, static_cast<T>(b));

    return clipped_val;
}

template <typename T, typename U>
T SafeArccos(const T &val, U a, U b)
{
    return acos(ScalarClip<T,U>(val, a, b));
}

// Smooth (C1) exterior penalty for a SafeArccos argument falling outside
// [-clip, clip]: zero inside the band, grows quadratically outside, with
// zero value and zero derivative exactly at the boundary. Used to build a
// differentiable loss for optimizing psi against the self-motion-manifold
// constraints, in place of (or alongside) the hard reject that SafeArccos's
// silent clipping otherwise requires.
template <typename T, typename U>
T HingeSqPenalty(const T &val, U clip)
{
    T over = CondExpGe(val - static_cast<T>(clip), static_cast<T>(0), val - static_cast<T>(clip), static_cast<T>(0));
    T under = CondExpGe(-val - static_cast<T>(clip), static_cast<T>(0), -val - static_cast<T>(clip), static_cast<T>(0));
    return over * over + under * under;
}

// Result of a parameterization: the joint configuration `q`, plus the 4
// pre-clip arguments that were fed to SafeArccos (in call order: phi,
// theta_4v, shoulder q_subordinate(1), wrist q_subordinate(5)). A value
// outside [-1, 1] here means the requested pose/psi/GC combination has no
// valid IK solution on this branch -- SafeArccos silently clips it instead
// of failing, so callers must check `unclipped` themselves to detect and
// reject those cases.
template <typename T>
struct IKParamResult
{
    Eigen::VectorX<T> q;
    Eigen::Vector4<T> unclipped;
    T loss{};
};

template <typename T, typename InputVector>
auto IiwaBimanualParameterization(
    InputVector &ad_inp
) {
    // now distribute the input values into different variables
    auto q_and_psi = ad_inp.segment(0, 8);
    // const auto shoulder_up = CondExpGt(ad_inp[8], 0.5, ADCG(1.0), ADCG(-1.0));
    // const auto elbow_up = CondExpGt(ad_inp[9], 0.5, ADCG(1.0), ADCG(-1.0));
    // const auto wrist_up = CondExpGt(ad_inp[10], 0.5, ADCG(1.0), ADCG(-1.0));
    const T rel_x = ad_inp[11];
    const T rel_y = ad_inp[12];
    const T rel_z = ad_inp[13];
    const T rel_qx = ad_inp[14];
    const T rel_qy = ad_inp[15];
    const T rel_qz = ad_inp[16];
    const T rel_qw = ad_inp[17];

    // It is
    const auto GC2 = ad_inp[8];
    const auto GC4 = ad_inp[9];
    const auto GC6 = ad_inp[10];

    // it is assumed that q_and_psi is of size (8)

    const Eigen::VectorX<T> q_controlled = q_and_psi.head(7);
    const T psi = q_and_psi.tail(1)[0];

    Eigen::VectorX<T> q_subordinate(7);
    Eigen::VectorX<T> q_full(14);
    q_full.head(7) = q_controlled;

    Eigen::Vector4<T> unclipped;

    // iiwa kinematic parameters.
    Eigen::VectorX<T> iiwa_alpha(7);
    iiwa_alpha(0) = -M_PI_2;
    iiwa_alpha(1) = M_PI_2;
    iiwa_alpha(2) = M_PI_2;
    iiwa_alpha(3) = -M_PI_2;
    iiwa_alpha(4) = -M_PI_2;
    iiwa_alpha(5) = M_PI_2;
    iiwa_alpha(6) = 0.0;
    // iiwa_alpha << -M_PI_2, M_PI_2, M_PI_2, -M_PI_2, -M_PI_2, M_PI_2, 0.0;
    Eigen::VectorX<T> iiwa_d(7);
    iiwa_d(0) = 0.36;
    iiwa_d(1) = 0.0;
    iiwa_d(2) = 0.42;
    iiwa_d(3) = 0.0;
    iiwa_d(4) = 0.4;
    iiwa_d(5) = 0.0;
    iiwa_d(6) = 0.126 - 0.045;

    const T d_bs = iiwa_d[0];
    const T d_se = iiwa_d[2];
    const T d_ew = iiwa_d[4];
    const T d_wf = iiwa_d[6];

    const Eigen::Vector3d base_translation(0, -0.765, 0);
    const T clip = (T) (1.0 - 1e-4);

    // Forward kinematics.
    Eigen::Matrix4<T> tf_goal = ComputeDHMatrix(q_controlled[0], iiwa_alpha[0], iiwa_d[0]);
    for (int i = 1; i < 7; ++i)
    {
        tf_goal = tf_goal * ComputeDHMatrix(q_controlled[i], iiwa_alpha[i], iiwa_d[i]);
    }
    std::cout << "FK result: " << std::endl;
    std::cout << tf_goal << std::endl;

    // Compute rotation matrix from relative quaternion (qx, qy, qz, qw)
    const T q_norm = sqrt(rel_qx * rel_qx + rel_qy * rel_qy + rel_qz * rel_qz + rel_qw * rel_qw + static_cast<T>(1e-12));
    const T qx = rel_qx / q_norm;
    const T qy = rel_qy / q_norm;
    const T qz = rel_qz / q_norm;
    const T qw = rel_qw / q_norm;

    Eigen::Matrix3<T> R_rel;
    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);
    R_rel << one - two * (qy * qy + qz * qz), two * (qx * qy - qw * qz),       two * (qx * qz + qw * qy),
             two * (qx * qy + qw * qz),       one - two * (qx * qx + qz * qz), two * (qy * qz - qw * qx),
             two * (qx * qz - qw * qy),       two * (qy * qz + qw * qx),       one - two * (qx * qx + qy * qy);

    // Apply the relative transform and add base translation to position
    Eigen::Matrix3<T> R_fk = tf_goal.template block<3, 3>(0, 0);
    Eigen::Matrix<T, 3, 1> p_fk = tf_goal.template block<3, 1>(0, 3);

    tf_goal.template block<3, 3>(0, 0) = R_fk * R_rel;
    tf_goal.template block<3, 1>(0, 3) = p_fk + R_fk * Eigen::Matrix<T, 3, 1>(rel_x, rel_y, rel_z);

    tf_goal.template block<3, 1>(0, 3) += base_translation.cast<T>();
    std::cout << "Transformed FK result: " << std::endl;
    std::cout << tf_goal << std::endl;

    // Do the IK!!!!
    Eigen::Matrix<T, 3, 1> p_02((T)0.0, (T)0.0, (T)d_bs);
    Eigen::Matrix<T, 3, 1> p_24((T)0.0, (T)d_se, (T)0.0);
    Eigen::Matrix<T, 3, 1> p_46((T)0.0, (T)0.0, (T)d_ew);
    Eigen::Matrix<T, 3, 1> p_67((T)0.0, (T)0.0, (T)d_wf);

    Eigen::Matrix<T, 3, 1> p_07 = tf_goal.template block<3, 1>(0, 3);
    Eigen::Matrix3<T> R_07 = tf_goal.template block<3, 3>(0, 0);

    // EQ (3)
    Eigen::Matrix<T, 3, 1> p_26 = p_07 - p_02 - R_07 * p_67;
    // normalized has branching statements, so we write a custom version of it that is differentiable.
    Eigen::Matrix<T, 3, 1> p_26_hat = p_26 / (p_26.norm() + static_cast<T>(1e-8));

    // EQ (5)
    T theta_1v = atan2(p_26(1), p_26(0));

    // EQ (7)
    T p_26_norm = p_26.norm();
    T p_26_dot = p_26.dot(p_26);  // = ||p_26||²

    T arccos_in = (d_se * d_se + p_26_dot - d_ew * d_ew) / (2.0 * d_se * p_26_norm);
    unclipped(0) = arccos_in;

    T phi = SafeArccos(arccos_in, -clip, clip);
    T theta_2v = atan2(p_26.template head<2>().norm(), p_26(2)) + GC4 * phi;

    T theta_3v = T(0);  // This joint is fixed

    // EQ (4)
    arccos_in = (p_26_dot - d_se * d_se - d_ew * d_ew) / (2.0 * d_se * d_ew);
    unclipped(1) = arccos_in;

    T theta_4v = GC4 * SafeArccos(arccos_in, -clip, clip);
    q_subordinate[3] = theta_4v;

    // Build list of transforms T_01, T_12, T_23 using Ts[0], Ts[1], Ts[2] and
    // theta_[0:3]
    std::vector<Eigen::Matrix4<T>> T_vs;
    T_vs.push_back(ComputeDHMatrix(theta_1v, iiwa_alpha(0), iiwa_d(0)));
    T_vs.push_back(ComputeDHMatrix(theta_2v, iiwa_alpha(1), iiwa_d(1)));
    T_vs.push_back(ComputeDHMatrix(theta_3v, iiwa_alpha(2), iiwa_d(2)));

    Eigen::Matrix4<T> T_03_v = T_vs[0] * T_vs[1] * T_vs[2];
    Eigen::Matrix3<T> R_03_v = T_03_v.template block<3, 3>(0, 0);

    // EQ (15)
    Eigen::Matrix3<T> cprod_p_26 = CrossProductMatrix(p_26_hat);
    Eigen::Matrix3<T> A_s = cprod_p_26 * R_03_v;
    Eigen::Matrix3<T> B_s = -cprod_p_26 * cprod_p_26 * R_03_v;
    Eigen::Matrix3<T> C_s = p_26_hat * p_26_hat.transpose() * R_03_v;

    // EQ (17)-(19)
    q_subordinate(0) = atan2(
        GC2 * (A_s(1, 1) * sin(psi) + B_s(1, 1) * cos(psi) + C_s(1, 1)),
        GC2 * (A_s(0, 1) * sin(psi) + B_s(0, 1) * cos(psi) + C_s(0, 1)));

    arccos_in = A_s(2, 1) * sin(psi) + B_s(2, 1) * cos(psi) + C_s(2, 1);
    unclipped(2) = arccos_in;
    q_subordinate(1) = GC2 * SafeArccos(arccos_in, -clip, clip);

    q_subordinate(2) = atan2(
        GC2 * (-A_s(2, 2) * sin(psi) - B_s(2, 2) * cos(psi) - C_s(2, 2)),
        GC2 * (-A_s(2, 0) * sin(psi) - B_s(2, 0) * cos(psi) - C_s(2, 0)));

    // EQ (20)
    Eigen::Matrix4<T> T_34 = ComputeDHMatrix(theta_4v, iiwa_alpha(3), iiwa_d(3));
    Eigen::Matrix3<T> R_34 = T_34.template block<3, 3>(0, 0);

    Eigen::Matrix3<T> A_w = R_34.transpose() * A_s.transpose() * R_07;
    Eigen::Matrix3<T> B_w = R_34.transpose() * B_s.transpose() * R_07;
    Eigen::Matrix3<T> C_w = R_34.transpose() * C_s.transpose() * R_07;

    // EQ (22)-(24)
    q_subordinate(4) = atan2(
        GC6 * (A_w(1, 2) * sin(psi) + B_w(1, 2) * cos(psi) + C_w(1, 2)),
        GC6 * (A_w(0, 2) * sin(psi) + B_w(0, 2) * cos(psi) + C_w(0, 2)));

    arccos_in = A_w(2, 2) * sin(psi) + B_w(2, 2) * cos(psi) + C_w(2, 2);
    unclipped(3) = arccos_in;
    q_subordinate(5) = GC6 * SafeArccos(arccos_in, -clip, clip);

    q_subordinate(6) = atan2(
        GC6 * (A_w(2, 1) * sin(psi) + B_w(2, 1) * cos(psi) + C_w(2, 1)),
        GC6 * (-A_w(2, 0) * sin(psi) - B_w(2, 0) * cos(psi) - C_w(2, 0)));

    q_full.tail(7) = q_subordinate;
    return IKParamResult<T>{q_full, unclipped};
}
template <typename T, typename InputVector>
auto IiwaSE3Parameterization(
    InputVector &ad_inp
) {
    /*
    Get the IIWA joint angles from the end-effector pose and a free parameter psi. 
    The input is a vector of size 11, 
        where the first 7 elements are the end-effector pose (x, y, z, qx, qy, qz, qw) and 
        the 8th element is psi. The next 3 elements are the GC2, GC4, and GC6 parameters.
        which are shoulder, elbow, and wrist configuration parameters.
    The output is a vector of size 7 containing the joint angles.
    @param ad_inp: Input vector of size 11, containing the end-effector pose and psi.
    @return q_subordinate: Output vector of size 7, containing the joint angles.
    */
    const T x = ad_inp[0];
    const T y = ad_inp[1];
    const T z = ad_inp[2];
    const T qx = ad_inp[3];
    const T qy = ad_inp[4];
    const T qz = ad_inp[5];
    const T qw = ad_inp[6];
    const T psi = ad_inp[7];
    const T clip = (T) (1.0 - 1e-4);


    Eigen::Matrix4<T> tf_goal;
    // First create a rotation matrix from the quaternion
    // const T q_norm = sqrt(qx * qx + qy * qy + qz * qz + qw * qw + static_cast<T>(1e-12));
    // const T qx = qx / q_norm;
    // const T qy = qy / q_norm;
    // const T qz = qz / q_norm;
    // const T qw = qw / q_norm;

    Eigen::Matrix3<T> R;
    const T one = static_cast<T>(1);
    const T two = static_cast<T>(2);
    R << one - two * (qy * qy + qz * qz), two * (qx * qy - qw * qz),       two * (qx * qz + qw * qy),
         two * (qx * qy + qw * qz),       one - two * (qx * qx + qz * qz), two * (qy * qz - qw * qx),
         two * (qx * qz - qw * qy),       two * (qy * qz + qw * qx),       one - two * (qx * qx + qy * qy);

    Eigen::Matrix<T, 3, 1> p(x, y, z);
    tf_goal.template block<3, 3>(0, 0) = R;
    tf_goal.template block<3, 1>(0, 3) = p;

    // // It is
    const auto GC2 = ad_inp[8];
    const auto GC4 = ad_inp[9];
    const auto GC6 = ad_inp[10];


    Eigen::VectorX<T> q_subordinate(7);
    Eigen::Vector4<T> unclipped;

    // iiwa kinematic parameters.
    Eigen::VectorX<T> iiwa_alpha(7);
    iiwa_alpha(0) = -M_PI_2;
    iiwa_alpha(1) = M_PI_2;
    iiwa_alpha(2) = M_PI_2;
    iiwa_alpha(3) = -M_PI_2;
    iiwa_alpha(4) = -M_PI_2;
    iiwa_alpha(5) = M_PI_2;
    iiwa_alpha(6) = 0.0;
    // iiwa_alpha << -M_PI_2, M_PI_2, M_PI_2, -M_PI_2, -M_PI_2, M_PI_2, 0.0;
    Eigen::VectorX<T> iiwa_d(7);
    iiwa_d(0) = 0.36;
    iiwa_d(1) = 0.0;
    iiwa_d(2) = 0.42;
    iiwa_d(3) = 0.0;
    iiwa_d(4) = 0.4;
    iiwa_d(5) = 0.0;
    iiwa_d(6) = 0.126 - 0.045;

    const T d_bs = iiwa_d[0];
    const T d_se = iiwa_d[2];
    const T d_ew = iiwa_d[4];
    const T d_wf = iiwa_d[6];


    // Do the IK!!!!
    Eigen::Matrix<T, 3, 1> p_02((T)0.0, (T)0.0, (T)d_bs);
    Eigen::Matrix<T, 3, 1> p_24((T)0.0, (T)d_se, (T)0.0);
    Eigen::Matrix<T, 3, 1> p_46((T)0.0, (T)0.0, (T)d_ew);
    Eigen::Matrix<T, 3, 1> p_67((T)0.0, (T)0.0, (T)d_wf);

    Eigen::Matrix<T, 3, 1> p_07 = tf_goal.template block<3, 1>(0, 3);
    Eigen::Matrix3<T> R_07 = tf_goal.template block<3, 3>(0, 0);

    // EQ (3)
    Eigen::Matrix<T, 3, 1> p_26 = p_07 - p_02 - R_07 * p_67;
    // normalized has branching statements, so we write a custom version of it that is differentiable.
    Eigen::Matrix<T, 3, 1> p_26_hat = p_26 / (p_26.norm() + static_cast<T>(1e-8));

    // EQ (5)
    T theta_1v = atan2(p_26(1), p_26(0));

    // EQ (7)
    T p_26_norm = p_26.norm();
    T p_26_dot = p_26.dot(p_26);  // = ||p_26||²

    T arccos_in = (d_se * d_se + p_26_dot - d_ew * d_ew) / (2.0 * d_se * p_26_norm);
    unclipped(0) = arccos_in;

    T phi = SafeArccos(arccos_in, -clip, clip);
    T theta_2v = atan2(p_26.template head<2>().norm(), p_26(2)) + GC4 * phi;

    T theta_3v = T(0);  // This joint is fixed

    // EQ (4)
    arccos_in = (p_26_dot - d_se * d_se - d_ew * d_ew) / (2.0 * d_se * d_ew);
    unclipped(1) = arccos_in;

    T theta_4v = GC4 * SafeArccos(arccos_in, -clip, clip);
    q_subordinate[3] = theta_4v;

    // Build list of transforms T_01, T_12, T_23 using Ts[0], Ts[1], Ts[2] and
    // theta_[0:3]
    std::vector<Eigen::Matrix4<T>> T_vs;
    T_vs.push_back(ComputeDHMatrix(theta_1v, iiwa_alpha(0), iiwa_d(0)));
    T_vs.push_back(ComputeDHMatrix(theta_2v, iiwa_alpha(1), iiwa_d(1)));
    T_vs.push_back(ComputeDHMatrix(theta_3v, iiwa_alpha(2), iiwa_d(2)));

    Eigen::Matrix4<T> T_03_v = T_vs[0] * T_vs[1] * T_vs[2];
    Eigen::Matrix3<T> R_03_v = T_03_v.template block<3, 3>(0, 0);

    // EQ (15)
    Eigen::Matrix3<T> cprod_p_26 = CrossProductMatrix(p_26_hat);
    Eigen::Matrix3<T> A_s = cprod_p_26 * R_03_v;
    Eigen::Matrix3<T> B_s = -cprod_p_26 * cprod_p_26 * R_03_v;
    Eigen::Matrix3<T> C_s = p_26_hat * p_26_hat.transpose() * R_03_v;

    // EQ (17)-(19)
    q_subordinate(0) = atan2(
        GC2 * (A_s(1, 1) * sin(psi) + B_s(1, 1) * cos(psi) + C_s(1, 1)),
        GC2 * (A_s(0, 1) * sin(psi) + B_s(0, 1) * cos(psi) + C_s(0, 1)));

    arccos_in = A_s(2, 1) * sin(psi) + B_s(2, 1) * cos(psi) + C_s(2, 1);
    unclipped(2) = arccos_in;
    q_subordinate(1) = GC2 * SafeArccos(arccos_in, -clip, clip);

    q_subordinate(2) = atan2(
        GC2 * (-A_s(2, 2) * sin(psi) - B_s(2, 2) * cos(psi) - C_s(2, 2)),
        GC2 * (-A_s(2, 0) * sin(psi) - B_s(2, 0) * cos(psi) - C_s(2, 0)));

    // EQ (20)
    Eigen::Matrix4<T> T_34 = ComputeDHMatrix(theta_4v, iiwa_alpha(3), iiwa_d(3));
    Eigen::Matrix3<T> R_34 = T_34.template block<3, 3>(0, 0);

    Eigen::Matrix3<T> A_w = R_34.transpose() * A_s.transpose() * R_07;
    Eigen::Matrix3<T> B_w = R_34.transpose() * B_s.transpose() * R_07;
    Eigen::Matrix3<T> C_w = R_34.transpose() * C_s.transpose() * R_07;

    // EQ (22)-(24)
    q_subordinate(4) = atan2(
        GC6 * (A_w(1, 2) * sin(psi) + B_w(1, 2) * cos(psi) + C_w(1, 2)),
        GC6 * (A_w(0, 2) * sin(psi) + B_w(0, 2) * cos(psi) + C_w(0, 2)));

    arccos_in = A_w(2, 2) * sin(psi) + B_w(2, 2) * cos(psi) + C_w(2, 2);
    unclipped(3) = arccos_in;
    q_subordinate(5) = GC6 * SafeArccos(arccos_in, -clip, clip);

    q_subordinate(6) = atan2(
        GC6 * (A_w(2, 1) * sin(psi) + B_w(2, 1) * cos(psi) + C_w(2, 1)),
        GC6 * (-A_w(2, 0) * sin(psi) - B_w(2, 0) * cos(psi) - C_w(2, 0)));

    return IKParamResult<T>{q_subordinate, unclipped};
}

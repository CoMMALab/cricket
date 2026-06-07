// TODO (siyer) -- ask tommy
// 1. URDF of the iiwa along with end effector
// 2. Why not sample

#pragma once

#include "housekeeping.hh"
#include "robot_info.hh"
#include "tracer_utils.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

// define a macro constant for ang
#define ang ((180.0 - 2.0 * 68.0) * M_PI / 180.0)

// Using declarations (optional, but clearer)
// using std::atan2;
// using std::cos;
// using std::max;
// using std::min;
// using std::sin;

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
    // A(0, 0) = 0;
    // A(0, 1) = -a(2);
    // A(0, 2) = a(1);
    // A(1, 0) = a(2);
    // A(1, 1) = 0;
    // A(1, 2) = -a(0);
    // A(2, 0) = -a(1);
    // A(2, 1) = a(0);
    // A(2, 2) = 0;
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

template <typename T>
auto IiwaBimanualParameterization(
    const std::string &language,
    bool compute_gradient = false
)
{

    std::cout << "Generating parameterized IK code for iiwa..." << std::endl;
    const size_t num_inp = 8 + 1 + 1 + 1 + 1;

    ADVectorXs ad_inp(num_inp);  // 3 4x4 matrices
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = ADCG(0.0001);
    }
    Independent(ad_inp);
    
    // now distribute the input values into different variables
    ADVectorXs q_and_psi = ad_inp.segment(0, 8);
    // const auto shoulder_up = CondExpGt(ad_inp[8], 0.5, ADCG(1.0), ADCG(-1.0));
    // const auto elbow_up = CondExpGt(ad_inp[9], 0.5, ADCG(1.0), ADCG(-1.0));
    // const auto wrist_up = CondExpGt(ad_inp[10], 0.5, ADCG(1.0), ADCG(-1.0));
    const ADCG grasp_distance = ad_inp[11];

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
    // std::cout << "T_02: " << std::endl << tf_goal << std::endl;

    // Adjust for grasp distance, etc.
    // const T ang = static_cast<T> ((180.0 - 2.0 * 68.0) * M_PI / 180.0);
    const T c = cos((T)ang);
    const T s = sin((T)ang);

    Eigen::Matrix3<T> R1;
    R1(0) = -1; R1(1) = 0; R1(2) = 0;
    R1(3) = 0; R1(4) = 1; R1(5) = 0;
    R1(6) = 0; R1(7) = 0; R1(8) = -1;
    // R1 << -1, 0, 0, 0, 1, 0, 0, 0, -1;

    Eigen::Matrix3<T> R2;
    R2(0) = c; R2(1) = -s; R2(2) = 0;
    R2(3) = s; R2(4) = c; R2(5) = 0;
    R2(6) = 0; R2(7) = 0; R2(8) = 1;
    // R2 << c, -s, 0, s, c, 0, 0, 0, 1;

    Eigen::Matrix3<T> R3;
    R3(0) = -1; R3(1) = 0; R3(2) = 0;
    R3(3) = 0; R3(4) = -1; R3(5) = 0;
    R3(6) = 0; R3(7) = 0; R3(8) = 1;
    // R3 << -1, 0, 0, 0, -1, 0, 0, 0, 1;

    tf_goal.template block<3, 3>(0, 0) = tf_goal.template block<3, 3>(0, 0) * R1 * R2 * R3;

    Eigen::Matrix<T, 3, 1> grasp_distance_vec((T)0, (T)0, -grasp_distance);

    tf_goal.template block<3, 1>(0, 3) +=
        tf_goal.template block<3, 3>(0, 0) * grasp_distance_vec;

    tf_goal.template block<3, 1>(0, 3) += base_translation.cast<T>();

    // std::cout << "goal is : " << tf_goal << std::endl;

    // // if (unclipped_vals != nullptr) {
    // //   unclipped_vals->resize(4);
    // // }

    // // // Do the IK!!!!
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
    // if (unclipped_vals != nullptr) {
    //   (*unclipped_vals)(0) = arccos_in;
    // }

    T phi = SafeArccos(arccos_in, -clip, clip);
    T theta_2v = atan2(p_26.template head<2>().norm(), p_26(2)) + GC4 * phi;

    T theta_3v = T(0);  // This joint is fixed

    // EQ (4)
    arccos_in = (p_26_dot - d_se * d_se - d_ew * d_ew) / (2.0 * d_se * d_ew);
    // if (unclipped_vals != nullptr) {
    //   (*unclipped_vals)(1) = arccos_in;
    // }

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
    // if (unclipped_vals != nullptr) {
    //   (*unclipped_vals)(2) = arccos_in;
    // }
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
    // if (unclipped_vals != nullptr) {
    //   (*unclipped_vals)(3) = arccos_in;
    // }
    q_subordinate(5) = GC6 * SafeArccos(arccos_in, -clip, clip);

    q_subordinate(6) = atan2(
        GC6 * (A_w(2, 1) * sin(psi) + B_w(2, 1) * cos(psi) + C_w(2, 1)),
        GC6 * (-A_w(2, 0) * sin(psi) - B_w(2, 0) * cos(psi) - C_w(2, 0)));

    q_full.tail(7) = q_subordinate;
    std::cout << "q_full: " << q_full.transpose() << std::endl;
    // return q_full;
    const size_t n_out = 14;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_out; ++i)
    {
        data[i] = q_full[i];
    }

    std::cout << "Copied to data." << std::endl;
    ADFun<CGD> iiwa_param_func(ad_inp, data);
    std::cout << "Created the AD function." << std::endl;
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);

    handler.makeVariables(ind_vars);


    CppAD::vector<CGD> result = iiwa_param_func.Forward(0, ind_vars);
    std::cout << "Ran the AD function." << std::endl;


    if (compute_gradient)
    {
      // this is the full jacobian
      CppAD::vector<CGD> jac_e_q = iiwa_param_func.Jacobian(ind_vars);
    //   CppAD::vector<CGD> jac_e_q(n_out * nq);  // this is jacobian with respect to joint configs only.

    //   for (auto i = 0U; i < n_out; i++)
    //       for (auto j = 0U; j < nq; j++)
    //           jac_e_q[i * nq + j] = jac[i * num_inp + j];

      std::move(jac_e_q.begin(), jac_e_q.end(), std::back_inserter(result));              
    }

    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;

    if (language == "c++")
    {
        LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    std::cout << "Generated the parameterized IK code." << std::endl;
    return Traced{function_code.str(), handler.getTemporaryVariableCount(), result.size()};    

}

// template <typename T>
// Eigen::VectorX<T> IiwaBimanualParameterization(
//     const Eigen::VectorX<T> &q_and_psi,
//     const bool shoulder_up,
//     const bool elbow_up,
//     const bool wrist_up,
//     std::nullptr_t,
//     const double grasp_distance)
// {
//     return IiwaBimanualParameterization(
//         q_and_psi,
//         shoulder_up,
//         elbow_up,
//         wrist_up,
//         static_cast<Eigen::VectorX<T> *>(nullptr),
//         grasp_distance);
// }

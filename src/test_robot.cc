#include "cholesky_decomp.hh"
#include "pinocchio/math/fwd.hpp"

#include <cmath>
#include <boost/mpl/int.hpp>
// #include <cppad/cg/support/cppadcg_eigen.hpp>

#include "pinocchio/autodiff/cppad.hpp"

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <optional>
#include "se3_ops.hh"

#include "robot_info.hh"

using namespace pinocchio;
using namespace CppAD;
using namespace std;

// Typedef for AD types
using CGD = double;
using ADCG = AD<CGD>;

using ADModel = ModelTpl<ADCG>;
using ADData = DataTpl<ADCG>;
using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;
using ADMatrixXs = Eigen::Matrix<ADCG, Eigen::Dynamic, Eigen::Dynamic>;

auto compute_com(const RobotInfo &info, ADVectorXs ad_inp) -> CppAD::vector<CGD>
{
    std::cout << info.spheres.size() << std::endl;
    auto nq = info.model.nq;
    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    const size_t num_inp = nq;

    Independent(ad_inp);

    ADVectorXs ad_q(nq);

    // Copying inputs from ad_inp into individual matrices
    for (auto i = 0U; i < nq; i++)
    {
        ad_q[i] = ad_inp[i];  // This is the first 7 vars for nq
    }

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    const auto CoM = centerOfMass(ad_model, ad_data, ad_q, true);

    std::cout << "->" << info.end_effector_indexes[0] << ", " << info.end_effector_indexes[1] << " -- "
              << info.end_effector_indexes.size() << std::endl;
    const auto lTw = ad_data.oMf[info.end_effector_indexes[0]];
    const auto rTw = ad_data.oMf[info.end_effector_indexes[1]];

    const auto lTr_rob = lTw.inverse() * rTw;
    // const auto errT = lTr_rob * lTr.inverse();

    std::cout << "lTw = \n" << lTw << std::endl;
    std::cout << "rTw = \n" << rTw << std::endl;
    std::cout << "rTobj = \n" << lTr_rob << std::endl;

    std::size_t n_out = 3;
    ADVectorXs data(n_out);
    for (auto i = 0U; i < n_out; i++)
    {
        data[i] = CoM[i];
    }

    // Create the AD function
    ADFun<CGD> jacobian_error_func(ad_inp, data);
    CppAD::vector<CGD> ind_vars(num_inp);
    for (auto i = 0U; i < num_inp; i++)
    {
        ind_vars[i] = CppAD::Value(ad_inp[i]);
    }

    CppAD::vector<CGD> result = jacobian_error_func.Forward(0, ind_vars);
    CppAD::vector<CGD> jac = jacobian_error_func.Jacobian(ind_vars);

    // // print result
    std::cout << "COM is: ";
    for (auto i = 0U; i < result.size(); i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    CppAD::vector<CGD> jac_e_q(n_out * nq);  // this is jacobian with respect to joint configs only.
    for (auto i = 0U; i < n_out; i++)
    {
        for (auto j = 0U; j < nq; j++)
        {
            jac_e_q[i * nq + j] = jac[i * num_inp + j];
        }
    }

    std::move(jac_e_q.begin(), jac_e_q.end(), std::back_inserter(result));
    return result;
}

auto get_bounding_sphere_info(
    const RobotInfo &info,
    const ADData &ad_data,
    int bounding_sphere_index,
    double *radius)
{
    // auto link_bs = info.bounding_sphere_index[bounding_sphere_index];
    auto sphere_it = info.bounding_spheres.find(bounding_sphere_index);
    if (sphere_it == info.bounding_spheres.end())
    {
        std::cout << "Looking for bounding sphere with index " << bounding_sphere_index << " but found none "
                  << std::endl;
        throw std::runtime_error("Bounding sphere not found");
    }
    auto sph_info = sphere_it->second;
    const auto &joint_placement = ad_data.oMi[sph_info.parent_joint];

    Eigen::Matrix<ADCG, 3, 1> local_translation;
    local_translation[0] = sph_info.relative.translation()[0];
    local_translation[1] = sph_info.relative.translation()[1];
    local_translation[2] = sph_info.relative.translation()[2];

    Eigen::Matrix<ADCG, 3, 1> world_position =
        joint_placement.rotation() * local_translation + joint_placement.translation();

    std::vector<ADCG> sphere_info;
    sphere_info.push_back(world_position[0]);
    sphere_info.push_back(world_position[1]);
    sphere_info.push_back(world_position[2]);
    std::cout << "Sphere info: " << sphere_info[0] << ", " << sphere_info[1] << ", " << sphere_info[2] << " "
              << sph_info.radius << std::endl;
    *radius = sph_info.radius;
    return sphere_info;
}

auto compute_self_collision(const RobotInfo &info, ADVectorXs ad_inp) -> CppAD::vector<CGD>
{
    std::cout << info.spheres.size() << std::endl;
    auto nq = info.model.nq;
    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    const size_t num_inp = nq;

    Independent(ad_inp);

    ADVectorXs ad_q(nq);

    // Copying inputs from ad_inp into individual matrices
    for (auto i = 0U; i < nq; i++)
    {
        ad_q[i] = ad_inp[i];  // This is the first 7 vars for nq
    }

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    std::size_t n_out = info.num_valid_bounding_spheres;
    ADVectorXs data(n_out);

    for (auto &pair : info.allowed_link_pairs)
    {
        double sphere_1_radius = 0.0;
        double sphere_2_radius = 0.0;

        auto sphere_link_1_info = get_bounding_sphere_info(info, ad_data, pair.first, &sphere_1_radius);
        auto sphere_link_2_info = get_bounding_sphere_info(info, ad_data, pair.second, &sphere_2_radius);

        // now calculate the penetration between the two spheres
        auto sum =
            (sphere_link_1_info[0] - sphere_link_2_info[0]) *
                (sphere_link_1_info[0] - sphere_link_2_info[0]) +
            (sphere_link_1_info[1] - sphere_link_2_info[1]) *
                (sphere_link_1_info[1] - sphere_link_2_info[1]) +
            (sphere_link_1_info[2] - sphere_link_2_info[2]) * (sphere_link_1_info[2] - sphere_link_2_info[2]);
        auto rs = sphere_1_radius + sphere_2_radius;
        auto penetration = rs * rs - sum;
        auto penetration_soft_hinge = 0.5 * (penetration + sqrt(penetration * penetration + 1e-6));
        data[info.bounding_sphere_index[pair.first]] =
            data[info.bounding_sphere_index[pair.first]] + penetration_soft_hinge;
        std::cout << "Penetration between " << pair.first << " and " << pair.second << ": " << penetration
                  << " " << penetration_soft_hinge << " inserted at "
                  << info.bounding_sphere_index[pair.first] << " - " << info.model.frames[pair.first].name
                  << " and " << info.bounding_sphere_index[pair.second] << " - "
                  << info.model.frames[pair.second].name << " with radius : " << sphere_1_radius << " and "
                  << sphere_2_radius << std::endl;
    }
    // Create the AD function
    ADFun<CGD> jacobian_error_func(ad_inp, data);
    CppAD::vector<CGD> ind_vars(num_inp);
    for (auto i = 0U; i < num_inp; i++)
    {
        ind_vars[i] = CppAD::Value(ad_inp[i]);
    }

    CppAD::vector<CGD> result = jacobian_error_func.Forward(0, ind_vars);
    CppAD::vector<CGD> jac = jacobian_error_func.Jacobian(ind_vars);

    // // print result
    std::cout << "Penetration is: ";
    for (auto i = 0U; i < result.size(); i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Done processing" << std::endl;

    CppAD::vector<CGD> jac_e_q(n_out * nq);  // this is jacobian with respect to joint configs only.
    for (auto i = 0U; i < n_out; i++)
    {
        for (auto j = 0U; j < nq; j++)
        {
            jac_e_q[i * nq + j] = jac[i * num_inp + j];
        }
    }

    std::move(jac_e_q.begin(), jac_e_q.end(), std::back_inserter(result));
    return result;
}

auto com_polygon_constraint(ADVectorXs ad_inp) -> CppAD::vector<CGD>
{
    const size_t num_inp = 4 + 3;  // polygon constraint, com constraint

    Independent(ad_inp);

    ADVectorXs ad_com(3);
    ADVectorXs A(2);
    ADVectorXs B(2);

    // for(size_t i = 0; i < num_inp; i++)
    //     std::cout << ad_inp[i] << ", ";
    // std::cout << std::endl;

    // Copying inputs from ad_inp into individual matrices
    for (auto i = 0U; i < 3; i++)
    {
        ad_com[i] = ad_inp[i];  // This is the first 7 vars for nq
    }

    for (auto i = 0U; i < 2; i++)
    {
        A[i] = ad_inp[i + 3];  // This is the first 7 vars for nq
    }

    for (auto i = 0U; i < 2; i++)
    {
        B[i] = ad_inp[i + 3 + 2];  // This is the first 7 vars for nq
    }

    ADVectorXs com(2);
    com << ad_com[0], ad_com[1];
    // auto n = B - A;
    ADVectorXs n(2);
    n << (B - A).y(), (A - B).x();
    n.normalize();
    // std::cout << n.transpose() << std::endl;

    auto r = (n.dot(com - A));
    auto error = r * n;
    // std::cout << error.transpose() << ", " << r << ", " <<com.transpose() <<", " << A.transpose() << ", "
    // <<(com - A).transpose() << std::endl;

    // using ACGDMatrixXs = Eigen::Matrix<ADCG, Eigen::Dynamic, Eigen::Dynamic>;
    // ACGDMatrixXs identity2(2, 2);
    // identity2.setIdentity();

    // auto P = identity2 - (n * n.transpose()) / n.dot(n);
    // auto error = (P * (com - A));
    // std::cout << "error is " << error[0] << ", " << error[1] << ", " <<  error.size() << std::endl;

    std::size_t n_out = 2 + 2;
    ADVectorXs data(n_out);
    for (auto i = 0U; i < n_out; i++)
    {
        data[i] = error[i];
    }
    for (auto i = 0U; i < n_out; i++)
    {
        data[i + 2] = n[i];
    }

    // Create the AD function
    ADFun<CGD> jacobian_error_func(ad_inp, data);
    CppAD::vector<CGD> ind_vars(num_inp);
    for (auto i = 0U; i < num_inp; i++)
    {
        ind_vars[i] = CppAD::Value(ad_inp[i]);
    }

    CppAD::vector<CGD> result = jacobian_error_func.Forward(0, ind_vars);
    CppAD::vector<CGD> jac = jacobian_error_func.Jacobian(ind_vars);

    // bool outside = (result[0] * result[2] + result[1] * result[3]) > 0.0;
    bool outside = CondExpGt((result[0] * result[2] + result[1] * result[3]), 0.0, 1.0, 0.0);

    CppAD::vector<CGD> error_vector(2);

    error_vector[0] = result[0] * outside;
    error_vector[1] = result[1] * outside;
    // error_vector << outside * result[0], outside * result[1];

    // // print result
    // std::cout << "Error: ";
    // for (auto i=0U; i < result.size(); i++)
    //     std::cout << error_vector[i] << " ";
    // std::cout << std::endl;

    CppAD::vector<CGD> jac_e_q(2 * 3);  // this is jacobian with respect to joint configs only.
    for (auto i = 0U; i < 2; i++)
    {
        for (auto j = 0U; j < 3; j++)
        {
            jac_e_q[i * 3 + j] = jac[i * num_inp + j] * outside;
        }
    }

    std::move(jac_e_q.begin(), jac_e_q.end(), std::back_inserter(error_vector));
    return error_vector;
}

auto trace_full_tsr_project(const RobotInfo &info, ADVectorXs ad_inp) -> CppAD::vector<CGD>
{
    const double DT = 0.1;
    const double damp = 1e-6;
    auto nq = info.model.nq;
    auto nv = info.model.nv;
    const size_t nt = 6;  // task space is se3
    // const size_t ntnt = 16; // for 4x4 matrix

    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    // Total inputs is:
    // 1 * 4x4 matrices for constraint space
    // 2 * 6 bounds for constraint space
    // nq  for configuration space
    const size_t num_inp = 7 * 1 + nt * 2 + nq;

    Independent(ad_inp);

    Eigen::Vector3<ADCG> lTrp;  // left joint in right joint frame

    ADVectorXs lb(nt);
    ADVectorXs ub(nt);
    ADVectorXs ad_q(nq);

    // Copying inputs from ad_inp into individual matrices
    for (auto i = 0U; i < nq; i++)
    {
        ad_q[i] = ad_inp[i];  // This is the first 7 vars for nq
    }

    Eigen::Quaternion<ADCG> lTrq(
        ad_inp[nq + 0 + 0], ad_inp[nq + 0 + 1], ad_inp[nq + 0 + 2], ad_inp[nq + 0 + 3]);  // Next 7 for rTe

    for (auto i = 0U; i < 3; i++)
    {
        lTrp[i] = ad_inp[nq + 4 + i];
    }

    for (auto i = 0U; i < nt; i++)
    {
        lb[i] = ad_inp[nq + 1 * 7 + i];
        ub[i] = ad_inp[nq + 1 * 7 + nt + i];
    }

    SE3Tpl<ADCG, 0> lTr(lTrq, lTrp);  // it is assumed that this err is expressed in the eef joint frame
    // input setup done

    // std::cout << "lTrq" << lTrq.coeffs().transpose() << " " << lTrp.transpose() << std::endl;
    // std::cout << "wtrq" << wTrq.coeffs().transpose() << " " << wTrp.transpose() << std::endl;

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    const auto CoM = centerOfMass(ad_model, ad_data, ad_q, true);

    std::cout << "\nCOM is : " << CoM.x() << ", " << CoM.y() << ", " << CoM.z() << ", " << info.spheres.size()
              << std::endl;

    for (auto i = 0U; i < info.spheres.size(); ++i)
    {
        const auto &sphere = info.spheres[i];
        // trace_sphere(sphere, ad_data, data, sphere.geom_index * 4);
        const auto &joint_placement = ad_data.oMi[sphere.parent_joint];
        Eigen::Matrix<ADCG, 3, 1> local_translation;
        local_translation[0] = sphere.relative.translation()[0];
        local_translation[1] = sphere.relative.translation()[1];
        local_translation[2] = sphere.relative.translation()[2];

        Eigen::Matrix<ADCG, 3, 1> world_position =
            joint_placement.rotation() * local_translation + joint_placement.translation();

        // if ((i > 63 && i < 67) || (i > 130 && i < 134))

        // std::cout << i << " - " << local_translation.transpose() << ", " << world_position.transpose() << "
        // -> " << sphere.radius << std::endl;
    }
    // A is (0, 1)
    // B is (0, -1)
    // C is (-1, 1)

    // std::cout << "rteq = \n" << rTe.toHomogeneousMatrix_impl() << std::endl;
    // std::cout << "wtrq = \n" << wTr.toHomogeneousMatrix_impl() << std::endl;
    // std::cout << "eef = \n" << ad_data.oMf[info.end_effector_index].toHomogeneousMatrix_impl() <<
    // std::endl;

    // compute error term

    // ADVectorXs A(2);
    // A << ADCG(0.001), ADCG(1.0);
    // ADVectorXs B(2);
    // B << ADCG(-0.001), ADCG(-1.0);

    using ACGDMatrixXs = Eigen::Matrix<ADCG, Eigen::Dynamic, Eigen::Dynamic>;
    ACGDMatrixXs identity2(2, 2);
    identity2.setIdentity();

    ADMatrixXs polygon(4, 2);
    polygon << ADCG(0.001), ADCG(1.0), ADCG(1.0), ADCG(1.0), ADCG(1.0), ADCG(-1.0), ADCG(-0.001), ADCG(-1.0);

    // ADMatrixXs polygon(2, 2);
    // polygon << ADCG(0.001), ADCG(1.0), ADCG(-0.001), ADCG(-1.0);

    ADVectorXs direction(2);
    direction << ADCG(0.0), ADCG(0.0);

    for (size_t i = 0; i < polygon.rows(); i++)
    {
        const auto j = (i + 1) % 4;
        auto A = polygon.row(i).transpose();
        auto B = polygon.row(j).transpose();

        auto n = B - A;
        // std::cout << n << std::endl;

        // ADVectorXs n(2);
        // n << (B - A).y(), (A - B).x();
        // n << 0 - 0, -1 -1;
        // n.normalize();

        // ADVectorXs A(2);
        // A << 0, 1;
        // ADVectorXs com(2);
        // com << CoM.x() , CoM.y();
        // ADVectorXs n(2);
        // n << (B - A).y(), (A - B).x();
        // n.normalize();

        // auto error = (n.transpose() * (com - A));

        ADVectorXs com(2);
        com << CoM.x(), CoM.y();

        auto P = identity2 - (n * n.transpose()) / n.dot(n);
        auto error = (P * (com - A));
        // std::cout << "error is " << error[0] << ", " << error[1] << ", " <<  error.size() << std::endl;
        direction = direction + error;
    }

    // auto direction = n.transpose() * (com - A);
    // std::cout << "dir is " << direction[0] << ", " << direction[1] << ", " <<  direction.size() <<
    // std::endl;

    std::cout << "->" << info.end_effector_indexes[0] << ", " << info.end_effector_indexes[1] << " -- "
              << info.end_effector_indexes.size() << std::endl;
    const auto lTw = ad_data.oMf[info.end_effector_indexes[0]];
    const auto rTw = ad_data.oMf[info.end_effector_indexes[1]];

    const auto lTr_rob = lTw.inverse() * rTw;
    // const auto errT = lTr_rob * lTr.inverse();

    std::cout << "lTw = \n" << lTw << std::endl;
    std::cout << "rTw = \n" << rTw << std::endl;
    std::cout << "rTobj = \n" << lTr_rob << std::endl;
    // std::cout << "errT = \n" << errT << std::endl;

    // ADVectorXs displacement(nt);
    // displacement.setZero();
    // displacement << errT.translation_impl(), log3(errT.rotation_impl());
    // ADCG zero(0.0);

    // const auto iMd = ad_data.oMf[info.end_effector_index].actInv(wTobj);
    // std::cout << "oMe = \n" << ad_data.oMf[info.end_effector_index].toHomogeneousMatrix_impl() <<
    // std::endl; std::cout << "iMd = \n" << iMd.toHomogeneousMatrix_impl() << std::endl;

    // auto displacement = log6(iMd); // in joint frame
    // std::cout << "displacement = " << displacement.transpose() << std::endl;

    // ADVectorXs displacement(nt);
    // displacement.setZero();
    // displacement << iMd.translation_impl(), log3(iMd.rotation_impl());

    std::size_t n_out = 2;
    ADVectorXs data(n_out);
    data[0] = direction[0];
    data[1] = direction[1];
    // for (auto i = 0U; i < nt; i++){
    //     // ;
    //     // data[i] = min(displacement[i] - lb[i], displacement[i] * 1e-6) + max(displacement[i] - ub[i],
    //     displacement[i] * 1e-6); data[i] = displacement[i];
    //     // std::cout << displacement[i] << " ";
    // }
    // std::cout << std::endl;
    // for (auto i = 0U; i < n_out; i++)
    //     std::cout << "Error[" << i << "] = " << data[i] << std::endl;

    // Create the AD function
    ADFun<CGD> jacobian_error_func(ad_inp, data);
    CppAD::vector<CGD> ind_vars(num_inp);
    for (auto i = 0U; i < num_inp; i++)
    {
        ind_vars[i] = CppAD::Value(ad_inp[i]);
    }

    CppAD::vector<CGD> result = jacobian_error_func.Forward(0, ind_vars);
    CppAD::vector<CGD> jac = jacobian_error_func.Jacobian(ind_vars);

    // // print result
    std::cout << "Error: ";
    for (auto i = 0U; i < result.size(); i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    using CGDVectorXs = Eigen::Matrix<CGD, Eigen::Dynamic, 1>;
    using CGDMatrixXs = Eigen::Matrix<CGD, Eigen::Dynamic, Eigen::Dynamic>;
    CGDMatrixXs ad_J(n_out, nq);

    for (auto i = 0U; i < n_out; i++)
    {
        for (auto j = 0U; j < nq; j++)
        {
            ad_J(i, j) = jac[i * num_inp + j];
        }
    }

    std::cout << "Jacobian is " << std::endl;
    for (auto i = 0U; i < n_out; i++)
    {
        for (auto j = 0U; j < nq; j++)
        {
            std::cout << std::setprecision(5) << ad_J(i, j) << " ";
        }
        std::cout << std::endl;
    }

    CGDMatrixXs identity(n_out, n_out);
    identity.setIdentity();
    CGDVectorXs ad_e(n_out);

    // set up ad_e
    for (auto i = 0U; i < n_out; i++)
    {
        if (i < 3)
        {
            ad_e(i) = result[i];  // for position, we want to reduce the err in 0.1s
        }
        else
        {
            ad_e(i) = result[i];  // for rotation, we want to reduce the err in 1 step
        }
    }

    // compute solution here directly.
    auto decomposed = cholesky_factor<CGDMatrixXs, CGD>(ad_J * ad_J.transpose() + identity * 1e-4);
    CGDVectorXs grad = ad_J.transpose() * cholesky_solve<CGDMatrixXs, CGDVectorXs, CGD>(decomposed, ad_e);

    // std::cout << "Grad: ";
    // for (auto i=0U; i < grad.size(); i++)
    //     std::cout << grad[i] << " ";
    // std::cout << std::endl;

    // // CGDVectorXs grad = ad_J.transpose() * (ad_J * ad_J.transpose() + identity * 1e-4).llt().solve(ad_e);
    // CGDVectorXs grad = ad_J.transpose() * ad_e;

    CppAD::vector<CGD> grad_vec(nq);
    for (auto i = 0U; i < nq; i++)
    {
        grad_vec[i] = grad(i);
    }

    std::move(result.begin(), result.end(), std::back_inserter(grad_vec));

    return grad_vec;
}

auto tsr_error_function(const RobotInfo &info, ADVectorXs ad_inp) -> CppAD::vector<CGD>
{
    const double DT = 0.1;
    const double damp = 1e-6;
    auto nq = info.model.nq;
    auto nv = info.model.nv;
    const size_t nt = 6;  // task space is se3
    const size_t n_eef = info.end_effector_indexes.size();

    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    const size_t num_inp_eef = (7 * 2 + nt * 2);
    const size_t num_inp = num_inp_eef * n_eef + nq;
    std::cout << "num_inp: " << num_inp << std::endl;

    Independent(ad_inp);

    std::size_t n_out = nt * n_eef;
    ADVectorXs data(n_out);

    // First copy over configs and run FK
    ADVectorXs ad_q(nq);

    for (auto i = 0U; i < nq; i++)
    {
        ad_q[i] = ad_inp[i];  // This is the first 7 vars for nq
    }

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    for (size_t eef_idx = 0; eef_idx < n_eef; eef_idx++)
    {
        auto eef_offset = num_inp_eef * eef_idx + nq;
        std::cout << "\n Cur EEF offset is : " << eef_offset << std::endl;

        auto eef_out_offset = nt * eef_idx;

        // Form the matrices
        Eigen::Vector3<ADCG> rTep;
        Eigen::Vector3<ADCG> wTrp;
        Eigen::Quaternion<ADCG> rTeq(
            ad_inp[eef_offset + 0 + 0],
            ad_inp[eef_offset + 0 + 1],
            ad_inp[eef_offset + 0 + 2],
            ad_inp[eef_offset + 0 + 3]);  // Next 7 for rTe
        // std::cout << "rTeq : " << eef_idx << rTeq << std::endl;
        Eigen::Quaternion<ADCG> wTrq(
            ad_inp[eef_offset + 7 + 0],
            ad_inp[eef_offset + 7 + 1],
            ad_inp[eef_offset + 7 + 2],
            ad_inp[eef_offset + 7 + 3]);  // 7 after that for wTr
        // std::cout << "wTrq : " << eef_idx << wTrq << std::endl;
        for (auto i = 0U; i < 3; i++)
        {
            rTep[i] = ad_inp[eef_offset + 4 + i];
            wTrp[i] = ad_inp[eef_offset + 7 + 4 + i];
        }
        SE3Tpl<ADCG, 0> rTe(rTeq, rTep);  // it is assumed that this err is expressed in the eef joint frame
        SE3Tpl<ADCG, 0> wTr(wTrq, wTrp);

        // Form bounds
        ADVectorXs lb(nt);
        ADVectorXs ub(nt);
        for (auto i = 0U; i < nt; i++)
        {
            lb[i] = ad_inp[eef_offset + 2 * 7 + i];
            ub[i] = ad_inp[eef_offset + 2 * 7 + nt + i];
        }
        // input setup done

        // compute error term
        const auto wTobj = ad_data.oMf[info.end_effector_indexes[eef_idx]] * rTe.inverse();
        const auto rTobj = wTr.inverse() * wTobj;
        // std::cout << "wTobj : " << eef_idx << std::endl;
        // std::cout << rTobj.translation_impl().transpose() << std::endl;
        // std::cout << rTobj.rotation_impl() << std::endl;

        ADVectorXs displacement(nt);
        displacement.setZero();
        displacement << rTobj.translation_impl(), log3(rTobj.rotation_impl());
        std::cout << "Rot errs are : " <<  log3(rTobj.rotation_impl()).transpose() << std::endl;
        std::cout << "Rot errs are : " << so3_log_smooth<ADMatrixXs, ADCG>(rTobj.rotation_impl()).transpose() << std::endl;
        // displacement << rTobj.translation_impl(), so3_log_smooth<ADMatrixXs, ADCG>(rTobj.rotation_impl());

        for (auto i = 0U; i < nt; i++)
        {
            data[eef_out_offset + i] = displacement[i];
        }
    }
    // print data
    for (auto i = 0U; i < n_out; i++)
    {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    // Create the AD function
    ADFun<CGD> jacobian_error_func(ad_inp, data);
    CppAD::vector<CGD> ind_vars(num_inp);
    for (auto i = 0U; i < num_inp; i++)
    {
        ind_vars[i] = CppAD::Value(ad_inp[i]);
    }

    CppAD::vector<CGD> error_vec = jacobian_error_func.Forward(0, ind_vars);
    std::cout << "error vec: " << error_vec << std::endl;

    // this is the full jacobian
    CppAD::vector<CGD> jac = jacobian_error_func.Jacobian(ind_vars);
    CppAD::vector<CGD> jac_e_q(n_out * nq);  // this is jacobian with respect to joint configs only.

    for (auto i = 0U; i < n_out; i++)
    {
        for (auto j = 0U; j < nq; j++)
        {
            jac_e_q[i * nq + j] = jac[i * num_inp + j];
        }
    }

    // now correct the error vector with the lower and upper bounds.

    // CGD zero_cgd(0.0);
    // for (auto eef_idx = 0U; eef_idx < n_eef; eef_idx++)
    // {
    //     const size_t eef_inp_offset = num_inp_eef * eef_idx;
    //     const size_t eef_out_offset = nt * eef_idx;
    //     for (auto i = 0U; i < nt; i++)
    //     {
    //         CGD lb = ad_inp[eef_inp_offset + nq + 2 * 7 + i];
    //         CGD ub = ad_inp[eef_inp_offset + nq + 2 * 7 + nt + i];
    //         error_vec[eef_out_offset + i] =
    //             CondExpLt(error_vec[eef_out_offset + i], lb, error_vec[eef_out_offset + i] - lb, zero_cgd) +
    //             CondExpGt(error_vec[eef_out_offset + i], ub, error_vec[eef_out_offset + i] - ub, zero_cgd);
    //     }
    // }

    std::move(error_vec.begin(), error_vec.end(), std::back_inserter(jac_e_q));

    return jac_e_q;
}


int main(int argc, char **argv)
{
    cxxopts::Options options(argv[0], "Tracing compiler for forward kinematics and collision checking");

    options.positional_help("[JSON configuration filename]").show_positional_help();

    options.add_options()                                                                       //
        ("f,configuration_file", "JSON configuration filename", cxxopts::value<std::string>())  //
        ("o,output_filename", "Output JSON filename", cxxopts::value<std::string>())            //
        ("t,output_template",
         "Output template filename (override configuration file)",
         cxxopts::value<std::string>())  //
        ("h,help", "Print usage")        //
        ;

    options.parse_positional({"configuration_file"});

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (not result.count("configuration_file"))
    {
        throw std::runtime_error(fmt::format("Must provide configuration file!"));
    }

    std::filesystem::path json_path(result["configuration_file"].as<std::string>());
    auto parent_path = json_path.parent_path();

    if (not std::filesystem::exists(json_path))
    {
        throw std::runtime_error(fmt::format("JSON file {} does not exist!", json_path.string()));
    }

    if (not std::filesystem::exists(json_path))
    {
    }

    std::ifstream json_file(json_path);
    nlohmann::json data;

    try
    {
        data = nlohmann::json::parse(json_file);
    }
    catch (std::exception &e)
    {
        throw std::runtime_error(fmt::format("Failed to parse JSON file! Error: \n{}", e.what()));
    }

    std::optional<std::filesystem::path> srdf_path = {};
    if (data.contains("srdf"))
    {
        srdf_path = parent_path / data["srdf"];
    }

    std::vector<std::string> end_effector_names;
    if (data.contains("end_effectors"))
    {
        for (const auto end_effector_name : data["end_effectors"])
        {
            end_effector_names.push_back(end_effector_name);
        }
    }

    RobotInfo robot(parent_path / data["urdf"], srdf_path, end_effector_names);

    // std::cout << "Robot has " << robot.spheres.size() << " spheres " << std::endl;

    // for(size_t i = 0; i < robot.per_link_spheres.size(); i++)
    // {
    //     std::cout << "\nLink : " << i << " - ";
    //     for(size_t j = 0; j < robot.per_link_spheres[i].size(); j++)
    //     {
    //         std::cout << robot.per_link_spheres[i][j] << ", ";

    //     }

    // }
    // std::cout << std::endl;

    std::array<float, 6> lower_bound = {-0.00002, -0.00002, -0.00002, -0.00002, -0.00002, -0.00002};
    std::array<float, 6> upper_bound = {0.00002, 0.00002, 0.00002, 0.00002, 0.00002, 0.00002};

    // std::array<float, 14> q_init = {
    //    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    // };
    // std::array<float, 14> q_init = {
    //    -0.816, -0.203, 0.716, -0.961, 0.770, 2.055, -0.360
    // };
    std::array<float, 35> q_init = {0.035,0.0,0.031,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

    Eigen::Matrix<float, 4, 4> T;
    T << -0.71933, 0.69467, -6.9968e-07, 3.01e-06, 0.69467, 0.71933, 4.8424e-06, -0.00015324, 3.8415e-06,
        3.0337e-06, -1, 0.41, 0, 0, 0, 1;

    const Eigen::Transform<float, 3, Eigen::Isometry> target_pose(T);
    std::cout << "Target pose is : " << target_pose.matrix() << std::endl;

    Eigen::Quaternion<float> q2(target_pose.linear());
    std::array<float, 7> target_pose_7 = {
        q2.w(),
        q2.x(),
        q2.y(),
        q2.z(),
        target_pose.translation().x(),
        target_pose.translation().y(),
        target_pose.translation().z()};

    // compose a new input of ADVectorXs ad_inp of q_init, target_pose, in_hand_pose, lower_bound, upper_bound
    ADVectorXs ad_inp((7 * 2 + 6 * 2) * 4 + robot.model.nq);  // 3 4x4 matrices + 3 6D vectors + nq

    std::array<Eigen::Transform<float, 3, Eigen::Isometry>, 4> eef_transforms;

    T << 1, 0, 0, 0.13, 0, 1, 0, 0.11, 0, 0, 1, -0.725, 0, 0, 0, 1;
    eef_transforms[0] = Eigen::Transform<float, 3, Eigen::Isometry>(T);
    T << 1, 0, 0, 0.13, 0, 1, 0, 0.11, 0, 0, 1, -0.725, 0, 0, 0, 1;
    eef_transforms[1] = Eigen::Transform<float, 3, Eigen::Isometry>(T);
    T << 1, 0, 0, 0.13, 0, 1, 0, 0.11, 0, 0, 1, -0.725, 0, 0, 0, 1;
    eef_transforms[2] = Eigen::Transform<float, 3, Eigen::Isometry>(T);
    T << 1, 0, 0, 0.13, 0, 1, 0, -0.11, 0, 0, 1, -0.725, 0, 0, 0, 1;
    eef_transforms[3] = Eigen::Transform<float, 3, Eigen::Isometry>(T);
    std::array<float, 6 * 4> tsr_lower_bound = {
        -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -0.01, -0.01, -0.01, -10.0, -10.0, -10.0, -0.01, -0.01, -0.01, -10.0, -10.0, -10.0
    };

    std::array<float, 6 * 4> tsr_upper_bound = {
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        0.01, 0.01, 0.01,
        10.0, 10.0, 10.0,
        0.01, 0.01, 0.01,
        10.0, 10.0, 10.0
    };
    for (auto eef_idx = 0U; eef_idx < 4; ++eef_idx)
    {
        const size_t eef_offset = (7 * 2 + 6 * 2) * eef_idx + robot.model.nq;


        ad_inp[eef_offset + (0 + 0)] = 1.0;
        ad_inp[eef_offset + (0 + 1)] = 0.0;
        ad_inp[eef_offset + (0 + 2)] = 0.0;
        ad_inp[eef_offset + (0 + 3)] = 0.0;
        ad_inp[eef_offset + (0 + 4)] = 0.0;
        ad_inp[eef_offset + (0 + 5)] = 0.0;
        ad_inp[eef_offset + (0 + 6)] = 0.0;

        Eigen::Quaternion<float> q2(eef_transforms[eef_idx].linear());
        std::cout << "Quaternion: " << q2.w() << ", " << q2.x() << ", " << q2.y() << ", " << q2.z() <<  " -- " << eef_offset << ", " << eef_idx <<std::endl;
        ad_inp[eef_offset + (7 + 0)] = q2.w();
        ad_inp[eef_offset + (7 + 1)] = q2.x();
        ad_inp[eef_offset + (7 + 2)] = q2.y();
        ad_inp[eef_offset + (7 + 3)] = q2.z();
        ad_inp[eef_offset + (7 + 4)] = eef_transforms[eef_idx].translation().x();
        ad_inp[eef_offset + (7 + 5)] = eef_transforms[eef_idx].translation().y();
        ad_inp[eef_offset + (7 + 6)] = eef_transforms[eef_idx].translation().z();

        ad_inp[eef_offset + (14 + 0)] = tsr_lower_bound[eef_idx * 6 + 0];
        ad_inp[eef_offset + (14 + 1)] = tsr_lower_bound[eef_idx * 6 + 1];
        ad_inp[eef_offset + (14 + 2)] = tsr_lower_bound[eef_idx * 6 + 2];
        ad_inp[eef_offset + (14 + 3)] = tsr_lower_bound[eef_idx * 6 + 3];
        ad_inp[eef_offset + (14 + 4)] = tsr_lower_bound[eef_idx * 6 + 4];
        ad_inp[eef_offset + (14 + 5)] = tsr_lower_bound[eef_idx * 6 + 5];

        ad_inp[eef_offset + (14 + 6 + 0)] = tsr_upper_bound[eef_idx * 6 + 0];
        ad_inp[eef_offset + (14 + 6 + 1)] = tsr_upper_bound[eef_idx * 6 + 1];
        ad_inp[eef_offset + (14 + 6 + 2)] = tsr_upper_bound[eef_idx * 6 + 2];
        ad_inp[eef_offset + (14 + 6 + 3)] = tsr_upper_bound[eef_idx * 6 + 3];
        ad_inp[eef_offset + (14 + 6 + 4)] = tsr_upper_bound[eef_idx * 6 + 4];
        ad_inp[eef_offset + (14 + 6 + 5)] = tsr_upper_bound[eef_idx * 6 + 5];

        // now set the corresponding ad_inp values
    }
    for (auto i = 0U; i < robot.model.nq; ++i)
    {
        ad_inp[i] = ADCG(q_init[i]);
    }

    auto tsr_err_jac = tsr_error_function(robot, ad_inp);
    // first 6 * 4 * robot.model.nq are jacobians
    // Next 6 * 4 are the errors
    //
    std::cout << "Jacobian matrix : " << std::endl;
    for(auto i = 0; i < 6 * 4; ++i)
    {
        for(auto j = 0; j < robot.model.nq; ++j)
        {
            std::cout << tsr_err_jac[i * robot.model.nq + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Error vector : " << std::endl;
    for(auto i = 0; i < 4; ++i)
    {
        for(auto j = 0; j < 6; ++j)
        {
            std::cout << tsr_err_jac[robot.model.nq * 6 * 4 + i * 6 + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;


    // ADMatrixXs polygon(4, 2);
    // polygon << ADCG(0.001), ADCG(1.0), ADCG(-0.001), ADCG(-1.0), ADCG(1.0), ADCG(-1.0), ADCG(1.0), ADCG(1.0);
    // // polygon << ADCG(0.0), ADCG(0.2), ADCG(0.0), ADCG(1.0), ADCG(1.0), ADCG(1.0), ADCG(1.0), ADCG(0.2) ;
    // // polygon << ADCG(1.0), ADCG(-0.3), ADCG(1.0), ADCG(0.3),  ADCG(0.0), ADCG(0.3), ADCG(0.0), ADCG(-0.3);
    // // polygon << ADCG(0.0), ADCG(0.2), ADCG(1.0), ADCG(0.2), ADCG(1.0), ADCG(1.0), ADCG(0.0), ADCG(1.0) ;

    // ADVectorXs ad_q(robot.model.nq);
    // for (auto i = 0U; i < robot.model.nq; ++i)
    // {
    //     ad_q[i] = ADCG(q_init[i]);
    // }
    // auto selfcoll_jac = compute_self_collision(robot, ad_q);
    // std::cout << selfcoll_jac.size() << std::endl;
    // for (auto i = 0U; i < robot.num_valid_bounding_spheres; ++i)
    // {
    //     std::cout << selfcoll_jac[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Done printing error" << std::endl;

    // // for(auto i =0; i < robot.model.frames.size(); i++)
    // //     std::cout << i << " " << robot.model.frames[i].name << std::endl;

    // // ADVectorXs com_constraint_inp(4 + 3);
    // // for (auto i = 0U; i < 3; ++i)
    // //     com_constraint_inp[i] = ADCG(com_jac[i]);

    // // using CGDVectorXs = Eigen::Matrix<CGD, Eigen::Dynamic, 1>;
    // // CGDVectorXs error(2);
    // // error.setZero();
    // // CGDVectorXs com_constraint_grad(2 * 3);
    // // com_constraint_grad.setZero();

    // // for(size_t i = 0; i < polygon.rows(); i++){

    // //     std::cout << "-----Starting iteration " << i << std::endl;
    // //     const auto j = (i + 1) % 4;
    // //     auto A = polygon.row(i);
    // //     auto B = polygon.row(j);

    // //     com_constraint_inp[3 + 0] = A[0];
    // //     com_constraint_inp[3 + 1] = A[1];
    // //     com_constraint_inp[3 + 2] = B[0];
    // //     com_constraint_inp[3 + 3] = B[1];

    // //     auto com_val = com_polygon_constraint(com_constraint_inp);
    // //     // auto n =
    // //     std::cout << com_val << std::endl;
    // //     CGDVectorXs err(2);
    // //     // auto err = com_val.head(2);
    // //     err[0] = (com_val[0] > 0) ? com_val[0] : com_val[0] * 0.0;
    // //     err[1] = (com_val[1] > 0) ? com_val[1] : com_val[1] * 0.0;

    // //     for(size_t j = 0; j < 2 * 3; j++)
    // //         com_constraint_grad[j] = com_constraint_grad[j] + com_val[2 + j];

    // //     std::cout << err.transpose() << std::endl;
    // //     std::cout << "-----Completed iteration " << i << std::endl;
    // //     error = error + err;

    // // }
    // // std::cout << com_constraint_grad.transpose() << std::endl;
    // // std::cout << error.transpose() << std::endl;

    // // for (auto iteration=0U; iteration < 5; iteration++){
    // //     std::cout << "-----------------Iteration " << iteration << " : ";
    // //     for (auto j=0U; j < robot.model.nq; j++)
    // //         std::cout << CppAD::Value(ad_q[j]) << " ";
    // //     std::cout << std::endl;
    // //     auto com_jac = compute_com(robot, ad_q);
    // //     std::cout << "Computed COM" << std::endl;
    // //     ADVectorXs com_constraint_inp(4 + 3);
    // //     std::cout << com_jac << std::endl;

    // //     for (auto i = 0U; i < 3; ++i)
    // //         com_constraint_inp[i] = ADCG(com_jac[i]);

    // //     using CGDVectorXs = Eigen::Matrix<CGD, Eigen::Dynamic, 1>;
    // //     CGDVectorXs error(2);
    // //     error.setZero();
    // //     CGDVectorXs com_constraint_grad(2 * 3);
    // //     com_constraint_grad.setZero();

    // //     for(size_t polygon_idx = 0; polygon_idx < polygon.rows(); polygon_idx++){

    // //         // std::cout << "----- Polygon " << polygon_idx << std::endl;
    // //         const auto polygon_next_idx = (polygon_idx + 1) % 4;
    // //         auto A = polygon.row(polygon_idx);
    // //         auto B = polygon.row(polygon_next_idx);

    // //         com_constraint_inp[3 + 0] = A[0];
    // //         com_constraint_inp[3 + 1] = A[1];
    // //         com_constraint_inp[3 + 2] = B[0];
    // //         com_constraint_inp[3 + 3] = B[1];

    // //         auto com_val = com_polygon_constraint(com_constraint_inp);
    // //     //     // auto n =
    // //         // std::cout << com_val << com_val.size() << std::endl;
    // //         CGDVectorXs err(2);
    // //     //     // // auto err = com_val.head(2);
    // //         // err[0] = (com_val[0] > 0) ? com_val[0] : com_val[0] * 0.0;
    // //         // err[1] = (com_val[1] > 0) ? com_val[1] : com_val[1] * 0.0;
    // //         err[0] = com_val[0];
    // //         err[1] = com_val[1];

    // //         for(size_t j = 0; j < 2 * 3; j++)
    // //             com_constraint_grad[j] = com_constraint_grad[j] + com_val[2 + j];

    // //         std::cout << err.transpose() << std::endl;
    // //         // std::cout << "-----Completed iteration " << iteration << std::endl;
    // //         error = error + err;
    // //     }
    // //     std::cout << com_constraint_grad.transpose() << std::endl;
    // //     std::cout << "Error : " << error.transpose() << std::endl;

    // //     // reshape the jacobian
    // //     using CGDMatrixXs = Eigen::Matrix<CGD, Eigen::Dynamic, Eigen::Dynamic>;
    // //     CGDMatrixXs ad_J_constraint(2, 3);
    // //     CGDMatrixXs ad_J_com(3, robot.model.nq);

    // //     for(auto i=0U; i < 2; i++)
    // //         for(auto j=0U; j < 3; j++)
    // //             ad_J_constraint(i, j) = com_constraint_grad[i * 3 + j];

    // //     // std::cout << "Jacobian is " <<  ad_J_constraint << std::endl;
    // //     // for (auto i=0U; i < 2; i++){
    // //     //     for (auto j=0U; j < 3; j++)
    // //     //         std::cout << std::setprecision(5) << ad_J_constraint(i, j) << " ";
    // //     //     std::cout << std::endl;
    // //     // }

    // //     for(auto i=0U; i < 3; i++)
    // //         for(auto j=0U; j < robot.model.nq; j++)
    // //             ad_J_com(i, j) = com_jac[3 + i * robot.model.nq + j];
    // //     // std::cout << "Jacobian 2 is " << ad_J_com <<  std::endl;
    // //     // for (auto i=0U; i < 3; i++){
    // //     //     for (auto j=0U; j < 7; j++)
    // //     //         std::cout << std::setprecision(5) << ad_J_com(i, j) << " ";
    // //     //     std::cout << std::endl;
    // //     // }

    // //     const auto ad_J = ad_J_constraint * ad_J_com;
    // //     // std::cout << ad_J.rows() << ", " << ad_J.cols() << std::endl;
    // //     std::cout << ad_J;

    // //     CGDMatrixXs identity(2, 2);
    // //     identity.setIdentity();

    // //     // compute solution here directly.
    // //     auto decomposed = cholesky_factor<CGDMatrixXs, CGD>(ad_J * ad_J.transpose() + identity * 1e-6);
    // //     CGDVectorXs grad = ad_J.transpose() * cholesky_solve<CGDMatrixXs, CGDVectorXs, CGD>(decomposed,
    // //     error);

    // //     std::cout << "\nGradient : " ;
    // //     for (auto j=0U; j < robot.model.nq; j++){
    // //         ad_q[j] = ad_q[j] - ADCG(1.0) * ADCG(grad[j]);
    // //         std::cout << grad[j] << ", ";
    // //     }
    // //     std::cout << std::endl;
    // // }
    // // std::cout << "Final q is : ";
    // // for (auto j=0U; j < robot.model.nq; j++)
    // //     std::cout << CppAD::Value(ad_q[j]) << " ";
    // // std::cout << std::endl;

    // // for(size_t i = 0; i < polygon.rows(); i++){

    // //     std::cout << "-----Starting iteration " << i << std::endl;
    // //     const auto j = (i + 1) % 4;
    // //     auto A = polygon.row(i);
    // //     auto B = polygon.row(j);

    // //     // ADVectorXs com(2);
    // //     // com << ad_com[0] , ad_com[1];
    // //     // // auto n = B - A;
    // //     // ADVectorXs n(2);
    // //     // n << (B - A).y(), (A - B).x();
    // //     // n.normalize();
    // //     // std::cout << n.transpose() << std::endl;

    // //     // auto r = (n.dot(com - A));
    // //     // auto error = r * n;

    // //     auto com_val = com_polygon_constraint(com_constraint_inp);
    // //     std::cout << com_val << std::endl;
    // //     CGDVectorXs err(2);
    // //     // auto err = com_val.head(2);
    // //     err[0] = (com_val[0] > 0) ? com_val[0] : com_val[0] * 0.0;
    // //     err[1] = (com_val[1] > 0) ? com_val[1] : com_val[1] * 0.0;

    // //     for(size_t j = 0; j < 2 * 3; j++)
    // //         com_constraint_grad[j] = com_constraint_grad[j] + com_val[2 + j];

    // //     std::cout << err.transpose() << std::endl;
    // //     std::cout << "-----Completed iteration " << i << std::endl;

    // // }
    // // std::cout << com_constraint_grad.transpose() << std::endl;

    // // for (auto i=0U; i < 1; i++){
    // //     std::cout << "\nIteration " << i << " : ";
    // //     for (auto j=0U; j < robot.model.nq; j++)
    // //         std::cout << CppAD::Value(ad_inp[j]) << " ";
    // //     std::cout << "-->";

    // //     auto val = trace_full_tsr_project(robot, ad_inp);
    // //     for (auto j=0U; j < robot.model.nq; j++){
    // //         ad_inp[j] = ad_inp[j] - ADCG(1.0) * ADCG(val[j]);
    // //         std::cout << val[j] << ", ";
    // //     }
    // // }
    // // std::cout << "Final q is : ";
    // // for (auto j=0U; j < robot.model.nq; j++)
    // //     std::cout << CppAD::Value(ad_inp[j]) << " ";
    // // std::cout << std::endl;

    // // for (auto i=0U; i < val.size(); i++)
    // //     std::cout << val[i] << " ";
    // // std::cout << std::endl;

    // // int num_inp = 7 * 2 + 6 * 2 + robot.model.nq; // 3 4x4 matrices + 3 6D vectors + nq
    // // ADVectorXs ad_inp(num_inp); // 3 4x4 matrices
    // // for (auto i = 0U; i < num_inp; ++i)
    // //     ad_inp[i] = ADCG(0.0);

    // // auto new_q = ad_inp - val;

    // // for (auto i=0U; i < robot.model.nq; i++) {
    // //     ad_inp[i] = ad_inp[i] + val[i];
    // //     std::cout << ad_inp[i] << " ";
    // // }
    // // std::cout << std::endl;

    // // std::cout << "New q is : ";
    // // for (auto i=0U; i < robot.model.nq; i++)
    // //     std::cout << CppAD::Value(new_q[i]) << " ";
    // // std::cout << std::endl;

    // // val = trace_full_tsr_project(robot, ad_inp2);
    // // for (auto i=0U; i < val.size(); i++)
    // //     std::cout << val[i] << " ";
    // // std::cout << std::endl;

    return 0;
}

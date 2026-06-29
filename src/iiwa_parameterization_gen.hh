// TODO (siyer) -- ask tommy
// 1. URDF of the iiwa along with end effector
// 2. Why not sample

#pragma once

#include "housekeeping.hh"
#include "robot_info.hh"
#include "tracer_utils.hh"
#include "iiwa_parameterization.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>



template <typename T>
auto IiwaBimanualParameterizationCG(
    const std::string &language,
    bool compute_gradient = false
)
{

    std::cout << "Generating parameterized IK code for iiwa..." << std::endl;
    const size_t num_inp = 8 + 1 + 1 + 1 + 7;

    ADVectorXs ad_inp(num_inp);  // 3 4x4 matrices
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    auto q_full = IiwaBimanualParameterization<T>(ad_inp);
    
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

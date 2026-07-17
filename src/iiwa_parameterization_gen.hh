// TODO (siyer) -- ask tommy
// 1. URDF of the iiwa along with end effector
// 2. Why not sample

#pragma once

#include "housekeeping.hh"
#include "robot_info.hh"
#include "tracer_utils.hh"
#include "iiwa_parameterization.hh"
#include "internal.hh"

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


template <typename T>
auto IiwaSE3ParameterizationCG(
    const std::string &language,
    bool compute_gradient = false
)
{

    std::cout << "Generating task parameterized IK code for iiwa..." << std::endl;
    const size_t num_inp = 7 + 1 + 1 + 1 + 1; // 6 for the pose, 1 for psi, 3 for GC2, GC4, GC6

    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    auto q_out = IiwaSE3Parameterization<T>(ad_inp);
    
    const size_t n_out = 7;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_out; ++i)
    {
        data[i] = q_out[i];
    }

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

    // `pose` already varies per-lane (it's the block itself), so it's
    // indexed normally like any other block array. GC2/GC4/GC6 (self-motion-
    // manifold selectors) are fixed for the whole planning problem, so
    // instead of being read from the input they're read off the `smm` class
    // member directly: this segment occupies the same 3 tape positions that
    // used to be GC2/GC4/GC6, but is named "smm" so the generated code emits
    // `smm[0]`, `smm[1]`, `smm[2]`. Output keeps the default `y[i]` naming,
    // matching the `y` variable already declared in parameterized_ik.
    const std::string lang = (language == "c++") ? "c++_block" : language;

    SegmentedVariableNameGenerator<double> nameGen(
        {{"pose", 7, true},
         {"psi", 1, false},
         {"smm", 3, true}});

    std::cout << "Generated the parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}

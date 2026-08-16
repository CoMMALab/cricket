#pragma once

#include <cricket/codegen.hh>

#include "../tracing/internal.hh"
#include "iiwa_parameterization.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cricket
{
    using namespace CppAD;
    using namespace CppAD::cg;

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

    auto ik_result = IiwaBimanualParameterization<T>(ad_inp);

    // Output layout: first n_q joint angles ("y"), followed by the 4 pre-clip
    // SafeArccos arguments ("u"). A |u[i]| > 1 means the pose/psi/GC
    // combination has no valid IK solution on this branch -- SafeArccos would
    // have silently clipped it -- so callers must reject rather than trust y.
    const size_t n_q = 14;
    const size_t n_unclipped = 4;
    const size_t n_out = n_q + n_unclipped;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_q; ++i)
    {
        data[i] = ik_result.q[i];
    }
    for (int i = 0; i < n_unclipped; ++i)
    {
        data[n_q + i] = ik_result.unclipped[i];
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

    // Codegen needs the block-oriented language for fk_template.hh's
    // FloatVector-based parameterized_ik, same remap used by the SE3 path.
    const std::string lang = (language == "c++") ? "c++_block" : language;

    SegmentedVariableNameGenerator<double> nameGen(
        {},
        {{"y", n_q, true}, {"u", n_unclipped, true}});

    std::cout << "Generated the parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}


// CppAD-CodeGen wrapper around IiwaSE3Parameterization -- the single-arm
// counterpart of RainbowIkCG (rainbow_ik_cg.hh), used by fk_template.hh's
// ParameterizedSpace for the `param_kind == "iiwa_se3"` case. Tape input
// layout (size 11): [0:7) end-effector pose (x, y, z, qx, qy, qz, qw) in the
// robot's own base frame, [7] psi (self-motion-manifold free parameter),
// [8:11) GC2/GC4/GC6 (shoulder/elbow/wrist configuration selectors, expected
// +-1). Output: "y" (7 joint angles), "u" (4 pre-clip SafeArccos arguments --
// see IKParamResult in iiwa_parameterization.hh for what a value outside
// [-1, 1] means and why callers must check it themselves rather than trust
// `y`). `loss` (see IKParamResult) is intentionally not part of the CG
// output here -- ParameterizedSpace::resolve_block only needs the hard
// accept/reject `u` gives it, not a smooth optimization target.
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

    auto ik_result = IiwaSE3Parameterization<T>(ad_inp);

    // Output layout: first n_q joint angles ("y"), followed by the 4 pre-clip
    // SafeArccos arguments ("u"). A |u[i]| > 1 means the pose/psi/GC
    // combination has no valid IK solution on this branch -- SafeArccos would
    // have silently clipped it -- so callers must reject rather than trust y.
    const size_t n_q = 7;
    const size_t n_unclipped = 4;
    const size_t n_out = n_q + n_unclipped;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_q; ++i)
    {
        data[i] = ik_result.q[i];
    }
    for (int i = 0; i < n_unclipped; ++i)
    {
        data[n_q + i] = ik_result.unclipped[i];
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
    // `smm[0]`, `smm[1]`, `smm[2]`. Output is split into `y[i]` (joint
    // angles, matching the `y` variable already declared in
    // parameterized_ik) and `u[i]` (pre-clip SafeArccos arguments, matching
    // the `u` variable declared there for the joint-limit-style rejection
    // check).
    const std::string lang = (language == "c++") ? "c++_block" : language;

    SegmentedVariableNameGenerator<double> nameGen(
        {{"pose", 7, true},
         {"psi", 1, false},
         {"smm", 3, true}},
        {{"y", n_q, true}, {"u", n_unclipped, true}});

    std::cout << "Generated the parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}
}  // namespace cricket

#pragma once

#include <cricket/codegen.hh>

#include "../tracing/internal.hh"
#include "fr3_parameterization.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cricket
{
    using namespace CppAD;
    using namespace CppAD::cg;

// CppAD-CodeGen wrapper around Fr3SE3Parameterization -- the FR3/Panda counterpart of
// IiwaSE3ParameterizationCG (iiwa_parameterization_gen.hh), used by fk_template.hh's
// ParameterizedSpace for the `param_kind == "fr3_se3"` case. Tape input layout (size 11):
// [0:7) end-effector pose (x, y, z, qx, qy, qz, qw) in the robot's own base frame, [7] `psi`
// (named to match IiwaSE3ParameterizationCG's tape/State layout, even though here it IS
// joint 7's angle directly rather than a shoulder-elbow-wrist self-motion parameter -- see
// Fr3SE3Parameterization's header in fr3_parameterization.hh), [8:11) case6_sel,
// case1_sel, unused (SolveArmBranchTaped's two real branch selectors, plus one deliberately
// unused slot kept only so this arm's `smm` stays 3-wide like iiwa_se3's GC2/GC4/GC6 -- see
// that function's header in fr3_parameterization.hh). Output: "y" (7 joint
// angles), "u" (1: `reach_violation` -- `> 0` means no valid IK solution on this branch;
// see Fr3IKParamResult for what this means and why callers must check it themselves rather
// than trust `y`). Unlike IiwaSE3ParameterizationCG's 4-element "u" (raw pre-clip SafeArccos
// arguments each individually checked against [-1, 1]), FR3's closed form already folds
// every domain-clamp and joint-limit residual into this one nonnegative scalar, so
// fk_template.hh's `param_kind == "fr3_se3"` resolve_block rejects on `u[0] > 0` instead of
// the iiwa_se3 branch's per-element [-1, 1] range check.
template <typename T>
auto Fr3SE3ParameterizationCG(
    const std::string &language,
    bool compute_gradient = false
)
{

    std::cout << "Generating task parameterized IK code for fr3..." << std::endl;
    const size_t num_inp = 7 + 1 + 1 + 1 + 1; // 7 for the pose, 1 for psi, 3 for case6_sel/case1_sel/unused

    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    auto ik_result = Fr3SE3Parameterization<T>(ad_inp);

    // Output layout: 7 joint angles ("y"), followed by the single folded reach-violation
    // scalar ("u") -- see this file's header comment for why FR3 exposes 1 unclipped value
    // where iiwa_se3 exposes 4.
    const size_t n_q = 7;
    const size_t n_unclipped = 1;
    const size_t n_out = n_q + n_unclipped;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_q; ++i)
    {
        data[i] = ik_result.q[i];
    }
    data[n_q] = ik_result.reach_violation;

    ADFun<CGD> fr3_param_func(ad_inp, data);
    std::cout << "Created the AD function." << std::endl;
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);

    handler.makeVariables(ind_vars);


    CppAD::vector<CGD> result = fr3_param_func.Forward(0, ind_vars);
    std::cout << "Ran the AD function." << std::endl;


    if (compute_gradient)
    {
      // this is the full jacobian
      CppAD::vector<CGD> jac_e_q = fr3_param_func.Jacobian(ind_vars);
      std::move(jac_e_q.begin(), jac_e_q.end(), std::back_inserter(result));
    }

    // Codegen needs the block-oriented language for fk_template.hh's FloatVector-based
    // parameterized_ik, same remap used by the iiwa_se3 path.
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

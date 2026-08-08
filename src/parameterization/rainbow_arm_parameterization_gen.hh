#pragma once

#include <cricket/codegen.hh>

#include "../tracing/internal.hh"
#include "rainbow_arm_parameterization.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <string>

namespace cricket
{
    using namespace CppAD;
    using namespace CppAD::cg;

// CppAD-CodeGen wrappers around RainbowLeftArmParameterization and
// RainbowRightArmParameterization (RainbowRightArmParameterizationCG is
// defined further down), in the same style as IiwaSE3ParameterizationCG in
// iiwa_parameterization_gen.hh.
//
// Tape input layout (size 11), grouped for SegmentedVariableNameGenerator:
//   "pose"  (7, block-indexed) -- end-effector (x, y, z, qx, qy, qz, qw)
//   "j15"   (1, not block-indexed) -- the free self-motion joint angle,
//           this arm's analogue of iiwa's "psi"; callers search over this
//           the same way IiwaSE3ParameterizationCG's callers search over psi
//   "gcp"   (3, block-indexed) -- elbow_sel (0/1), shoulder_sel (0/1),
//           wrist_sel (+-1), fixed for the whole planning problem, same
//           role as the "smm" segment (GC2/GC4/GC6) in
//           IiwaSE3ParameterizationCG
//
// Output layout: "y" (7 joint angles), "u" (the 3 pre-clip
// RainbowAsin/RainbowAcos arguments -- |u[i]| > 1 means the pose/j15/gcp
// combination has no valid IK solution on this branch), "reach_violation"
// (1, a smooth always-nonnegative measure of how far any of "u" overshot
// [-1, 1]; zero iff "y" is an exact IK solution), and "loss" (1, a smooth
// penalty over the same three arguments for gradient-based search over
// j15).
template <typename T>
auto RainbowLeftArmParameterizationCG(
    const std::string &language,
    bool compute_gradient = false
)
{

    std::cout << "Generating task parameterized IK code for the rainbow left arm..." << std::endl;
    const size_t num_inp = 7 + 1 + 3; // 7 for the pose, 1 for j15_free, 3 for elbow_sel/shoulder_sel/wrist_sel

    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    auto ik_result = RainbowLeftArmParameterization<T>(ad_inp);

    const size_t n_q = 7;
    const size_t n_unclipped = 3;
    const size_t n_reach_violation = 1;
    const size_t n_loss = 1;
    const size_t n_out = n_q + n_unclipped + n_reach_violation + n_loss;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_q; ++i)
    {
        data[i] = ik_result.q[i];
    }
    for (int i = 0; i < n_unclipped; ++i)
    {
        data[n_q + i] = ik_result.unclipped[i];
    }
    data[n_q + n_unclipped] = ik_result.reach_violation;
    data[n_q + n_unclipped + n_reach_violation] = ik_result.loss;

    ADFun<CGD> rainbow_param_func(ad_inp, data);
    std::cout << "Created the AD function." << std::endl;
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);

    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = rainbow_param_func.Forward(0, ind_vars);
    std::cout << "Ran the AD function." << std::endl;

    // Codegen needs the block-oriented language for fk_template.hh-style
    // FloatVector-based parameterized_ik, same remap used by the iiwa SE3
    // path.
    const std::string lang = (language == "c++") ? "c++_block" : language;

    if (compute_gradient)
    {
        // Gradient of every output (y, u, reach_violation, loss) w.r.t.
        // every input.
        CppAD::vector<CGD> jac_e_q = rainbow_param_func.Jacobian(ind_vars);
        // Callers only ever search over j15 (input index 7) to satisfy the
        // j15-dependent constraints, so slice out just that column. Row
        // n_q + n_unclipped + n_reach_violation of jac_j15 is
        // d(loss)/d(j15), the gradient used to drive that search.
        CppAD::vector<CGD> jac_e_j15(n_out);
        for (auto i = 0U; i < n_out; i++)
        {
            jac_e_j15[i] = jac_e_q[i * num_inp + 7]; // j15 is the 8th input
        }
        std::move(jac_e_j15.begin(), jac_e_j15.end(), std::back_inserter(result));

        SegmentedVariableNameGenerator<double> nameGen(
            {{"pose", 7, true},
             {"j15", 1, false},
             {"gcp", 3, true}},
            {{"y", n_q, true},
             {"u", n_unclipped, true},
             {"reach_violation", n_reach_violation, true},
             {"loss", n_loss, true},
             {"jac_j15", n_out, true}});

        std::cout << "Generated the parameterized IK code." << std::endl;
        return Traced{
            generate_code(handler, result, lang, nameGen),
            handler.getTemporaryVariableCount(),
            result.size()};
    }

    SegmentedVariableNameGenerator<double> nameGen(
        {{"pose", 7, true},
         {"j15", 1, false},
         {"gcp", 3, true}},
        {{"y", n_q, true},
         {"u", n_unclipped, true},
         {"reach_violation", n_reach_violation, true},
         {"loss", n_loss, true}});

    std::cout << "Generated the parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}

// CppAD-CodeGen wrapper around RainbowRightArmParameterization -- the
// right-arm counterpart of RainbowLeftArmParameterizationCG above (same
// input/output layout, "j24" in place of "j15" for the free self-motion
// joint angle since that's this arm's free joint index).
template <typename T>
auto RainbowRightArmParameterizationCG(
    const std::string &language,
    bool compute_gradient = false
)
{

    std::cout << "Generating task parameterized IK code for the rainbow right arm..." << std::endl;
    const size_t num_inp = 7 + 1 + 3; // 7 for the pose, 1 for j24_free, 3 for elbow_sel/shoulder_sel/wrist_sel

    ADVectorXs ad_inp(num_inp);
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = (T)(0.0001);
    }
    Independent(ad_inp);

    auto ik_result = RainbowRightArmParameterization<T>(ad_inp);

    const size_t n_q = 7;
    const size_t n_unclipped = 3;
    const size_t n_reach_violation = 1;
    const size_t n_loss = 1;
    const size_t n_out = n_q + n_unclipped + n_reach_violation + n_loss;
    ADVectorXs data(n_out);
    for (int i = 0; i < n_q; ++i)
    {
        data[i] = ik_result.q[i];
    }
    for (int i = 0; i < n_unclipped; ++i)
    {
        data[n_q + i] = ik_result.unclipped[i];
    }
    data[n_q + n_unclipped] = ik_result.reach_violation;
    data[n_q + n_unclipped + n_reach_violation] = ik_result.loss;

    ADFun<CGD> rainbow_param_func(ad_inp, data);
    std::cout << "Created the AD function." << std::endl;
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);

    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = rainbow_param_func.Forward(0, ind_vars);
    std::cout << "Ran the AD function." << std::endl;

    // Codegen needs the block-oriented language for fk_template.hh-style
    // FloatVector-based parameterized_ik, same remap used by the iiwa SE3
    // path.
    const std::string lang = (language == "c++") ? "c++_block" : language;

    if (compute_gradient)
    {
        // Gradient of every output (y, u, reach_violation, loss) w.r.t.
        // every input.
        CppAD::vector<CGD> jac_e_q = rainbow_param_func.Jacobian(ind_vars);
        // Callers only ever search over j24 (input index 7) to satisfy the
        // j24-dependent constraints, so slice out just that column. Row
        // n_q + n_unclipped + n_reach_violation of jac_j24 is
        // d(loss)/d(j24), the gradient used to drive that search.
        CppAD::vector<CGD> jac_e_j24(n_out);
        for (auto i = 0U; i < n_out; i++)
        {
            jac_e_j24[i] = jac_e_q[i * num_inp + 7]; // j24 is the 8th input
        }
        std::move(jac_e_j24.begin(), jac_e_j24.end(), std::back_inserter(result));

        SegmentedVariableNameGenerator<double> nameGen(
            {{"pose", 7, true},
             {"j24", 1, false},
             {"gcp", 3, true}},
            {{"y", n_q, true},
             {"u", n_unclipped, true},
             {"reach_violation", n_reach_violation, true},
             {"loss", n_loss, true},
             {"jac_j24", n_out, true}});

        std::cout << "Generated the parameterized IK code." << std::endl;
        return Traced{
            generate_code(handler, result, lang, nameGen),
            handler.getTemporaryVariableCount(),
            result.size()};
    }

    SegmentedVariableNameGenerator<double> nameGen(
        {{"pose", 7, true},
         {"j24", 1, false},
         {"gcp", 3, true}},
        {{"y", n_q, true},
         {"u", n_unclipped, true},
         {"reach_violation", n_reach_violation, true},
         {"loss", n_loss, true}});

    std::cout << "Generated the parameterized IK code." << std::endl;
    return Traced{
        generate_code(handler, result, lang, nameGen),
        handler.getTemporaryVariableCount(),
        result.size()};
}
}  // namespace cricket

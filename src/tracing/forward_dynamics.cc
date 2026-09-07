#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/aba.hpp>

namespace cricket
{
    auto trace_forward_dynamics(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        const auto nq = static_cast<std::size_t>(model.nq);
        const auto nv = static_cast<std::size_t>(model.nv);
        const auto n_input = nq + 2 * nv;

        ADModel ad_model = model.cast<ADCG>();
        ADData ad_data(ad_model);
        ADVectorXs ad_input(n_input);
        ADVectorXs ad_acceleration(nv);

        for (std::size_t i = 0; i < n_input; ++i)
        {
            ad_input[i] = ADCG(0.0);
        }

        CppAD::Independent(ad_input);

        const auto ad_q = ad_input.head(nq);
        const auto ad_v = ad_input.segment(nq, nv);
        const auto ad_tau = ad_input.tail(nv);
        ad_acceleration = pinocchio::aba(ad_model, ad_data, ad_q, ad_v, ad_tau);

        CppAD::ADFun<CGD> dynamics_func(ad_input, ad_acceleration);

        CppAD::cg::CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(n_input);
        handler.makeVariables(ind_vars);
        CppAD::vector<CGD> result = dynamics_func.Forward(0, ind_vars);

        return emit_traced(handler, result, language, nv);
    }
}  // namespace cricket

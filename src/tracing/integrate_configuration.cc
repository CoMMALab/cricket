#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/joint-configuration.hpp>

namespace cricket
{
    auto trace_integrate_configuration(const pinocchio::Model &model, const std::string &language)
        -> Traced
    {
        const auto nq = static_cast<std::size_t>(model.nq);
        const auto nv = static_cast<std::size_t>(model.nv);
        const auto n_input = nq + nv;

        ADModel ad_model = model.cast<ADCG>();
        ADVectorXs ad_input(n_input);
        ADVectorXs ad_output(nq);

        for (std::size_t i = 0; i < n_input; ++i)
        {
            ad_input[i] = ADCG(0.0);
        }

        CppAD::Independent(ad_input);

        const auto ad_q = ad_input.head(nq);
        const auto ad_dq = ad_input.tail(nv);
        pinocchio::integrate(ad_model, ad_q, ad_dq, ad_output);

        CppAD::ADFun<CGD> integrate_func(ad_input, ad_output);

        CppAD::cg::CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(n_input);
        handler.makeVariables(ind_vars);
        CppAD::vector<CGD> result = integrate_func.Forward(0, ind_vars);

        return emit_traced(handler, result, language, nq);
    }
}  // namespace cricket

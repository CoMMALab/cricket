#include "tracing.hh"
#include "tracing/internal.hh"

#include <pinocchio/algorithm/joint-configuration.hpp>

namespace cricket
{

auto trace_distance(const pinocchio::Model &model, const std::string &language) -> Traced
{
    auto nq = model.nq;
    std::size_t n_input = 2 * nq;  // a, b

    // Cast model to AD scalar type for use with Pinocchio's distance
    ADModel ad_model = model.cast<ADCG>();

    ADVectorXs ad_input(n_input);
    ADVectorXs ad_out(1);

    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_input);

    // Extract a, b from input
    ADVectorXs ad_a = ad_input.head(nq);
    ADVectorXs ad_b = ad_input.tail(nq);

    ad_out[0] = pinocchio::distance(ad_model, ad_a, ad_b);

    CppAD::ADFun<CGD> dist_func(ad_input, ad_out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = dist_func.Forward(0, ind_vars);

    auto nq_size = static_cast<std::size_t>(nq);
    SegmentedVariableNameGenerator<double> nameGen({{"a", nq_size, true}, {"b", nq_size, true}});

    return Traced{generate_code(handler, result, language, nameGen), handler.getTemporaryVariableCount(), 1};
}

}  // namespace cricket

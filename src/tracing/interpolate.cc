#include "tracing.hh"
#include "tracing/internal.hh"

#include <pinocchio/algorithm/joint-configuration.hpp>

namespace cricket
{

auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced
{
    auto nq = model.nq;
    std::size_t n_input = 2 * nq + 1;  // a, b, t

    // Cast model to AD scalar type for use with Pinocchio's interpolate
    ADModel ad_model = model.cast<ADCG>();

    ADVectorXs ad_input(n_input);
    ADVectorXs ad_out(nq);

    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_input);

    // Extract a, b, t from input
    ADVectorXs ad_a = ad_input.head(nq);
    ADVectorXs ad_b = ad_input.segment(nq, nq);
    ADCG t = ad_input[2 * nq];

    pinocchio::interpolate(ad_model, ad_a, ad_b, t, ad_out);

    CppAD::ADFun<CGD> interp_func(ad_input, ad_out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = interp_func.Forward(0, ind_vars);

    // Use custom variable naming: a[0..nq-1], b[0..nq-1], t
    SegmentedVariableNameGenerator<double> nameGen({
        {"a", static_cast<std::size_t>(nq), true},
        {"b", static_cast<std::size_t>(nq), true},
        {"t", 1, false}
    });

    return Traced{generate_code(handler, result, language, nameGen), handler.getTemporaryVariableCount(),
                  static_cast<std::size_t>(nq)};
}

}  // namespace cricket

#include "tracing.hh"
#include "tracing/internal.hh"

#include <pinocchio/algorithm/joint-configuration.hpp>

namespace cricket
{

namespace
{

auto trace_interpolate_impl(
    const pinocchio::Model &model,
    const std::string &language,
    std::vector<VarSegment> input_segments,
    std::vector<VarSegment> output_segments) -> Traced
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

    SegmentedVariableNameGenerator<double> nameGen(std::move(input_segments), std::move(output_segments));

    return Traced{generate_code(handler, result, language, nameGen), handler.getTemporaryVariableCount(),
                  static_cast<std::size_t>(nq)};
}

}  // namespace

auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced
{
    auto nq = static_cast<std::size_t>(model.nq);
    return trace_interpolate_impl(
        model,
        language,
        {{"a", nq, true}, {"b", nq, true}, {"t", 1, false}},
        {});
}

auto trace_interpolate_block(const pinocchio::Model &model, const std::string &language) -> Traced
{
    auto nq = static_cast<std::size_t>(model.nq);
    return trace_interpolate_impl(
        model,
        language,
        {{"a", nq, true, ".broadcast(", ")"}, {"b", nq, true, ".broadcast(", ")"}, {"t", 1, false}},
        {{"out", nq, true}});
}

}  // namespace cricket

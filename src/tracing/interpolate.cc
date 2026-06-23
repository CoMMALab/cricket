#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/joint-configuration.hpp>

#include <utility>
#include <vector>

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
            const auto nq = model.nq;
            const std::size_t n_input = 2 * nq + 1;  // a, b, t

            ADModel ad_model = model.cast<ADCG>();

            ADVectorXs ad_input(n_input);
            ADVectorXs ad_out(nq);

            for (std::size_t i = 0U; i < n_input; ++i)
            {
                ad_input[i] = ADCG(0.0);
            }

            CppAD::Independent(ad_input);

            ADVectorXs ad_a = ad_input.head(nq);
            ADVectorXs ad_b = ad_input.segment(nq, nq);
            ADCG t = ad_input[2 * nq];

            pinocchio::interpolate(ad_model, ad_a, ad_b, t, ad_out);

            CppAD::ADFun<CGD> interp_func(ad_input, ad_out);

            CppAD::cg::CodeHandler<double> handler;
            CppAD::vector<CGD> ind_vars(n_input);
            handler.makeVariables(ind_vars);

            CppAD::vector<CGD> result = interp_func.Forward(0, ind_vars);

            SegmentedVariableNameGenerator<double> nameGen(
                std::move(input_segments), std::move(output_segments));

            return Traced{
                generate_code(handler, result, language, nameGen),
                handler.getTemporaryVariableCount(),
                static_cast<std::size_t>(nq)};
        }
    }  // namespace

    auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        const auto nq = static_cast<std::size_t>(model.nq);
        return trace_interpolate_impl(
            model,
            language,
            {{"a", nq, true}, {"b", nq, true}, {"t", 1, false}},
            {});
    }

    auto trace_interpolate_block(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        // For C++ we need the SIMD-friendly LanguageCVampBlock, which emits
        // mask/blend instead of `if (cond) y = ...; else ...` so the trace
        // lowers cleanly to FloatVector<rake, 1>. Rust path is unchanged.
        const std::string lang = (language == "c++") ? "c++_block" : language;
        const auto nq = static_cast<std::size_t>(model.nq);
        return trace_interpolate_impl(
            model,
            lang,
            {{"a", nq, true, ".broadcast(", ")"}, {"b", nq, true, ".broadcast(", ")"}, {"t", 1, false}},
            {{"out", nq, true}});
    }
}  // namespace cricket

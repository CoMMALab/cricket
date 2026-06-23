#pragma once

#include "../codegen/pinocchio_cppadcg.hh"
#include "../codegen/lang_cpp.hh"
#include "../codegen/lang_cpp_block.hh"
#include "../codegen/lang_rust.hh"
#include "../codegen/lang_name_gen.hh"

#include <fmt/format.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace cricket
{
    using CGD = CppAD::cg::CG<double>;
    using ADCG = CppAD::AD<CGD>;

    using ADModel = pinocchio::ModelTpl<ADCG>;
    using ADData = pinocchio::DataTpl<ADCG>;
    using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;

    template <typename NameGen>
    inline auto generate_code(
        CppAD::cg::CodeHandler<double> &handler,
        CppAD::vector<CGD> &result,
        const std::string &language,
        NameGen &nameGen) -> std::string
    {
        std::ostringstream function_code;

        if (language == "c++")
        {
            CppAD::cg::LanguageCCustom<double> langC("double");
            handler.generateCode(function_code, langC, result, nameGen);
        }
        else if (language == "c++_block")
        {
            // SIMD-rake variant: emits VAMP mask/blend for conditional
            // assignments instead of if/else. Used by trace_interpolate_block
            // so pinocchio's branching geodesic kernels lower to vectorizable
            // code over a FloatVector<rake, 1>.
            CppAD::cg::LanguageCVampBlock<double> langC("double");
            handler.generateCode(function_code, langC, result, nameGen);
        }
        else if (language == "rust")
        {
            CppAD::cg::LanguageRust<double> langRust("double");
            handler.generateCode(function_code, langRust, result, nameGen);
        }
        else
        {
            throw std::runtime_error(fmt::format("unsupported language {}", language));
        }

        return function_code.str();
    }

    inline auto generate_code(
        CppAD::cg::CodeHandler<double> &handler,
        CppAD::vector<CGD> &result,
        const std::string &language) -> std::string
    {
        CppAD::cg::LangCDefaultVariableNameGenerator<double> nameGen;
        return generate_code(handler, result, language, nameGen);
    }
}  // namespace cricket

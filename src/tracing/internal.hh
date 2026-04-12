#pragma once

#include "codegen/pinocchio_cppadcg.hh"
#include "codegen/lang_cpp.hh"
#include "codegen/lang_rust.hh"
#include "codegen/lang_name_gen.hh"

#include <fmt/core.h>

#include <sstream>
#include <stdexcept>

namespace cricket
{

// Typedef for AD types
using CGD = CppAD::cg::CG<double>;
using ADCG = CppAD::AD<CGD>;

using ADModel = pinocchio::ModelTpl<ADCG>;
using ADData = pinocchio::DataTpl<ADCG>;
using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;

// Struct-of-arrays sphere output layout (optional)
struct SphereOutputLayout
{
    std::size_t x_offset;
    std::size_t y_offset;
    std::size_t z_offset;
    std::size_t r_offset;
};

template <typename NameGen>
auto generate_code(
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

#pragma once

#include "pinocchio_cppadcg.hh"
#include "lang_cpp.hh"
#include "lang_cpp_block.hh"
#include "lang_rust.hh"
#include "lang_name_gen.hh"

#include <fmt/format.h>

#include <sstream>
#include <stdexcept>
#include <string>

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


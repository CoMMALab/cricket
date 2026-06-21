#pragma once

#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include <memory>
#include <string>
#include <vector>

namespace cricket::jit
{
    struct CompileOptions
    {
        std::string std_flag = "-std=c++17";
        std::string opt_flag = "-O3";
        std::vector<std::string> default_flags;
        std::vector<std::string> include_dirs;
        std::vector<std::string> system_include_dirs;
        std::vector<std::string> defines;
        std::vector<std::string> extra_flags;
        std::string module_id;
    };

    class ClangCompiler
    {
    public:
        ClangCompiler();
        ~ClangCompiler();

        ClangCompiler(const ClangCompiler &) = delete;
        ClangCompiler &operator=(const ClangCompiler &) = delete;

        auto compile(const std::string &source, const CompileOptions &opts) -> llvm::orc::ThreadSafeModule;
    };

    auto hash_source(const std::string &source, const CompileOptions &opts) -> std::string;
}  // namespace cricket::jit

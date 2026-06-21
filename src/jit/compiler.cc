#include <cricket/jit/compiler.hh>

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Job.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/SHA1.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>

#include <stdexcept>
#include <utility>

namespace
{
    constexpr const char *kVirtualSourceName = "cricket_jit_input.cc";
}  // namespace

namespace cricket::jit
{
    auto hash_source(const std::string &source, const CompileOptions &opts) -> std::string
    {
        llvm::SHA1 sha;
        sha.update(source);
        sha.update(opts.std_flag);
        sha.update(opts.opt_flag);

        for (const auto &f : opts.default_flags)
        {
            sha.update(f);
        }

        for (const auto &f : opts.extra_flags)
        {
            sha.update(f);
        }

        for (const auto &d : opts.defines)
        {
            sha.update(d);
        }

        for (const auto &i : opts.include_dirs)
        {
            sha.update(i);
        }

        for (const auto &i : opts.system_include_dirs)
        {
            sha.update(i);
        }

        return "cricket-" + llvm::toHex(sha.final(), true);
    }
}  // namespace cricket::jit

namespace cricket::jit
{
    ClangCompiler::ClangCompiler() = default;
    ClangCompiler::~ClangCompiler() = default;

    auto ClangCompiler::compile(const std::string &source, const CompileOptions &opts)
        -> llvm::orc::ThreadSafeModule
    {
        constexpr std::size_t kFixedArgCount = 8;

        std::vector<std::string> arg_storage;
        arg_storage.emplace_back("clang");
        arg_storage.emplace_back(opts.std_flag);
        arg_storage.emplace_back(opts.opt_flag);
        arg_storage.emplace_back("-c");
        arg_storage.emplace_back("-emit-llvm");

        for (const auto &f : opts.default_flags)
        {
            arg_storage.emplace_back(f);
        }

        for (const auto &i : opts.include_dirs)
        {
            arg_storage.emplace_back("-I" + i);
        }

        for (const auto &i : opts.system_include_dirs)
        {
            arg_storage.emplace_back("-isystem" + i);
        }

        for (const auto &d : opts.defines)
        {
            arg_storage.emplace_back("-D" + d);
        }

        for (const auto &f : opts.extra_flags)
        {
            arg_storage.emplace_back(f);
        }

        arg_storage.emplace_back("-x");
        arg_storage.emplace_back("c++");
        arg_storage.emplace_back(kVirtualSourceName);

        std::vector<const char *> driver_args;
        driver_args.reserve(arg_storage.size());
        for (const auto &s : arg_storage)
        {
            driver_args.emplace_back(s.c_str());
        }

        std::string diag_text;
        llvm::raw_string_ostream diag_stream(diag_text);
        clang::DiagnosticOptions diag_opts;
        auto *diag_printer = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
        auto diag_ids = llvm::makeIntrusiveRefCnt<clang::DiagnosticIDs>();
        clang::DiagnosticsEngine diags(diag_ids, diag_opts, diag_printer, true);

        auto clang_path = llvm::sys::findProgramByName("clang");
        if (not clang_path)
        {
            throw std::runtime_error("cricket::jit: cannot find clang on PATH");
        }

        clang::driver::Driver driver(*clang_path, llvm::sys::getDefaultTargetTriple(), diags);
        driver.setCheckInputsExist(false);

        std::unique_ptr<clang::driver::Compilation> compilation(driver.BuildCompilation(driver_args));

        if (not compilation or compilation->containsError())
        {
            diag_stream.flush();
            throw std::runtime_error("cricket::jit: driver compilation build failed:\n" + diag_text);
        }

        const auto &jobs = compilation->getJobs();
        if (jobs.size() != 1)
        {
            throw std::runtime_error("cricket::jit: expected 1 cc1 job, got " + std::to_string(jobs.size()));
        }

        const auto &cmd = *jobs.begin();
        const auto &cc1_args_list = cmd.getArguments();

        std::vector<const char *> cc1_args;
        cc1_args.reserve(cc1_args_list.size());
        for (const auto &a : cc1_args_list)
        {
            cc1_args.push_back(a);
        }

        auto invocation = std::make_shared<clang::CompilerInvocation>();
        if (not clang::CompilerInvocation::CreateFromArgs(*invocation, cc1_args, diags))
        {
            diag_stream.flush();
            throw std::runtime_error(
                "cricket::jit: CompilerInvocation::CreateFromArgs failed:\n" + diag_text);
        }

        clang::CompilerInstance ci(invocation);
        ci.createDiagnostics(diag_printer, false);

        auto source_buf = llvm::MemoryBuffer::getMemBufferCopy(source, kVirtualSourceName);
        ci.getPreprocessorOpts().addRemappedFile(kVirtualSourceName, source_buf.release());

        auto context = std::make_unique<llvm::LLVMContext>();
        auto action = std::make_unique<clang::EmitLLVMOnlyAction>(context.get());

        if (not ci.ExecuteAction(*action))
        {
            diag_stream.flush();
            throw std::runtime_error("cricket::jit: compilation failed:\n" + diag_text);
        }

        auto module = action->takeModule();
        if (not module)
        {
            diag_stream.flush();
            throw std::runtime_error("cricket::jit: action produced no module:\n" + diag_text);
        }

        std::string id = opts.module_id.empty() ? hash_source(source, opts) : opts.module_id;
        module->setModuleIdentifier(id);
        module->setSourceFileName(id);

        return llvm::orc::ThreadSafeModule(std::move(module), std::move(context));
    }
}  // namespace cricket::jit

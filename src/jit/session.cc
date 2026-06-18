#include <cricket/jit/session.hh>

#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/Support/TargetSelect.h>

#include <mutex>
#include <stdexcept>
#include <utility>

namespace lo = llvm::orc;

namespace
{
    std::once_flag g_init_targets;

    auto init_native_target() -> void
    {
        std::call_once(
            g_init_targets,
            []()
            {
                llvm::InitializeNativeTarget();
                llvm::InitializeNativeTargetAsmPrinter();
                llvm::InitializeNativeTargetAsmParser();
            });
    }

    auto build_jit(std::shared_ptr<cricket::jit::DiskObjectCache> cache) -> std::unique_ptr<lo::LLJIT>
    {
        init_native_target();

        lo::LLJITBuilder builder;

        builder.setCompileFunctionCreator(
            [cache](lo::JITTargetMachineBuilder JTMB)
                -> llvm::Expected<std::unique_ptr<lo::IRCompileLayer::IRCompiler>>
            {
                JTMB.setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
                auto TM = JTMB.createTargetMachine();
                if (not TM)
                {
                    return TM.takeError();
                }
                return std::make_unique<lo::TMOwningSimpleCompiler>(std::move(*TM), cache.get());
            });

        auto jit = builder.create();
        if (not jit)
        {
            throw std::runtime_error(
                "cricket::jit: LLJIT construction failed: " + llvm::toString(jit.takeError()));
        }
        return std::move(*jit);
    }
}  // namespace

namespace cricket::jit
{
    JitSession::JitSession(std::shared_ptr<DiskObjectCache> cache)
      : cache_(std::move(cache)), jit_(build_jit(cache_))
    {
        auto generator =
            lo::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit_->getDataLayout().getGlobalPrefix());
        if (not generator)
        {
            throw std::runtime_error(
                "cricket::jit: DynamicLibrarySearchGenerator failed: " +
                llvm::toString(generator.takeError()));
        }
        jit_->getMainJITDylib().addGenerator(std::move(*generator));
    }

    JitSession::~JitSession() = default;

    auto JitSession::add_module(lo::ThreadSafeModule tsm) -> llvm::Error
    {
        return jit_->addIRModule(std::move(tsm));
    }

    auto JitSession::add_external_symbol(const std::string &name, void *addr) -> llvm::Error
    {
        // intern() mutates the session's string pool — needs a non-const ref.
        auto &es = jit_->getExecutionSession();
        const auto mangled = es.intern(name);
        lo::SymbolMap syms;
        syms[mangled] = {
            lo::ExecutorAddr::fromPtr(addr), llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Absolute};
        return jit_->getMainJITDylib().define(lo::absoluteSymbols(std::move(syms)));
    }

    auto JitSession::lookup(const std::string &symbol) -> llvm::Expected<lo::ExecutorAddr>
    {
        return jit_->lookup(symbol);
    }
}  // namespace cricket::jit

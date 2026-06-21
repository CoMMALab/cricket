#pragma once

#include <cricket/jit/object_cache.hh>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h>

#include <memory>
#include <string>

namespace lo = llvm::orc;

namespace cricket::jit
{
    class JitSession
    {
    public:
        explicit JitSession(std::shared_ptr<DiskObjectCache> cache = nullptr);
        ~JitSession();

        JitSession(const JitSession &) = delete;
        JitSession &operator=(const JitSession &) = delete;

        auto add_module(lo::ThreadSafeModule tsm) -> llvm::Error;

        // Add a pre-compiled object file directly, skipping clang + LLVM IR
        // codegen. Pair with hash_source + DiskObjectCache::load_object for
        // sub-second warm loads on cache hits.
        auto add_object_file(std::unique_ptr<llvm::MemoryBuffer> obj) -> llvm::Error;

        auto add_external_symbol(const std::string &name, void *addr) -> llvm::Error;

        auto lookup(const std::string &symbol) -> llvm::Expected<lo::ExecutorAddr>;

        template <typename Fn>
        auto lookup_fn(const std::string &symbol) -> llvm::Expected<Fn *>
        {
            auto addr = lookup(symbol);
            if (not addr)
            {
                return addr.takeError();
            }
            return addr->toPtr<Fn *>();
        }

    private:
        std::shared_ptr<DiskObjectCache> cache_;
        std::unique_ptr<lo::LLJIT> jit_;
    };
}  // namespace cricket::jit

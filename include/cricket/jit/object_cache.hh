#pragma once

#include <llvm/ExecutionEngine/ObjectCache.h>

#include <filesystem>
#include <memory>

namespace cricket::jit
{
    class DiskObjectCache : public llvm::ObjectCache
    {
    public:
        explicit DiskObjectCache(std::filesystem::path dir);

        auto notifyObjectCompiled(const llvm::Module *M, llvm::MemoryBufferRef Obj) -> void override;

        auto getObject(const llvm::Module *M) -> std::unique_ptr<llvm::MemoryBuffer> override;

        auto directory() const -> const std::filesystem::path &
        {
            return dir_;
        }

    private:
        std::filesystem::path dir_;
    };

    auto default_cache_dir() -> std::filesystem::path;
}  // namespace cricket::jit

#include <cricket/jit/object_cache.hh>

#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <unistd.h>

#include <cstdlib>
#include <stdexcept>
#include <system_error>

namespace cricket::jit
{
    DiskObjectCache::DiskObjectCache(std::filesystem::path dir) : dir_(std::move(dir))
    {
        std::error_code ec;
        std::filesystem::create_directories(dir_, ec);
        if (ec)
        {
            throw std::runtime_error(
                "cricket::jit: cannot create cache dir " + dir_.string() + ": " + ec.message());
        }
    }

    auto DiskObjectCache::notifyObjectCompiled(const llvm::Module *M, llvm::MemoryBufferRef Obj) -> void
    {
        const auto path = dir_ / (M->getModuleIdentifier() + ".o");
        const auto tmp = std::filesystem::path(path).concat(".tmp");

        std::error_code ec;
        llvm::raw_fd_ostream out(tmp.string(), ec, llvm::sys::fs::OF_None);

        if (ec)
        {
            return;
        }

        out.write(Obj.getBufferStart(), Obj.getBufferSize());

        if (out.has_error())
        {
            out.clear_error();
            std::filesystem::remove(tmp, ec);
            return;
        }

        out.close();

        std::filesystem::rename(tmp, path, ec);
        if (ec)
        {
            std::filesystem::remove(tmp, ec);
        }
    }

    auto DiskObjectCache::getObject(const llvm::Module *M) -> std::unique_ptr<llvm::MemoryBuffer>
    {
        const auto path = dir_ / (M->getModuleIdentifier() + ".o");
        auto buf = llvm::MemoryBuffer::getFile(path.string(), false, false);
        if (not buf)
        {
            return nullptr;
        }
        return std::move(*buf);
    }

    auto default_cache_dir() -> std::filesystem::path
    {
        if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr and xdg[0] != '\0')
        {
            return std::filesystem::path(xdg) / "cricket";
        }

        if (const char *home = std::getenv("HOME"); home != nullptr and home[0] != '\0')
        {
            return std::filesystem::path(home) / ".cache" / "cricket";
        }

        return std::filesystem::temp_directory_path() / "cricket";
    }
}  // namespace cricket::jit

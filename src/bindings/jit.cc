// Python bindings for cricket's runtime JIT (LLVM ORC) so callers can compile a
// C++ translation unit at runtime and pull out function pointers — used by
// pyroffi to JIT-compile a robot-specialised VAMP collision checker and hand its
// XLA FFI handler to JAX.
//
// Kept deliberately small: a single `JitSession` wrapper that owns a clang
// front-end, an LLJIT, and an on-disk object cache.  `add_source` compiles +
// links a TU; `handler_capsule` looks up an `extern "C" void *getter()` symbol,
// calls it, and returns the resulting handler pointer wrapped in a PyCapsule
// suitable for `jax.ffi.register_ffi_target`.

#include <cricket/jit/compiler.hh>
#include <cricket/jit/object_cache.hh>
#include <cricket/jit/session.hh>

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <llvm/Support/Error.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{
    // A niladic getter emitted (with C linkage) by the per-robot translation
    // unit; returns the address of an XLA FFI custom-call handler.
    using HandlerGetter = void *(*)();

    class PyJitSession
    {
    public:
        explicit PyJitSession(const std::optional<std::filesystem::path> &cache_dir)
        {
            const auto dir = cache_dir.value_or(cricket::jit::default_cache_dir());
            std::filesystem::create_directories(dir);
            cache_ = std::make_shared<cricket::jit::DiskObjectCache>(dir);
            session_ = std::make_unique<cricket::jit::JitSession>(cache_);
        }

        // Fast path: if an object compiled for `module_id` is already on disk,
        // link it directly and skip the clang front-end entirely.  Returns true
        // on a cache hit.
        auto try_load_cached(const std::string &module_id) -> bool
        {
            auto obj = cache_->load_object(module_id);
            if (not obj)
            {
                return false;
            }
            if (auto err = session_->add_object_file(std::move(obj)))
            {
                throw std::runtime_error(
                    "cricket.jit: add_object_file failed: " + llvm::toString(std::move(err)));
            }
            return true;
        }

        auto add_source(const std::string &source, const cricket::jit::CompileOptions &opts) -> void
        {
            auto tsm = compiler_.compile(source, opts);
            if (auto err = session_->add_module(std::move(tsm)))
            {
                throw std::runtime_error(
                    "cricket.jit: add_module failed: " + llvm::toString(std::move(err)));
            }
        }

        auto add_external_symbol(const std::string &name, std::uintptr_t addr) -> void
        {
            if (auto err = session_->add_external_symbol(name, reinterpret_cast<void *>(addr)))
            {
                throw std::runtime_error(
                    "cricket.jit: add_external_symbol failed: " + llvm::toString(std::move(err)));
            }
        }

        auto lookup_address(const std::string &symbol) -> std::uintptr_t
        {
            auto addr = session_->lookup(symbol);
            if (not addr)
            {
                throw std::runtime_error(
                    "cricket.jit: lookup('" + symbol + "') failed: " + llvm::toString(addr.takeError()));
            }
            return addr->getValue();
        }

        // Look up a `void *getter()` symbol, call it, and wrap the returned
        // handler pointer in a PyCapsule for jax.ffi.register_ffi_target.
        auto handler_capsule(const std::string &getter_symbol) -> nb::capsule
        {
            auto fn = session_->lookup_fn<void *()>(getter_symbol);
            if (not fn)
            {
                throw std::runtime_error(
                    "cricket.jit: lookup_fn('" + getter_symbol +
                    "') failed: " + llvm::toString(fn.takeError()));
            }
            void *handler = (*fn.get())();
            if (handler == nullptr)
            {
                throw std::runtime_error("cricket.jit: handler getter '" + getter_symbol + "' returned null");
            }
            return nb::capsule(handler);
        }

        auto cache_dir() const -> std::filesystem::path
        {
            return cache_->directory();
        }

    private:
        std::shared_ptr<cricket::jit::DiskObjectCache> cache_;
        std::unique_ptr<cricket::jit::JitSession> session_;
        cricket::jit::ClangCompiler compiler_;
    };
}  // namespace

namespace cricket
{
    void init_jit(nb::module_ &m)
    {
        auto jit = m.def_submodule("jit", "Runtime JIT compilation (LLVM ORC).");

        nb::class_<jit::CompileOptions>(jit, "CompileOptions")
            .def(nb::init<>())
            .def_rw("std_flag", &jit::CompileOptions::std_flag)
            .def_rw("opt_flag", &jit::CompileOptions::opt_flag)
            .def_rw("default_flags", &jit::CompileOptions::default_flags)
            .def_rw("include_dirs", &jit::CompileOptions::include_dirs)
            .def_rw("system_include_dirs", &jit::CompileOptions::system_include_dirs)
            .def_rw("defines", &jit::CompileOptions::defines)
            .def_rw("extra_flags", &jit::CompileOptions::extra_flags)
            .def_rw("module_id", &jit::CompileOptions::module_id);

        nb::class_<PyJitSession>(jit, "JitSession")
            .def(
                nb::init<const std::optional<std::filesystem::path> &>(),
                "cache_dir"_a = nb::none(),
                "Create a JIT session backed by an on-disk object cache.")
            .def(
                "try_load_cached",
                &PyJitSession::try_load_cached,
                "module_id"_a,
                "Link a previously-cached object for module_id, skipping clang. "
                "Returns True on a cache hit.")
            .def(
                "add_source",
                &PyJitSession::add_source,
                "source"_a,
                "options"_a,
                "Compile a C++ translation unit and add it to the session.")
            .def(
                "add_external_symbol",
                &PyJitSession::add_external_symbol,
                "name"_a,
                "address"_a,
                "Define an absolute external symbol (address as an integer).")
            .def(
                "lookup_address",
                &PyJitSession::lookup_address,
                "symbol"_a,
                "Return the JIT address of a symbol as an integer.")
            .def(
                "handler_capsule",
                &PyJitSession::handler_capsule,
                "getter_symbol"_a,
                "Call an `extern \"C\" void *getter()` symbol and return its result "
                "wrapped in a PyCapsule for jax.ffi.register_ffi_target.")
            .def_prop_ro("cache_dir", &PyJitSession::cache_dir);

        jit.def(
            "hash_source",
            [](const std::string &source, const jit::CompileOptions &opts)
            { return jit::hash_source(source, opts); },
            "source"_a,
            "options"_a,
            "Stable content hash of (source, options) for cache keying.");

        jit.def(
            "default_cache_dir",
            []() { return jit::default_cache_dir(); },
            "Default on-disk object cache directory.");
    }
}  // namespace cricket

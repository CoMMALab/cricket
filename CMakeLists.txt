cmake_minimum_required(VERSION 3.10)
project(cricket
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0175 OLD)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CRICKET_BUILD_JIT    "Build the JIT runtime" ON)
option(CRICKET_BUILD_PYTHON "Build the cricket Python extension" OFF)

# Download and setup CPM
if(NOT DEFINED ENV{CPM_SOURCE_CACHE})
  set(ENV{CPM_SOURCE_CACHE} ${CMAKE_CURRENT_SOURCE_DIR}/.cpm-cache)
endif()

file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

set(_cricket_inja_include
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/inja/include")

CPMAddPackage(
  NAME cxxopts
  GITHUB_REPOSITORY jarro2783/cxxopts
  GIT_TAG v3.3.1
)

find_package(cppad QUIET)
if(NOT cppad_FOUND)
  pkg_check_modules(cppad REQUIRED cppad)
endif()

CPMAddPackage(
  NAME CppADCodeGen
  GITHUB_REPOSITORY joaoleal/CppADCodeGen
  GIT_TAG v2.5.0
  DOWNLOAD_ONLY YES
)

if(cppadcodegen_ADDED OR CppADCodeGen_ADDED)
  if(NOT cppadcodegen_SOURCE_DIR)
    set(cppadcodegen_SOURCE_DIR "${CppADCodeGen_SOURCE_DIR}")
    set(cppadcodegen_BINARY_DIR "${CppADCodeGen_BINARY_DIR}")
  endif()

  # Patch application not idempotent through CPM...
  set(_cppadcg_patch "${CMAKE_CURRENT_SOURCE_DIR}/CppADCodeGen.patch")

  execute_process(
    COMMAND git apply --reverse --check "${_cppadcg_patch}"
    WORKING_DIRECTORY "${cppadcodegen_SOURCE_DIR}"
    RESULT_VARIABLE _cppadcg_patch_applied
    OUTPUT_QUIET ERROR_QUIET
  )
  if(NOT _cppadcg_patch_applied EQUAL 0)
    execute_process(
      COMMAND git apply "${_cppadcg_patch}"
      WORKING_DIRECTORY "${cppadcodegen_SOURCE_DIR}"
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()

  set(ENABLE_THREAD_POOL_TESTS Off CACHE BOOL "" FORCE)
  add_subdirectory("${cppadcodegen_SOURCE_DIR}" "${cppadcodegen_BINARY_DIR}" EXCLUDE_FROM_ALL)
endif()

list(APPEND cppadcg_INCLUDE_DIRS
  "${cppadcg_SOURCE_DIR}/include"
  "${cppadcg_BINARY_DIR}/include")

if(CRICKET_BUILD_JIT)
  find_package(LLVM REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)
endif()

# Load FK templates into embedded strings
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/resources/templates/fk_template.hh"   CRICKET_FK_TEMPLATE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/resources/templates/ccfk_template.hh" CRICKET_CCFK_TEMPLATE)
configure_file(
    include/cricket/embedded_templates.hh.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/cricket/embedded_templates.hh
    @ONLY
)

add_library(cricket
    src/robot_info.cc
    src/codegen.cc
    src/tracing/mapping.cc
    src/tracing/interpolate.cc
    src/tracing/distance.cc
    src/tracing/flask.cc
)
add_library(cricket::cricket ALIAS cricket)
set_target_properties(cricket PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CRICKET_BUILD_PYTHON)
  set(_cricket_install_include_iface "cricket/include")
else()
  set(_cricket_install_include_iface "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

target_include_directories(cricket
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
        $<INSTALL_INTERFACE:${_cricket_install_include_iface}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${cppad_INCLUDE_DIRS}
        ${cppadcg_INCLUDE_DIRS}
)

target_link_libraries(cricket
    PUBLIC
        # Only the pinocchio subtargets we actually use, the *_parallel targets bring in OpenMP
        pinocchio::pinocchio_default
        pinocchio::pinocchio_parsers
        pinocchio::pinocchio_collision
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        CGAL::CGAL
    PRIVATE
        ${cppad_LINK_LIBRARIES}
        fmt::fmt
)

target_include_directories(cricket PRIVATE ${_cricket_inja_include})
target_compile_definitions(cricket PUBLIC
    ${pinocchio_CFLAGS_OTHER}
)

# Expose the vendored inja header for downstream packages (i.e., VAMP)
add_library(cricket_inja INTERFACE)
add_library(cricket::inja ALIAS cricket_inja)
target_include_directories(cricket_inja INTERFACE
    $<BUILD_INTERFACE:${_cricket_inja_include}>
    $<INSTALL_INTERFACE:${_cricket_install_include_iface}>
)

add_executable(fkcc_gen src/fkcc_gen.cc)

target_link_libraries(fkcc_gen
    PRIVATE
        cricket::cricket
        fmt::fmt
        cxxopts::cxxopts
)

target_include_directories(fkcc_gen PRIVATE ${_cricket_inja_include})
target_compile_definitions(fkcc_gen PRIVATE
    ${pinocchio_CFLAGS_OTHER}
)

if(CRICKET_BUILD_JIT)
  add_library(cricket_jit
      src/jit/compiler.cc
      src/jit/object_cache.cc
      src/jit/session.cc
  )
  add_library(cricket::jit ALIAS cricket_jit)
  set_target_properties(cricket_jit PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_include_directories(cricket_jit
      PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:${_cricket_install_include_iface}>
          ${LLVM_INCLUDE_DIRS}
          ${CLANG_INCLUDE_DIRS}
  )

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  target_compile_definitions(cricket_jit PUBLIC ${LLVM_DEFINITIONS_LIST})

  target_link_libraries(cricket_jit
      PUBLIC
          clang-cpp
          LLVM
  )
endif()

if(CRICKET_BUILD_PYTHON)
  find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(nanobind CONFIG QUIET)
  if(NOT nanobind_FOUND)
    CPMAddPackage("gh:wjakob/nanobind#9a25aed8a7edfe60ef9ad1c911e57667bc4916c4")
  endif()

  nanobind_add_module(_core_ext
      NB_STATIC
      STABLE_ABI
      src/bindings/python.cc
  )
  target_link_libraries(_core_ext PRIVATE cricket::cricket)
  target_include_directories(_core_ext PRIVATE
      third_party/nanobind_json/include
  )
endif()

if(CRICKET_BUILD_PYTHON)
  set(_cricket_lib_dest       cricket/lib)
  set(_cricket_include_dest   cricket/include)
  set(_cricket_resources_dest cricket/resources)
  set(_cricket_cmake_dest     cricket/cmake)
  set(_cricket_bin_dest       cricket/bin)
  set(_cricket_python_dest    cricket)
else()
  include(GNUInstallDirs)
  set(_cricket_lib_dest       ${CMAKE_INSTALL_LIBDIR})
  set(_cricket_include_dest   ${CMAKE_INSTALL_INCLUDEDIR})
  set(_cricket_resources_dest ${CMAKE_INSTALL_DATAROOTDIR}/cricket/resources)
  set(_cricket_cmake_dest     ${CMAKE_INSTALL_DATAROOTDIR}/cricket/cmake)
  set(_cricket_bin_dest       ${CMAKE_INSTALL_BINDIR})
endif()

install(TARGETS cricket
    EXPORT cricketTargets
    LIBRARY DESTINATION ${_cricket_lib_dest}
    ARCHIVE DESTINATION ${_cricket_lib_dest}
    RUNTIME DESTINATION ${_cricket_bin_dest}
)
if(TARGET cricket_jit)
  install(TARGETS cricket_jit
      EXPORT cricketTargets
      LIBRARY DESTINATION ${_cricket_lib_dest}
      ARCHIVE DESTINATION ${_cricket_lib_dest}
      RUNTIME DESTINATION ${_cricket_bin_dest}
  )
endif()
install(DIRECTORY include/cricket/ DESTINATION ${_cricket_include_dest}/cricket)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/cricket/embedded_templates.hh
        DESTINATION ${_cricket_include_dest}/cricket)
install(DIRECTORY resources/        DESTINATION ${_cricket_resources_dest})

install(DIRECTORY third_party/inja/include/inja
        DESTINATION ${_cricket_include_dest})
install(TARGETS cricket_inja EXPORT cricketTargets)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/cricketConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cricketConfig.cmake
    INSTALL_DESTINATION ${_cricket_cmake_dest}
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cricketConfig.cmake
        DESTINATION ${_cricket_cmake_dest})
install(EXPORT cricketTargets
        FILE cricketTargets.cmake
        NAMESPACE cricket::
        DESTINATION ${_cricket_cmake_dest})

install(TARGETS fkcc_gen
    RUNTIME DESTINATION ${_cricket_bin_dest}
    COMPONENT Runtime
)

if(CRICKET_BUILD_PYTHON)
  install(TARGETS _core_ext LIBRARY DESTINATION ${_cricket_python_dest})
endif()
cmake_minimum_required(VERSION 3.10)
project(cricket
  LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CMake Policies
set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0175 OLD)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

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

find_package(inja QUIET)
if(NOT inja_FOUND)
  CPMAddPackage(
    NAME inja
    GITHUB_REPOSITORY pantor/inja
    GIT_TAG v3.4.0
  )
endif()

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

option(CRICKET_BUILD_JIT "Build the JIT runtime" ON)
if(CRICKET_BUILD_JIT)
  find_package(LLVM REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)
endif()

add_library(cricket
    src/robot_info.cc
    src/codegen.cc
)
add_library(cricket::cricket ALIAS cricket)
set_target_properties(cricket PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(cricket
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${cppad_INCLUDE_DIRS}
        ${cppadcg_INCLUDE_DIRS}
)

target_link_libraries(cricket
    PUBLIC
        pinocchio::pinocchio
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        CGAL::CGAL
    PRIVATE
        ${cppad_LINK_LIBRARIES}
        fmt::fmt
        pantor::inja
)

# Pinocchio's compile flags affect structs inside coal, so they must be visible.
target_compile_definitions(cricket PUBLIC
    ${pinocchio_CFLAGS_OTHER}
)

# CGAL's breaks under -ffast-math, remove these flags
target_compile_options(cricket PRIVATE
    -fno-fast-math
    -frounding-math
    -fsigned-zeros
    -fno-associative-math
    -ffp-contract=off
    -fno-finite-math-only
    -ftrapping-math
    -fmath-errno
)

add_executable(fkcc_gen src/fkcc_gen.cc)

target_link_libraries(fkcc_gen
    PRIVATE
        cricket::cricket
        pantor::inja
        fmt::fmt
        cxxopts::cxxopts
)

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
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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

  install(TARGETS cricket_jit
      EXPORT cricketTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

# Install
include(GNUInstallDirs)
install(TARGETS cricket
    EXPORT cricketTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/cricket
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(TARGETS fkcc_gen
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Runtime
)
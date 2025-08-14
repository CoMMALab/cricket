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
find_package(Boost REQUIRED COMPONENTS system filesystem)

# inja may or may not be installed
find_package(inja QUIET)
if(NOT inja_FOUND)
  CPMAddPackage(
    URI "gh:pantor/inja@3.4.0"
  )
endif()

# Add cxxopts
CPMAddPackage(
  URI "gh:jarro2783/cxxopts@3.3.1"
)

# Find CppAD
find_package(cppad QUIET)
if(NOT cppad_FOUND)
  pkg_check_modules(cppad REQUIRED cppad)
endif()

# Setup CppADCG
CPMAddPackage(
  URI "gh:joaoleal/CppADCodeGen@2.5.0"
  PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/CppADCodeGen.patch"
  OPTIONS "ENABLE_THREAD_POOL_TESTS Off"
)

list(APPEND cppadcg_INCLUDE_DIRS
  "${cppadcg_SOURCE_DIR}/include"
  "${cppadcg_BINARY_DIR}/include")

# Create the executable
add_executable(fkcc_gen src/fkcc_gen.cc)

# Set include directories
target_include_directories(fkcc_gen PRIVATE
    ${pinocchio_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${cppad_INCLUDE_DIRS}
    ${cppadcg_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(fkcc_gen
    pinocchio::pinocchio
    ${Boost_LIBRARIES}
    ${cppad_LIBRARIES}
    pantor::inja
    nlohmann_json::nlohmann_json
    CGAL::CGAL
    fmt::fmt
    cxxopts::cxxopts
)

# Add compiler definitions
target_compile_definitions(fkcc_gen PRIVATE
    ${pinocchio_CFLAGS_OTHER}
)

include(GNUInstallDirs)
install(TARGETS fkcc_gen
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT Runtime
)

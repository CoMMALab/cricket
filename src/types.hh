#pragma once

#include <pinocchio/spatial/se3.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <string>
#include <vector>

namespace cricket
{

struct SphereInfo
{
    std::size_t geom_index;
    float radius;
    std::size_t parent_joint;
    std::size_t parent_frame;
    pinocchio::SE3 relative;
};

// Joint type classification for mapToConfiguration
enum class JointType
{
    Bounded,           // Revolute/Prismatic with limits: nq=1, nu=1
    UnboundedRevolute, // Unbounded revolute (cos,sin): nq=2, nu=1
    SO3,               // Spherical quaternion: nq=4, nu=3
    SE3,               // FreeFlyer: nq=7, nu=6
    SE2,               // Planar: nq=4, nu=3
    Unsupported
};

struct JointMapping
{
    JointType type;
    std::size_t joint_id;
    std::size_t idx_q;  // Start index in configuration vector
    std::size_t idx_u;  // Start index in [0,1] input vector
    std::size_t nq;     // Configuration DOFs
    std::size_t nu;     // Number of [0,1] inputs needed
};

struct Bounds
{
    Eigen::Vector3d lower;
    Eigen::Vector3d upper;
};

struct Traced
{
    std::string code;
    std::size_t temp_variables;
    std::size_t outputs;
};

}  // namespace cricket

#pragma once

#include <cricket/robot_info.hh>

#include <pinocchio/multibody/model.hpp>

#include <fmt/format.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cricket
{
    inline auto classify_joint_type(const std::string &shortname, int nq) -> JointType
    {
        if (shortname.find("FreeFlyer") != std::string::npos)
        {
            return JointType::SE3;
        }
        if (shortname.find("Planar") != std::string::npos and nq == 4)
        {
            return JointType::SE2;
        }
        if (shortname.find("Spherical") != std::string::npos and nq == 4)
        {
            return JointType::SO3;
        }
        if ((shortname.find("Unbounded") != std::string::npos or
             shortname.find("RUB") != std::string::npos) and
            nq == 2)
        {
            return JointType::UnboundedRevolute;
        }
        if (nq == 1)
        {
            return JointType::Bounded;
        }
        return JointType::Unsupported;
    }

    inline auto get_nu_for_type(JointType type) -> std::size_t
    {
        switch (type)
        {
            case JointType::Bounded:
                return 1;
            case JointType::UnboundedRevolute:
                return 1;
            case JointType::SO3:
                return 3;
            case JointType::SE3:
                return 6;
            case JointType::SE2:
                return 3;
            default:
                return 0;
        }
    }

    inline auto classify_joints(const pinocchio::Model &model)
        -> std::pair<std::size_t, std::vector<JointMapping>>
    {
        std::vector<JointMapping> mappings;
        std::size_t total_nu = 0;

        for (auto joint_id = 1U; joint_id < model.joints.size(); ++joint_id)
        {
            const auto &joint = model.joints[joint_id];
            const std::string shortname = joint.shortname();
            const auto nq = joint.nq();

            if (nq == 0)
            {
                continue;
            }

            const JointType type = classify_joint_type(shortname, nq);
            const std::size_t nu = get_nu_for_type(type);

            if (type == JointType::Unsupported)
            {
                throw std::runtime_error(
                    fmt::format(
                        "Unsupported joint type: {} (shortname: {}, nq: {})",
                        model.names[joint_id],
                        shortname,
                        nq));
            }

            JointMapping mapping;
            mapping.type = type;
            mapping.joint_id = joint_id;
            mapping.idx_q = joint.idx_q();
            mapping.idx_u = total_nu;
            mapping.nq = static_cast<std::size_t>(nq);
            mapping.nu = nu;

            mappings.push_back(mapping);
            total_nu += nu;
        }

        return {total_nu, mappings};
    }

    inline auto get_randomness_dimension(const pinocchio::Model &model) -> std::size_t
    {
        auto [nu, _] = classify_joints(model);
        return nu;
    }

    template <typename Scalar>
    auto map_bounded(Scalar u, double lower, double upper) -> Scalar
    {
        return Scalar(lower) + u * Scalar(upper - lower);
    }

    template <typename Scalar>
    void map_unbounded_revolute(Scalar u, Scalar &cos_out, Scalar &sin_out)
    {
        constexpr double two_pi = 2.0 * M_PI;
        Scalar theta = u * Scalar(two_pi);
        cos_out = cos(theta);
        sin_out = sin(theta);
    }

    template <typename Scalar>
    void map_so3_shoemake(Scalar u1, Scalar u2, Scalar u3, Scalar &x, Scalar &y, Scalar &z, Scalar &w)
    {
        constexpr double two_pi = 2.0 * M_PI;

        Scalar sqrt1_minus_u1 = sqrt(Scalar(1.0) - u1);
        Scalar sqrt_u1 = sqrt(u1);
        Scalar theta1 = u2 * Scalar(two_pi);
        Scalar theta2 = u3 * Scalar(two_pi);

        x = sqrt1_minus_u1 * sin(theta1);
        y = sqrt1_minus_u1 * cos(theta1);
        z = sqrt_u1 * sin(theta2);
        w = sqrt_u1 * cos(theta2);
    }
}  // namespace cricket

#include "tracing.hh"
#include "tracing/internal.hh"
#include "robot_info.hh"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace cricket
{

namespace
{

// Compute world position for a sphere
auto compute_sphere_world_position(const SphereInfo &sphere, const ADData &ad_data)
    -> Eigen::Matrix<ADCG, 3, 1>
{
    const auto &joint_placement = ad_data.oMi[sphere.parent_joint];

    Eigen::Matrix<ADCG, 3, 1> local_translation;
    local_translation[0] = sphere.relative.translation()[0];
    local_translation[1] = sphere.relative.translation()[1];
    local_translation[2] = sphere.relative.translation()[2];

    return joint_placement.rotation() * local_translation + joint_placement.translation();
}

// Interleaved layout: [x0, y0, z0, r0, x1, y1, z1, r1, ...]
auto trace_sphere(const SphereInfo &sphere, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    auto world_position = compute_sphere_world_position(sphere, ad_data);

    data[index + 0] = world_position[0];
    data[index + 1] = world_position[1];
    data[index + 2] = world_position[2];
    data[index + 3] = ADCG(sphere.radius);
}

// Struct-of-arrays layout: [all x][all y][all z][all r]
auto trace_sphere(
    const SphereInfo &sphere,
    const ADData &ad_data,
    ADVectorXs &data,
    std::size_t sphere_index,
    const SphereOutputLayout &layout)
{
    auto world_position = compute_sphere_world_position(sphere, ad_data);

    data[layout.x_offset + sphere_index] = world_position[0];
    data[layout.y_offset + sphere_index] = world_position[1];
    data[layout.z_offset + sphere_index] = world_position[2];
    data[layout.r_offset + sphere_index] = ADCG(sphere.radius);
}

auto trace_frame(std::size_t ee_index, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &oMf = ad_data.oMf[ee_index];

    data[index + 0] = oMf.translation()[0];
    data[index + 1] = oMf.translation()[1];
    data[index + 2] = oMf.translation()[2];

    const auto &R = oMf.rotation();

    // Eigen stores as column major
    data[index + 3] = R(0, 0);
    data[index + 4] = R(1, 0);
    data[index + 5] = R(2, 0);
    data[index + 6] = R(0, 1);
    data[index + 7] = R(1, 1);
    data[index + 8] = R(2, 1);
    data[index + 9] = R(0, 2);
    data[index + 10] = R(1, 2);
    data[index + 11] = R(2, 2);
}

}  // namespace

auto trace_sphere_cc_fk(
    const RobotInfo &info,
    const std::string &language,
    bool spheres,
    bool bounding_spheres,
    bool fk,
    bool use_soa_output) -> Traced
{
    auto nq = info.model.nq;
    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    ADVectorXs ad_q(nq);
    for (auto i = 0U; i < nq; ++i)
    {
        ad_q[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_q);

    pinocchio::forwardKinematics(ad_model, ad_data, ad_q);
    pinocchio::updateFramePlacements(ad_model, ad_data);

    // Count spheres
    std::size_t n_spheres = (spheres) ? info.spheres.size() : 0;
    std::size_t n_bounding = (bounding_spheres) ? info.bounding_spheres.size() : 0;
    std::size_t total_spheres = n_spheres + n_bounding;
    std::size_t n_fk_data = (fk) ? 12 : 0;

    std::size_t n_out = total_spheres * 4 + n_fk_data;

    ADVectorXs data(n_out);

    if (use_soa_output)
    {
        // Struct-of-arrays layout: [all x][all y][all z][all r][fk data]
        SphereOutputLayout layout{
            .x_offset = 0,
            .y_offset = total_spheres,
            .z_offset = total_spheres * 2,
            .r_offset = total_spheres * 3
        };

        if (spheres)
        {
            for (auto i = 0U; i < info.spheres.size(); ++i)
            {
                const auto &sphere = info.spheres[i];
                trace_sphere(sphere, ad_data, data, sphere.geom_index, layout);
            }
        }

        if (bounding_spheres)
        {
            for (auto i = 0U; i < info.model.frames.size(); ++i)
            {
                auto sphere_it = info.bounding_spheres.find(i);
                if (sphere_it != info.bounding_spheres.end())
                {
                    const auto &sphere = sphere_it->second;
                    trace_sphere(sphere, ad_data, data, n_spheres + sphere.geom_index, layout);
                }
            }
        }

        if (fk)
        {
            trace_frame(info.end_effector_index, ad_data, data, total_spheres * 4);
        }
    }
    else
    {
        // Interleaved layout: [x0, y0, z0, r0, x1, y1, z1, r1, ...][fk data]
        if (spheres)
        {
            for (auto i = 0U; i < info.spheres.size(); ++i)
            {
                const auto &sphere = info.spheres[i];
                trace_sphere(sphere, ad_data, data, sphere.geom_index * 4);
            }
        }

        if (bounding_spheres)
        {
            for (auto i = 0U; i < info.model.frames.size(); ++i)
            {
                auto sphere_it = info.bounding_spheres.find(i);
                if (sphere_it != info.bounding_spheres.end())
                {
                    const auto &sphere = sphere_it->second;
                    trace_sphere(sphere, ad_data, data, (n_spheres + sphere.geom_index) * 4);
                }
            }
        }

        if (fk)
        {
            trace_frame(info.end_effector_index, ad_data, data, total_spheres * 4);
        }
    }

    // Create the AD function
    CppAD::ADFun<CGD> collision_sphere_func(ad_q, data);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = collision_sphere_func.Forward(0, ind_vars);

    if (use_soa_output)
    {
        // Build output segments for struct-of-arrays naming (out.x, out.y, etc.)
        std::vector<VarSegment> output_segments;
        if (total_spheres > 0)
        {
            output_segments.push_back({"out.x", total_spheres, true});
            output_segments.push_back({"out.y", total_spheres, true});
            output_segments.push_back({"out.z", total_spheres, true});
            output_segments.push_back({"out.r", total_spheres, true});
        }
        if (fk)
        {
            // FK output: translation (3) + rotation matrix (9)
            output_segments.push_back({"out.translation", 3, true});
            output_segments.push_back({"out.rotation", 9, true});
        }

        SegmentedVariableNameGenerator<double> nameGen(
            {{"x", static_cast<std::size_t>(nq), true}},  // input
            output_segments
        );

        return Traced{generate_code(handler, result, language, nameGen), handler.getTemporaryVariableCount(), n_out};
    }
    else
    {
        // Use default y[] output naming
        return Traced{generate_code(handler, result, language), handler.getTemporaryVariableCount(), n_out};
    }
}

}  // namespace cricket

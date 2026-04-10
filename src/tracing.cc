#include "tracing.hh"
#include "joint_utils.hh"
#include "pinocchio_cppadcg.hh"
#include "lang_cpp.hh"
#include "lang_rust.hh"

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <fmt/core.h>

#include <sstream>
#include <stdexcept>

namespace cricket
{

// Typedef for AD types
using CGD = CppAD::cg::CG<double>;
using ADCG = CppAD::AD<CGD>;

using ADModel = pinocchio::ModelTpl<ADCG>;
using ADData = pinocchio::DataTpl<ADCG>;
using ADVectorXs = Eigen::Matrix<ADCG, Eigen::Dynamic, 1>;

namespace
{

auto trace_sphere(const SphereInfo &sphere, const ADData &ad_data, ADVectorXs &data, std::size_t index)
{
    const auto &joint_placement = ad_data.oMi[sphere.parent_joint];

    Eigen::Matrix<ADCG, 3, 1> local_translation;
    local_translation[0] = sphere.relative.translation()[0];
    local_translation[1] = sphere.relative.translation()[1];
    local_translation[2] = sphere.relative.translation()[2];

    Eigen::Matrix<ADCG, 3, 1> world_position =
        joint_placement.rotation() * local_translation + joint_placement.translation();

    data[index + 0] = world_position[0];
    data[index + 1] = world_position[1];
    data[index + 2] = world_position[2];
    data[index + 3] = ADCG(sphere.radius);
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

auto generate_code(
    CppAD::cg::CodeHandler<double> &handler,
    CppAD::vector<CGD> &result,
    const std::string &language) -> std::string
{
    CppAD::cg::LangCDefaultVariableNameGenerator<double> nameGen;
    std::ostringstream function_code;

    if (language == "c++")
    {
        CppAD::cg::LanguageCCustom<double> langC("double");
        handler.generateCode(function_code, langC, result, nameGen);
    }
    else if (language == "rust")
    {
        CppAD::cg::LanguageRust<double> langRust("double");
        handler.generateCode(function_code, langRust, result, nameGen);
    }
    else
    {
        throw std::runtime_error(fmt::format("unsupported language {}", language));
    }

    return function_code.str();
}

}  // namespace

auto trace_sphere_cc_fk(
    const RobotInfo &info,
    const std::string &language,
    bool spheres,
    bool bounding_spheres,
    bool fk) -> Traced
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

    std::size_t n_spheres_data = (spheres) ? info.spheres.size() * 4 : 0;
    std::size_t n_bounding_spheres_data = (bounding_spheres) ? info.bounding_spheres.size() * 4 : 0;
    std::size_t n_fk_data = (fk) ? 12 : 0;
    std::size_t n_out = n_spheres_data + n_bounding_spheres_data + n_fk_data;

    ADVectorXs data(n_out);

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
                trace_sphere(sphere, ad_data, data, sphere.geom_index * 4 + n_spheres_data);
            }
        }
    }

    if (fk)
    {
        trace_frame(info.end_effector_index, ad_data, data, n_spheres_data + n_bounding_spheres_data);
    }

    // Create the AD function
    CppAD::ADFun<CGD> collision_sphere_func(ad_q, data);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = collision_sphere_func.Forward(0, ind_vars);

    return Traced{generate_code(handler, result, language), handler.getTemporaryVariableCount(), n_out};
}

auto trace_map_to_configuration(
    const pinocchio::Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds) -> Traced
{
    auto [nu, joint_mappings] = classify_joints(model);
    auto nq = model.nq;

    // Check if bounds are required
    for (const auto &jm : joint_mappings)
    {
        if (jm.type == JointType::SE3 && !bounds)
        {
            throw std::runtime_error(
                "FreeFlyer joint detected but no bounds provided. "
                "Please specify lower and upper in the configuration.");
        }
        if (jm.type == JointType::SE2 && !bounds)
        {
            throw std::runtime_error(
                "Planar joint detected but no bounds provided. "
                "Please specify lower and upper in the configuration.");
        }
    }

    ADVectorXs ad_u(nu);   // [0,1] inputs
    ADVectorXs ad_q(nq);   // Configuration output

    // Initialize inputs
    for (auto i = 0U; i < nu; ++i)
    {
        ad_u[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_u);

    // Apply joint-specific mappings
    for (const auto &jm : joint_mappings)
    {
        switch (jm.type)
        {
            case JointType::Bounded:
            {
                // Linear mapping: q = lower + u * (upper - lower)
                double lower = model.lowerPositionLimit[jm.idx_q];
                double upper = model.upperPositionLimit[jm.idx_q];
                ad_q[jm.idx_q] = map_bounded(ad_u[jm.idx_u], lower, upper);
                break;
            }
            case JointType::UnboundedRevolute:
            {
                // Maps [0,1] to (cos, sin)
                ADCG cos_val, sin_val;
                map_unbounded_revolute(ad_u[jm.idx_u], cos_val, sin_val);
                ad_q[jm.idx_q] = cos_val;
                ad_q[jm.idx_q + 1] = sin_val;
                break;
            }
            case JointType::SO3:
            {
                // Shoemake's algorithm for uniform quaternion
                ADCG x, y, z, w;
                map_so3_shoemake(ad_u[jm.idx_u], ad_u[jm.idx_u + 1], ad_u[jm.idx_u + 2], x, y, z, w);
                // Pinocchio quaternion order: (x, y, z, w)
                ad_q[jm.idx_q] = x;
                ad_q[jm.idx_q + 1] = y;
                ad_q[jm.idx_q + 2] = z;
                ad_q[jm.idx_q + 3] = w;
                break;
            }
            case JointType::SE3:
            {
                // FreeFlyer: position (3 inputs) + orientation (3 inputs)
                // Position mapping
                for (int i = 0; i < 3; ++i)
                {
                    double lo = bounds->lower[i];
                    double hi = bounds->upper[i];
                    ad_q[jm.idx_q + i] = map_bounded(ad_u[jm.idx_u + i], lo, hi);
                }
                // Orientation mapping (Shoemake)
                ADCG x, y, z, w;
                map_so3_shoemake(ad_u[jm.idx_u + 3], ad_u[jm.idx_u + 4], ad_u[jm.idx_u + 5], x, y, z, w);
                ad_q[jm.idx_q + 3] = x;
                ad_q[jm.idx_q + 4] = y;
                ad_q[jm.idx_q + 5] = z;
                ad_q[jm.idx_q + 6] = w;
                break;
            }
            case JointType::SE2:
            {
                // Planar: position (2 inputs) + orientation (1 input)
                // Position mapping (x, y)
                for (int i = 0; i < 2; ++i)
                {
                    double lo = bounds->lower[i];
                    double hi = bounds->upper[i];
                    ad_q[jm.idx_q + i] = map_bounded(ad_u[jm.idx_u + i], lo, hi);
                }
                // Orientation mapping (cos(θ), sin(θ))
                ADCG cos_val, sin_val;
                map_unbounded_revolute(ad_u[jm.idx_u + 2], cos_val, sin_val);
                ad_q[jm.idx_q + 2] = cos_val;
                ad_q[jm.idx_q + 3] = sin_val;
                break;
            }
            default:
                throw std::runtime_error("Unsupported joint type in trace_map_to_configuration");
        }
    }

    // Create the AD function
    CppAD::ADFun<CGD> map_func(ad_u, ad_q);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nu);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = map_func.Forward(0, ind_vars);

    return Traced{generate_code(handler, result, language), handler.getTemporaryVariableCount(),
                  static_cast<std::size_t>(nq)};
}

auto trace_check_bounds(
    const pinocchio::Model &model,
    const std::string &language,
    const std::optional<Bounds> &bounds) -> Traced
{
    auto [nu, joint_mappings] = classify_joints(model);
    auto nq = model.nq;

    // Check if bounds are required
    for (const auto &jm : joint_mappings)
    {
        if (jm.type == JointType::SE3 && !bounds)
        {
            throw std::runtime_error(
                "FreeFlyer joint detected but no bounds provided. "
                "Please specify lower and upper in the configuration.");
        }
        if (jm.type == JointType::SE2 && !bounds)
        {
            throw std::runtime_error(
                "Planar joint detected but no bounds provided. "
                "Please specify lower and upper in the configuration.");
        }
    }

    ADVectorXs ad_q(nq);
    ADVectorXs ad_out(1);

    for (auto i = 0; i < nq; ++i)
    {
        ad_q[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_q);

    // Start with valid = 1.0, multiply by per-joint validity
    ADCG valid = ADCG(1.0);

    for (const auto &jm : joint_mappings)
    {
        switch (jm.type)
        {
            case JointType::Bounded:
            {
                double lo = model.lowerPositionLimit[jm.idx_q];
                double hi = model.upperPositionLimit[jm.idx_q];
                ADCG q = ad_q[jm.idx_q];
                ADCG above_lo = CppAD::CondExpGe(q, ADCG(lo), ADCG(1.0), ADCG(0.0));
                ADCG below_hi = CppAD::CondExpLe(q, ADCG(hi), ADCG(1.0), ADCG(0.0));
                valid *= above_lo * below_hi;
                break;
            }
            case JointType::UnboundedRevolute:
            case JointType::SO3:
                // Always valid by construction
                break;
            case JointType::SE3:
            {
                for (int i = 0; i < 3; ++i)
                {
                    double lo = bounds->lower[i];
                    double hi = bounds->upper[i];
                    ADCG q = ad_q[jm.idx_q + i];
                    ADCG above_lo = CppAD::CondExpGe(q, ADCG(lo), ADCG(1.0), ADCG(0.0));
                    ADCG below_hi = CppAD::CondExpLe(q, ADCG(hi), ADCG(1.0), ADCG(0.0));
                    valid *= above_lo * below_hi;
                }
                break;
            }
            case JointType::SE2:
            {
                for (int i = 0; i < 2; ++i)
                {
                    double lo = bounds->lower[i];
                    double hi = bounds->upper[i];
                    ADCG q = ad_q[jm.idx_q + i];
                    ADCG above_lo = CppAD::CondExpGe(q, ADCG(lo), ADCG(1.0), ADCG(0.0));
                    ADCG below_hi = CppAD::CondExpLe(q, ADCG(hi), ADCG(1.0), ADCG(0.0));
                    valid *= above_lo * below_hi;
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported joint type in trace_check_bounds");
        }
    }

    ad_out[0] = valid;

    CppAD::ADFun<CGD> check_func(ad_q, ad_out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(nq);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = check_func.Forward(0, ind_vars);

    return Traced{generate_code(handler, result, language), handler.getTemporaryVariableCount(), 1};
}

auto trace_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced
{
    auto nq = model.nq;
    std::size_t n_input = 2 * nq + 1;  // q0, q1, t

    // Cast model to AD scalar type for use with Pinocchio's interpolate
    ADModel ad_model = model.cast<ADCG>();

    ADVectorXs ad_input(n_input);
    ADVectorXs ad_out(nq);

    for (std::size_t i = 0U; i < n_input; ++i)
    {
        ad_input[i] = ADCG(0.0);
    }

    CppAD::Independent(ad_input);

    // Extract q0, q1, t from input
    ADVectorXs ad_q0 = ad_input.head(nq);
    ADVectorXs ad_q1 = ad_input.segment(nq, nq);
    ADCG t = ad_input[2 * nq];

    pinocchio::interpolate(ad_model, ad_q0, ad_q1, t, ad_out);

    CppAD::ADFun<CGD> interp_func(ad_input, ad_out);

    CppAD::cg::CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(n_input);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = interp_func.Forward(0, ind_vars);

    return Traced{generate_code(handler, result, language), handler.getTemporaryVariableCount(),
                  static_cast<std::size_t>(nq)};
}

}  // namespace cricket

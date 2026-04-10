#include "tracing.hh"
#include "tracing/internal.hh"
#include "joint_utils.hh"

namespace cricket
{

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

}  // namespace cricket

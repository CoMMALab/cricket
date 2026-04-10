#include "tracing.hh"
#include "tracing/internal.hh"
#include "joint_utils.hh"

namespace cricket
{

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

}  // namespace cricket

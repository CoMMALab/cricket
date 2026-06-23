#include <cricket/codegen.hh>

#include "internal.hh"
#include "../joint_utils.hh"

#include <stdexcept>

namespace cricket
{
    auto trace_map_to_configuration(
        const pinocchio::Model &model,
        const std::string &language,
        const std::optional<Bounds> &bounds) -> Traced
    {
        auto [nu, joint_mappings] = classify_joints(model);
        const auto nq = model.nq;

        for (const auto &jm : joint_mappings)
        {
            if ((jm.type == JointType::SE3 or jm.type == JointType::SE2) and not bounds)
            {
                throw std::runtime_error(
                    "FreeFlyer/Planar joint detected but no Cartesian bounds provided; "
                    "set GenOptions::bounds (or `bounds` in the JSON config).");
            }
        }

        ADVectorXs ad_u(nu);
        ADVectorXs ad_q(nq);

        for (auto i = 0U; i < nu; ++i)
        {
            ad_u[i] = ADCG(0.0);
        }

        CppAD::Independent(ad_u);

        for (const auto &jm : joint_mappings)
        {
            switch (jm.type)
            {
                case JointType::Bounded:
                {
                    const double lower = model.lowerPositionLimit[jm.idx_q];
                    const double upper = model.upperPositionLimit[jm.idx_q];
                    ad_q[jm.idx_q] = map_bounded(ad_u[jm.idx_u], lower, upper);
                    break;
                }
                case JointType::UnboundedRevolute:
                {
                    ADCG c, s;
                    map_unbounded_revolute(ad_u[jm.idx_u], c, s);
                    ad_q[jm.idx_q] = c;
                    ad_q[jm.idx_q + 1] = s;
                    break;
                }
                case JointType::SO3:
                {
                    ADCG x, y, z, w;
                    map_so3_shoemake(
                        ad_u[jm.idx_u], ad_u[jm.idx_u + 1], ad_u[jm.idx_u + 2], x, y, z, w);
                    ad_q[jm.idx_q] = x;
                    ad_q[jm.idx_q + 1] = y;
                    ad_q[jm.idx_q + 2] = z;
                    ad_q[jm.idx_q + 3] = w;
                    break;
                }
                case JointType::SE3:
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        ad_q[jm.idx_q + i] =
                            map_bounded(ad_u[jm.idx_u + i], bounds->lower[i], bounds->upper[i]);
                    }
                    ADCG x, y, z, w;
                    map_so3_shoemake(
                        ad_u[jm.idx_u + 3], ad_u[jm.idx_u + 4], ad_u[jm.idx_u + 5], x, y, z, w);
                    ad_q[jm.idx_q + 3] = x;
                    ad_q[jm.idx_q + 4] = y;
                    ad_q[jm.idx_q + 5] = z;
                    ad_q[jm.idx_q + 6] = w;
                    break;
                }
                case JointType::SE2:
                {
                    for (int i = 0; i < 2; ++i)
                    {
                        ad_q[jm.idx_q + i] =
                            map_bounded(ad_u[jm.idx_u + i], bounds->lower[i], bounds->upper[i]);
                    }
                    ADCG c, s;
                    map_unbounded_revolute(ad_u[jm.idx_u + 2], c, s);
                    ad_q[jm.idx_q + 2] = c;
                    ad_q[jm.idx_q + 3] = s;
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported joint type in trace_map_to_configuration");
            }
        }

        CppAD::ADFun<CGD> map_func(ad_u, ad_q);

        CppAD::cg::CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(nu);
        handler.makeVariables(ind_vars);

        CppAD::vector<CGD> result = map_func.Forward(0, ind_vars);

        return Traced{
            generate_code(handler, result, language),
            handler.getTemporaryVariableCount(),
            static_cast<std::size_t>(nq)};
    }
}  // namespace cricket

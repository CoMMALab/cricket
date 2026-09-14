#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

namespace cricket
{
    auto trace_ee_fk_jacobian(const RobotInfo &info, const std::string &language) -> Traced
    {
        const auto nq = info.model.nq;
        const auto nv = info.model.nv;
        const auto n_out = static_cast<std::size_t>(12 + 6 * nv);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        ADVectorXs ad_q(nq);
        for (auto i = 0; i < nq; ++i)
        {
            ad_q[i] = ADCG(0.0);
        }

        CppAD::Independent(ad_q);

        // A single forward pass feeds both the placement and the Jacobian, so the shared
        // transcendentals are traced once.
        pinocchio::computeJointJacobians(ad_model, ad_data, ad_q);
        pinocchio::updateFramePlacements(ad_model, ad_data);

        Eigen::Matrix<ADCG, 6, Eigen::Dynamic> J(6, nv);
        J.setZero();
        pinocchio::getFrameJacobian(
            ad_model, ad_data, info.end_effector_index, pinocchio::LOCAL_WORLD_ALIGNED, J);

        const auto &oMf = ad_data.oMf[info.end_effector_index];
        const auto &R = oMf.rotation();

        ADVectorXs data(n_out);

        data[0] = oMf.translation()[0];
        data[1] = oMf.translation()[1];
        data[2] = oMf.translation()[2];

        // Eigen stores as column major
        data[3] = R(0, 0);
        data[4] = R(1, 0);
        data[5] = R(2, 0);
        data[6] = R(0, 1);
        data[7] = R(1, 1);
        data[8] = R(2, 1);
        data[9] = R(0, 2);
        data[10] = R(1, 2);
        data[11] = R(2, 2);

        for (auto r = 0; r < 6; ++r)
        {
            for (auto c = 0; c < nv; ++c)
            {
                data[12 + r * nv + c] = J(r, c);
            }
        }

        CppAD::ADFun<CGD> jacobian_func(ad_q, data);

        CppAD::cg::CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(nq);
        handler.makeVariables(ind_vars);

        CppAD::vector<CGD> result = jacobian_func.Forward(0, ind_vars);

        return emit_traced(handler, result, language, n_out);
    }
}  // namespace cricket

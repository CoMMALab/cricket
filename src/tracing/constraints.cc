#include <cricket/codegen.hh>

#include "internal.hh"
#include "se3_ops.hh"
#include "cholesky.hh"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <Eigen/Geometry>

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace cricket
{
    using namespace pinocchio;
    using namespace CppAD;
    using namespace CppAD::cg;

    using ADMatrixXs = Eigen::Matrix<ADCG, Eigen::Dynamic, Eigen::Dynamic>;

    namespace
    {
        constexpr std::size_t task_dim = 6;  // se(3) task-space error
        constexpr std::size_t tf_dim = 7;    // wxyz quaternion + xyz translation

        // Read a transform laid out as [qw, qx, qy, qz, tx, ty, tz] off the input tape.
        auto read_transform(const ADVectorXs &inp, std::size_t offset) -> SE3Tpl<ADCG>
        {
            Eigen::Quaternion<ADCG> rotation(
                inp[offset + 0], inp[offset + 1], inp[offset + 2], inp[offset + 3]);

            Eigen::Vector3<ADCG> translation;
            for (auto i = 0U; i < 3; ++i)
            {
                translation[i] = inp[offset + 4 + i];
            }

            return SE3Tpl<ADCG>(rotation, translation);
        }

        // se(3) displacement of a relative transform: [translation; log3(rotation)].
        auto se3_displacement(const SE3Tpl<ADCG> &transform) -> ADVectorXs
        {
            ADVectorXs displacement(task_dim);
            displacement << transform.translation_impl(),
                so3_log_smooth<ADMatrixXs, ADCG>(transform.rotation_impl());
            return displacement;
        }

        // Emit code computing [d(err)/dq (row-major), err] for an error function whose first
        // nq inputs are the configuration. The bound inputs are part of the tape but do not
        // affect the emitted error: hinging the error against the bounds during tracing would
        // wrongly zero gradients, so the hinge is applied at runtime in vamp instead.
        auto emit_error_and_jacobian(
            ADFun<CGD> &error_func,
            std::size_t num_inp,
            std::size_t nq,
            std::size_t n_out,
            const std::string &language) -> Traced
        {
            CodeHandler<double> handler;
            CppAD::vector<CGD> ind_vars(num_inp);
            handler.makeVariables(ind_vars);

            CppAD::vector<CGD> error_vec = error_func.Forward(0, ind_vars);
            CppAD::vector<CGD> jac = error_func.Jacobian(ind_vars);

            // Jacobian with respect to the configuration only
            CppAD::vector<CGD> jac_e_q(n_out * nq);
            for (auto i = 0U; i < n_out; ++i)
            {
                for (auto j = 0U; j < nq; ++j)
                {
                    jac_e_q[i * nq + j] = jac[i * num_inp + j];
                }
            }

            std::move(error_vec.begin(), error_vec.end(), std::back_inserter(jac_e_q));

            return Traced{
                generate_code(handler, jac_e_q, language),
                handler.getTemporaryVariableCount(),
                jac_e_q.size()};
        }

        // Emit code computing just [err] for an error function, skipping the Jacobian.
        auto emit_error(ADFun<CGD> &error_func, std::size_t num_inp, const std::string &language)
            -> Traced
        {
            CodeHandler<double> handler;
            CppAD::vector<CGD> ind_vars(num_inp);
            handler.makeVariables(ind_vars);

            CppAD::vector<CGD> error_vec = error_func.Forward(0, ind_vars);

            return Traced{
                generate_code(handler, error_vec, language),
                handler.getTemporaryVariableCount(),
                error_vec.size()};
        }
    }  // namespace

    auto trace_tsr_error(const RobotInfo &info, const std::string &language, bool compute_jac)
        -> Traced
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto n_eef = info.end_effector_indexes.size();

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        // Input layout (must match vamp's TSRComputeInput::operator[]):
        // q (nq), then per end-effector: rTe (7), wTr (7), lb (6), ub (6)
        const std::size_t num_inp_eef = 2 * tf_dim + 2 * task_dim;
        const std::size_t num_inp = nq + num_inp_eef * n_eef;

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);

        const std::size_t n_out = task_dim * n_eef;
        ADVectorXs data(n_out);

        for (auto eef_idx = 0U; eef_idx < n_eef; ++eef_idx)
        {
            const std::size_t offset = nq + num_inp_eef * eef_idx;

            // end-effector in the object frame; TSR reference frame in the world frame
            const auto rTe = read_transform(ad_inp, offset);
            const auto wTr = read_transform(ad_inp, offset + tf_dim);

            const auto wTobj = ad_data.oMf[info.end_effector_indexes[eef_idx]] * rTe.inverse();
            const auto rTobj = wTr.inverse() * wTobj;

            data.segment(task_dim * eef_idx, task_dim) = se3_displacement(rTobj);
        }

        ADFun<CGD> error_func(ad_inp, data);
        return compute_jac ? emit_error_and_jacobian(error_func, num_inp, nq, n_out, language)
                            : emit_error(error_func, num_inp, language);
    }

    auto trace_tsr_bimanual_error(
        const RobotInfo &info,
        const std::string &language,
        std::size_t eef1,
        std::size_t eef2,
        bool compute_jac) -> Traced
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);

        if (info.end_effector_indexes.size() < 2)
        {
            throw std::runtime_error("trace_tsr_bimanual_error requires at least two end-effectors");
        }

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        // Input layout: q (nq), then the reference relative transform lTr (7), lb (6), ub (6)
        const std::size_t num_inp = nq + tf_dim + 2 * task_dim;

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);

        const auto lTr = read_transform(ad_inp, nq);

        const auto &lTw = ad_data.oMf[info.end_effector_indexes[eef1]];
        const auto &rTw = ad_data.oMf[info.end_effector_indexes[eef2]];

        const auto lTr_rob = lTw.inverse() * rTw;
        const auto errT = lTr_rob * lTr.inverse();

        ADVectorXs data = se3_displacement(errT);

        ADFun<CGD> error_func(ad_inp, data);
        return compute_jac ? emit_error_and_jacobian(error_func, num_inp, nq, task_dim, language)
                            : emit_error(error_func, num_inp, language);
    }

    auto trace_solve_tsr(
        const RobotInfo &info,
        const std::string &language,
        ProjMethod method,
        bool relative) -> Traced
    {
        return trace_solve_jacobian(
            info, language, method, task_dim * (relative ? 1 : info.end_effector_indexes.size()));
    }

    auto trace_solve_jacobian(
        const RobotInfo &info,
        const std::string &language,
        ProjMethod method,
        std::size_t err_size) -> Traced
    {
        constexpr double damp = 1e-6;

        const auto nq = static_cast<std::size_t>(info.model.nq);

        // Input layout (must match vamp's JacobianProjectInp::operator[]):
        // J (err_size x nq, row-major), then err (err_size)
        const std::size_t num_inp = err_size * nq + err_size;

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        Independent(ad_inp);

        ADMatrixXs J(err_size, nq);
        for (auto i = 0U; i < err_size; ++i)
        {
            for (auto j = 0U; j < nq; ++j)
            {
                J(i, j) = ad_inp[i * nq + j];
            }
        }

        ADVectorXs err(err_size);
        for (auto i = 0U; i < err_size; ++i)
        {
            err[i] = ad_inp[err_size * nq + i];
        }

        ADVectorXs grad(nq);
        switch (method)
        {
            case ProjMethod::InnerLM:
            {
                ADMatrixXs identity = ADMatrixXs::Identity(err_size, err_size);
                const auto factor =
                    cholesky_factor<ADMatrixXs, ADCG>(J * J.transpose() + identity * damp);
                grad = J.transpose() * cholesky_solve<ADMatrixXs, ADVectorXs, ADCG>(factor, err);
                break;
            }
            case ProjMethod::OuterLM:
            {
                ADMatrixXs identity = ADMatrixXs::Identity(nq, nq);
                const auto factor =
                    cholesky_factor<ADMatrixXs, ADCG>(J.transpose() * J + identity * damp);
                grad = cholesky_solve<ADMatrixXs, ADVectorXs, ADCG>(factor, J.transpose() * err);
                break;
            }
            case ProjMethod::GradDesc:
            {
                grad = J.transpose() * err;
                break;
            }
        }

        ADFun<CGD> solve_func(ad_inp, grad);

        CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(num_inp);
        handler.makeVariables(ind_vars);

        CppAD::vector<CGD> result = solve_func.Forward(0, ind_vars);

        return Traced{
            generate_code(handler, result, language),
            handler.getTemporaryVariableCount(),
            result.size()};
    }

    auto trace_com_jacobian(
        const RobotInfo &info,
        const std::vector<std::string> &reference_frames,
        const std::string &language,
        bool compute_jac) -> Traced
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        std::vector<FrameIndex> reference_ids;
        for (const auto &name : reference_frames)
        {
            if (not ad_model.existFrame(name, BODY))
            {
                throw std::runtime_error(
                    fmt::format("CoM reference frame `{}` does not exist in the model", name));
            }

            reference_ids.push_back(ad_model.getFrameId(name, BODY));
        }

        ADVectorXs ad_inp(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);
        const auto com = centerOfMass(ad_model, ad_data, ad_q, true);

        Eigen::Vector3<ADCG> reference = Eigen::Vector3<ADCG>::Zero();
        for (const auto id : reference_ids)
        {
            reference += ad_data.oMf[id].translation();
        }

        if (not reference_ids.empty())
        {
            reference /= ADCG(static_cast<double>(reference_ids.size()));
        }

        ADVectorXs data(3);
        for (auto i = 0U; i < 3; ++i)
        {
            data[i] = com[i] - reference[i];
        }

        ADFun<CGD> com_func(ad_inp, data);
        return compute_jac ? emit_error_and_jacobian(com_func, nq, nq, 3, language)
                            : emit_error(com_func, nq, language);
    }

    auto trace_closed_loop_error(
        const RobotInfo &info,
        const std::vector<ClosedLoop> &loops,
        const std::string &language,
        bool compute_jac) -> Traced
    {
        if (loops.empty())
        {
            throw std::runtime_error("trace_closed_loop_error requires at least one loop");
        }

        const auto nq = static_cast<std::size_t>(info.model.nq);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        std::vector<std::pair<FrameIndex, FrameIndex>> frame_ids;
        for (const auto &loop : loops)
        {
            for (const auto &name : {loop.start_frame, loop.end_frame})
            {
                if (not ad_model.existFrame(name, BODY))
                {
                    throw std::runtime_error(
                        fmt::format("Closed-loop frame `{}` does not exist in the model", name));
                }
            }

            frame_ids.emplace_back(
                ad_model.getFrameId(loop.start_frame, BODY), ad_model.getFrameId(loop.end_frame, BODY));
        }

        ADVectorXs ad_inp(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);

        ADVectorXs data(loops.size());
        for (auto i = 0U; i < loops.size(); ++i)
        {
            const auto &start = ad_data.oMf[frame_ids[i].first].translation();
            const auto &end = ad_data.oMf[frame_ids[i].second].translation();
            data[i] = (end - start).norm() - ADCG(loops[i].length);
        }

        ADFun<CGD> error_func(ad_inp, data);
        return compute_jac ? emit_error_and_jacobian(error_func, nq, nq, loops.size(), language)
                            : emit_error(error_func, nq, language);
    }

    auto trace_lead_screw_error(const RobotInfo &info, const std::string &language, bool compute_jac)
        -> Traced
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        // Input layout (must match vamp's LeadScrewConstraint::Input::operator[]):
        // q (nq), then rTe (7), wTr (7), pitch (1)
        const std::size_t num_inp = nq + 2 * tf_dim + 1;

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);

        const auto rTe = read_transform(ad_inp, nq);
        const auto wTr = read_transform(ad_inp, nq + tf_dim);

        constexpr double two_pi = 6.283185307179586476925287;
        const ADCG advance_per_radian = ad_inp[nq + 2 * tf_dim] / ADCG(two_pi);

        // Same relative pose as the TSR error: the offset end-effector frame expressed
        // in the reference frame. Its se(3) displacement's z-translation is the axial
        // advance and its z-log-rotation the axial angle, so the screw invariant is a
        // scalar combination of the displacement rows.
        const auto wTobj = ad_data.oMf[info.end_effector_indexes[0]] * rTe.inverse();
        const auto rTobj = wTr.inverse() * wTobj;
        const auto displacement = se3_displacement(rTobj);

        ADVectorXs data(1);
        data[0] = displacement[2] - advance_per_radian * displacement[5];

        ADFun<CGD> error_func(ad_inp, data);
        return compute_jac ? emit_error_and_jacobian(error_func, num_inp, nq, 1, language)
                            : emit_error(error_func, num_inp, language);
    }

    auto trace_twist_jacobians(const RobotInfo &info, const std::string &language) -> Traced
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        // Input layout (must match vamp's TwistConstraint::Input::operator[]):
        // q (nq), then rTe (7), wTr (7)
        const std::size_t num_inp = nq + 2 * tf_dim;

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        computeJointJacobians(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);

        const auto rTe = read_transform(ad_inp, nq);
        const auto wTr = read_transform(ad_inp, nq + tf_dim);

        const auto frame = info.end_effector_indexes[0];

        // World-aligned frame Jacobian of the end-effector: linear rows are the velocity
        // of the frame origin, angular rows the angular velocity, both in world axes.
        ADData::Matrix6x jac(6, ad_model.nv);
        jac.setZero();
        getFrameJacobian(ad_model, ad_data, frame, LOCAL_WORLD_ALIGNED, jac);

        const auto wTobj = ad_data.oMf[frame] * rTe.inverse();

        // Shift the linear rows to the offset frame's origin: v_obj = v_eef + w x r.
        const Eigen::Vector3<ADCG> r =
            wTobj.translation_impl() - ad_data.oMf[frame].translation_impl();
        ADMatrixXs lin(3, nq);
        ADMatrixXs ang(3, nq);
        for (auto k = 0U; k < nq; ++k)
        {
            const Eigen::Vector3<ADCG> jw = jac.col(k).tail(3);
            ang.col(k) = jw;
            lin.col(k) = Eigen::Vector3<ADCG>(jac.col(k).head(3)) + jw.cross(r);
        }

        // Twist of the offset frame expressed in the reference frame's axes, then in the
        // offset frame's own (body) axes.
        const Eigen::Matrix3<ADCG> ref_rotation = wTr.rotation_impl().transpose();
        const Eigen::Matrix3<ADCG> loc_rotation = wTobj.rotation_impl().transpose();

        ADMatrixXs rows(2 * task_dim, nq);
        rows.topRows(3) = ref_rotation * lin;
        rows.middleRows(3, 3) = ref_rotation * ang;
        rows.middleRows(6, 3) = loc_rotation * lin;
        rows.bottomRows(3) = loc_rotation * ang;

        ADVectorXs data(2 * task_dim * nq);
        for (auto i = 0U; i < 2 * task_dim; ++i)
        {
            for (auto j = 0U; j < nq; ++j)
            {
                data[i * nq + j] = rows(i, j);
            }
        }

        ADFun<CGD> func(ad_inp, data);

        CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(num_inp);
        handler.makeVariables(ind_vars);

        CppAD::vector<CGD> result = func.Forward(0, ind_vars);

        return Traced{
            generate_code(handler, result, language),
            handler.getTemporaryVariableCount(),
            2 * task_dim * nq};
    }
}  // namespace cricket

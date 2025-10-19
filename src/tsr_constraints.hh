#include "housekeeping.hh"
#include "robot_info.hh"
#include "tracer_utils.hh"
#include "cholesky_decomp.hh"

#include <vector>

auto trace_tsr_error_function(const RobotInfo &info) -> Traced
{
    const double DT = 0.1;
    const double damp = 1e-6;
    auto nq = info.model.nq;
    auto nv = info.model.nv;
    const size_t nt = 6;  // task space is se3
    const size_t n_eef = info.end_effector_indexes.size();

    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    // Total inputs is:
    // 2 * 4x4 matrices for constraint space
    // 2 * 6 bounds for constraint space
    // nq  for configuration space
    // It is repeated for all end effectors.
    const size_t num_inp_eef = (7 * 2 + nt * 2 + nq);
    const size_t num_inp = num_inp_eef * n_eef;

    ADVectorXs ad_inp(num_inp);  // 3 4x4 matrices
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = ADCG(0.0);
    }

    Independent(ad_inp);

    std::size_t n_out = nt * n_eef;
    ADVectorXs data(n_out);

    // First copy over configs and run FK

    for (auto eef_idx = 0U; eef_idx < n_eef; eef_idx++)
    {
        const size_t eef_offset = num_inp_eef * eef_idx;
        ADVectorXs ad_q(nq);

        // Copying inputs from ad_inp into individual matrices
        for (auto i = 0U; i < nq; i++)
        {
            ad_q[i] = ad_inp[eef_offset + i];  // This is the first 7 vars for nq
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);
    }

    for (auto eef_idx = 0U; eef_idx < n_eef; eef_idx++)
    {
        const size_t eef_offset = num_inp_eef * eef_idx;
        const size_t eef_out_offset = nt * eef_idx;

        // Form the matrices
        Eigen::Vector3<ADCG> rTep;
        Eigen::Vector3<ADCG> wTrp;
        Eigen::Quaternion<ADCG> rTeq(
            ad_inp[eef_offset + nq + 0 + 0],
            ad_inp[eef_offset + nq + 0 + 1],
            ad_inp[eef_offset + nq + 0 + 2],
            ad_inp[eef_offset + nq + 0 + 3]);  // Next 7 for rTe
        Eigen::Quaternion<ADCG> wTrq(
            ad_inp[eef_offset + nq + 7 + 0],
            ad_inp[eef_offset + nq + 7 + 1],
            ad_inp[eef_offset + nq + 7 + 2],
            ad_inp[eef_offset + nq + 7 + 3]);  // 7 after that for wTr
        for (auto i = 0U; i < 3; i++)
        {
            rTep[i] = ad_inp[eef_offset + nq + 4 + i];
            wTrp[i] = ad_inp[eef_offset + nq + 7 + 4 + i];
        }
        SE3Tpl<ADCG, 0> rTe(rTeq, rTep);  // it is assumed that this err is expressed in the eef joint frame
        SE3Tpl<ADCG, 0> wTr(wTrq, wTrp);

        // Form bounds
        ADVectorXs lb(nt);
        ADVectorXs ub(nt);
        for (auto i = 0U; i < nt; i++)
        {
            lb[i] = ad_inp[eef_offset + nq + 2 * 7 + i];
            ub[i] = ad_inp[eef_offset + nq + 2 * 7 + nt + i];
        }
        // input setup done

        // compute error term
        const auto wTobj = ad_data.oMf[info.end_effector_indexes[eef_idx]] * rTe.inverse();
        const auto rTobj = wTr.inverse() * wTobj;

        ADVectorXs displacement(nt);
        displacement.setZero();
        displacement << rTobj.translation_impl(), log3(rTobj.rotation_impl());

        // TODO (siyer) -- we ignore the bounds here, since
        // it would set some gradients to be zero by mistake while tracing.
        // we want the tracing to be generic
        // leaving this here, since we may try out some hinge loss to make it
        // differentiable probably.
        for (auto i = 0U; i < nt; i++)
        {
            // data[i] = CondExpLt(displacement[i], zero, displacement[i] - lb[i], displacement[i] - ub[i]);
            // data[i] = data[i] + CondExpGt(displacement[i], ub[i], displacement[i] - ub[i], zero);
            // data[eef_offset + i] = CondExpLt(displacement[i], lb[i], displacement[i] - lb[i],
            // (displacement[i] - lb[i]) * 1e-6) + CondExpGt(displacement[i], ub[i], displacement[i] - ub[i],
            // (displacement[i] - ub[i]) * 1e-6);
            data[eef_out_offset + i] = displacement[i];
        }
    }

    // Create the AD function
    ADFun<CGD> jacobian_error_func(ad_inp, data);
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> error_vec = jacobian_error_func.Forward(0, ind_vars);

    // this is the full jacobian
    CppAD::vector<CGD> jac = jacobian_error_func.Jacobian(ind_vars);
    CppAD::vector<CGD> jac_e_q(n_out * nq);  // this is jacobian with respect to joint configs only.

    for (auto i = 0U; i < n_out; i++)
    {
        for (auto j = 0U; j < nq; j++)
        {
            jac_e_q[i * nq + j] = jac[i * num_inp + j];
        }
    }

    // now correct the error vector with the lower and upper bounds.

    // CGD zero_cgd(0.0);
    // for (auto eef_idx = 0U; eef_idx < n_eef; eef_idx++)
    // {
    //     const size_t eef_inp_offset = num_inp_eef * eef_idx;
    //     const size_t eef_out_offset = nt * eef_idx;
    //     for (auto i = 0U; i < nt; i++)
    //     {
    //         CGD lb = ad_inp[eef_inp_offset + nq + 2 * 7 + i];
    //         CGD ub = ad_inp[eef_inp_offset + nq + 2 * 7 + nt + i];
    //         error_vec[eef_out_offset + i] =
    //             CondExpLt(error_vec[eef_out_offset + i], lb, error_vec[eef_out_offset + i] - lb, zero_cgd) +
    //             CondExpGt(error_vec[eef_out_offset + i], ub, error_vec[eef_out_offset + i] - ub, zero_cgd);
    //     }
    // }

    std::move(error_vec.begin(), error_vec.end(), std::back_inserter(jac_e_q));

    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, jac_e_q, nameGen);

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), jac_e_q.size()};
}

enum ProjMethod {
    InnerLM, // J.T * ((J.JT + l)-1 . err) --> (6,6) matrix inv
    OuterLM, // (JT.T + l)-1 . (J.T * err) -> (nq,nq) matrix inv
    GradDesc // J.T * err
};

auto solve_error_function_wrt_joints(
    ADVectorXs ad_inp,
    const size_t err_vec_size,
    const size_t nq,
    ProjMethod projection_method
)
{

    const double damp = 1e-6;
    ADVectorXs ad_e(err_vec_size);
    ADMatrixXs ad_J(err_vec_size, nq);

    // Copying inputs from ad_inp into individual matrices
    for (auto i = 0U; i < err_vec_size; i++)
        ad_e[i] = ad_inp[i + err_vec_size * nq]; // First err_vec_size * nq is the jacobian

    // Copying inputs from ad_inp into individual matrices
    for(auto i=0U; i < err_vec_size; i++)
        for(auto j=0U; j < nq; j++)
            ad_J(i, j) = ad_inp[i * nq + j];


    ADVectorXs grad(nq);

    if (projection_method == ProjMethod::InnerLM)
    {
        // grad = ad_J.transpose() * (ad_J * ad_J.transpose()).llt().solve(ad_e);
        ADMatrixXs identity(err_vec_size, err_vec_size);
        identity.setIdentity();

        auto decomposed = cholesky_factor<ADMatrixXs, ADCG>(ad_J * ad_J.transpose() + identity * damp);
        grad = ad_J.transpose() * cholesky_solve<ADMatrixXs, ADVectorXs, ADCG>(decomposed, ad_e);
    }
    else if (projection_method == ProjMethod::OuterLM)
    {

        ADMatrixXs identity(nq, nq);
        identity.setIdentity();
        auto decomposed = cholesky_factor<ADMatrixXs, ADCG>(ad_J.transpose() * ad_J + identity * damp);
        grad = cholesky_solve<ADMatrixXs, ADVectorXs, ADCG>(decomposed, ad_J.transpose() * ad_e);
    }
    else if (projection_method == ProjMethod::GradDesc)
    {
        grad = ad_J.transpose() * ad_e;
    }
    else
    {
        throw std::runtime_error("Invalid method specified");
    }

    return grad;

}



auto trace_solve_tsr_function(
    const RobotInfo &info,
    ProjMethod projection_method
    ) -> Traced
{

    const double DT = 0.1;
    auto nq = info.model.nq;
    auto nv = info.model.nv;
    const size_t nt = 6; // task space is se3
    const size_t n_eef = info.end_effector_indexes.size();

    const size_t err_vec_size = nt * n_eef;
    const size_t num_inp = err_vec_size + err_vec_size * nq;

    ADVectorXs ad_inp(num_inp); // 3 4x4 matrices
    for (auto i = 0U; i < num_inp; ++i)
        ad_inp[i] = ADCG(0.0);
    Independent(ad_inp);

    auto grad = solve_error_function_wrt_joints(
        ad_inp,
        err_vec_size,
        nq,
        projection_method
    );


    std::size_t n_out = nq;
    ADVectorXs data(n_out);

    for (auto i = 0U; i < nq; i++)
        data[i] = grad(i);


    // Create the AD function
    ADFun<CGD> solve_func(ad_inp, data);
    CodeHandler<double> handler;
    CppAD::vector<CGD> ind_vars(num_inp);
    handler.makeVariables(ind_vars);

    CppAD::vector<CGD> result = solve_func.Forward(0, ind_vars);


    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, result, nameGen);

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), result.size()};
}

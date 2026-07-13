#include <cricket/codegen.hh>

#include "internal.hh"

#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <utility>
#include <vector>

namespace cricket
{
    namespace
    {
        auto check_flat_manipulator(const pinocchio::Model &model)
        {
            if (model.nq != model.nv)
            {
                throw std::runtime_error(
                    fmt::format(
                        "flask codegen requires nq == nv (1-DoF joints only), got nq={} nv={}",
                        model.nq,
                        model.nv));
            }
        }

        auto check_has_dynamics(const pinocchio::Model &model)
        {
            double total_mass = 0.;
            for (std::size_t i = 1U; i < model.inertias.size(); ++i)
            {
                total_mass += model.inertias[i].mass();
            }

            if (not (total_mass > 0.))
            {
                throw std::runtime_error(
                    "flask kinetic energy codegen requires link masses; the URDF has no <inertial> data");
            }
        }

        // Minimum-acceleration (LQMT, r = 2) cubic between flat states a = (y0, yd0) and
        // b = (yf, ydf) over duration T, evaluated at fraction t of T.
        // Outputs 3n rows: [y(tT); yd(tT); ydd(tT)].
        auto trace_flask_interpolate_impl(
            const pinocchio::Model &model,
            const std::string &language,
            std::vector<VarSegment> input_segments,
            std::vector<VarSegment> output_segments) -> Traced
        {
            check_flat_manipulator(model);

            const auto nq = static_cast<std::size_t>(model.nq);
            const std::size_t n_input = 4 * nq + 2;  // a (2n), b (2n), T, t

            ADVectorXs ad_input(n_input);
            for (std::size_t i = 0U; i < n_input; ++i)
            {
                ad_input[i] = ADCG(0.0);
            }

            ad_input[4 * nq] = ADCG(1.0);  // T: nonzero value at the taping point

            CppAD::Independent(ad_input);

            const ADCG T = ad_input[4 * nq];
            const ADCG t = ad_input[4 * nq + 1];

            const ADCG tau = t * T;
            const ADCG T_inv = ADCG(1.0) / T;
            const ADCG T2_inv = T_inv * T_inv;
            const ADCG T3_inv = T2_inv * T_inv;

            ADVectorXs ad_out(3 * nq);
            for (std::size_t j = 0U; j < nq; ++j)
            {
                const ADCG y0 = ad_input[j];
                const ADCG yd0 = ad_input[nq + j];
                const ADCG yf = ad_input[2 * nq + j];
                const ADCG ydf = ad_input[3 * nq + j];

                const ADCG d1 = yf - y0 - T * yd0;
                const ADCG d2 = ydf - yd0;

                const ADCG c3 = ADCG(-2.0) * T3_inv * d1 + T2_inv * d2;
                const ADCG c2 = ADCG(3.0) * T2_inv * d1 - T_inv * d2;

                ad_out[j] = ((c3 * tau + c2) * tau + yd0) * tau + y0;
                ad_out[nq + j] = (ADCG(3.0) * c3 * tau + ADCG(2.0) * c2) * tau + yd0;
                ad_out[2 * nq + j] = ADCG(6.0) * c3 * tau + ADCG(2.0) * c2;
            }

            CppAD::ADFun<CGD> flask_func(ad_input, ad_out);

            CppAD::cg::CodeHandler<double> handler;
            CppAD::vector<CGD> ind_vars(n_input);
            handler.makeVariables(ind_vars);

            CppAD::vector<CGD> result = flask_func.Forward(0, ind_vars);

            SegmentedVariableNameGenerator<double> nameGen(
                std::move(input_segments), std::move(output_segments));

            return Traced{
                generate_code(handler, result, language, nameGen),
                handler.getTemporaryVariableCount(),
                3 * nq};
        }

        // Inverse dynamics tau = RNEA(q, v, a) with input x = [q; v; a] (3n rows), n outputs.
        auto trace_flask_rnea_impl(
            const pinocchio::Model &model,
            const std::string &language,
            std::vector<VarSegment> input_segments,
            std::vector<VarSegment> output_segments) -> Traced
        {
            check_flat_manipulator(model);

            const auto nq = static_cast<std::size_t>(model.nq);
            const std::size_t n_input = 3 * nq;

            ADModel ad_model = model.cast<ADCG>();
            ADData ad_data(ad_model);

            ADVectorXs ad_input(n_input);
            for (std::size_t i = 0U; i < n_input; ++i)
            {
                ad_input[i] = ADCG(0.0);
            }

            CppAD::Independent(ad_input);

            ADVectorXs ad_q = ad_input.head(nq);
            ADVectorXs ad_v = ad_input.segment(nq, nq);
            ADVectorXs ad_a = ad_input.segment(2 * nq, nq);

            const auto &tau = pinocchio::rnea(ad_model, ad_data, ad_q, ad_v, ad_a);

            ADVectorXs ad_out(nq);
            for (std::size_t j = 0U; j < nq; ++j)
            {
                ad_out[j] = tau[j];
            }

            CppAD::ADFun<CGD> rnea_func(ad_input, ad_out);

            CppAD::cg::CodeHandler<double> handler;
            CppAD::vector<CGD> ind_vars(n_input);
            handler.makeVariables(ind_vars);

            CppAD::vector<CGD> result = rnea_func.Forward(0, ind_vars);

            SegmentedVariableNameGenerator<double> nameGen(
                std::move(input_segments), std::move(output_segments));

            return Traced{
                generate_code(handler, result, language, nameGen),
                handler.getTemporaryVariableCount(),
                nq};
        }

        // Kinetic energy (1/2) v^T M(q) v with input x = [q; v] (2n rows), 1 output.
        auto trace_flask_kinetic_energy_impl(
            const pinocchio::Model &model,
            const std::string &language,
            std::vector<VarSegment> input_segments,
            std::vector<VarSegment> output_segments) -> Traced
        {
            check_flat_manipulator(model);
            check_has_dynamics(model);

            const auto nq = static_cast<std::size_t>(model.nq);
            const std::size_t n_input = 2 * nq;

            ADModel ad_model = model.cast<ADCG>();
            ADData ad_data(ad_model);

            ADVectorXs ad_input(n_input);
            for (std::size_t i = 0U; i < n_input; ++i)
            {
                ad_input[i] = ADCG(0.0);
            }

            CppAD::Independent(ad_input);

            ADVectorXs ad_q = ad_input.head(nq);
            ADVectorXs ad_v = ad_input.segment(nq, nq);

            ADVectorXs ad_out(1);
            ad_out[0] = pinocchio::computeKineticEnergy(ad_model, ad_data, ad_q, ad_v);

            CppAD::ADFun<CGD> ke_func(ad_input, ad_out);

            CppAD::cg::CodeHandler<double> handler;
            CppAD::vector<CGD> ind_vars(n_input);
            handler.makeVariables(ind_vars);

            CppAD::vector<CGD> result = ke_func.Forward(0, ind_vars);

            SegmentedVariableNameGenerator<double> nameGen(
                std::move(input_segments), std::move(output_segments));

            return Traced{
                generate_code(handler, result, language, nameGen),
                handler.getTemporaryVariableCount(),
                1};
        }

        // World-aligned linear velocity of every end-effector origin with input
        // x = [q; v] (2n rows), 3 outputs per end-effector.
        auto trace_flask_eef_velocity_impl(
            const RobotInfo &info,
            const std::string &language,
            std::vector<VarSegment> input_segments,
            std::vector<VarSegment> output_segments) -> Traced
        {
            check_flat_manipulator(info.model);

            const auto nq = static_cast<std::size_t>(info.model.nq);
            const auto n_eef = info.end_effector_indexes.size();
            const std::size_t n_input = 2 * nq;

            ADModel ad_model = info.model.cast<ADCG>();
            ADData ad_data(ad_model);

            ADVectorXs ad_input(n_input);
            for (std::size_t i = 0U; i < n_input; ++i)
            {
                ad_input[i] = ADCG(0.0);
            }

            CppAD::Independent(ad_input);

            ADVectorXs ad_q = ad_input.head(nq);
            ADVectorXs ad_v = ad_input.segment(nq, nq);

            pinocchio::forwardKinematics(ad_model, ad_data, ad_q, ad_v);

            ADVectorXs ad_out(3 * n_eef);
            for (std::size_t eef_idx = 0U; eef_idx < n_eef; ++eef_idx)
            {
                const auto vel = pinocchio::getFrameVelocity(
                    ad_model,
                    ad_data,
                    info.end_effector_indexes[eef_idx],
                    pinocchio::LOCAL_WORLD_ALIGNED);
                ad_out.segment(3 * eef_idx, 3) = vel.linear();
            }

            CppAD::ADFun<CGD> eev_func(ad_input, ad_out);

            CppAD::cg::CodeHandler<double> handler;
            CppAD::vector<CGD> ind_vars(n_input);
            handler.makeVariables(ind_vars);

            CppAD::vector<CGD> result = eev_func.Forward(0, ind_vars);

            SegmentedVariableNameGenerator<double> nameGen(
                std::move(input_segments), std::move(output_segments));

            return Traced{
                generate_code(handler, result, language, nameGen),
                handler.getTemporaryVariableCount(),
                3 * n_eef};
        }
    }  // namespace

    auto trace_flask_interpolate(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        const auto nz = 2 * static_cast<std::size_t>(model.nq);
        return trace_flask_interpolate_impl(
            model,
            language,
            {{"a", nz, true}, {"b", nz, true}, {"T", 1, false}, {"t", 1, false}},
            {});
    }

    auto trace_flask_interpolate_block(const pinocchio::Model &model, const std::string &language)
        -> Traced
    {
        const std::string lang = (language == "c++") ? "c++_block" : language;
        const auto nq = static_cast<std::size_t>(model.nq);
        const auto nz = 2 * nq;
        return trace_flask_interpolate_impl(
            model,
            lang,
            {{"a", nz, true, ".broadcast(", ")"},
             {"b", nz, true, ".broadcast(", ")"},
             {"T", 1, false},
             {"t", 1, false}},
            {{"out", 3 * nq, true}});
    }

    auto trace_flask_rnea(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        const auto nq = static_cast<std::size_t>(model.nq);
        return trace_flask_rnea_impl(model, language, {{"x", 3 * nq, true}}, {});
    }

    auto trace_flask_rnea_block(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        // Input x is a ConfigurationBlock: each row is already a full lane vector, so index directly.
        const std::string lang = (language == "c++") ? "c++_block" : language;
        const auto nq = static_cast<std::size_t>(model.nq);
        return trace_flask_rnea_impl(model, lang, {{"x", 3 * nq, true}}, {{"tau", nq, true}});
    }

    auto trace_flask_kinetic_energy(const pinocchio::Model &model, const std::string &language) -> Traced
    {
        const auto nq = static_cast<std::size_t>(model.nq);
        return trace_flask_kinetic_energy_impl(model, language, {{"x", 2 * nq, true}}, {});
    }

    auto trace_flask_kinetic_energy_block(const pinocchio::Model &model, const std::string &language)
        -> Traced
    {
        const std::string lang = (language == "c++") ? "c++_block" : language;
        const auto nq = static_cast<std::size_t>(model.nq);
        return trace_flask_kinetic_energy_impl(model, lang, {{"x", 2 * nq, true}}, {{"ke", 1, true}});
    }

    auto trace_flask_eef_velocity(const RobotInfo &info, const std::string &language) -> Traced
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);
        return trace_flask_eef_velocity_impl(info, language, {{"x", 2 * nq, true}}, {});
    }

    auto trace_flask_eef_velocity_block(const RobotInfo &info, const std::string &language) -> Traced
    {
        const std::string lang = (language == "c++") ? "c++_block" : language;
        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto n_eef = info.end_effector_indexes.size();
        return trace_flask_eef_velocity_impl(info, lang, {{"x", 2 * nq, true}}, {{"eev", 3 * n_eef, true}});
    }
}  // namespace cricket

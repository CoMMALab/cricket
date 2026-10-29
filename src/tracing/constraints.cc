#include <cricket/codegen.hh>

#include "internal.hh"
#include "se3_ops.hh"
#include "cholesky.hh"
#include "bounded_ls.hh"
#include "spline_trace.hh"
#include "catmull_rom.hh"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <Eigen/Geometry>

#include <fmt/format.h>

#include <algorithm>
#include <array>
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

        // Emit code computing [d(err)/d(first n_vars inputs) (row-major), err] for an error function whose
        // first n_vars inputs are the variables to differentiate: the configuration (n_vars = nq), or for
        // the spline constraint the configuration and the curve parameter (n_vars = nq + 1).
        // The bound inputs are part of the tape but do not
        // affect the emitted error: hinging the error against the bounds during tracing would
        // wrongly zero gradients, so the hinge is applied at runtime in vamp instead.
        //
        // `conditionals` marks errors whose tape contains CondExp operations (the se(3) log map, through
        // CppAD's atan2). Those are generated with the block language, which turns each into a SIMD
        // blend; the scalar C++ language would print `if/else`, which cannot take SIMD vectors, and the
        // Rust backend cannot emit conditionals at all.
        auto emit_error_and_jacobian(
            ADFun<CGD> &error_func,
            std::size_t num_inp,
            std::size_t n_vars,
            std::size_t n_out,
            const std::string &language_in,
            bool conditionals = false) -> Traced
        {
            if (conditionals and language_in == "rust")
            {
                throw std::runtime_error("TSR-style errors use the se(3) log map, which needs conditionals the rust language cannot emit");
            }

            // "c++_scalar" is plain scalar C++, where each conditional prints as if/else: it runs on doubles,
            // for tests and for reading the code. The vamp kernels use the block language.
            const std::string language = language_in == "c++_scalar"           ? "c++"
                                         : (conditionals and language_in == "c++") ? "c++_block"
                                                                                   : language_in;

            CodeHandler<double> handler;
            CppAD::vector<CGD> ind_vars(num_inp);
            handler.makeVariables(ind_vars);

            CppAD::vector<CGD> error_vec = error_func.Forward(0, ind_vars);
            CppAD::vector<CGD> jac = error_func.Jacobian(ind_vars);

            // Jacobian with respect to the first n_vars inputs only
            CppAD::vector<CGD> jac_e_q(n_out * n_vars);
            for (auto i = 0U; i < n_out; ++i)
            {
                for (auto j = 0U; j < n_vars; ++j)
                {
                    jac_e_q[i * n_vars + j] = jac[i * num_inp + j];
                }
            }

            std::move(error_vec.begin(), error_vec.end(), std::back_inserter(jac_e_q));

            return Traced{
                generate_code(handler, jac_e_q, language),
                handler.getTemporaryVariableCount(),
                jac_e_q.size()};
        }
    }  // namespace

    auto trace_tsr_error(const RobotInfo &info, const std::string &language) -> Traced
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
        return emit_error_and_jacobian(error_func, num_inp, nq, n_out, language, true);
    }

    template <std::size_t N>
    auto trace_spline_error(const RobotInfo &info, const std::string &language, bool fixed_parameter) -> Traced
    {
        using Layout = SplineInputLayout<N>;

        if (info.end_effector_indexes.empty())
        {
            throw std::runtime_error("trace_spline_error requires an end-effector");
        }

        const auto nq = static_cast<std::size_t>(info.model.nq);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        // Input layout (see SplineInputLayout): q (nq), t, control points (3 N), knots (N + 4)
        const std::size_t num_inp = Layout::total(nq);

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        // The tape is recorded at these values, so they must be a valid spline: the basis recursion divides by
        // knot differences, and all-zero knots would give 0 / 0 = NaN on the tape (which CppAD rejects). The
        // recorded operations (including each CondExp) do not depend on the values, only the check does.
        // Clamped uniform knots, t in the middle.
        ad_inp[Layout::t_idx(nq)] = ADCG(0.5);
        for (std::size_t i = 0; i < Layout::knots; ++i)
        {
            const double knot = i < CubicBSpline::degree + 1 ? 0.0
                                : i >= N                      ? 1.0
                                                              : static_cast<double>(i - CubicBSpline::degree) /
                                                                    static_cast<double>(N - CubicBSpline::degree);
            ad_inp[Layout::knot_idx(nq, i)] = ADCG(knot);
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);

        // position error of the first end-effector against the curve point at t
        const auto spline = read_spline<ADCG, N>(ad_inp, nq);
        const Eigen::Vector3<ADCG> curve_point = spline_point<ADCG, N>(spline);
        const Eigen::Vector3<ADCG> eef_position = ad_data.oMf[info.end_effector_indexes[0]].translation();

        constexpr std::size_t n_out = 3;
        ADVectorXs data(n_out);
        data = eef_position - curve_point;

        ADFun<CGD> error_func(ad_inp, data);

        // d(err)/d(q, t): the curve parameter is the input right after q, so it is the last column. With a fixed
        // parameter t stays on the tape as an input but only the first nq inputs are differentiated.
        return emit_error_and_jacobian(error_func, num_inp, fixed_parameter ? nq : nq + 1, n_out, language, true);
    }

    template auto trace_spline_error<20>(const RobotInfo &info, const std::string &language, bool fixed_parameter)
        -> Traced;

    template <std::size_t N>
    auto trace_catmull_rom_error(const RobotInfo &info, const std::string &language) -> Traced
    {
        using Layout = CatmullRomInputLayout<N>;

        if (info.end_effector_indexes.empty())
        {
            throw std::runtime_error("trace_catmull_rom_error requires an end-effector");
        }

        const auto nq = static_cast<std::size_t>(info.model.nq);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        // Input layout (see CatmullRomInputLayout): q (nq), t, extended points (3 (N + 2)), extended knots (N + 2)
        const std::size_t num_inp = Layout::total(nq);

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        // The tape is recorded at these values, so they must be a valid curve: the blends divide by knot differences,
        // and all-zero knots would give 0 / 0 = NaN on the tape (which CppAD rejects). Strictly increasing knots, t in
        // the middle; the recorded operations (including each CondExp) do not depend on the values.
        ad_inp[Layout::t_idx(nq)] = ADCG(0.5);
        for (std::size_t i = 0; i < Layout::knots; ++i)
        {
            ad_inp[Layout::knot_idx(nq, i)] = ADCG((static_cast<double>(i) - 1.0) / static_cast<double>(N - 1));
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);

        // position error of the first end-effector against the curve point at t
        const auto curve = read_catmull_rom<ADCG, N>(ad_inp, nq);
        const Eigen::Vector3<ADCG> curve_point = catmull_rom_point<ADCG, N>(curve);
        const Eigen::Vector3<ADCG> eef_position = ad_data.oMf[info.end_effector_indexes[0]].translation();

        constexpr std::size_t n_out = 3;
        ADVectorXs data(n_out);
        data = eef_position - curve_point;

        ADFun<CGD> error_func(ad_inp, data);

        // d(err)/dq only: t and the curve data stay on the tape as inputs
        return emit_error_and_jacobian(error_func, num_inp, nq, n_out, language, true);
    }

    template auto trace_catmull_rom_error<20>(const RobotInfo &info, const std::string &language) -> Traced;

    // Starting guess for the spline's curve parameter t, from the end-effector position alone: weighted_waypoint_seed
    // (see spline_trace.hh), the parameter between the nearest fitted waypoint and its nearer neighbour, weighted by the
    // distances to the two. The waypoints and their parameters are inputs (see SplineSeedLayout), so one kernel serves
    // any spline with N waypoints; the closest-t kernel uses the same function for its own seed.
    //
    // CALL-OUT: this is still a simple seed. It is exact for a point on the segment between the two waypoints and an
    // approximation off it, and it ignores that the curve bends between waypoints. A better seed, left for later,
    // projects the end-effector onto the two chords W_{k-1} -> W_k and W_k -> W_{k+1} and interpolates t along the
    // closer one, which also handles points off the segment. A projection or a Newton refinement then needs fewer
    // iterations from it.
    template <std::size_t N>
    auto trace_spline_seed(const RobotInfo &info, const std::string &language) -> Traced
    {
        using Layout = SplineSeedLayout<N>;

        if (info.end_effector_indexes.empty())
        {
            throw std::runtime_error("trace_spline_seed requires an end-effector");
        }

        // The nearest-waypoint search is made of conditionals: SIMD blends from the block language for C++, plain
        // if/else for "c++_scalar", and the Rust backend cannot emit them
        if (language == "rust")
        {
            throw std::runtime_error("trace_spline_seed uses conditionals, which the rust language cannot emit");
        }

        const std::string lang = language == "c++_scalar" ? "c++" : language == "c++" ? "c++_block" : language;

        const auto nq = static_cast<std::size_t>(info.model.nq);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        const std::size_t num_inp = Layout::total(nq);

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
        const Eigen::Vector3<ADCG> eef_position = ad_data.oMf[info.end_effector_indexes[0]].translation();

        // squared distance from the end-effector to every waypoint, and the waypoints' parameters
        std::array<ADCG, Layout::waypoints> distance2;
        std::array<ADCG, Layout::waypoints> parameters;
        for (std::size_t point = 0; point < Layout::waypoints; ++point)
        {
            distance2[point] = ADCG(0.0);
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const ADCG diff = eef_position[axis] - ad_inp[Layout::waypoint_idx(nq, point, axis)];
                distance2[point] += diff * diff;
            }

            parameters[point] = ad_inp[Layout::parameter_idx(nq, point)];
        }

        const ADCG seed = weighted_waypoint_seed<ADCG, Layout::waypoints>(distance2, parameters);

        ADVectorXs data(1);
        data[0] = seed;
        ADFun<CGD> seed_func(ad_inp, data);

        // Only the value is needed (no derivative: the argmin is not differentiated)
        CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(num_inp);
        handler.makeVariables(ind_vars);
        CppAD::vector<CGD> result = seed_func.Forward(0, ind_vars);

        return Traced{generate_code(handler, result, lang), handler.getTemporaryVariableCount(), result.size()};
    }

    template auto trace_spline_seed<20>(const RobotInfo &info, const std::string &language) -> Traced;

    // The closest point of the spline to the end-effector, as a function of q alone: the parameter t* and the squared
    // distance |FK(q) - P(t*)|^2 (the "distance to the spline"). Inputs: see SplineClosestLayout; outputs: [t*, squared
    // distance].
    //
    //   1. x = FK(q).translation, computed once.
    //   2. Seed: weighted_waypoint_seed, the parameter between the nearest waypoint and its nearer neighbour (in the
    //      fitted sequence), weighted by the distances to the two.
    //   3. `iterations` Gauss-Newton steps on t alone, with q fixed. With r = x - P(t), minimising |r|^2 over t gives
    //      t <- t + (r . P'(t)) / (P'(t) . P'(t)). Each step is limited to one waypoint spacing and t is clamped to
    //      [0, 1]. At a point of the curve the residual vanishes at the answer, so the step is Newton's method for a
    //      zero-residual problem there and converges quadratically from a seed within half a waypoint spacing.
    //   4. The squared distance at the final t.
    //
    // This does not certify anything: it finds the closest point on the branch the seed lands on. A residual below
    // tolerance proves the point is on the curve; a larger one only says that no nearby point was found.
    // The Newton steps are not accepted conditionally (no check that the residual dropped): left for later.
    template <std::size_t N>
    auto trace_spline_closest_t(const RobotInfo &info, const std::string &language) -> Traced
    {
        using Layout = SplineClosestLayout<N>;
        constexpr std::size_t iterations = 3;

        if (info.end_effector_indexes.empty())
        {
            throw std::runtime_error("trace_spline_closest_t requires an end-effector");
        }

        if (language == "rust")
        {
            throw std::runtime_error("trace_spline_closest_t uses conditionals, which the rust language cannot emit");
        }

        const std::string lang = language == "c++_scalar" ? "c++" : language == "c++" ? "c++_block" : language;

        const auto nq = static_cast<std::size_t>(info.model.nq);

        ADModel ad_model = info.model.cast<ADCG>();
        ADData ad_data(ad_model);

        const std::size_t num_inp = Layout::total(nq);

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        // The tape is recorded at these values, so the knots must be valid (the basis divides by knot differences):
        // clamped uniform knots, as for the error kernel
        for (std::size_t i = 0; i < Layout::knots; ++i)
        {
            const double knot = i < CubicBSpline::degree + 1 ? 0.0
                                : i >= N                      ? 1.0
                                                              : static_cast<double>(i - CubicBSpline::degree) /
                                                                    static_cast<double>(N - CubicBSpline::degree);
            ad_inp[Layout::knot_idx(nq, i)] = ADCG(knot);
        }

        Independent(ad_inp);

        ADVectorXs ad_q(nq);
        for (auto i = 0U; i < nq; ++i)
        {
            ad_q[i] = ad_inp[i];
        }

        forwardKinematics(ad_model, ad_data, ad_q);
        updateFramePlacements(ad_model, ad_data);
        const Eigen::Vector3<ADCG> x = ad_data.oMf[info.end_effector_indexes[0]].translation();

        // The spline as a SplineInputs: its data from the input tape, t to be set per evaluation
        SplineInputs<ADCG, N> spline;
        for (std::size_t point = 0; point < N; ++point)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                spline.control_points[point][axis] = ad_inp[Layout::control_point_idx(nq, point, axis)];
            }
        }

        for (std::size_t i = 0; i < Layout::knots; ++i)
        {
            spline.knots[i] = ad_inp[Layout::knot_idx(nq, i)];
        }

        // 2. seed, the same function the seed kernel uses: squared distances to the waypoints, then the distance-weighted
        // parameter between the nearest waypoint and its nearer neighbour
        std::array<ADCG, N> distance2;
        std::array<ADCG, N> parameters;
        for (std::size_t point = 0; point < N; ++point)
        {
            distance2[point] = ADCG(0.0);
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const ADCG diff = x[axis] - ad_inp[Layout::waypoint_idx(nq, point, axis)];
                distance2[point] += diff * diff;
            }

            parameters[point] = ad_inp[Layout::parameter_idx(nq, point)];
        }

        ADCG t = weighted_waypoint_seed<ADCG, N>(distance2, parameters);

        // 3. Gauss-Newton on t alone, q fixed
        const ADCG max_step = ADCG(1.0 / static_cast<double>(N - 1));
        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            spline.t = t;
            const auto evaluated = spline_point_and_velocity<ADCG, N>(spline);

            ADCG numerator = ADCG(0.0);
            ADCG denominator = ADCG(1e-12);  // keeps the step finite at a stationary curve point
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                numerator += (x[axis] - evaluated.point[axis]) * evaluated.velocity[axis];
                denominator += evaluated.velocity[axis] * evaluated.velocity[axis];
            }

            ADCG step = numerator / denominator;
            step = CppAD::CondExpGt(step, max_step, max_step, step);
            step = CppAD::CondExpLt(step, ADCG(0.0) - max_step, ADCG(0.0) - max_step, step);

            t = t + step;
            t = CppAD::CondExpLt(t, ADCG(0.0), ADCG(0.0), t);
            t = CppAD::CondExpGt(t, ADCG(1.0), ADCG(1.0), t);
        }

        // 4. the squared distance at the final t
        spline.t = t;
        const Eigen::Vector3<ADCG> closest = spline_point<ADCG, N>(spline);
        ADCG squared_distance = ADCG(0.0);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const ADCG diff = x[axis] - closest[axis];
            squared_distance += diff * diff;
        }

        ADVectorXs data(2);
        data[0] = t;
        data[1] = squared_distance;
        ADFun<CGD> closest_func(ad_inp, data);

        // Only the values are needed: the closest point is not differentiated here
        CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(num_inp);
        handler.makeVariables(ind_vars);
        CppAD::vector<CGD> result = closest_func.Forward(0, ind_vars);

        return Traced{generate_code(handler, result, lang), handler.getTemporaryVariableCount(), result.size()};
    }

    template auto trace_spline_closest_t<20>(const RobotInfo &info, const std::string &language) -> Traced;

    auto trace_tsr_bimanual_error(
        const RobotInfo &info,
        const std::string &language,
        std::size_t eef1,
        std::size_t eef2) -> Traced
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
        return emit_error_and_jacobian(error_func, num_inp, nq, task_dim, language, true);
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

    auto trace_solve_spline(
        const RobotInfo &info,
        const std::string &language,
        ProjMethod method,
        bool fixed_parameter) -> Traced
    {
        // err is the 3-vector position error, and the unknowns are the joints plus the curve parameter t (the joints
        // alone when t is given)
        return trace_solve_jacobian(info, language, method, 3, fixed_parameter ? 0 : 1);
    }

    auto trace_solve_jacobian(
        const RobotInfo &info,
        const std::string &language,
        ProjMethod method,
        std::size_t err_size,
        std::size_t extra_vars) -> Traced
    {
        constexpr double damp = 1e-6;

        // The unknowns: the joints, and any extra variables the constraint adds (the spline's curve parameter)
        const auto n_vars = static_cast<std::size_t>(info.model.nq) + extra_vars;

        const bool bounded = method == ProjMethod::BoundedLS;
        if (bounded and language == "rust")
        {
            // The Rust backend cannot emit conditionals, which the active-set solver is made of.
            throw std::runtime_error("ProjMethod::BoundedLS is not supported for the rust language");
        }

        // The SIMD blends the conditionals compile to come from the block language; "c++_scalar" is plain
        // scalar C++ (conditionals as if/else) that runs on doubles, for tests
        const std::string lang = language == "c++_scalar"           ? "c++"
                                 : (bounded and language == "c++") ? "c++_block"
                                                                   : language;

        // Input layout (must match vamp's JacobianProjectInp::operator[]):
        // J (err_size x n_vars, row-major), then err (err_size), then, for BoundedLS only, the allowed step
        // to the lower bound (lower - x, n_vars) and to the upper bound (upper - x, n_vars), where x is the
        // joints followed by any extra variables
        const std::size_t num_inp = err_size * n_vars + err_size + (bounded ? 2 * n_vars : 0);

        ADVectorXs ad_inp(num_inp);
        for (auto i = 0U; i < num_inp; ++i)
        {
            ad_inp[i] = ADCG(0.0);
        }

        Independent(ad_inp);

        ADMatrixXs J(err_size, n_vars);
        for (auto i = 0U; i < err_size; ++i)
        {
            for (auto j = 0U; j < n_vars; ++j)
            {
                J(i, j) = ad_inp[i * n_vars + j];
            }
        }

        ADVectorXs err(err_size);
        for (auto i = 0U; i < err_size; ++i)
        {
            err[i] = ad_inp[err_size * n_vars + i];
        }

        ADVectorXs grad(n_vars);
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
                ADMatrixXs identity = ADMatrixXs::Identity(n_vars, n_vars);
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
            case ProjMethod::BoundedLS:
            {
                ADVectorXs lower(n_vars);
                ADVectorXs upper(n_vars);
                for (auto i = 0U; i < n_vars; ++i)
                {
                    lower[i] = ad_inp[err_size * n_vars + err_size + i];
                    upper[i] = ad_inp[err_size * n_vars + err_size + n_vars + i];
                }

                // Every iteration either fixes a joint at a bound, frees one, or finishes the solve
                const std::size_t iterations = std::min<std::size_t>(n_vars + 2, 10);
                grad = bounded_ls_step<ADMatrixXs, ADVectorXs, ADCG>(J, err, lower, upper, damp, iterations);
                break;
            }
        }

        ADFun<CGD> solve_func(ad_inp, grad);

        CodeHandler<double> handler;
        CppAD::vector<CGD> ind_vars(num_inp);
        handler.makeVariables(ind_vars);

        CppAD::vector<CGD> result = solve_func.Forward(0, ind_vars);

        return Traced{
            generate_code(handler, result, lang),
            handler.getTemporaryVariableCount(),
            result.size()};
    }

    auto trace_com_jacobian(
        const RobotInfo &info,
        const std::vector<std::string> &reference_frames,
        const std::string &language) -> Traced
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
        return emit_error_and_jacobian(com_func, nq, nq, 3, language);
    }

    auto trace_closed_loop_error(
        const RobotInfo &info,
        const std::vector<ClosedLoop> &loops,
        const std::string &language) -> Traced
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
        return emit_error_and_jacobian(error_func, nq, nq, loops.size(), language);
    }

    auto trace_lead_screw_error(const RobotInfo &info, const std::string &language) -> Traced
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
        return emit_error_and_jacobian(error_func, num_inp, nq, 1, language, true);
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

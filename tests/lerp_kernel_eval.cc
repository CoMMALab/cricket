// Evaluates the lerp constraint kernels two ways on the same inputs, runs a fixed-t projection on the generated code,
// and writes a dump that tests/view_lerp.py shows in viser. The curve is a straight-line interpolation through 32
// keyframes: positions in R^3 (curve=r3), or poses in SO(3) x R^3 with the rotation slerped (curve=so3r3).
//
//   1. reference: forward kinematics from pinocchio, the curve point (and its slerped rotation) from plain host code
//      (Eigen), the error written out, and the Jacobian by central differences. No CppAD, no generated code.
//   2. traced: the generated kernels (trace_lerp_r3_error / trace_lerp_so3r3_error and trace_solve_jacobian, in the
//      "c++_scalar" language, plain C++ where each conditional is an if/else). The code is written to a temporary file,
//      compiled into a shared library with the compiler this was built with, loaded, and called.
//
// Both give  y = [ d(err)/dq (m x nq, row-major), err (m) ]  with m = 3 for r3 and 6 for so3r3, and
//   r3:    err = FK(q).position - P(t)
//   so3r3: err = [ R_t^T (p - p_t) ; weight * log(R_t^T R) ]   (the TSR error against the pose P(t) of the curve)
// t is held fixed: it is an input of the kernel and the unknowns of a projection are the joints alone.
//
//   lerp_kernel_eval [curve=r3] [mode=qt] [t=0.4] [weight=1] [turn=60] [q=a,b,c,...] [method=bounded] [iterations=15]
//                    [dump=out.json] [sweep=1] [repeat=1000] [tolerance=1e-8] [max_step=0] [flags="-O3 -march=native"] [keep=1]
//
//   mode       qt (default): t is held fixed, the end-effector must reach the point P(t), the joints alone are solved for.
//              qs: t only picks the segment (the one it starts on); the end-effector may move along that segment, t is
//              solved for next to the joints (kernel: lerp_*_segment_error, with t as an extra Jacobian column) and kept
//              in the segment by the bounded solve; the projection ends wherever on the segment it gets to. Then
//              y = [ d(err)/d(q, t) (m x (nq + 1)), err (m), tau_i - t, tau_{i + 1} - t ].
//
//   curve      r3 or so3r3
//   t          curve parameter in [0, 1]: held fixed in the projection (qt), or where it starts (qs)
//   scale      the size of the curve: the S that the keyframes follow has extents 20 x 12 x 8 cm at scale 1, and a larger
//              scale makes its segments longer
//   weight     so3r3: the weight of the rotation error against the translation error (m / rad); it also sets how the
//              keyframes' parameters taus are spaced, since a segment's length is sqrt(|dp|^2 + weight^2 theta^2)
//   turn       so3r3: the rotation about the world z axis (degrees) between the first and last keyframe
//   q          joint values, one per joint of the robot
//   method     projection solve: bounded (default), inner, outer or gradient (the four generated solves)
//   iterations projection steps from q onto the point P(t), on the generated kernels
//   dump       write a JSON file that tests/view_lerp.py shows
//   sweep      1 = also compare the two evaluations at many (q, t), among them t on the segment boundaries, 0 and 1
//   tolerance  the squared error at which a projection has converged (as in vamp); it stops there, or after `iterations`
//   max_step   a cap on the norm of the joint part of a projection step (rad; 0 = none, the default): a longer step is
//              scaled down to that length, keeping its direction (t's step is scaled with it). The cap is in this tool's
//              loop only; the generated kernels and solvers are not changed.
//   flags      the compiler flags of the generated kernels (default "-O3 -march=native"; try "-O1" to compare)
//   repeat     calls timed for each of the error kernel, the solve and the non-traced reference, and projections timed
//              from the start to convergence (0 = no timing). The kernels are the generated code in scalar C++, one
//              lane, so the times compare the two curves and the two modes with each other; vamp's kernels run the same
//              code as SIMD blends over a whole vector of lanes.
//   keep       keep the generated source and library (their paths are printed)
//
// The curve starts at the end-effector's pose at the default q, so t = 0 is already satisfied and a larger t makes the
// projection move the arm. To plug in other numbers, change the arguments, or edit the `Inputs` section in main().

#include <cricket/codegen.hh>

#include "tracing/lerp.hh"

#include <nlohmann/json.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <Eigen/Geometry>

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr std::size_t N = cricket::lerp_points;
    constexpr double pi = 3.14159265358979323846;

    // The lerp curve: N keyframes at the parameters taus
    struct Curve
    {
        bool pose = false;
        double weight = 1.0;
        std::vector<Eigen::Vector3d> positions;
        std::vector<Eigen::Quaterniond> rotations;  // so3r3 only
        std::vector<double> taus;

        struct Value
        {
            Eigen::Vector3d position;
            Eigen::Quaterniond rotation;
            std::size_t segment;
            double s;  // the fraction along the segment, in [0, 1]
        };

        // The parameters are the normalised lengths of the segments: |dp|, or for poses sqrt(|dp|^2 + w^2 theta^2),
        // theta the angle between the two rotations
        void set_taus()
        {
            taus.assign(1, 0.0);
            for (std::size_t i = 0; i + 1 < positions.size(); ++i)
            {
                double length2 = (positions[i + 1] - positions[i]).squaredNorm();
                if (pose)
                {
                    const double theta = rotations[i].angularDistance(rotations[i + 1]);
                    length2 += weight * weight * theta * theta;
                }

                taus.push_back(taus.back() + std::sqrt(length2));
            }

            for (auto &tau : taus)
            {
                tau /= taus.back();
            }
        }

        // The segment t is on: the first whose end is after t; t = 1 stays in the last one
        auto segment_of(double t) const -> std::size_t
        {
            t = std::clamp(t, 0.0, 1.0);

            std::size_t segment = 0;
            while (segment + 2 < positions.size() and t >= taus[segment + 1])
            {
                ++segment;
            }

            return segment;
        }

        // The pose at the parameter t on the line of a given segment, which may be a little past its ends (the
        // segment projection keeps t in the segment, but a finite difference steps out of it): the lerp, and the
        // slerp R_i exp(s log(R_i^T R_{i + 1})) with s not clamped
        auto value_on(std::size_t segment, double t) const -> Value
        {
            const double s = (t - taus[segment]) / (taus[segment + 1] - taus[segment]);
            Value value;
            value.segment = segment;
            value.s = s;
            value.position = positions[segment] + s * (positions[segment + 1] - positions[segment]);
            value.rotation = Eigen::Quaterniond::Identity();
            if (pose)
            {
                const Eigen::AngleAxisd relative(rotations[segment].conjugate() * rotations[segment + 1]);
                value.rotation = rotations[segment] * Eigen::Quaterniond(Eigen::AngleAxisd(s * relative.angle(), relative.axis()));
            }

            return value;
        }

        auto value(double t) const -> Value
        {
            t = std::clamp(t, 0.0, 1.0);
            return value_on(segment_of(t), t);
        }

        auto rows() const -> std::size_t
        {
            return pose ? 6 : 3;
        }
    };

    auto end_effector(const cricket::RobotInfo &info, const Eigen::VectorXd &q) -> pinocchio::SE3
    {
        pinocchio::Data data(info.model);
        pinocchio::forwardKinematics(info.model, data, q);
        pinocchio::updateFramePlacements(info.model, data);
        return data.oMf[info.end_effector_indexes.at(0)];
    }

    auto rotation_vector(const Eigen::Matrix3d &rotation) -> Eigen::Vector3d
    {
        const Eigen::AngleAxisd angle_axis(rotation);
        return angle_axis.angle() * angle_axis.axis();
    }

    // The error of the constraint at (q, t), written out from the definition
    // (on the segment `segment` when given, else the one t is on)
    auto reference_error(
        const cricket::RobotInfo &info,
        const Curve &curve,
        const Eigen::VectorXd &q,
        double t,
        std::optional<std::size_t> segment = std::nullopt) -> Eigen::VectorXd
    {
        const auto pose = end_effector(info, q);
        const auto target = segment ? curve.value_on(*segment, t) : curve.value(t);

        Eigen::VectorXd error(curve.rows());
        if (not curve.pose)
        {
            error = pose.translation() - target.position;
        }
        else
        {
            const Eigen::Matrix3d target_rotation = target.rotation.toRotationMatrix();
            error.head(3) = target_rotation.transpose() * (pose.translation() - target.position);
            error.tail(3) = curve.weight * rotation_vector(target_rotation.transpose() * pose.rotation());
        }

        return error;
    }

    // y = [d(err)/dq (m x nq, row-major), err (m)] from the reference error, the Jacobian by central differences
    auto reference_output(const cricket::RobotInfo &info, const Curve &curve, const Eigen::VectorXd &q, double t)
        -> std::vector<double>
    {
        constexpr double h = 1e-6;

        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto m = curve.rows();

        std::vector<double> y(m * nq + m);
        for (std::size_t joint = 0; joint < nq; ++joint)
        {
            Eigen::VectorXd plus = q;
            Eigen::VectorXd minus = q;
            plus[static_cast<Eigen::Index>(joint)] += h;
            minus[static_cast<Eigen::Index>(joint)] -= h;

            const Eigen::VectorXd column = (reference_error(info, curve, plus, t) - reference_error(info, curve, minus, t)) / (2.0 * h);
            for (std::size_t row = 0; row < m; ++row)
            {
                y[row * nq + joint] = column[static_cast<Eigen::Index>(row)];
            }
        }

        const Eigen::VectorXd error = reference_error(info, curve, q, t);
        for (std::size_t row = 0; row < m; ++row)
        {
            y[m * nq + row] = error[static_cast<Eigen::Index>(row)];
        }

        return y;
    }

    // The same for the segment form: y = [d(err)/d(q, t) (m x (nq + 1), t the last column, by central differences on the
    // line of the segment), err (m), tau_i - t, tau_{i + 1} - t]
    auto reference_segment_output(
        const cricket::RobotInfo &info, const Curve &curve, const Eigen::VectorXd &q, double t, std::size_t segment)
        -> std::vector<double>
    {
        constexpr double h = 1e-6;

        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto m = curve.rows();
        const auto n = nq + 1;

        std::vector<double> y(m * n + m + 2);
        for (std::size_t column = 0; column < n; ++column)
        {
            Eigen::VectorXd plus = q;
            Eigen::VectorXd minus = q;
            double t_plus = t;
            double t_minus = t;
            if (column < nq)
            {
                plus[static_cast<Eigen::Index>(column)] += h;
                minus[static_cast<Eigen::Index>(column)] -= h;
            }
            else
            {
                t_plus += h;
                t_minus -= h;
            }

            const Eigen::VectorXd derivative = (reference_error(info, curve, plus, t_plus, segment) - reference_error(info, curve, minus, t_minus, segment)) / (2.0 * h);
            for (std::size_t row = 0; row < m; ++row)
            {
                y[row * n + column] = derivative[static_cast<Eigen::Index>(row)];
            }
        }

        const Eigen::VectorXd error = reference_error(info, curve, q, t, segment);
        for (std::size_t row = 0; row < m; ++row)
        {
            y[m * n + row] = error[static_cast<Eigen::Index>(row)];
        }

        y[m * n + m] = curve.taus[segment] - t;
        y[m * n + m + 1] = curve.taus[segment + 1] - t;
        return y;
    }

    // The flat input vector of the traced kernel (see LerpR3InputLayout and LerpPoseInputLayout in tracing/lerp.hh).
    // With Segment, the segment kernel's: t is the parameter on the segment that t_segment is on.
    template <bool Segment>
    auto kernel_inputs(const Curve &curve, const Eigen::VectorXd &q, double t, double t_segment = 0.0) -> std::vector<double>
    {
        const auto nq = static_cast<std::size_t>(q.size());

        std::vector<double> x;
        if (not curve.pose)
        {
            using Layout = cricket::LerpR3InputLayout<N, Segment>;
            x.assign(Layout::total(nq), 0.0);
            x[Layout::t_idx(nq)] = t;
            if constexpr (Segment)
            {
                x[Layout::segment_idx(nq)] = t_segment;
            }

            for (std::size_t i = 0; i < N; ++i)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    x[Layout::point_idx(nq, i, axis)] = curve.positions[i][static_cast<Eigen::Index>(axis)];
                }

                x[Layout::tau_idx(nq, i)] = curve.taus[i];
            }
        }
        else
        {
            using Layout = cricket::LerpPoseInputLayout<N, Segment>;
            x.assign(Layout::total(nq), 0.0);
            x[Layout::t_idx(nq)] = t;
            if constexpr (Segment)
            {
                x[Layout::segment_idx(nq)] = t_segment;
            }

            x[Layout::weight_idx(nq)] = curve.weight;
            for (std::size_t i = 0; i < N; ++i)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    x[Layout::position_idx(nq, i, axis)] = curve.positions[i][static_cast<Eigen::Index>(axis)];
                }

                const auto &rotation = curve.rotations[i];
                x[Layout::quaternion_idx(nq, i, 0)] = rotation.w();
                x[Layout::quaternion_idx(nq, i, 1)] = rotation.x();
                x[Layout::quaternion_idx(nq, i, 2)] = rotation.y();
                x[Layout::quaternion_idx(nq, i, 3)] = rotation.z();
                x[Layout::tau_idx(nq, i)] = curve.taus[i];
            }
        }

        for (std::size_t joint = 0; joint < nq; ++joint)
        {
            x[joint] = q[static_cast<Eigen::Index>(joint)];
        }

        return x;
    }

    // A traced function compiled into a shared library and loaded, so it can be called many times:
    // kernel(x) runs the generated code on the flat input vector x and returns its outputs.
    class LoadedKernel
    {
    public:
        LoadedKernel(const cricket::Traced &traced, const std::string &name, const std::string &flags, bool keep)
          : outputs_(traced.outputs)
          , directory_(
                std::filesystem::temp_directory_path() /
                ("lerp_kernel_eval_" + std::to_string(::getpid()) + "_" + name))
          , keep_(keep)
        {
            std::filesystem::create_directories(directory_);

            const auto source = directory_ / "kernel.cc";
            const auto library = directory_ / "kernel.so";
            {
                std::ofstream out(source);
                out << "#include <cmath>\n"
                    << "using std::sin; using std::cos; using std::sqrt; using std::atan2; using std::fabs;\n"
                    << "extern \"C\" void kernel(const double *x, double *y)\n{\n"
                    << "    double v[" << std::max<std::size_t>(traced.temp_variables, 1) << "];\n"
                    << traced.code << "\n}\n";
            }

            const std::string command = std::string(CRICKET_CXX) + " " + flags + " -shared -fPIC -o " + library.string() +
                                        " " + source.string() + " 2>&1";
            if (std::system(command.c_str()) != 0)
            {
                throw std::runtime_error("compiling the generated kernel failed: " + command);
            }

            handle_ = dlopen(library.c_str(), RTLD_NOW);
            if (handle_ == nullptr)
            {
                throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
            }

            kernel_ = reinterpret_cast<Function>(dlsym(handle_, "kernel"));
            if (kernel_ == nullptr)
            {
                throw std::runtime_error("the generated library has no `kernel` symbol");
            }

            if (keep_)
            {
                std::printf("kept %s: %s, %s\n\n", name.c_str(), source.c_str(), library.c_str());
            }
        }

        ~LoadedKernel()
        {
            if (handle_ != nullptr)
            {
                dlclose(handle_);
            }

            if (not keep_)
            {
                std::filesystem::remove_all(directory_);
            }
        }

        LoadedKernel(const LoadedKernel &) = delete;
        auto operator=(const LoadedKernel &) -> LoadedKernel & = delete;

        auto operator()(const std::vector<double> &x) const -> std::vector<double>
        {
            std::vector<double> y(outputs_, 0.0);
            kernel_(x.data(), y.data());
            return y;
        }

    private:
        using Function = void (*)(const double *, double *);

        std::size_t outputs_;
        std::filesystem::path directory_;
        bool keep_;
        void *handle_ = nullptr;
        Function kernel_ = nullptr;
    };

    auto max_difference(const std::vector<double> &a, const std::vector<double> &b) -> double
    {
        double difference = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            difference = std::max(difference, std::abs(a[i] - b[i]));
        }

        return difference;
    }

    void print_output(const char *label, const std::vector<double> &y, std::size_t nq, std::size_t m)
    {
        std::printf("%s\n", label);
        std::printf("  d(err)/d(q%s), %zu x %zu:\n", y.size() == m * nq + m + 2 ? ", t" : "", m, nq);
        for (std::size_t row = 0; row < m; ++row)
        {
            std::printf("   ");
            for (std::size_t col = 0; col < nq; ++col)
            {
                std::printf(" %10.5f", y[row * nq + col]);
            }

            std::printf("\n");
        }

        std::printf("  err:");
        for (std::size_t row = 0; row < m; ++row)
        {
            std::printf(" %10.5f", y[m * nq + row]);
        }

        std::printf("\n");
        if (y.size() == m * nq + m + 2)
        {
            std::printf("  allowed step of t to the start and end of the segment: %10.5f %10.5f\n", y[m * nq + m], y[m * nq + m + 1]);
        }

        std::printf("\n");
    }

    // The mean time of one call of `call`, in microseconds, over `repeat` calls after a warm-up; the sum of the results keeps
    // the calls from being optimised away
    template <typename Call>
    auto time_microseconds(std::size_t repeat, Call call) -> double
    {
        volatile double sink = 0.0;
        for (std::size_t i = 0; i < 10; ++i)
        {
            sink = sink + call();
        }

        const auto started = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < repeat; ++i)
        {
            sink = sink + call();
        }

        const auto elapsed = std::chrono::steady_clock::now() - started;
        return std::chrono::duration<double, std::micro>(elapsed).count() / static_cast<double>(repeat);
    }

    auto parse_numbers(const std::string &text) -> std::vector<double>
    {
        std::vector<double> values;
        std::stringstream stream(text);
        std::string item;
        while (std::getline(stream, item, ','))
        {
            values.push_back(std::stod(item));
        }

        return values;
    }

    // Joint origins (base to last joint) and then the end-effector, in the world frame
    auto arm_positions(const cricket::RobotInfo &info, const Eigen::VectorXd &q) -> std::vector<Eigen::Vector3d>
    {
        pinocchio::Data data(info.model);
        pinocchio::forwardKinematics(info.model, data, q);
        pinocchio::updateFramePlacements(info.model, data);

        std::vector<Eigen::Vector3d> positions;
        for (int joint = 1; joint < info.model.njoints; ++joint)  // joint 0 is the universe
        {
            positions.push_back(data.oMi[static_cast<std::size_t>(joint)].translation());
        }

        positions.push_back(data.oMf[info.end_effector_indexes.at(0)].translation());
        return positions;
    }

    auto to_vector(const Eigen::VectorXd &v) -> std::vector<double>
    {
        return std::vector<double>(v.data(), v.data() + v.size());
    }

    auto to_vector(const Eigen::Vector3d &v) -> std::vector<double>
    {
        return {v[0], v[1], v[2]};
    }

    auto to_vector(const Eigen::Quaterniond &q) -> std::vector<double>
    {
        return {q.w(), q.x(), q.y(), q.z()};  // w x y z
    }

    struct Iterate
    {
        Eigen::VectorXd q;
        double t = 0.0;
    };

    // Projection steps with t held fixed, on the generated code: each step runs the error kernel (J and err at (q, t),
    // which is also the start of the solve's input), appends the allowed step to every joint limit for the bounded
    // solve, runs the solve, and applies  q <- q - gradient.  Prints one row per iterate.
    auto project_traced(
        const cricket::RobotInfo &info,
        const Curve &curve,
        const LoadedKernel &error_kernel,
        const LoadedKernel &solve_kernel,
        const Eigen::VectorXd &q0,
        double t,
        std::size_t iterations,
        const std::string &method_name,
        bool bounded,
        double tolerance,
        double max_step,
        bool verbose) -> std::vector<Iterate>
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto m = curve.rows();

        if (verbose) std::printf("projection with t = %.4f held fixed, on the generated kernels (error kernel, then the %s solve)\n", t, method_name.c_str());
        if (verbose) std::printf("  step   |err|            position (m)   rotation (rad)   |gradient|   step scale (1 = not capped)\n");

        std::vector<Iterate> trail{{q0, t}};
        for (std::size_t iteration = 0; iteration <= iterations; ++iteration)
        {
            const auto &current = trail.back();

            auto solve_input = error_kernel(kernel_inputs<false>(curve, current.q, t));

            // the error is split for printing: the translation rows and, for poses, the (weighted) rotation rows
            double position_error = 0.0;
            double rotation_error = 0.0;
            for (std::size_t row = 0; row < m; ++row)
            {
                const double value = solve_input[m * nq + row];
                (row < 3 ? position_error : rotation_error) += value * value;
            }

            if (bounded)
            {
                for (std::size_t joint = 0; joint < nq; ++joint)
                {
                    solve_input.push_back(
                        info.model.lowerPositionLimit[static_cast<Eigen::Index>(joint)] -
                        current.q[static_cast<Eigen::Index>(joint)]);
                }

                for (std::size_t joint = 0; joint < nq; ++joint)
                {
                    solve_input.push_back(
                        info.model.upperPositionLimit[static_cast<Eigen::Index>(joint)] -
                        current.q[static_cast<Eigen::Index>(joint)]);
                }
            }

            const auto gradient = solve_kernel(solve_input);
            double gradient_norm = 0.0;
            for (const double g : gradient)
            {
                gradient_norm += g * g;
            }

            // the cap: a step longer than max_step is scaled down, direction kept
            double joint_norm2 = 0.0;
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                joint_norm2 += gradient[joint] * gradient[joint];
            }

            const double scale = (max_step > 0.0 and joint_norm2 > max_step * max_step) ? max_step / std::sqrt(joint_norm2) : 1.0;

            if (verbose) std::printf(
                "  %4zu   %.3e     %.3e        %.3e         %.3e      %.3f\n",
                iteration,
                std::sqrt(position_error + rotation_error),
                std::sqrt(position_error),
                std::sqrt(rotation_error),
                std::sqrt(gradient_norm),
                scale);
            if (iteration == iterations or position_error + rotation_error <= tolerance)
            {
                break;
            }

            Iterate next = current;
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                next.q[static_cast<Eigen::Index>(joint)] -= scale * gradient[joint];
            }

            trail.push_back(next);
        }

        if (verbose) std::printf("\n");
        return trail;
    }

    // Projection steps with t an unknown, on the generated code (qs): each step runs the segment error kernel at (q, t) on
    // the segment t0 is on, which gives J (m x (nq + 1)), err, and the allowed step of t to the two ends of the segment.
    // The solve's input is J and err, then for the bounded solve lower - x and upper - x: the joints' limits minus q, and for
    // t those two last outputs of the kernel. Then  x <- x - gradient  with t kept in the segment. Prints one row per
    // iterate, with t and the fraction s along the segment.
    auto project_traced_segment(
        const cricket::RobotInfo &info,
        const Curve &curve,
        const LoadedKernel &error_kernel,
        const LoadedKernel &solve_kernel,
        const Eigen::VectorXd &q0,
        double t0,
        std::size_t iterations,
        const std::string &method_name,
        bool bounded,
        double tolerance,
        double max_step,
        bool verbose,
        const LoadedKernel *free_solve) -> std::vector<Iterate>
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto n = nq + 1;  // the unknowns: the joints and t
        const auto m = curve.rows();

        const auto segment = curve.segment_of(t0);
        const double lower_t = curve.taus[segment];
        const double upper_t = curve.taus[segment + 1];

        if (verbose) std::printf(
            "projection with t free on segment %zu (t in [%.4f, %.4f]), on the generated kernels (error kernel, then the %s solve)\n",
            segment,
            lower_t,
            upper_t,
            method_name.c_str());
        if (verbose) std::printf("  step   |err|            position (m)   rotation (rad)         t        s   |gradient|    gradient_t   s wanted (unbounded step)   step scale\n");

        std::vector<Iterate> trail{{q0, t0}};
        for (std::size_t iteration = 0; iteration <= iterations; ++iteration)
        {
            const auto &current = trail.back();

            // [J, err, lo, hi]; the solve takes J and err, then the bounds
            const auto output = error_kernel(kernel_inputs<true>(curve, current.q, current.t, t0));
            std::vector<double> solve_input(output.begin(), output.begin() + static_cast<std::ptrdiff_t>(m * n + m));

            // where the same step would put t with no bound on it: the unbounded solve (InnerLM) on the same J and err. If
            // that s is far outside [0, 1] the end-effector is off the segment's line by more than the segment is long, and
            // the segment's end is where the projection belongs.
            double wanted_s = std::nan("");
            if (free_solve != nullptr)
            {
                const auto free_gradient = (*free_solve)(solve_input);
                wanted_s = (current.t - free_gradient[nq] - lower_t) / (upper_t - lower_t);
            }

            double position_error = 0.0;
            double rotation_error = 0.0;
            for (std::size_t row = 0; row < m; ++row)
            {
                const double value = output[m * n + row];
                (row < 3 ? position_error : rotation_error) += value * value;
            }

            if (bounded)
            {
                for (std::size_t joint = 0; joint < nq; ++joint)
                {
                    solve_input.push_back(
                        info.model.lowerPositionLimit[static_cast<Eigen::Index>(joint)] -
                        current.q[static_cast<Eigen::Index>(joint)]);
                }

                solve_input.push_back(output[m * n + m]);

                for (std::size_t joint = 0; joint < nq; ++joint)
                {
                    solve_input.push_back(
                        info.model.upperPositionLimit[static_cast<Eigen::Index>(joint)] -
                        current.q[static_cast<Eigen::Index>(joint)]);
                }

                solve_input.push_back(output[m * n + m + 1]);
            }

            const auto gradient = solve_kernel(solve_input);
            double gradient_norm = 0.0;
            for (const double g : gradient)
            {
                gradient_norm += g * g;
            }

            // the cap: a step longer than max_step is scaled down, direction kept
            double joint_norm2 = 0.0;
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                joint_norm2 += gradient[joint] * gradient[joint];
            }

            const double scale = (max_step > 0.0 and joint_norm2 > max_step * max_step) ? max_step / std::sqrt(joint_norm2) : 1.0;

            if (verbose) std::printf(
                "  %4zu   %.3e     %.3e        %.3e      %.5f   %.4f   %.3e    %.3e   %9.3f     %.3f\n",
                iteration,
                std::sqrt(position_error + rotation_error),
                std::sqrt(position_error),
                std::sqrt(rotation_error),
                current.t,
                (current.t - lower_t) / (upper_t - lower_t),
                std::sqrt(gradient_norm),
                gradient[nq],
                wanted_s,
                scale);
            if (iteration == iterations or position_error + rotation_error <= tolerance)
            {
                break;
            }

            Iterate next = current;
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                next.q[static_cast<Eigen::Index>(joint)] -= scale * gradient[joint];
            }

            // the bounded solve keeps t in the segment; the others do not, so it is clamped
            next.t = std::clamp(current.t - scale * gradient[nq], lower_t, upper_t);
            trail.push_back(next);
        }

        if (verbose) std::printf("\n");
        return trail;
    }

    // Everything the viewer (tests/view_lerp.py) draws: the keyframes and their parameters, the sampled curve, the two
    // evaluations, and the projection iterations with the arm at each one.
    void write_dump(
        const std::string &path,
        const cricket::RobotInfo &info,
        const Curve &curve,
        const std::string &mode,
        std::size_t segment,
        double t,
        const std::vector<double> &expected,
        const std::vector<double> &actual,
        const std::vector<Iterate> &trail)
    {
        nlohmann::json dump;
        dump["kind"] = curve.pose ? "so3r3" : "r3";
        dump["mode"] = mode;
        dump["label"] = std::string(curve.pose ? "lerp so3r3" : "lerp r3") + ", " + mode + ", t = " + std::to_string(t);
        dump["nq"] = info.model.nq;
        dump["weight"] = curve.weight;
        dump["t_start"] = t;
        dump["taus"] = curve.taus;

        auto &keyframes = dump["keyframes"];
        auto &quaternions = dump["keyframe_quaternions"];
        for (std::size_t i = 0; i < curve.positions.size(); ++i)
        {
            keyframes.push_back(to_vector(curve.positions[i]));
            if (curve.pose)
            {
                quaternions.push_back(to_vector(curve.rotations[i]));
            }
        }

        // the curve, sampled finely
        auto &curve_t = dump["curve_t"];
        auto &samples = dump["curve"];
        for (int i = 0; i <= 400; ++i)
        {
            curve_t.push_back(i / 400.0);
            samples.push_back(to_vector(curve.value(i / 400.0).position));
        }

        dump["reference_output"] = expected;
        dump["traced_output"] = actual;

        auto &iterations = dump["iterations"];
        for (const auto &iterate : trail)
        {
            const auto arm = arm_positions(info, iterate.q);
            const auto target = curve.value_on(segment, iterate.t);
            const auto error = reference_error(info, curve, iterate.q, iterate.t, segment);

            nlohmann::json step;
            step["t"] = iterate.t;
            step["q"] = to_vector(iterate.q);
            auto &arm_json = step["arm"];
            for (const auto &point : arm)
            {
                arm_json.push_back(to_vector(point));
            }

            step["curve_point"] = to_vector(target.position);
            step["segment"] = target.segment;
            step["s"] = target.s;
            step["error_norm"] = error.norm();
            step["position_error"] = error.head(3).norm();
            if (curve.pose)
            {
                step["rotation_error"] = error.tail(3).norm();  // weighted: weight * angle
                step["curve_quaternion"] = to_vector(target.rotation);
                step["eef_quaternion"] = to_vector(Eigen::Quaterniond(end_effector(info, iterate.q).rotation()));
            }

            iterations.push_back(step);
        }

        std::ofstream out(path);
        if (not out)
        {
            throw std::runtime_error("cannot write " + path);
        }

        out << dump.dump(1) << "\n";
    }
}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const auto root = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/panda";
        cricket::RobotInfo info(root / "panda_spherized.urdf", root / "panda.srdf", std::string("panda_grasptarget"));
        const auto nq = static_cast<std::size_t>(info.model.nq);

        // ---- Inputs: change these, or pass them as arguments ----
        std::string curve_kind = "r3";
        std::string mode = "qt";
        double t = 0.4;
        double weight = 1.0;
        double curve_scale = 1.0;  // the size of the curve: its S has extents 20 x 12 x 8 cm at scale 1
        double turn = 60.0;  // degrees, between the first and last keyframe

        Eigen::VectorXd q(nq);
        q << 1.489049, -0.146671, -2.0884975, -2.174557, 0.709923, 2.35321, 1.061964;  // default joint values

        bool keep = false;
        bool sweep = false;
        std::string method = "bounded";  // projection solve: bounded, inner, outer or gradient
        std::string dump;                // write a JSON file for tests/view_lerp.py
        std::size_t iterations = 15;
        std::size_t repeat = 1000;  // calls to time, 0 for no timing
        double tolerance = 1e-8;    // squared error at which a projection has converged
        double max_step = 0.0;      // cap on the norm of a step's joint part (rad), 0 for none
        std::string flags = "-O3 -march=native";  // how the generated kernels are compiled
        // ---------------------------------------------------------

        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            const auto equals = argument.find('=');
            const auto key = argument.substr(0, equals);
            const auto value = equals == std::string::npos ? std::string() : argument.substr(equals + 1);

            if (key == "curve")
            {
                curve_kind = value;
            }
            else if (key == "mode")
            {
                mode = value;
            }
            else if (key == "t")
            {
                t = std::stod(value);
            }
            else if (key == "scale")
            {
                curve_scale = std::stod(value);
            }
            else if (key == "weight")
            {
                weight = std::stod(value);
            }
            else if (key == "turn")
            {
                turn = std::stod(value);
            }
            else if (key == "q")
            {
                const auto numbers = parse_numbers(value);
                if (numbers.size() != nq)
                {
                    throw std::runtime_error("q needs " + std::to_string(nq) + " values");
                }

                for (std::size_t j = 0; j < nq; ++j)
                {
                    q[static_cast<Eigen::Index>(j)] = numbers[j];
                }
            }
            else if (key == "method")
            {
                method = value;
            }
            else if (key == "iterations")
            {
                iterations = static_cast<std::size_t>(std::stoul(value));
            }
            else if (key == "dump")
            {
                dump = value;
            }
            else if (key == "sweep")
            {
                sweep = value != "0";
            }
            else if (key == "tolerance")
            {
                tolerance = std::stod(value);
            }
            else if (key == "max_step")
            {
                max_step = std::stod(value);
            }
            else if (key == "flags")
            {
                flags = value;
            }
            else if (key == "repeat")
            {
                repeat = static_cast<std::size_t>(std::stoul(value));
            }
            else if (key == "keep")
            {
                keep = value != "0";
            }
            else
            {
                throw std::runtime_error(
                    "unknown argument `" + argument +
                    "`; arguments are key=value, with keys curve, mode, t, scale, weight, turn, q, method, iterations, dump, sweep, repeat, tolerance, max_step, flags, keep "
                    "(for example: curve=so3r3 t=0.4 dump=lerp.json)");
            }
        }

        if (curve_kind != "r3" and curve_kind != "so3r3")
        {
            throw std::runtime_error("curve is r3 or so3r3");
        }

        if (mode != "qt" and mode != "qs")
        {
            throw std::runtime_error("mode is qt or qs");
        }

        const bool segment_mode = mode == "qs";

        // The curve starts at the end-effector's pose at q, and wanders from there: positions along a gentle S, and for
        // so3r3 the rotation turning about the world z axis
        Curve curve;
        curve.pose = curve_kind == "so3r3";
        curve.weight = weight;

        const auto start = end_effector(info, q);
        for (std::size_t i = 0; i < N; ++i)
        {
            const double s = static_cast<double>(i) / static_cast<double>(N - 1);
            curve.positions.push_back(start.translation() + curve_scale * Eigen::Vector3d(0.20 * s, 0.12 * std::sin(2.0 * pi * s), 0.08 * s));
            if (curve.pose)
            {
                const Eigen::Quaterniond about_z(Eigen::AngleAxisd(turn * pi / 180.0 * s, Eigen::Vector3d::UnitZ()));
                curve.rotations.push_back((about_z * Eigen::Quaterniond(start.rotation())).normalized());
            }
        }

        // consecutive quaternions on the same side, so the interpolation takes the short way round
        for (std::size_t i = 1; i < curve.rotations.size(); ++i)
        {
            if (curve.rotations[i].dot(curve.rotations[i - 1]) < 0.0)
            {
                curve.rotations[i].coeffs() *= -1.0;
            }
        }

        curve.set_taus();

        std::printf("%s curve through %zu keyframes (weight %.3g), %s, parameters tau:\n  ", curve_kind.c_str(), N, weight, mode.c_str());
        for (std::size_t i = 0; i < N; ++i)
        {
            std::printf("%.4f%s", curve.taus[i], i + 1 < N ? " " : "\n");
        }

        const auto target = curve.value(t);
        std::printf("\nt = %.4f is on segment %zu (keyframes %zu to %zu, tau %.4f to %.4f), s = %.4f along it\n", t, target.segment, target.segment, target.segment + 1, curve.taus[target.segment], curve.taus[target.segment + 1], target.s);
        std::printf("  P(t) position: %.4f %.4f %.4f\n", target.position[0], target.position[1], target.position[2]);
        if (curve.pose)
        {
            std::printf("  P(t) rotation (w x y z): %.4f %.4f %.4f %.4f\n", target.rotation.w(), target.rotation.x(), target.rotation.y(), target.rotation.z());
        }

        std::printf("  FK(q) position: %.4f %.4f %.4f\n\n", start.translation()[0], start.translation()[1], start.translation()[2]);

        // The traced kernels, in plain scalar C++ (conditionals as if/else), compiled and loaded
        const auto m = curve.rows();
        const auto error_traced =
            segment_mode ? (curve.pose ? cricket::trace_lerp_so3r3_segment_error<N>(info, "c++_scalar")
                                : cricket::trace_lerp_r3_segment_error<N>(info, "c++_scalar"))
                  : (curve.pose ? cricket::trace_lerp_so3r3_error<N>(info, "c++_scalar")
                                : cricket::trace_lerp_r3_error<N>(info, "c++_scalar"));
        const LoadedKernel error_kernel(error_traced, "error", flags, keep);

        // the segment kernel starts on the segment t is on, with t itself as the parameter
        const auto segment = curve.segment_of(t);
        const auto expected = segment_mode ? reference_segment_output(info, curve, q, t, segment) : reference_output(info, curve, q, t);
        const auto actual = segment_mode ? error_kernel(kernel_inputs<true>(curve, q, t, t)) : error_kernel(kernel_inputs<false>(curve, q, t));

        const auto columns = segment_mode ? nq + 1 : nq;
        print_output("non-traced (pinocchio and host curve, Jacobian by central differences)", expected, columns, m);
        print_output("traced (generated kernel)", actual, columns, m);
        std::printf("max |difference| = %.3e\n\n", max_difference(expected, actual));

        if (sweep)
        {
            // t on every boundary between segments, at both ends and in between, with random configurations near q
            std::mt19937 random(7);
            std::normal_distribution<double> noise(0.0, 0.2);
            std::vector<double> parameters{0.0, 1.0};
            for (std::size_t i = 1; i + 1 < N; ++i)
            {
                parameters.push_back(curve.taus[i]);
                parameters.push_back(0.5 * (curve.taus[i] + curve.taus[i + 1]));
            }

            double worst = 0.0;
            double worst_t = 0.0;
            std::size_t evaluations = 0;
            const auto compare = [&](const Eigen::VectorXd &perturbed, double parameter, double selector)
            {
                // qs: t = parameter on the segment selector is on; qt: both are t
                const auto on = curve.segment_of(selector);
                const double difference = segment_mode
                                              ? max_difference(
                                                    reference_segment_output(info, curve, perturbed, parameter, on),
                                                    error_kernel(kernel_inputs<true>(curve, perturbed, parameter, selector)))
                                              : max_difference(
                                                    reference_output(info, curve, perturbed, parameter),
                                                    error_kernel(kernel_inputs<false>(curve, perturbed, parameter)));
                ++evaluations;
                if (difference > worst)
                {
                    worst = difference;
                    worst_t = parameter;
                }
            };

            const auto perturb = [&]()
            {
                Eigen::VectorXd perturbed = q;
                for (Eigen::Index j = 0; j < perturbed.size(); ++j)
                {
                    perturbed[j] += noise(random);
                }

                return perturbed;
            };

            if (not segment_mode)
            {
                for (const double parameter : parameters)
                {
                    compare(perturb(), parameter, parameter);
                }
            }
            else
            {
                // every segment, picked by its middle, with t at its start, middle and end
                for (std::size_t i = 0; i + 1 < N; ++i)
                {
                    const double selector = 0.5 * (curve.taus[i] + curve.taus[i + 1]);
                    for (const double fraction : {0.0, 0.5, 1.0})
                    {
                        compare(perturb(), curve.taus[i] + fraction * (curve.taus[i + 1] - curve.taus[i]), selector);
                    }
                }
            }

            std::printf(
                "sweep over %zu evaluations (%s): max |difference| = %.3e, at t = %.4f\n\n",
                evaluations,
                segment_mode ? "every segment, t at its start, middle and end" : "every tau, midpoints, 0 and 1",
                worst,
                worst_t);
        }

        // the projection solves
        const bool bounded = method == "bounded";
        const auto solver = [&]() -> cricket::ProjMethod
        {
            if (method == "bounded")
            {
                return cricket::ProjMethod::BoundedLS;
            }

            if (method == "inner")
            {
                return cricket::ProjMethod::InnerLM;
            }

            if (method == "outer")
            {
                return cricket::ProjMethod::OuterLM;
            }

            if (method == "gradient")
            {
                return cricket::ProjMethod::GradDesc;
            }

            throw std::runtime_error("method is bounded, inner, outer or gradient");
        }();

        const LoadedKernel solve_kernel(
            cricket::trace_solve_jacobian(info, "c++_scalar", solver, m, segment_mode ? 1 : 0), "solve", flags, keep);

        if (repeat > 0)
        {
            // The input of the solve at (q, t), as a projection step builds it: J and err from the error kernel, then for the
            // bounded solve the allowed step to every bound (the joints' limits minus q, and for the segment form t's
            // two ends from the kernel's last outputs)
            const auto columns_solved = segment_mode ? nq + 1 : nq;
            std::vector<double> solve_input(actual.begin(), actual.begin() + static_cast<std::ptrdiff_t>(m * columns_solved + m));
            if (bounded)
            {
                for (std::size_t joint = 0; joint < nq; ++joint)
                {
                    solve_input.push_back(info.model.lowerPositionLimit[static_cast<Eigen::Index>(joint)] - q[static_cast<Eigen::Index>(joint)]);
                }

                if (segment_mode)
                {
                    solve_input.push_back(actual[m * columns_solved + m]);
                }

                for (std::size_t joint = 0; joint < nq; ++joint)
                {
                    solve_input.push_back(info.model.upperPositionLimit[static_cast<Eigen::Index>(joint)] - q[static_cast<Eigen::Index>(joint)]);
                }

                if (segment_mode)
                {
                    solve_input.push_back(actual[m * columns_solved + m + 1]);
                }
            }

            const auto error_input = segment_mode ? kernel_inputs<true>(curve, q, t, t) : kernel_inputs<false>(curve, q, t);

            const double reference_time = time_microseconds(repeat, [&] {
                return (segment_mode ? reference_segment_output(info, curve, q, t, segment) : reference_output(info, curve, q, t))[0];
            });
            const double error_time = time_microseconds(repeat, [&] { return error_kernel(error_input)[0]; });
            const double solve_time = time_microseconds(repeat, [&] { return solve_kernel(solve_input)[0]; });

            std::printf("time per call, mean of %zu (generated kernels: scalar C++, %s, one lane):\n", repeat, flags.c_str());
            std::printf("  non-traced reference (pinocchio, host curve, finite differences)  %10.2f us\n", reference_time);
            std::printf("  error kernel (%s)  %10.2f us\n", segment_mode ? "J with the t column, err, t's range" : "J, err", error_time);
            std::printf("  solve (%s, %zu unknowns)  %10.2f us\n", method.c_str(), columns_solved, solve_time);
            std::printf("  one projection step (error kernel + solve)  %10.2f us\n\n", error_time + solve_time);
        }

        // the unbounded solve that shows where the step would put t without the segment's bound (qs only)
        std::unique_ptr<LoadedKernel> free_solve;
        if (segment_mode)
        {
            free_solve = std::make_unique<LoadedKernel>(
                cricket::trace_solve_jacobian(info, "c++_scalar", cricket::ProjMethod::InnerLM, m, 1), "free", flags, keep);
        }

        // start from q moved away from the point, so that the projection has something to do
        Eigen::VectorXd candidate = q;
        std::mt19937 random(1);
        std::normal_distribution<double> noise(0.0, 0.15);
        for (Eigen::Index j = 0; j < candidate.size(); ++j)
        {
            candidate[j] += noise(random);
        }

        const auto project = [&](bool verbose)
        {
            return segment_mode ? project_traced_segment(info, curve, error_kernel, solve_kernel, candidate, t, iterations, method, bounded, tolerance, max_step, verbose, verbose ? free_solve.get() : nullptr)
                                : project_traced(info, curve, error_kernel, solve_kernel, candidate, t, iterations, method, bounded, tolerance, max_step, verbose);
        };

        const auto trail = project(true);

        // from the start to convergence: the same projection, silent, repeated
        {
            const auto steps = trail.size() - 1;
            const auto error = reference_error(info, curve, trail.back().q, trail.back().t, segment_mode ? std::optional<std::size_t>(segment) : std::nullopt);
            const bool converged = error.squaredNorm() <= 10.0 * tolerance;  // the reference error, so a little slack for rounding
            std::printf(
                "projection: %s after %zu iterations, final |err| %.3e%s\n",
                converged ? "converged" : "did NOT converge",
                steps,
                error.norm(),
                segment_mode ? (", t " + std::to_string(trail.back().t)).c_str() : "");

            if (repeat > 0)
            {
                const double total = time_microseconds(repeat, [&] { return static_cast<double>(project(false).size()); });
                std::printf(
                    "  time: %.2f us per projection (mean of %zu, from the start to convergence), %.2f us per iteration\n\n",
                    total,
                    repeat,
                    total / static_cast<double>(std::max<std::size_t>(steps, 1)));
            }
        }

        if (not dump.empty())
        {
            write_dump(dump, info, curve, mode, segment, t, expected, actual, trail);
            std::printf("written to %s: python3 tests/view_lerp.py %s\n", dump.c_str(), dump.c_str());
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}

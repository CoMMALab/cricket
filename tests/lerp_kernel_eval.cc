// Evaluates the lerp constraint kernels two ways on the same inputs, runs a fixed-t projection on the generated code,
// and writes a dump that tests/view_lerp.py shows in viser. The curve is a straight-line interpolation through 20
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
//   lerp_kernel_eval [curve=r3] [t=0.4] [weight=1] [turn=60] [q=a,b,c,...] [method=bounded] [iterations=15]
//                    [dump=out.json] [sweep=1] [keep=1]
//
//   curve      r3 or so3r3
//   t          curve parameter in [0, 1], held fixed in the projection
//   weight     so3r3: the weight of the rotation error against the translation error (m / rad); it also sets how the
//              keyframes' parameters taus are spaced, since a segment's length is sqrt(|dp|^2 + weight^2 theta^2)
//   turn       so3r3: the rotation about the world z axis (degrees) between the first and last keyframe
//   q          joint values, one per joint of the robot
//   method     projection solve: bounded (default), inner, outer or gradient (the four generated solves)
//   iterations projection steps from q onto the point P(t), on the generated kernels
//   dump       write a JSON file that tests/view_lerp.py shows
//   sweep      1 = also compare the two evaluations at many (q, t), among them t on the segment boundaries, 0 and 1
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
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr std::size_t N = 20;
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

        auto value(double t) const -> Value
        {
            t = std::clamp(t, 0.0, 1.0);

            // the first segment whose end is after t; t = 1 stays in the last one
            std::size_t segment = 0;
            while (segment + 2 < positions.size() and t >= taus[segment + 1])
            {
                ++segment;
            }

            const double s = (t - taus[segment]) / (taus[segment + 1] - taus[segment]);
            Value value;
            value.segment = segment;
            value.s = s;
            value.position = positions[segment] + s * (positions[segment + 1] - positions[segment]);
            value.rotation = pose ? rotations[segment].slerp(s, rotations[segment + 1]) : Eigen::Quaterniond::Identity();
            return value;
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
    auto reference_error(const cricket::RobotInfo &info, const Curve &curve, const Eigen::VectorXd &q, double t)
        -> Eigen::VectorXd
    {
        const auto pose = end_effector(info, q);
        const auto target = curve.value(t);

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

    // The flat input vector of the traced kernel (see LerpR3InputLayout and LerpPoseInputLayout in tracing/lerp.hh)
    auto kernel_inputs(const Curve &curve, const Eigen::VectorXd &q, double t) -> std::vector<double>
    {
        const auto nq = static_cast<std::size_t>(q.size());

        std::vector<double> x;
        if (not curve.pose)
        {
            using Layout = cricket::LerpR3InputLayout<N>;
            x.assign(Layout::total(nq), 0.0);
            x[Layout::t_idx(nq)] = t;
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
            using Layout = cricket::LerpPoseInputLayout<N>;
            x.assign(Layout::total(nq), 0.0);
            x[Layout::t_idx(nq)] = t;
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
        LoadedKernel(const cricket::Traced &traced, const std::string &name, bool keep)
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

            const std::string command = std::string(CRICKET_CXX) + " -O1 -shared -fPIC -o " + library.string() + " " +
                                        source.string() + " 2>&1";
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
        std::printf("  d(err)/dq, %zu x %zu:\n", m, nq);
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

        std::printf("\n\n");
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
        bool bounded) -> std::vector<Iterate>
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto m = curve.rows();

        std::printf("projection with t = %.4f held fixed, on the generated kernels (error kernel, then the %s solve)\n", t, method_name.c_str());
        std::printf("  step   |err|            position (m)   rotation (rad)   |gradient|\n");

        std::vector<Iterate> trail{{q0}};
        for (std::size_t iteration = 0; iteration <= iterations; ++iteration)
        {
            const auto &current = trail.back();

            auto solve_input = error_kernel(kernel_inputs(curve, current.q, t));

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

            std::printf(
                "  %4zu   %.3e     %.3e        %.3e         %.3e\n",
                iteration,
                std::sqrt(position_error + rotation_error),
                std::sqrt(position_error),
                std::sqrt(rotation_error),
                std::sqrt(gradient_norm));
            if (iteration == iterations)
            {
                break;
            }

            Iterate next = current;
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                next.q[static_cast<Eigen::Index>(joint)] -= gradient[joint];
            }

            trail.push_back(next);
        }

        std::printf("\n");
        return trail;
    }

    // Everything the viewer (tests/view_lerp.py) draws: the keyframes and their parameters, the sampled curve, the two
    // evaluations, and the projection iterations with the arm at each one.
    void write_dump(
        const std::string &path,
        const cricket::RobotInfo &info,
        const Curve &curve,
        double t,
        const std::vector<double> &expected,
        const std::vector<double> &actual,
        const std::vector<Iterate> &trail)
    {
        nlohmann::json dump;
        dump["kind"] = curve.pose ? "so3r3" : "r3";
        dump["label"] = std::string(curve.pose ? "lerp so3r3" : "lerp r3") + ", t = " + std::to_string(t);
        dump["nq"] = info.model.nq;
        dump["weight"] = curve.weight;
        dump["t_fixed"] = t;
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
            const auto target = curve.value(t);
            const auto error = reference_error(info, curve, iterate.q, t);

            nlohmann::json step;
            step["t"] = t;
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
        double t = 0.4;
        double weight = 1.0;
        double turn = 60.0;  // degrees, between the first and last keyframe

        Eigen::VectorXd q(nq);
        q << 1.489049, -0.146671, -2.0884975, -2.174557, 0.709923, 2.35321, 1.061964;  // default joint values

        bool keep = false;
        bool sweep = false;
        std::string method = "bounded";  // projection solve: bounded, inner, outer or gradient
        std::string dump;                // write a JSON file for tests/view_lerp.py
        std::size_t iterations = 15;
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
            else if (key == "t")
            {
                t = std::stod(value);
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
            else if (key == "keep")
            {
                keep = value != "0";
            }
            else
            {
                throw std::runtime_error(
                    "unknown argument `" + argument +
                    "`; arguments are key=value, with keys curve, t, weight, turn, q, method, iterations, dump, sweep, keep "
                    "(for example: curve=so3r3 t=0.4 dump=lerp.json)");
            }
        }

        if (curve_kind != "r3" and curve_kind != "so3r3")
        {
            throw std::runtime_error("curve is r3 or so3r3");
        }

        // The curve starts at the end-effector's pose at q, and wanders from there: positions along a gentle S, and for
        // so3r3 the rotation turning about the world z axis
        Curve curve;
        curve.pose = curve_kind == "so3r3";
        curve.weight = weight;

        const auto start = end_effector(info, q);
        for (std::size_t i = 0; i < N; ++i)
        {
            const double s = static_cast<double>(i) / static_cast<double>(N - 1);
            curve.positions.push_back(start.translation() + Eigen::Vector3d(0.20 * s, 0.12 * std::sin(2.0 * pi * s), 0.08 * s));
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

        std::printf("%s curve through %zu keyframes (weight %.3g), parameters tau:\n  ", curve_kind.c_str(), N, weight);
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
        const auto error_traced = curve.pose ? cricket::trace_lerp_so3r3_error<N>(info, "c++_scalar")
                                             : cricket::trace_lerp_r3_error<N>(info, "c++_scalar");
        const LoadedKernel error_kernel(error_traced, "error", keep);

        const auto expected = reference_output(info, curve, q, t);
        const auto actual = error_kernel(kernel_inputs(curve, q, t));

        print_output("non-traced (pinocchio and host curve, Jacobian by central differences)", expected, nq, m);
        print_output("traced (generated kernel)", actual, nq, m);
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
            for (const double parameter : parameters)
            {
                Eigen::VectorXd perturbed = q;
                for (Eigen::Index j = 0; j < perturbed.size(); ++j)
                {
                    perturbed[j] += noise(random);
                }

                const double difference = max_difference(
                    reference_output(info, curve, perturbed, parameter), error_kernel(kernel_inputs(curve, perturbed, parameter)));
                if (difference > worst)
                {
                    worst = difference;
                    worst_t = parameter;
                }
            }

            std::printf("sweep over %zu parameters (every tau, midpoints, 0 and 1): max |difference| = %.3e, at t = %.4f\n\n", parameters.size(), worst, worst_t);
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

        const LoadedKernel solve_kernel(cricket::trace_solve_jacobian(info, "c++_scalar", solver, m), "solve", keep);

        // start from q moved away from the point, so that the projection has something to do
        Eigen::VectorXd candidate = q;
        std::mt19937 random(1);
        std::normal_distribution<double> noise(0.0, 0.15);
        for (Eigen::Index j = 0; j < candidate.size(); ++j)
        {
            candidate[j] += noise(random);
        }

        const auto trail = project_traced(info, curve, error_kernel, solve_kernel, candidate, t, iterations, method, bounded);

        if (not dump.empty())
        {
            write_dump(dump, info, curve, t, expected, actual, trail);
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

// Evaluates the spline constraint two ways on the same inputs and prints both, so they can be compared:
//
//   1. non-traced: forward kinematics and the Jacobian from pinocchio, the curve point and its derivative
//      from the host spline (src/tracing/spline.hh). Plain code, no CppAD.
//   2. traced: the generated kernels (trace_spline_error and trace_solve_spline, in the "c++_scalar" language, plain C++ where each
//      conditional is an if/else). The code is written to a temporary file, compiled into a shared library
//      with the compiler this was built with, loaded, and called.
//
// Both give  y = [ d(err)/d(q, t) (3 x (nq + 1), row-major, t last), err (3) ]  with  err = FK(q) - P(t).
//
//   spline_kernel_eval [t=0.4] [q=a,b,c,...] [waypoints=file.txt] [keep=1] [dump=out.json] [iterations=15]
//
//   t          curve parameter in [0, 1]
//   q          joint values, one per joint of the robot
//   waypoints  text file with 20 lines of "x y z"; the spline is fitted through them
//   trajectory JSON from evaluate_mbm.py --save_eef; the first 20 end-effector points of its first trajectory
//              are the waypoints (default: tests/eef_bookshelf_small.json; a helix if that file is missing)
//   keep       keep the generated source and library (their paths are printed)
//   dump       write a JSON file (spline, waypoints, arm, projection steps) that tests/view_spline.py shows in viser
//   iterations projection steps from (q, t) onto the curve, run on the generated error kernel and the generated
//              bounded least-squares solve (t kept in [0, 1]); the iterates are printed and go into the dump
//
// To plug in other numbers, change the arguments, or edit the `Inputs` section in main().

#include <cricket/codegen.hh>

#include "tracing/spline.hh"

#include <nlohmann/json.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr std::size_t N = 20;
    using Layout = cricket::SplineInputLayout<N>;

    // y = [d(err)/d(q, t) (3 x (nq + 1), row-major), err (3)], from pinocchio and the host spline
    auto reference_output(
        const cricket::RobotInfo &info,
        const Eigen::VectorXd &q,
        double t,
        const cricket::CubicBSpline &spline) -> std::vector<double>
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);
        const auto frame = info.end_effector_indexes.at(0);

        pinocchio::Data data(info.model);
        pinocchio::forwardKinematics(info.model, data, q);
        pinocchio::updateFramePlacements(info.model, data);
        pinocchio::computeJointJacobians(info.model, data, q);

        Eigen::MatrixXd frame_jacobian = Eigen::MatrixXd::Zero(6, info.model.nv);
        pinocchio::getFrameJacobian(info.model, data, frame, pinocchio::LOCAL_WORLD_ALIGNED, frame_jacobian);

        const Eigen::Vector3d position = data.oMf[frame].translation();
        const Eigen::Vector3d error = position - spline.value(t);
        const Eigen::Vector3d d_error_dt = -spline.derivative(t);

        std::vector<double> y;
        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                y.push_back(frame_jacobian(row, joint));
            }

            y.push_back(d_error_dt[row]);
        }

        for (std::size_t row = 0; row < 3; ++row)
        {
            y.push_back(error[row]);
        }

        return y;
    }

    // The flat input vector of the traced kernel: q, t, control points, knots
    auto kernel_inputs(const Eigen::VectorXd &q, double t, const cricket::CubicBSpline &spline) -> std::vector<double>
    {
        const auto nq = static_cast<std::size_t>(q.size());

        std::vector<double> x(Layout::total(nq), 0.0);
        for (std::size_t joint = 0; joint < nq; ++joint)
        {
            x[joint] = q[joint];
        }

        x[Layout::t_idx(nq)] = t;
        for (std::size_t point = 0; point < N; ++point)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                x[Layout::control_point_idx(nq, point, axis)] = spline.control_points[point][axis];
            }
        }

        for (std::size_t i = 0; i < Layout::knots; ++i)
        {
            x[Layout::knot_idx(nq, i)] = spline.knots[i];
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
                ("spline_kernel_eval_" + std::to_string(::getpid()) + "_" + name))
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

            const std::string command = std::string(CRICKET_CXX) + " -O1 -shared -fPIC -o " + library.string() +
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

    void print_output(const char *label, const std::vector<double> &y, std::size_t nq)
    {
        std::printf("%s\n", label);
        std::printf("  d(err)/d(q, t), 3 x %zu (last column is t):\n", nq + 1);
        for (std::size_t row = 0; row < 3; ++row)
        {
            std::printf("   ");
            for (std::size_t col = 0; col <= nq; ++col)
            {
                std::printf(" %11.6f", y[row * (nq + 1) + col]);
            }

            std::printf("\n");
        }

        std::printf("  err = FK(q) - P(t):  %11.6f %11.6f %11.6f\n\n", y[3 * (nq + 1)], y[3 * (nq + 1) + 1], y[3 * (nq + 1) + 2]);
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

    auto read_waypoints(const std::string &path) -> std::vector<Eigen::Vector3d>
    {
        std::ifstream in(path);
        if (not in)
        {
            throw std::runtime_error("cannot open waypoints file " + path);
        }

        std::vector<Eigen::Vector3d> points;
        double x, y, z;
        while (in >> x >> y >> z)
        {
            points.emplace_back(x, y, z);
        }

        return points;
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

    struct Iterate
    {
        Eigen::VectorXd q;
        double t;
    };

    // Damped Gauss-Newton steps on (q, t) that drive err = FK(q) - P(t) to zero, using the Jacobian of the
    // non-traced evaluation: the kind of step the projection takes. Joint limits are ignored. The first
    // entry is the start.
    auto project(
        const cricket::RobotInfo &info,
        const Eigen::VectorXd &q0,
        double t0,
        const cricket::CubicBSpline &spline,
        std::size_t iterations) -> std::vector<Iterate>
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);

        std::vector<Iterate> trail{{q0, t0}};
        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            const auto y = reference_output(info, trail.back().q, trail.back().t, spline);

            Eigen::MatrixXd jacobian(3, nq + 1);
            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t col = 0; col <= nq; ++col)
                {
                    jacobian(row, col) = y[row * (nq + 1) + col];
                }
            }

            const Eigen::Vector3d error(y[3 * (nq + 1)], y[3 * (nq + 1) + 1], y[3 * (nq + 1) + 2]);
            const Eigen::Matrix3d gram = jacobian * jacobian.transpose() + 1e-6 * Eigen::Matrix3d::Identity();
            const Eigen::VectorXd step = -jacobian.transpose() * gram.ldlt().solve(error);

            Iterate next = trail.back();
            next.q += step.head(static_cast<Eigen::Index>(nq));
            next.t = std::clamp(next.t + step[static_cast<Eigen::Index>(nq)], 0.0, 1.0);
            trail.push_back(next);
        }

        return trail;
    }

    // The same projection as project(), but on the generated code: each step runs the error kernel (J and err at
    // (q, t)), appends the allowed step to every bound (limit - x; t is bounded to [0, 1]), runs the bounded
    // least-squares solve, and applies  x <- x - gradient.  Prints one row per iterate.
    auto project_traced(
        const cricket::RobotInfo &info,
        const LoadedKernel &error_kernel,
        const LoadedKernel &solve_kernel,
        const Eigen::VectorXd &q0,
        double t0,
        const cricket::CubicBSpline &spline,
        std::size_t iterations) -> std::vector<Iterate>
    {
        const auto nq = static_cast<std::size_t>(info.model.nq);
        const std::size_t n = nq + 1;  // the unknowns: the joints and t

        std::printf("projection on the generated kernels (error kernel, then the bounded solve)\n");
        std::printf("  step   |err| (m)          t   |gradient|\n");

        std::vector<Iterate> trail{{q0, t0}};
        for (std::size_t iteration = 0; iteration <= iterations; ++iteration)
        {
            const auto &current = trail.back();

            // stage 1: [J (3 x n, row-major), err (3)], which is also the start of the solve's input
            auto solve_input = error_kernel(kernel_inputs(current.q, current.t, spline));
            const double error_norm =
                Eigen::Vector3d(solve_input[3 * n], solve_input[3 * n + 1], solve_input[3 * n + 2]).norm();

            // the allowed step to the lower bounds, then to the upper bounds
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                solve_input.push_back(info.model.lowerPositionLimit[static_cast<Eigen::Index>(joint)] - current.q[joint]);
            }

            solve_input.push_back(0.0 - current.t);
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                solve_input.push_back(info.model.upperPositionLimit[static_cast<Eigen::Index>(joint)] - current.q[joint]);
            }

            solve_input.push_back(1.0 - current.t);

            // stage 2: the step (nq joints, then t)
            const auto gradient = solve_kernel(solve_input);
            double gradient_norm = 0.0;
            for (const double g : gradient)
            {
                gradient_norm += g * g;
            }

            std::printf("  %4zu   %.6f   %.5f   %.3e\n", iteration, error_norm, current.t, std::sqrt(gradient_norm));
            if (iteration == iterations)
            {
                break;
            }

            Iterate next = current;
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                next.q[joint] -= gradient[joint];
            }

            next.t = std::clamp(current.t - gradient[nq], 0.0, 1.0);
            trail.push_back(next);
        }

        std::printf("\n");
        return trail;
    }

    void write_vector(std::ostream &out, const Eigen::VectorXd &v)
    {
        out << "[";
        for (Eigen::Index i = 0; i < v.size(); ++i)
        {
            out << (i == 0 ? "" : ", ") << v[i];
        }

        out << "]";
    }

    void write_numbers(std::ostream &out, const std::vector<double> &v)
    {
        out << "[";
        for (std::size_t i = 0; i < v.size(); ++i)
        {
            out << (i == 0 ? "" : ", ") << v[i];
        }

        out << "]";
    }

    void write_points(std::ostream &out, const std::vector<Eigen::Vector3d> &points)
    {
        out << "[";
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            out << (i == 0 ? "" : ", ");
            write_vector(out, points[i]);
        }

        out << "]";
    }

    // Everything the viewer (tests/view_spline.py) draws: the waypoints, the spline (control points, knots,
    // sampled curve), the two evaluations, and the projection iterations with the arm at each one.
    void write_dump(
        const std::string &path,
        const cricket::RobotInfo &info,
        const cricket::CubicBSpline &spline,
        const std::vector<Eigen::Vector3d> &waypoints,
        const std::vector<double> &expected,
        const std::vector<double> &actual,
        const std::vector<Iterate> &trail)
    {
        std::ofstream out(path);
        if (not out)
        {
            throw std::runtime_error("cannot write " + path);
        }

        out << std::setprecision(10);

        out << "{\n  \"nq\": " << info.model.nq << ",\n  \"waypoints\": ";
        write_points(out, waypoints);
        out << ",\n  \"control_points\": ";
        write_points(out, spline.control_points);
        out << ",\n  \"knots\": ";
        write_numbers(out, spline.knots);
        out << ",\n  \"parameters\": ";
        write_numbers(out, spline.parameters);

        // the curve, sampled finely, and the points where one cubic piece hands over to the next
        std::vector<double> curve_t;
        std::vector<Eigen::Vector3d> curve;
        for (int i = 0; i <= 400; ++i)
        {
            curve_t.push_back(i / 400.0);
            curve.push_back(spline.value(i / 400.0));
        }

        std::vector<Eigen::Vector3d> knot_points;
        for (std::size_t i = cricket::CubicBSpline::degree; i <= N; ++i)
        {
            knot_points.push_back(spline.value(spline.knots[i]));
        }

        out << ",\n  \"curve_t\": ";
        write_numbers(out, curve_t);
        out << ",\n  \"curve\": ";
        write_points(out, curve);
        out << ",\n  \"knot_points\": ";
        write_points(out, knot_points);

        out << ",\n  \"reference_output\": ";
        write_numbers(out, expected);
        out << ",\n  \"traced_output\": ";
        write_numbers(out, actual);

        out << ",\n  \"iterations\": [";
        for (std::size_t i = 0; i < trail.size(); ++i)
        {
            const auto &iterate = trail[i];
            const auto arm = arm_positions(info, iterate.q);
            const Eigen::Vector3d point = spline.value(iterate.t);

            out << (i == 0 ? "\n    " : ",\n    ") << "{\"t\": " << iterate.t << ", \"q\": ";
            write_vector(out, iterate.q);
            out << ", \"arm\": ";
            write_points(out, arm);
            out << ", \"curve_point\": ";
            write_vector(out, point);
            out << ", \"tangent\": ";
            write_vector(out, spline.derivative(iterate.t));
            out << ", \"error_norm\": " << (arm.back() - point).norm() << "}";
        }

        out << "\n  ]\n}\n";
    }

    // The first trajectory of a file written by vamp/scripts/evaluate_mbm.py --save_eef
    struct Trajectory
    {
        std::string problem;
        int problem_index = 0;
        std::vector<Eigen::Vector3d> positions;       // end-effector, one per configuration
        std::vector<Eigen::VectorXd> configurations;
    };

    auto read_first_trajectory(const std::string &path) -> Trajectory
    {
        std::ifstream in(path);
        if (not in)
        {
            throw std::runtime_error("cannot open trajectory file " + path);
        }

        const auto json = nlohmann::json::parse(in);
        const auto &problems = json.at("trajectories");
        if (problems.empty() or problems.begin().value().empty())
        {
            throw std::runtime_error("no trajectories in " + path);
        }

        Trajectory trajectory;
        trajectory.problem = problems.begin().key();
        const auto &entry = problems.begin().value().at(0);
        trajectory.problem_index = entry.at("problem_index").get<int>();

        for (const auto &position : entry.at("eef_positions"))
        {
            trajectory.positions.emplace_back(position[0].get<double>(), position[1].get<double>(), position[2].get<double>());
        }

        for (const auto &configuration : entry.at("configurations"))
        {
            const auto values = configuration.get<std::vector<double>>();
            trajectory.configurations.push_back(Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Eigen::Index>(values.size())));
        }

        return trajectory;
    }
}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const auto root = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/panda";
        cricket::RobotInfo info(
            root / "panda_spherized.urdf", root / "panda.srdf", std::string("panda_grasptarget"));
        const auto nq = static_cast<std::size_t>(info.model.nq);

        // ---- Inputs: change these, or pass them as arguments ----
        double t = 1.0;

        Eigen::VectorXd q(nq);
        q << 1.489049, -0.146671, -2.884975, -2.174557, 2.709923, 2.35321, 1.061964;  // default joint values

        // N waypoints; the spline is fitted through them (a helix with uneven spacing)
        std::vector<Eigen::Vector3d> waypoints;
        for (std::size_t i = 0; i < N; ++i)
        {
            const double s = static_cast<double>(i) / static_cast<double>(N - 1);
            const double angle = 6.28318530718 * std::pow(s, 1.3);
            waypoints.emplace_back(0.5 * std::cos(angle) + 0.6, 0.5 * std::sin(angle), 0.2 + 0.4 * s);
        }

        // The waypoints come from the first trajectory in this file (its first N end-effector positions), unless
        // waypoints= is given or the file does not exist (then the helix above is used)
        std::string trajectory_path = std::string(CRICKET_SOURCE_DIR) + "/tests/eef_bookshelf_small.json";
        bool waypoints_given = false;

        bool keep = false;
        std::string dump;                // write a JSON file for tests/view_spline.py
        std::size_t iterations = 15;     // projection steps in the dump
        // ---------------------------------------------------------

        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            const auto equals = argument.find('=');
            const auto key = argument.substr(0, equals);
            const auto value = equals == std::string::npos ? std::string() : argument.substr(equals + 1);

            if (key == "t")
            {
                t = std::stod(value);
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
                    q[j] = numbers[j];
                }
            }
            else if (key == "waypoints")
            {
                waypoints = read_waypoints(value);
                waypoints_given = true;
            }
            else if (key == "trajectory")
            {
                trajectory_path = value;
            }
            else if (key == "keep")
            {
                keep = value != "0";
            }
            else if (key == "dump")
            {
                dump = value;
            }
            else if (key == "iterations")
            {
                iterations = static_cast<std::size_t>(std::stoul(value));
            }
            else
            {
                throw std::runtime_error(
                    "unknown argument `" + argument +
                    "`; arguments are key=value, with keys t, q, waypoints, trajectory, keep, dump, iterations "
                    "(for example: dump=spline.json t=0.4)");
            }
        }

        if (not waypoints_given and std::filesystem::exists(trajectory_path))
        {
            const auto trajectory = read_first_trajectory(trajectory_path);
            if (trajectory.positions.size() < N)
            {
                throw std::runtime_error("the trajectory has fewer than " + std::to_string(N) + " points");
            }

            waypoints.assign(trajectory.positions.begin(), trajectory.positions.begin() + N);
            std::printf(
                "waypoints: the first %zu of %zu end-effector points of %s #%d in %s\n",
                N,
                trajectory.positions.size(),
                trajectory.problem.c_str(),
                trajectory.problem_index,
                trajectory_path.c_str());

            // Is the frame the trajectory was recorded in the one the spline kernel uses? FK of the
            // trajectory's own first configuration should land on its first recorded position.
            if (static_cast<std::size_t>(trajectory.configurations.front().size()) == nq)
            {
                const auto fk = arm_positions(info, trajectory.configurations.front()).back();
                std::printf(
                    "frame check: |FK(first configuration) - recorded first position| = %.2e m\n\n",
                    (fk - trajectory.positions.front()).norm());
            }
        }
        else if (not waypoints_given)
        {
            std::printf("waypoints: the helix (no trajectory file at %s)\n\n", trajectory_path.c_str());
        }

        if (waypoints.size() != N)
        {
            throw std::runtime_error("the spline constraint is traced for " + std::to_string(N) + " waypoints");
        }

        const auto spline = cricket::fit_cubic_bspline(waypoints);

        // The fitted spline: it passes through the waypoints at their parameters, and is defined by the control
        // points (not on the curve, except the first and last) and the knots
        std::printf("spline through %zu waypoints: %zu control points, %zu knots, %zu cubic pieces\n", waypoints.size(),
                    spline.size(), spline.knots.size(), spline.segments());
        std::printf("  %-3s %-26s %-26s %s\n", "i", "waypoint (x y z)", "control point (x y z)", "parameter t_i");
        for (std::size_t i = 0; i < spline.size(); ++i)
        {
            std::printf(
                "  %-3zu %8.4f %8.4f %8.4f      %8.4f %8.4f %8.4f      %.4f\n", i, waypoints[i][0], waypoints[i][1],
                waypoints[i][2], spline.control_points[i][0], spline.control_points[i][1],
                spline.control_points[i][2], spline.parameters[i]);
        }

        std::printf("  knots:");
        for (const double knot : spline.knots)
        {
            std::printf(" %.4f", knot);
        }

        std::printf("\n\n");

        std::printf("t = %g\nq =", t);
        for (std::size_t i = 0; i < nq; ++i)
        {
            std::printf(" %g", q[i]);
        }

        std::printf("\nP(t) = (%g, %g, %g),  dP/dt = (%g, %g, %g)\n\n",
                    spline.value(t)[0], spline.value(t)[1], spline.value(t)[2],
                    spline.derivative(t)[0], spline.derivative(t)[1], spline.derivative(t)[2]);

        const auto expected = reference_output(info, q, t, spline);
        print_output("non-traced (pinocchio + host spline)", expected, nq);

        const auto traced = cricket::trace_spline_error<N>(info, "c++_scalar");
        const LoadedKernel error_kernel(traced, "error", keep);
        const auto actual = error_kernel(kernel_inputs(q, t, spline));
        print_output("traced kernel (generated code, compiled and loaded)", actual, nq);

        double worst = 0.0;
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            worst = std::max(worst, std::abs(expected[i] - actual[i]));
        }

        std::printf("largest difference between the two: %.3e\n\n", worst);

        // The projection loop on the generated code: the error kernel and the bounded solve with t in [0, 1]
        if (iterations > 0)
        {
            const auto solve = cricket::trace_solve_spline(info, "c++_scalar", cricket::ProjMethod::BoundedLS);
            const LoadedKernel solve_kernel(solve, "solve", keep);
            const auto trail = project_traced(info, error_kernel, solve_kernel, q, t, spline, iterations);

            // ... against the stand-in (plain Gauss-Newton with pinocchio's Jacobian, no bounds)
            const auto reference_trail = project(info, q, t, spline, iterations);
            double dq = 0.0;
            for (std::size_t joint = 0; joint < nq; ++joint)
            {
                dq = std::max(dq, std::abs(trail.back().q[joint] - reference_trail.back().q[joint]));
            }

            std::printf(
                "final state, generated kernels vs the Gauss-Newton stand-in: max |dq| = %.2e, |dt| = %.2e\n"
                "(they need not match exactly: the traced solve is damped and keeps q and t inside their bounds)\n\n",
                dq,
                std::abs(trail.back().t - reference_trail.back().t));

            if (not dump.empty())
            {
                write_dump(dump, info, spline, waypoints, expected, actual, trail);
                std::printf(
                    "dump written to %s (%zu projection steps on the generated kernels: |err| %.4f -> %.6f); "
                    "view with tests/view_spline.py\n",
                    dump.c_str(),
                    iterations,
                    (arm_positions(info, trail.front().q).back() - spline.value(trail.front().t)).norm(),
                    (arm_positions(info, trail.back().q).back() - spline.value(trail.back().t)).norm());
            }
        }

        return worst < 1e-6 ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
}

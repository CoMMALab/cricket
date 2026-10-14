// Prints the generated spline-constraint kernel (trace_spline_error), to read it and play with it.
//
//   print_spline_kernel [language] [robot] [output file]
//
//   language  "c++" (default, SIMD blends as vamp uses them), or "c++_scalar" (plain scalar C++ with if/else,
//             easier to read); the kernel needs conditionals, so "rust" is rejected
//   robot     only "ur5" is wired up here
//   output    write the code to this file instead of stdout
//
// The summary (inputs, outputs, number of temporaries) goes to stderr, the code to stdout or the file.

#include <cricket/codegen.hh>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    const std::string language = argc > 1 ? argv[1] : "c++";
    const std::string robot = argc > 2 ? argv[2] : "ur5";
    const std::string output = argc > 3 ? argv[3] : "";

    if (robot != "ur5")
    {
        std::fprintf(stderr, "only the ur5 is wired up in this program\n");
        return 1;
    }

    constexpr std::size_t control_points = 20;

    const auto root = std::filesystem::path(CRICKET_SOURCE_DIR) / "resources/ur5";
    cricket::RobotInfo info(
        root / "ur5_spherized_no_offset.urdf", root / "ur5.srdf", std::string("robotiq_85_base_link"));

    const auto traced = cricket::trace_spline_error<control_points>(info, language);

    const auto nq = static_cast<std::size_t>(info.model.nq);
    std::fprintf(
        stderr,
        "spline kernel, %zu control points, %zu joints\n"
        "  inputs:  %zu = q (%zu) + t (1) + control points (%zu) + knots (%zu)\n"
        "  outputs: %zu = d(err)/d(q, t) (3 x %zu, row-major, t last) + err (3)\n"
        "  temporaries: %zu, code: %zu bytes, %zu lines\n",
        control_points,
        nq,
        nq + 1 + 3 * control_points + control_points + 4,
        nq,
        3 * control_points,
        control_points + 4,
        traced.outputs,
        nq + 1,
        traced.temp_variables,
        traced.code.size(),
        static_cast<std::size_t>(std::count(traced.code.begin(), traced.code.end(), '\n')));

    if (output.empty())
    {
        std::cout << traced.code;
    }
    else
    {
        std::ofstream(output) << traced.code;
        std::fprintf(stderr, "written to %s\n", output.c_str());
    }

    return 0;
}

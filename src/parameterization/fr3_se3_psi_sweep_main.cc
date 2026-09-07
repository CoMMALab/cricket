// Standalone helper for FR3/Panda's single-arm SE3+psi parameterized IK (see
// fr3_parameterization.hh / fr3_parameterization_gen.hh, wired into ParameterizedSpace for
// "param_kind": "fr3_se3" in fk_template.hh). Given a fixed end-effector pose and a fixed smm
// (case6_sel, case1_sel), this sweeps `psi` (FR3's joint 7 angle -- see
// Fr3SE3Parameterization's header comment for why it's still called `psi`) across its actual
// joint limits at a fixed step and prints the first value for which
// panda_ik_split_nobranch::SolveArmBranchTaped reports `reach_violation <= 0` (a genuine,
// in-limits IK solution -- see that function's header for what `reach_violation` means).
//
// No CLI args -- edit the constants below directly and rebuild (`ninja fr3_se3_psi_sweep`).
//
// Runs entirely in plain `double` -- no CppAD/CppADCodeGen trace, no pinocchio model, no JSON
// recipe -- since Fr3SE3Parameterization<T>/SolveArmBranchTaped<T> already compile for T =
// double (that's exactly what lets the double-only CondExp/IKabs/etc. overloads in
// fr3_parameterization.hh coexist with the CppAD-traced path -- see that file's header).
#include "fr3_parameterization.hh"

#include <fmt/format.h>

#include <array>
#include <optional>

int main()
{
    // --- Edit these ---------------------------------------------------------------------
    // pose = (x, y, z, qx, qy, qz, qw)
    const std::array<double, 7> pose = { 0.3868703544139862, -0.6284154653549194, 0.1560778248310089,
0.0F, -1.0F, 0.0F, 0.0F};

    const double case6_sel = 0.0;  // smm[0], expected 0 or 1
    const double case1_sel = 0.0;  // smm[1], expected 0 or 1
    // smm[2] is unused by FR3's IK -- see Fr3SE3Parameterization's header comment.

    const double resolution = 0.1;  // psi sweep step, radians

    // FR3 joint 7's own limits, exactly the q_min[6]/q_max[6] SolveArmBranchTaped folds into
    // reach_violation -- see fr3_parameterization.hh. Same default range
    // trace_map_to_se3(..., psi_dof_index) samples psi from for the traced/codegen path.
    const double psi_lower = -2.8973;
    const double psi_upper = 2.8973;
    // -------------------------------------------------------------------------------------

    fmt::print(
        "pose: t=({}, {}, {}) q=({}, {}, {}, {}), smm=({}, {}, unused), "
        "psi in [{}, {}] step {}\n",
        pose[0],
        pose[1],
        pose[2],
        pose[3],
        pose[4],
        pose[5],
        pose[6],
        case6_sel,
        case1_sel,
        psi_lower,
        psi_upper,
        resolution);

    Eigen::VectorXd ad_inp(11);
    ad_inp << pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6],
        /* psi placeholder */ 0.0, case6_sel, case1_sel, /* unused */ 0.0;

    std::optional<double> first_valid_psi;
    for (double psi = psi_lower; psi <= psi_upper + 1e-12; psi += resolution)
    {
        ad_inp[7] = psi;
        auto result_ik = Fr3SE3Parameterization<double>(ad_inp);

        const bool valid = result_ik.reach_violation <= 1e-9;
        fmt::print(
            "  psi={:.6f} reach_violation={:.9f} {}\n",
            psi,
            result_ik.reach_violation,
            valid ? "VALID" : "invalid");

        if (valid and not first_valid_psi)
        {
            first_valid_psi = psi;
            fmt::print(
                "\nFirst valid psi = {:.6f} (reach_violation={:.9f})\n"
                "  q = ({}, {}, {}, {}, {}, {}, {})\n",
                psi,
                result_ik.reach_violation,
                result_ik.q[0],
                result_ik.q[1],
                result_ik.q[2],
                result_ik.q[3],
                result_ik.q[4],
                result_ik.q[5],
                result_ik.q[6]);
            break;
        }
    }

    if (not first_valid_psi)
    {
        fmt::print("\nNo valid psi found in [{}, {}] at resolution {}\n", psi_lower, psi_upper, resolution);
        return 1;
    }

    return 0;
}

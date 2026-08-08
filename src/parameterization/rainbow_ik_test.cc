// Smoke test for RainbowLeftArmParameterization / RainbowRightArmParameterization
// (rainbow_arm_parameterization.hh): feed in random end-effector poses for
// both arms, across all 8 GCP branches, and print the resulting joint
// configuration + reachability info. Unlike iiwa_ik_test.cc's psi
// optimization tests, the free joint angle (j15_free / j24_free) here is
// just drawn randomly each trial -- there is no gradient search over it,
// since (unlike iiwa's psi) it doesn't gate feasibility: reach_violation
// only depends on the pose and the GCP branch, not on the free joint angle.
#include "rainbow_arm_parameterization.hh"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

void print_result(const std::string &label, const RainbowArmParamResult<double> &result)
{
    std::cout << label << ": q = [";
    for (Eigen::Index i = 0; i < result.q.size(); ++i)
    {
        std::cout << result.q[i] << (i + 1 < result.q.size() ? ", " : "");
    }
    std::cout << "], unclipped = [";
    for (Eigen::Index i = 0; i < result.unclipped.size(); ++i)
    {
        std::cout << result.unclipped[i] << (i + 1 < result.unclipped.size() ? ", " : "");
    }
    std::cout << "], reach_violation = " << result.reach_violation
               << (result.reach_violation > 0.0 ? "  (INFEASIBLE)" : "") << '\n';
}

struct GcpBranch
{
    double elbow_sel;
    double shoulder_sel;
    double wrist_sel;
    const char *name;
};

// All 8 combinations of (elbow_sel, shoulder_sel, wrist_sel), named to match
// the SolveLeftArm_W{p,m}_E{p,m}_S{p,m} / SolveRightArm_W{p,m}_E{p,m}_S{p,m}
// entry points in rainbow_{left,right}_arm_ik_split_nobranch.cpp.
const std::vector<GcpBranch> kGcpBranches = {
    {0.0, 0.0, 1.0, "Wp_Ep_Sp"},
    {0.0, 1.0, 1.0, "Wp_Ep_Sm"},
    {1.0, 0.0, 1.0, "Wp_Em_Sp"},
    {1.0, 1.0, 1.0, "Wp_Em_Sm"},
    {0.0, 0.0, -1.0, "Wm_Ep_Sp"},
    {0.0, 1.0, -1.0, "Wm_Ep_Sm"},
    {1.0, 0.0, -1.0, "Wm_Em_Sp"},
    {1.0, 1.0, -1.0, "Wm_Em_Sm"},
};

// Random end-effector pose (position in a box, uniformly random orientation)
// plus a random free joint angle. Indices [8:11) (the GCP selectors) are
// left unset here -- callers overwrite them per branch.
Eigen::VectorXd random_pose(std::mt19937 &rng)
{
    std::uniform_real_distribution<double> pos(-0.5, 0.5);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> two_pi(0.0, 2.0 * M_PI);

    Eigen::VectorXd ad_inp(11);
    ad_inp[0] = pos(rng);
    ad_inp[1] = pos(rng);
    ad_inp[2] = pos(rng);

    // Uniform random unit quaternion (Shoemake's method).
    const double u1 = unit(rng);
    const double u2 = two_pi(rng);
    const double u3 = two_pi(rng);
    ad_inp[3] = std::sqrt(1.0 - u1) * std::sin(u2);
    ad_inp[4] = std::sqrt(1.0 - u1) * std::cos(u2);
    ad_inp[5] = std::sqrt(u1) * std::sin(u3);
    ad_inp[6] = std::sqrt(u1) * std::cos(u3);

    ad_inp[7] = two_pi(rng);  // free joint angle (j15_free / j24_free)
    return ad_inp;
}

}  // namespace

int main()
{
    constexpr int kNumTrials = 1;
    std::mt19937 rng(std::random_device{}());

    for (int trial = 0; trial < kNumTrials; ++trial)
    {
        std::cout << "\n=== trial " << trial << " ===\n";

        // Eigen::VectorXd left_inp = random_pose(rng);
        // Eigen::VectorXd right_inp = random_pose(rng);

        // hardcoded pose for testing
        Eigen::VectorXd left_inp(11);
        left_inp << -0.6610796538183245, 0.15975837563311854, 0.15180355784364985, 0.12019512952277979, 0.7935748774528283, 0.18244358494379576, 0.5678964544946438, 0.0, 1, 1, 1;
        Eigen::VectorXd right_inp(11);
        right_inp << -0.5292091877761231, 0.18475614592708484, 0.03636226314465629, 0.20826067138009144, 0.7207931282753334, -0.26814259287247194, 0.6043048145389875, 0.0, 1, 1, 1;

        std::cout << "left pose  = [" << left_inp.head<7>().transpose() << "]\n";
        std::cout << "right pose = [" << right_inp.head<7>().transpose() << "]\n";

        for (const auto &branch : kGcpBranches)
        {
            left_inp[8] = branch.elbow_sel;
            left_inp[9] = branch.shoulder_sel;
            left_inp[10] = branch.wrist_sel;
            right_inp[8] = branch.elbow_sel;
            right_inp[9] = branch.shoulder_sel;
            right_inp[10] = branch.wrist_sel;

            const auto left_result = RainbowLeftArmParameterization<double, Eigen::VectorXd>(left_inp);
            // print_result(std::string("left  ") + branch.name, left_result);
            print_result(std::string(""), left_result);

            // const auto right_result = RainbowRightArmParameterization<double, Eigen::VectorXd>(right_inp);
            // print_result(std::string("right ") + branch.name, right_result);
        }
        std::cout << "--------------------------------\n";
        for (const auto &branch : kGcpBranches)
        {
            left_inp[8] = branch.elbow_sel;
            left_inp[9] = branch.shoulder_sel;
            left_inp[10] = branch.wrist_sel;
            right_inp[8] = branch.elbow_sel;
            right_inp[9] = branch.shoulder_sel;
            right_inp[10] = branch.wrist_sel;

            // const auto left_result = RainbowLeftArmParameterization<double, Eigen::VectorXd>(left_inp);
            // // print_result(std::string("left  ") + branch.name, left_result);
            // print_result(std::string(""), left_result);

            const auto right_result = RainbowRightArmParameterization<double, Eigen::VectorXd>(right_inp);
            // print_result(std::string("right ") + branch.name, right_result);
            print_result(std::string(""), right_result);
        }

    }

    return 0;
}

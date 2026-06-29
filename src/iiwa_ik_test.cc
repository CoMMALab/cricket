#include "iiwa_parameterization.hh"
#include <Eigen/Dense>
#include <iostream>


int main()
{
    Eigen::VectorXd ad_inp(8 + 1 + 1 + 1 + 7);  // 8 for q_and_psi, 4 for shoulder/elbow/wrist up/down and grasping mode
    ad_inp << -0.5997312520566763, 1.489780849654964, -1.4739679827359913, 1.2905366081785483, -0.04421061906813227, -0.8793712572715165, -1.1603461715511334, 1.45, 1.0, 1.0, -1.0, 0.0, 0.0, 0.6, 0.927184, -0.374607, 0.0, 0.0;

    const auto y = IiwaBimanualParameterization<double, Eigen::VectorXd>(ad_inp);

    for (std::size_t i = 0; i < y.size(); ++i)
    {
        std::cout << "y[" << i << "] = " << y[i] << '\n';
    }

    return 0;
}
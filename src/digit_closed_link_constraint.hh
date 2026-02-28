#include "housekeeping.hh"
#include "robot_info.hh"
#include "tracer_utils.hh"

bool isLoopPlanar(
                ADModel & model,
                ADData & data,
                const std::vector<std::string> &frame_names,
                double tol = 1e-6)
{

    // 2. Get frame positions
    std::vector<Eigen::Vector3<ADCG>> points;
    for (const auto &name : frame_names) {
        const auto &frame_id = model.getFrameId(name, BODY);
        std::cout << "Frame " << name << " is at position " << data.oMf[frame_id].translation().transpose() << "\n";
        points.push_back(data.oMf[frame_id].translation());
    }

    if (points.size() < 3) {
        std::cerr << "Need at least 3 points to define a plane\n";
        return false;
    }

    // 3. Define plane using first 3 points
    Eigen::Vector3<ADCG> A = points[0];
    Eigen::Vector3<ADCG> B = points[1];
    Eigen::Vector3<ADCG> C = points[2];
    Eigen::Vector3<ADCG> n = (B - A).cross(C - A);  // plane normal
    ADCG n_norm = n.norm();
    std::cout << "Plane normal: " << n.transpose() << n_norm << "\n";
    if (n_norm < 1e-12) return false; // degenerate points
    n.normalize();

    // 4. Check distance of all points to plane
    for (size_t i = 3; i < points.size(); ++i) {
        ADCG dist = abs((points[i] - A).dot(n));
        std::cout << "Point " << i << " is " << dist << " away from plane\n";
        if (dist > tol) {
            std::cout << "Point " << i << " is " << dist << " away from plane\n";
            return false; // not planar
        }
    }

    return true; // all points lie on the plane
}

ADCG distance_between_frames(
    ADData & data,
    pinocchio::FrameIndex frame_l1_start,
    pinocchio::FrameIndex frame_l1_end)
{

    const Eigen::Vector3<ADCG> & A =
        data.oMf[frame_l1_start].translation();
    const Eigen::Vector3<ADCG> & B =
        data.oMf[frame_l1_end].translation();

    ADCG distance = (B - A).norm();
    std::cout << "Rod length : " << distance - ADCG(0.503) << std::endl;

    return distance  - ADCG(0.503);
}

auto trace_closed_link_system(const RobotInfo &info) -> Traced
{
    auto nq = info.model.nq;

    ADModel ad_model = info.model.cast<ADCG>();
    ADData ad_data(ad_model);

    const size_t num_inp = nq;

    std::array<double, 30> reference_q = {0.00438741,    0.00952297,    -0.149783,    -0.00444245,    0.00167457,    -0.0013295,
    0.366443,    -0.000246321,    0.094638,    -0.102846, -0.0161417,     0.139485,   -0.087064,    -0.00352648,
    -0.102535,    0.843245,    -0.010127,    0.450957,
    -0.360956,    0.00656994,    -0.0956402,    0.0981786, 0.0160675,    -0.134632,    0.0859524,    0.0232191,
    0.103937,     -0.891019, 0.00527577,  -0.369154};

    std::array<double, 30> reference_q_squat = {    0.00352543,    0.0160413,    -0.48349,    -0.00927372,    0.000426254,    -0.00362023,
    0.372489,    0.007467,    -0.226769,    -0.850073, -0.0214815,    0.911696,    -0.430518,    0.0120371,
    -0.110171,    0.82905,    0.0310924,    0.456459,
    -0.342646,    0.0112654,    0.228212,    0.842371, 0.0213345,    -0.903408,   0.431292,    0.0528911,
    0.102618,    -0.891391, 0.00519742,    -0.369172,};

    ADVectorXs ad_inp(num_inp);  // 3 4x4 matrices
    for (auto i = 0U; i < num_inp; ++i)
    {
        ad_inp[i] = ADCG(reference_q[i]);
        // ad_inp[i] = ADCG(0.0);
    }


    Independent(ad_inp);

    std::size_t n_out = 2;
    ADVectorXs data(n_out);

    // First copy over configs and run FK
    ADVectorXs ad_q(nq);

    for (auto i = 0U; i < nq; i++)
    {
        ad_q[i] = ad_inp[i];  // This is the first 7 vars for nq
    }

    forwardKinematics(ad_model, ad_data, ad_q);
    updateFramePlacements(ad_model, ad_data);

    data[0] = distance_between_frames(ad_data, ad_model.getFrameId("left_l1_start", BODY), ad_model.getFrameId("left_l1_end", BODY));
    data[1] = distance_between_frames(ad_data, ad_model.getFrameId("right_l1_start", BODY), ad_model.getFrameId("right_l1_end", BODY));


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

    std::move(error_vec.begin(), error_vec.end(), std::back_inserter(jac_e_q));

    LanguageCCustom<double> langC("double");
    LangCDefaultVariableNameGenerator<double> nameGen;

    std::ostringstream function_code;
    handler.generateCode(function_code, langC, jac_e_q, nameGen);

    return Traced{function_code.str(), handler.getTemporaryVariableCount(), jac_e_q.size()};

}

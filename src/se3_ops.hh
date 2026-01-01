#include <stdlib.h>
#include <math.h>
#include <iostream>
constexpr double EPS = 1e-7;

template <typename scalar_type>
inline scalar_type sinc_smooth(scalar_type x)
{
    scalar_type x2 = x * x;

    scalar_type taylor =
        1.0 - x2 / 6.0 * (1.0 - x2 / 20.0);

    scalar_type full = sin(x) / x;
    scalar_type w = x2 / (x2 + EPS);

    return (1.0 - w) * taylor + w * full;
}



template <typename Matrix, typename scalar_type>
Matrix so3_log_smooth(Matrix const& R) {
    scalar_type cos_theta = (R.trace() - 1) / 2;

    scalar_type sin_theta2 = 1.0 - cos_theta * cos_theta;
    scalar_type w = sin_theta2 / (sin_theta2 + EPS); // smooth weight

    // vee = R - R^T
    Matrix vee = Matrix::Zero(3, 1);
    vee(0, 0) = R(2, 1) - R(1, 2);
    vee(1, 0) = R(0, 2) - R(2, 0);
    vee(2, 0) = R(1, 0) - R(0, 1);

    scalar_type theta = acos(cos_theta);

    // Small-angle Taylor approx
    auto taylor = 0.5 * vee;

    // Full formula with sinc_smooth
    scalar_type factor = 0.5 / (sinc_smooth(theta) + EPS); // factor = theta / (2 sin(theta))
    auto full = vee * factor;

    // Smooth blend between Taylor and full
    return (1.0 - w) * taylor + w * full;

}


template <typename Matrix, typename scalar_type>
Matrix so3_log(Matrix const& R) {
    scalar_type theta = acos((R.trace() - 1) / 2);
    Matrix w = Matrix::Zero(3, 1);
    w(0, 0) = R(2, 1) - R(1, 2);
    w(1, 0) = R(0, 2) - R(2, 0);
    w(2, 0) = R(1, 0) - R(0, 1);

    // auto texpand = 1 + (1 / 6) * theta^2 + (7 / 360) * theta^4;
    auto full = theta / (sin(theta) + 1e-10);

    auto logR_true = (w * full) / 2.0;
    // auto logR_approx = w * texpand;

    // Matrix logR = CondExpLt(abs(theta), 1e-6, logR_approx, logR_true);
    return logR_true;
}

// template <typename scalar_type>
// inline scalar_type coscc_smooth(scalar_type x)
// {
//     scalar_type x2 = x * x;

//     scalar_type taylor =
//         (1.0 / 12.0) *
//         (1.0 + x2 / 60.0 * (1.0 + x2 / 42.0));

//     scalar_type costerm = 2.0 * (1.0 - cos(x)) + 1e-8;
//     scalar_type full = (1.0 - x * sin(x) / costerm) / x2;

//     scalar_type eps2 = 1e-8;
//     scalar_type w = x2 / (x2 + eps2);

//     return (1.0 - w) * taylor + w * full;
// }


// template <typename Matrix, typename scalar_type>
// Matrix se3_log(Matrix const& T) {
//     Matrix logR = so3_log(T.block(0, 0, 3, 3));
//     Matrix logV = Matrix::Zero(3, 1);
//     scalar_type theta = logR.norm();


//     // computing coscc
//     //
//     // auto theta2 = theta * theta;
//     // auto texpand = 1 / 12 * (1 + theta2 / 60 * (1 + theta2 / 42 * (1 + theta2 / 40)));
//     // auto costerm = 2 * (1 - cos(theta));
//     // auto full = (1 - theta * sin(theta) / costerm) / theta2;

//     scalar_type c = cos(theta);
//     scalar_type s = sin(theta);
//     scalar_type t = 1 - c;
//     // scalar_type v = theta * (1 - c) / (theta * theta + 1e-10);
//     scalar_type coscc = ((1 - theta * s) / (2 * (1 - c) + 1e-10) / (theta * theta + 1e-10));
//     //
//     logV = v * (T.block(0, 3, 3, 1) - logR.cross(T.block(0, 3, 3, 1)));
//     Matrix logT = Matrix::Zero(4, 4);
//     logT.block(0, 0, 3, 3) = logR;
//     logT.block(0, 3, 3, 1) = logV;
//     logT(3, 3) = 1;
//     return logT;
// }

// 0.144161 0.0448875 0.851911 0.0201577 0.0100573 0.0198088
// 0.150196 -0.258288 0.845786 0.0200396 0.0100523 0.0201926
// -0.0138166 0.022942 -0.00329233 0.0280046 0.0399905 0.021589
// -0.0113997 0.00595423 -0.00572025 0.0280046 0.0399905 0.021589

#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd mean = VectorXd::Zero(estimations[0].rows());
    int n = estimations.size();
    for (int i = 0; i < n; ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array().pow(2);
        mean += residual;
    }
    mean = mean / n;
    mean = mean.array().sqrt();
    return mean;
}

double Tools::NormAngle(double angle){
    if (angle < -M_PI){
        return angle + 2*M_PI;
    } else if (angle > M_PI){
        return angle - 2*M_PI;
    }
}

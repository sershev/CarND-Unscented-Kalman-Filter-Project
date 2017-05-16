#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Weights of sigma points
  VectorXd weights_;

  // State dimension
  int n_x_ = 5;

  // Augmented state dimension
  int n_aug_ = 7;

  // Sigma point spreading parameter
  double lambda_ = 3 - n_x_;
  double lambda_aug_ = 3 - n_aug_;

  // the current NIS for radar
  double NIS_radar_;

  // the current NIS for laser
  double NIS_laser_;

  MatrixXd Xsig_pred_;

  long long time_us_;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}


void UKF::GenerateSigmaPoints(MatrixXd *Xsig_generated) {

    MatrixXd Xsig = MatrixXd::Zero(n_x_, 2 * n_x_ + 1);
    MatrixXd L = P_.llt().matrixL();
    MatrixXd Sig = sqrt(lambda_+n_x_)*L;

    Xsig.col(0) = x_;
    for (int i = 0; i < n_x_; ++i){
        Xsig.col(i+1) = x_ + Sig.col(i+1);
        Xsig.col(i+1+n_x_) = x_ - Sig.col(i+1+n_x_);
    }

    *Xsig_generated = Xsig;
}

void UKF::AugmentSigmaPoints(MatrixXd *Xsig_aug) {

    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.head(n_x_) = x_;
    x_aug(5) = x_aug(6) = 0;

    MatrixXd Q = MatrixXd(2,2);
    Q << std_a_*std_a_, 0,
            0, std_yawdd_*std_yawdd_;

    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug.bottomRightCorner(2,2) = Q;

    MatrixXd L_aug = P_aug.llt().matrixL();
    MatrixXd Sig_aug = sqrt(lambda_aug_+n_aug_)*L_aug;

    MatrixXd Xsig_aug_out = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);


    Xsig_aug_out.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i){
        Xsig_aug_out.col(i+1) = x_aug + Xsig_aug_out.col(i+1);
        Xsig_aug_out.col(i+1+n_x_) = x_aug - Xsig_aug_out.col(i+1+n_x_);
    }

    *Xsig_aug = Xsig_aug_out;
}

Eigen::MatrixXd UKF::SigmaPointPrediction(const MatrixXd & Xsig_aug, const double & delta_t){

    MatrixXd Xsig_pred = MatrixXd::Zero(n_x_, 2*n_aug_+1);

    for (int i = 0; i < 2*n_aug_+1 ; ++i){
        double px = Xsig_aug(0, i);
        double py = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yaw_d = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yaw_dd = Xsig_aug(6, i);

        double px_p, py_p;

        if (fabs(yaw_d) > 0.0001){
            px_p = v * cos(yaw)*delta_t;
            py_p = v * sin(yaw)*delta_t;
        }else{
            px_p = (v/yaw_d) * (sin(yaw+yaw_d*delta_t) - sin(yaw));
            px_p = (v/yaw_d) * (-cos(yaw+yaw_d*delta_t) + cos(yaw));
        }
        px_p += 0.5*delta_t*delta_t*cos(yaw)*nu_a;
        py_p += 0.5*delta_t*delta_t*sin(yaw)*nu_a;
        double v_p = v + delta_t*nu_a;

        double pyaw = yaw + (yaw_d * delta_t) + 0.5*delta_t*delta_t*nu_yaw_dd;
        double pyawd = yaw_d + 0 + delta_t * nu_yaw_dd;

        //write predicted sigma points into right column
        Xsig_pred.col(i) << px_p, py_p, v_p, pyaw, pyawd;
    }
    return Xsig_pred;
}

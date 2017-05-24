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
  P_ = MatrixXd::Zero(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

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

  // State dimension
  n_x_ = 5;
  n_z_ = 3;

  past_timestamp_ = 0;

  R_radar_ = MatrixXd::Zero(n_z_,n_z_);
  R_radar_(0,0) = std_radr_*std_radr_;
  R_radar_(1,1) = std_radphi_ *std_radphi_;
  R_radar_(2,2) = std_radrd_ * std_radrd_;

  R_laser_ = MatrixXd::Zero(2,2);
  R_laser_(0,0) = std_laspx_ * std_laspx_;
  R_laser_(1,1) = std_laspy_ * std_laspy_;

  Q_ = MatrixXd(2,2);
  Q_ << std_a_*std_a_, 0,
          0, std_yawdd_*std_yawdd_;

  // Augmented state dimension
  n_aug_ = 7;
  size_aug_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;
  lambda_aug_ = 3 - n_aug_;

  // the current NIS for radar
  NIS_radar_ = 0;

  // the current NIS for laser
  NIS_laser_ = 0;

  Xsig_pred_ = MatrixXd::Zero(n_x_, size_aug_);

  H_laser_ = MatrixXd::Zero(2,n_x_);
  H_laser_(0,0) = H_laser_(1,1) = 1;
  H_laser_t_ = H_laser_.transpose();

  // Weights of sigma points
  weights_ = VectorXd::Zero(size_aug_);


}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    if (! is_initialized_){
        past_timestamp_ = meas_package.timestamp_;
        is_initialized_ = true;

        if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
            auto ro = meas_package.raw_measurements_(0);
            auto phi = meas_package.raw_measurements_(1);
            auto ro_dot = meas_package.raw_measurements_(2);
            auto sin_phi = sin(phi);
            auto cos_phi = cos(phi);


            auto vx = ro_dot * cos_phi;
            auto vy = ro_dot * sin_phi;

            x_ << ro * cos_phi,  ro * sin_phi, sqrt(vx*vx+vy*vy), 0, 0;
            P_(0,0) = 1;
            P_(1,1) = 1;
            P_(2,2) = 1;
            P_(3,3) = 2;
            P_(4,4) = 2;

        }
        else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
            x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
            P_(0,0) = 1;
            P_(1,1) = 1;
            P_(2,2) = 2;
            P_(3,3) = 2;
            P_(4,4) = 2;
        }

        weights_(0)=lambda_aug_/(lambda_aug_+n_aug_);
        for (int i = 1; i < size_aug_; ++i){
           weights_(i)=0.5/(lambda_aug_+n_aug_);
        }

        return;
    }

    double delta_t = (meas_package.timestamp_ - past_timestamp_) / 1000000.0;
    past_timestamp_ = meas_package.timestamp_;

    if(delta_t < 0.001)
        return;
    if(!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
        return;
    if(!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR)
        return;

    //Predict Step
    Xsig_pred_ = MatrixXd::Zero(n_x_, size_aug_);
    MatrixXd Xsig_aug = AugmentSigmaPoints();
    SigmaPointPrediction(Xsig_aug, delta_t);
    PredictMeanAndCovariance();

    //Update Step
    if(meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR){
        UpdateRadar(meas_package);
        cout << "NIS radar: " << NIS_radar_ << endl;
    }else if(meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER){
        UpdateLidar(meas_package);
        cout << "NIS laser: " << NIS_laser_ << endl;
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    VectorXd y = meas_package.raw_measurements_ - H_laser_*x_;
    MatrixXd PH_t = P_ * H_laser_t_;
    MatrixXd S = H_laser_ * PH_t + R_laser_;
    MatrixXd S_inv = S.inverse();
    MatrixXd K = PH_t * S.inverse();
    x_ = x_ + K * y;
    MatrixXd I = MatrixXd::Identity(n_x_,n_x_);
    P_ = (I - K*H_laser_) * P_;

    NIS_laser_= y.transpose() * S_inv * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

      MatrixXd Zsig = MatrixXd::Zero(n_z_, size_aug_);
      double ro, phi, rodot;
      for (int i=0; i< Zsig.cols() ; ++i){
          VectorXd sigPoint = Xsig_pred_.col(i);
          ro = sqrt(sigPoint(0)*sigPoint(0)+sigPoint(1)*sigPoint(1));
          bool px_zero = fabs(sigPoint(0) < 0.0001);
          bool py_zero = fabs(sigPoint(0) < 0.0001);
          if(px_zero && py_zero ){
              phi = 0.78;
          }else if(px_zero){
              phi = M_PI;
              std::cout << "Ooops!" << std::endl; //exception
          }else{
              phi = atan2(sigPoint(1), sigPoint(0));
              phi = Tools::NormAngle(phi);
          }
          rodot = (sigPoint(0)*cos(sigPoint(3))*sigPoint(2)+sigPoint(1)*sin(sigPoint(3))*sigPoint(2))/ro;
          Zsig.col(i) << ro, phi, rodot;

      }
      //calculate mean predicted measurement
      VectorXd z_pred = VectorXd::Zero(n_z_);
      for (int i =0; i < Zsig.cols(); ++i){
          z_pred = z_pred + weights_(i)*Zsig.col(i);
      }
      //calculate measurement covariance matrix S
      MatrixXd S = MatrixXd::Zero(n_z_,n_z_);
      for (int i =0; i< Zsig.cols(); ++i){
          VectorXd diff = Zsig.col(i) - z_pred;
          S = S + weights_(i)*diff*diff.transpose();
      }
      S = S + R_radar_;
      MatrixXd T = MatrixXd::Zero(n_x_, n_z_);
      for (int i = 0; i < size_aug_; ++i){
          VectorXd diff_oriign_state = Xsig_pred_.col(i)-x_;
          VectorXd diff_radar_state = Zsig.col(i)-z_pred;
          diff_oriign_state(3) = Tools::NormAngle(diff_oriign_state(3));
          diff_radar_state(1) = Tools::NormAngle(diff_radar_state(1));
          T = T + weights_(i) * diff_oriign_state*diff_radar_state.transpose();
      }
      MatrixXd S_inv = S.inverse();
      MatrixXd K = T * S.inverse();
      VectorXd z_diff = (meas_package.raw_measurements_ - z_pred);
      z_diff(1) = Tools::NormAngle(z_diff(1));
      x_ = x_ + K * z_diff;
      P_ = P_ - K * S * K.transpose();
      NIS_radar_= z_diff.transpose() * S_inv * z_diff;
}


Eigen::MatrixXd UKF::AugmentSigmaPoints() {

    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, size_aug_);
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.head(n_x_) = x_;
    x_aug(5) = x_aug(6) = 0;

    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug.bottomRightCorner(2,2) = Q_;

    MatrixXd L_aug = P_aug.llt().matrixL();
    MatrixXd Sig_aug = sqrt(lambda_aug_+n_aug_)*L_aug;

    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i){
        Xsig_aug.col(i+1) = x_aug + Sig_aug.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - Sig_aug.col(i);
    }
   return Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd & Xsig_aug, const double & delta_t){

    for (int i = 0; i < size_aug_ ; ++i){
        double px = Xsig_aug(0, i);
        double py = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yaw_d = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yaw_dd = Xsig_aug(6, i);

        double px_p, py_p;

        if (fabs(yaw_d) < 0.0001){
            px_p = px + v * cos(yaw)*delta_t;
            py_p = py + v * sin(yaw)*delta_t;
        }else{
            px_p = px + (v/yaw_d) * (sin(yaw+yaw_d*delta_t) - sin(yaw));
            py_p = py + (v/yaw_d) * (-cos(yaw+yaw_d*delta_t) + cos(yaw));
        }
        const double delta_t_squered = delta_t*delta_t;
        px_p += 0.5*delta_t_squered*cos(yaw)*nu_a;
        py_p += 0.5*delta_t_squered*sin(yaw)*nu_a;
        double v_p = v + delta_t*nu_a;

        double pyaw = yaw + (yaw_d * delta_t) + 0.5*delta_t_squered*nu_yaw_dd;
        double pyawd = yaw_d + 0 + delta_t * nu_yaw_dd;

        //write predicted sigma points into right column
        Xsig_pred_.col(i) << px_p, py_p, v_p, pyaw, pyawd;
    }
}

void UKF::PredictMeanAndCovariance(){

      //predict state mean
      x_.fill(0.0);
      for(int i=0; i<size_aug_; ++i){
          x_ = x_ + weights_(i)*Xsig_pred_.col(i);
      }

      //predict state covariance matrix
      P_.fill(0.0);
      for(int i=0; i < size_aug_; ++i){
          VectorXd diff = Xsig_pred_.col(i)-x_;
          diff(3) = Tools::NormAngle(diff(3));

          P_ = P_ + weights_(i)*diff*diff.transpose();
      }
}

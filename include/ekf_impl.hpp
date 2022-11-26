#ifndef KF3D_HPP
#define KF3D_HPP

#include "Quaternion4D.hpp"
#include "Vector3D.hpp"

#include "Core"
#include "Dense"
#include <cassert>
#include <iostream>

class EKF_Impl{
private:
    float _dt;
    Vector3D<float> _raw_gyro, _old_gyro, _raw_acc, _meas_pos, _omega, _acc_inertial, _meas_ang;
    Quaternion4D<float> _pred_ang;
    Vector3D<float> _new_pos;
    Vector3D<float> _new_eul;
    bool initialized = false; //TODO: delete

    Eigen::Matrix<float, 13, 13> _Fx, _P, _Qtune;
    Eigen::Matrix<float, 13, 1> _x;
    Eigen::Matrix<float, 6,  13> _FQ;
    Eigen::DiagonalMatrix<float, 6> _Q;
    Eigen::Matrix<float, 3, 3> _R_pos;
    Eigen::Matrix<float, 4, 4> _R_ang;
    Eigen::Matrix<float, 3, 13> _H_pos;
    Eigen::Matrix<float, 4, 13> _H_ang;

    void readInputs();
    void predict(const float &gx, const float &gy, const float &gz, const float &ax, const float &ay, const float &az);
    void correct(const float &px, const float &py, const float &pz, const float &qw, const float &qx, const float &qy, const float &qz);
    void doCorrectionMath(const Eigen::Ref<const Eigen::MatrixXf> &, const Eigen::Ref<const Eigen::MatrixXf> &, const Eigen::Ref<const Eigen::MatrixXf> &);
    void predictAngle();
    void calcInertialAcceleration();
    void predictVelocity();
    void predictPosition();
    void updatePredictionCoveriance();
    void updatePredictionJacobian();
    void updateProcessNoiseJacobian();

public:
    void getPosition(float &px, float &py, float &pz);
    void getVelocity(float &vx, float &vy, float &vz);
    void getAccBias(float &ax_b, float &ay_b, float &az_b);
    void getOrientation( float &qw, float &qx, float &qy, float &qz);
    void freeze_acc_biases();

    void set_dt(float dt){
        this->_dt = dt;
    }

    EKF_Impl(float dt);
    void reset();
    ~EKF_Impl(){}
};


#endif
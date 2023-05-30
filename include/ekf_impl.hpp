#ifndef KF3D_HPP
#define KF3D_HPP

#include "Quaternion4D.hpp"
#include "Vector3D.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
// #include <boost/numeric/ublas/matrix_expression.hpp>
// #include <boost/qvm/mat_operations.hpp>
#include <cassert>
#include <iostream>
#include "invert_matrix.hpp"

using namespace boost::numeric::ublas;

class EKF_Impl
{
private:
    float _dt;
    Vector3D<float> _raw_gyro, _old_gyro, _raw_acc, _meas_pos, _omega, _acc_inertial, _meas_ang;
    Quaternion4D<float> _pred_ang;
    Vector3D<float> _new_pos;
    Vector3D<float> _new_eul;
    bool initialized = false; // TODO: delete

    matrix<float> _Fx, _P, _Qtune;
    matrix<float> _x = matrix<float>(13, 1, 0);
    matrix<float> _FQ = matrix<float>(6, 13, 0);
    diagonal_matrix<float> _Q = diagonal_matrix<float>(6);
    matrix<float> _R_pos = matrix<float>(3, 3, 0);
    matrix<float> _R_ang = matrix<float>(4, 4, 0);
    matrix<float> _H_pos = matrix<float>(3, 13, 0);
    matrix<float> _H_ang = matrix<float>(4, 13, 0);

    void doCorrectionMath(const matrix<float> &, const matrix<float>&, const matrix<float> &);
    void predictAngle();
    void calcInertialAcceleration();
    void predictVelocity();
    void predictPosition();
    void updatePredictionCoveriance();
    void updatePredictionJacobian();
    void updateProcessNoiseJacobian();

public:
    void init()
    {
        _Fx = matrix<float>(13, 13, 0);
        _P = matrix<float>(13, 13, 0);
        _Qtune = matrix<float>(13, 13, 0);
    }
    void predict(const float &gx, const float &gy, const float &gz, const float &ax, const float &ay, const float &az);
    void correct(const float &px, const float &py, const float &pz, const float &qw, const float &qx, const float &qy, const float &qz);
    void getPosition(float &px, float &py, float &pz);
    void getVelocity(float &vx, float &vy, float &vz);
    void getAccBias(float &ax_b, float &ay_b, float &az_b);
    void getOrientation(float &qw, float &qx, float &qy, float &qz);
    void freeze_acc_biases();

    void set_dt(float dt)
    {
        this->_dt = dt;
    }

    EKF_Impl(float dt);
    void reset();
    ~EKF_Impl() {}
};

#endif
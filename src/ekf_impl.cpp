#include "ekf_impl.hpp"

EKF_Impl::EKF_Impl(float dt) : _dt(dt) {
    this->reset();
}

void EKF_Impl::reset() {
    _x = zero_matrix<float>(_x.size1(),_x.size2());
    _x(6, 0) = 1;
    _x(10, 0) = 0.0;  // ax bias
    _x(11, 0) = 0.0;  // ay bias
    _x(12, 0) = 0.0;  // az bias
    _P = identity_matrix<float>(_P.size1());
    _P = _P * 0.1;
    assert(_x.size1() == 13 && _x.size2() == 1);
    assert(_P.size1() == 13 && _P.size2() == 13);
    _H_ang = zero_matrix<float>(_H_ang.size1(),_H_ang.size2()); _H_ang(0,6) = 1; _H_ang(1,7) = 1; _H_ang(2,8) = 1; _H_ang(3,9) = 1; 
    _H_pos = zero_matrix<float>(_H_pos.size1(),_H_pos.size2()); _H_pos(0,0) = 1; _H_pos(1,1) = 1; _H_pos(2,2) = 1;
    std::vector<float> diag ={0.1893, 0.2238, 1.5781, 0.0097, 0.0133, 0.000106};
    _Q = diagonal_matrix<float,row_major,std::vector<float>>(6, diag);

    _Qtune = zero_matrix<float>(_Qtune.size1(),_Qtune.size2()); 

    // position prediction trust parameters 

    subslice(_Qtune, 0,1,3, 0,1,3) = diagonal_matrix<float>(3,unbounded_array<float>(3,1e-9));

    // velocity prediction trust parameters
    subslice(_Qtune, 3,1,3, 3,1,3) = diagonal_matrix<float>(3, unbounded_array<float>(3, 1e-4));
    // _Qtune.block<3, 3>(3, 3) << 1E-4, 0,    0,
    //                             0,    1.5E-4, 0,
    //                             0,    0,    2E-4;

    // orientation prediction trust parameters
    subslice(_Qtune, 6,1,4, 6,1,4) = diagonal_matrix<float>(3, unbounded_array<float>(4, 1E-5));
    // _Qtune.block<4, 4>(6,6) << 1E-5, 0,     0,      0,
    //                            0,    1E-5,  0,      0,
    //                            0,    0,     1E-5,   0,
    //                            0,    0,     0,      1E-5;

    // acceleration bias prediction trust
    subslice(_Qtune, 10,1,3, 10,1,3) = zero_matrix<float>(3);
    // _Qtune.block<3, 3>(10, 10) << 0,    0,    0,
    //                               0,    0,    0,
    //                               0,    0,    0;

    // measurement trust parameters (position)
    _R_pos = diagonal_matrix<float>(3, unbounded_array<float>(3, 5e-2));
    // _R_pos << 5e-2, 0,    0,
    //           0,    5e-2, 0,
    //           0,    0,    5e-2;

    // measurement trust parameters (orientation)
    _R_ang = diagonal_matrix<float>(4, unbounded_array<float>(3, 1E-5));
    
    // _R_ang << 1E-5, 0,      0,    0,
    //           0,    1E-5,   0,    0,
    //           0,    0,      1E-5, 0,
    //           0,    0,      0,    1E-5;

    initialized = false; 
    std::cout << "Kalman filter Reset\n";   
}


void EKF_Impl::getPosition(float &px, float &py, float &pz) {
    px = _x(0, 0);
    py = _x(1, 0);
    pz = _x(2, 0);
}

void EKF_Impl::getVelocity(float &vx, float &vy, float &vz){
    vx = _x(3, 0);
    vy = _x(4, 0);
    vz = _x(5, 0);
}

void EKF_Impl::getAccBias(float &ax_b, float &ay_b, float &az_b){
    ax_b = _x(10, 0);
    ay_b = _x(11, 0);
    az_b = _x(12, 0);
}

void EKF_Impl::getOrientation( float &qw, float &qx, float &qy, float &qz){
    qw = _pred_ang.w;
    qx = _pred_ang.x;
    qy = _pred_ang.y;
    qz = _pred_ang.z;
}

void EKF_Impl::predict(const float &gx, const float &gy, const float &gz, const float &ax, const float &ay, const float &az){
    _raw_gyro = Vector3D<float>(gx, gy, gz);
    _raw_acc = Vector3D<float>(ax, ay, az);
    predictAngle();
    calcInertialAcceleration();
    predictVelocity();
    predictPosition();
    updatePredictionCoveriance();
    // _P = _Fx.transpose() * _P * _Fx + _FQ.transpose() * _Q *_FQ + _Qtune;
    _P = prod(matrix<float>(prod(trans(_Fx), _P)), _Fx) + prod(matrix<float>(prod(trans(_FQ), _Q)),_FQ) + _Qtune;
}

void EKF_Impl::correct(const float &px, const float &py, const float &pz, const float &qw, const float &qx, const float &qy, const float &qz) {
    if(!initialized){
        this->reset();
        _x(0, 0) = px; _x(1, 0) = py; _x(2, 0) = pz;
        _x(6, 0) = qw; _x(7, 0) = qx; _x(8, 0) = qy; _x(9, 0) = qz;
        initialized = true;
        return;
    }
    // Eigen::Matrix<float, 7, 13> H;
    // H.block<3, 13>(0, 0) << _H_pos;
    // H.block<4, 13>(3, 0) << _H_ang;
    matrix<float> H(7,13);
    subrange(H, 0,3, 0,13) = _H_pos;
    subrange(H, 3,7, 0,13) = _H_ang;
    // Eigen::Matrix<float, 7, 1> z;
    std::vector<float> z_data =  {px, py, pz, qw, qx, qy, qz};
    matrix<float, row_major, std::vector<float>> z(7,1, z_data);
    // Eigen::Matrix<float, 7, 7> R; R = Eigen::ArrayXXf::Zero(7, 7);
    // R.block<3, 3>(0, 0) << _R_pos;
    // R.block<4, 4>(3, 3) << _R_ang;
    matrix<float> R = zero_matrix<float>(7);
    subrange(R, 0,3, 0,3) = _R_pos;
    subrange(R, 3,7, 3,7) = _R_ang;
    doCorrectionMath(H, z, R);
}

void EKF_Impl::freeze_acc_biases(){
    // _Qtune.block<3, 3>(10, 10) << 0,    0,    0,
    //                                 0,    0,    0,
    //                                 0,    0,    0;
    subrange(_Qtune, 10,13, 10,13) = zero_matrix<float>(3);
    std::cout <<"Freezed Acceleration Biases\n";
}

void EKF_Impl::predictAngle() {
    _omega(_raw_gyro.x, _raw_gyro.y, _raw_gyro.z);
    Quaternion4D<float> q_dot(0, _omega.x, _omega.y, _omega.z);
    _pred_ang(_x(6,0), _x(7,0), _x(8,0), _x(9,0));
    q_dot = _pred_ang*0.5 * q_dot;
    q_dot = q_dot * _dt;
    _pred_ang = _pred_ang + (q_dot);
    _pred_ang.normalize();
    _x(6,0) = _pred_ang.w, _x(7,0) = _pred_ang.x, _x(8,0) = _pred_ang.y, _x(9,0) = _pred_ang.z;
};
void EKF_Impl::calcInertialAcceleration() {
    Vector3D<float> acc_bias(_x(10, 0), _x(11, 0), _x(12, 0));
    _acc_inertial = _pred_ang * (_raw_acc - acc_bias);
    _acc_inertial.z = _acc_inertial.z - 9.78909;
};
void EKF_Impl::predictVelocity() {
    _x(3,0) = _x(3,0)+_acc_inertial.x*_dt;
    _x(4,0) = _x(4,0)+_acc_inertial.y*_dt;
    _x(5,0) = _x(5,0)+_acc_inertial.z*_dt;
};
void EKF_Impl::predictPosition() {
    float t2 = _dt*_dt;
    _x(0,0) = _x(0,0)+(_acc_inertial.x*t2)/2.0+_dt*_x(3,0);
    _x(1,0) = _x(1,0)+(_acc_inertial.y*t2)/2.0+_dt*_x(4,0);
    _x(2,0) = _x(2,0)+(_acc_inertial.z*t2)/2.0+_dt*_x(5,0);
};

void EKF_Impl::updatePredictionCoveriance() {
    updatePredictionJacobian();
    updateProcessNoiseJacobian();
};
void EKF_Impl::updatePredictionJacobian() {
    float q1 = _x(6,0), q2 =_x(7,0), q3 = _x(8,0), q4 =_x(9,0);
    float t2 = _dt*_dt;
    float t3 = q1*q1;
    float t4 = q2*q2;
    float t5 = q3*q3;
    float t6 = q4*q4;
    float t7 = q1*q2*2.0;
    float t8 = q1*q3*2.0;
    float t9 = q1*q4*2.0;
    float t10 = q2*q3*2.0;
    float t11 = q2*q4*2.0;
    float t12 = q3*q4*2.0;
    float t13 = -_x(10,0);
    float t14 = -_x(11,0);
    float t15 = -_x(12,0);
    float t22 = (_dt*_omega.x)/2.0;
    float t23 = (_dt*_omega.y)/2.0;
    float t24 = (_dt*_omega.z)/2.0;
    float t16 = -t10;
    float t17 = -t11;
    float t18 = -t12;
    float t19 = -t4;
    float t20 = -t5;
    float t21 = -t6;
    float t25 = _raw_acc.x+t13;
    float t26 = _raw_acc.y+t14;
    float t27 = _raw_acc.z+t15;
    float t28 = -t22;
    float t29 = -t23;
    float t30 = -t24;
    float t43 = t7+t12;
    float t44 = t8+t11;
    float t45 = t9+t10;
    float t31 = q1*t25*2.0;
    float t32 = q2*t25*2.0;
    float t33 = q1*t26*2.0;
    float t34 = q3*t25*2.0;
    float t35 = q2*t26*2.0;
    float t36 = q4*t25*2.0;
    float t37 = q1*t27*2.0;
    float t38 = q3*t26*2.0;
    float t39 = q2*t27*2.0;
    float t40 = q4*t26*2.0;
    float t41 = q3*t27*2.0;
    float t42 = q4*t27*2.0;
    float t49 = t7+t18;
    float t50 = t8+t17;
    float t51 = t9+t16;
    float t52 = t3+t6+t19+t20;
    float t53 = t3+t5+t19+t21;
    float t54 = t3+t4+t20+t21;
    float t46 = -t34;
    float t47 = -t39;
    float t48 = -t40;
    float t55 = t32+t38+t42;
    float t56 = t35+t37+t46;
    float t57 = t33+t36+t47;
    float t58 = t31+t41+t48;
    float t59 = _dt*t55;
    float t63 = (t2*t55)/2.0;
    float t60 = _dt*t56;
    float t61 = _dt*t57;
    float t62 = _dt*t58;
    float t64 = (t2*t56)/2.0;
    float t65 = (t2*t57)/2.0;
    float t66 = (t2*t58)/2.0;
    std::vector<float> fx_data = {1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            _dt,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,_dt,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            0.0,0.0,_dt,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
            t66,t65,t64,t62,t61,t60,1.0,t22,t23,t24,0.0,0.0,0.0,
            t63,-t64,t65,t59,-t60,t61,t28,1.0,t30,t23,0.0,0.0,0.0,
            t64,t63,-t66,t60,t59,-t62,t29,t24,1.0,t28,0.0,0.0,0.0,
            -t65,t66,t63,-t61,t62,t59,t30,t29,t22,1.0,0.0,0.0,0.0,
            t2*t54*(-1.0f/2.0f),t2*t45*(-1.0f/2.0f),(t2*t50)/2.0f,-_dt*t54,-_dt*t45,_dt*t50,0.0,0.0,0.0,0.0,1.0,0.0,0.0,
            (t2*t51)/2.0f,t2*t53*(-1.0f/2.0f),t2*t43*(-1.0f/2.0f),_dt*t51,-_dt*t53,-_dt*t43,0.0,0.0,0.0,0.0,0.0,1.0,0.0,
            t2*t44*(-1.0f/2.0f),(t2*t49)/2.0f,t2*t52*(-1.0f/2.0f),-_dt*t44,_dt*t49,-_dt*t52,0.0,0.0,0.0,0.0,0.0,0.0,1.0}; //WARNING: This is the transposed representation}
    // _Fx << 1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    //         0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    //         0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    //         _dt,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    //         0.0,_dt,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    //         0.0,0.0,_dt,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    //         t66,t65,t64,t62,t61,t60,1.0,t22,t23,t24,0.0,0.0,0.0,
    //         t63,-t64,t65,t59,-t60,t61,t28,1.0,t30,t23,0.0,0.0,0.0,
    //         t64,t63,-t66,t60,t59,-t62,t29,t24,1.0,t28,0.0,0.0,0.0,
    //         -t65,t66,t63,-t61,t62,t59,t30,t29,t22,1.0,0.0,0.0,0.0,
    //         t2*t54*(-1.0/2.0),t2*t45*(-1.0/2.0),(t2*t50)/2.0,-_dt*t54,-_dt*t45,_dt*t50,0.0,0.0,0.0,0.0,1.0,0.0,0.0,
    //         (t2*t51)/2.0,t2*t53*(-1.0/2.0),t2*t43*(-1.0/2.0),_dt*t51,-_dt*t53,-_dt*t43,0.0,0.0,0.0,0.0,0.0,1.0,0.0,
    //         t2*t44*(-1.0/2.0),(t2*t49)/2.0,t2*t52*(-1.0/2.0),-_dt*t44,_dt*t49,-_dt*t52,0.0,0.0,0.0,0.0,0.0,0.0,1.0; //WARNING: This is the transposed representation
    _Fx = matrix<float, row_major, std::vector<float>>(13,13,fx_data);
    assert((_Fx.size1() == _Fx.size2()) && (_Fx.size1() == 13));
};
void EKF_Impl::updateProcessNoiseJacobian() {
    float q1 = _x(6,0), q2 =_x(7,0), q3 = _x(8,0), q4 =_x(9,0);
    float t2 = _dt*_dt;
    float t3 = q1*q1;
    float t4 = q2*q2;
    float t5 = q3*q3;
    float t6 = q4*q4;
    float t7 = q1*q2*2.0;
    float t8 = q1*q3*2.0;
    float t9 = q1*q4*2.0;
    float t10 = q2*q3*2.0;
    float t11 = q2*q4*2.0;
    float t12 = q3*q4*2.0;
    float t13 = -t10;
    float t14 = -t11;
    float t15 = -t12;
    float t16 = -t4;
    float t17 = -t5;
    float t18 = -t6;
    float t19 = t7+t12;
    float t20 = t8+t11;
    float t21 = t9+t10;
    float t22 = t7+t15;
    float t23 = t8+t14;
    float t24 = t9+t13;
    float t25 = t3+t6+t16+t17;
    float t26 = t3+t5+t16+t18;
    float t27 = t3+t4+t17+t18;
    std::vector<float> fq_data = {(t2*t27)/2.0f,        (t2*t21)/2.0f,       t2*t23*(-1.0f/2.0f),  _dt*t27,    _dt*t21,    -_dt*t23,   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
           t2*t24*(-1.0f/2.0f),   (t2*t26)/2.0f,       (t2*t19)/2.0f,       -_dt*t24,   _dt*t26,    _dt*t19,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
           (t2*t20)/2.0f,        t2*t22*(-1.0f/2.0f),  (t2*t25)/2.0f,       _dt*t20,    -_dt*t22,   _dt*t25,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
           0.0,                 0.0,                0.0,                0.0,        0.0,        0.0,        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
           0.0,                 0.0,                0.0,                0.0,        0.0,        0.0,        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
           0.0,                 0.0,                0.0,                0.0,        0.0,        0.0,        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0};
    _FQ = matrix<float, row_major, std::vector<float>>(6,13, fq_data);
    // _FQ << (t2*t27)/2.0,        (t2*t21)/2.0,       t2*t23*(-1.0/2.0),  _dt*t27,    _dt*t21,    -_dt*t23,   0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    //        t2*t24*(-1.0/2.0),   (t2*t26)/2.0,       (t2*t19)/2.0,       -_dt*t24,   _dt*t26,    _dt*t19,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    //        (t2*t20)/2.0,        t2*t22*(-1.0/2.0),  (t2*t25)/2.0,       _dt*t20,    -_dt*t22,   _dt*t25,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    //        0.0,                 0.0,                0.0,                0.0,        0.0,        0.0,        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    //        0.0,                 0.0,                0.0,                0.0,        0.0,        0.0,        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
    //        0.0,                 0.0,                0.0,                0.0,        0.0,        0.0,        0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0; 
           //WARNING: This is the transposed representation
    assert((_FQ.size1() == 6) && (_FQ.size2() == 13));
};

void EKF_Impl::doCorrectionMath(const matrix<float> &H,
                                const matrix<float> &z, 
                                const matrix<float> &R) {
    matrix<float> K, S, I_KH, S_inv;
    S = prod(matrix<float>(prod(H, _P)), trans(H)) + R;
    InvertMatrix(S, S_inv);
    K = prod(matrix<float>(prod(_P, trans(H))), S_inv);
    _x = _x + prod(K, (z - matrix<float>(prod(H, _x))));
    _pred_ang(_x(6,0), _x(7,0), _x(8,0), _x(9,0));
    _pred_ang.normalize();
    _x(6,0) = _pred_ang.w, _x(7,0) = _pred_ang.x, _x(8,0) = _pred_ang.y, _x(9,0) = _pred_ang.z;
    I_KH = identity_matrix<float>(_P.size1(), _P.size2()) - prod(K,H);
    _P = prod(matrix<float>(prod(I_KH, _P)), trans(I_KH)) + prod(matrix<float>(prod(K, R)), trans(K));
}
#include <iostream>
#include "ekf_impl.hpp"

#include <chrono>
using namespace std::chrono;

int main(int argc, char** argv){
    EKF_Impl ekf(0.005);
    float qx,qy,qz,qw;
    float px,py,pz;

    auto start = high_resolution_clock::now();
    ekf.correct(0.1, 0.2, 1, 0.9937607, 0.0497295, 0.0997087, 0.0049896);
    for (int i=0;i<1000;i++){
        ekf.predict(0.02, 0.04, 0.0001, 0.12, 0.11, 9.81);
        ekf.getOrientation(qw,qx,qy,qz);
        // ekf.getPosition(px,py,pz);
        // std::cout << px << ", " << py << ", " << pz << ", "
        //         << qw << ", " << qx << ", " << qy << ", " << qz << ", " << std::endl;   
        ekf.correct(0.3, 0.4, 1.1, qw, qx, qy, qz);
        // ekf.getOrientation(qw,qx,qy,qz);
        // ekf.getPosition(px,py,pz);
    }
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
 
    // To get the value of duration use the count()
    // member function on the duration object
    std::cout << "Total time : " << duration.count()/1e6 << std::endl;

    // std::cout << px << ", " << py << ", " << pz << ", "
    //           << qw << ", " << qx << ", " << qy << ", " << qz << ", " << std::endl;   
    return 0;

}
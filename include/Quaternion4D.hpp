
#ifndef QUATERNION4D_HPP
#define QUATERNION4D_HPP

#include <cstdint>
#include <math.h>
#include <initializer_list>
#include "Vector3D.hpp"

template <typename T>  
class Quaternion4D {
        public:
                T w = 1;
                T x = 0;
                T y = 0;
                T z = 0;
                inline Quaternion4D<T>(){}; 
                inline Quaternion4D<T>(const T &w_, const T &x_, const T &y_, const T &z_){
                        this->w = w_;
                        this->x = x_;
                        this->y = y_;
                        this->z = z_;
                }
                inline friend Quaternion4D<T> operator + (const Quaternion4D<T> &v1, const Quaternion4D<T> &v2) { //TODO: functions are not efficient, need rework
                        return Quaternion4D<T>(v1.w+v2.w, v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
                }
                inline friend Quaternion4D<T> operator - (const Quaternion4D<T> &v1, const Quaternion4D<T> &v2) {
                        return Quaternion4D<T>(v1.w+v2.w, v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
                }
                inline friend Quaternion4D<T> operator * (const Quaternion4D<T> &v1, const Quaternion4D<T> &v2) {
                        return Quaternion4D<T>(v1.w*v2.w - v1.x*v2.x - v1.y*v2.y - v1.z*v2.z,
                        v1.w*v2.x + v1.x*v2.w + v1.y*v2.z - v1.z*v2.y,
                        v1.w*v2.y - v1.x*v2.z + v1.y*v2.w + v1.z*v2.x,
                        v1.w*v2.z + v1.x*v2.y - v1.y*v2.x + v1.z*v2.w);
                }
                inline friend Quaternion4D<T> operator * (const Quaternion4D<T> &v1, const T& tmp) {
                        return Quaternion4D<T>(v1.w*tmp, v1.x*tmp, v1.y*tmp, v1.z*tmp);
                }
                inline friend Vector3D<T> operator * (const Quaternion4D<T> &q, const Vector3D<T>& v) {
                        Vector3D<T> vec(v.x*(q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z) - v.y*(2*q.w*q.z - 2*q.x*q.y) + v.z*(2*q.w*q.y + 2*q.x*q.z),
                                        v.y*(q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z) + v.x*(2*q.w*q.z + 2*q.x*q.y) - v.z*(2*q.w*q.x - 2*q.y*q.z),
                                        v.z*(q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z) - v.x*(2*q.w*q.y - 2*q.x*q.z) + v.y*(2*q.w*q.x + 2*q.y*q.z));
                        return vec;
                }
                inline Quaternion4D<T>& operator = (const std::initializer_list<T>& tmp) {
                        w = *tmp.begin(); x = *(tmp.begin()+1); y = *(tmp.begin()+2); z = *(tmp.begin()+3);
                        return *this;
                }
                void operator() (const T& w_, const T& x_, const T& y_, const T& z_) {
                        this->w = w_;
                        this->x = x_;
                        this->y = y_;
                        this->z = z_;
                }
                template <typename M>
                operator Quaternion4D<M>() {
                        Quaternion4D<M> tmp;
                        tmp.w = M(this->w);
                        tmp.x = M(this->x);
                        tmp.y = M(this->y);
                        tmp.z = M(this->z);

                        return tmp;
                }
                inline T dot(const Quaternion4D<T> &tmp){
                        return w*tmp.w + x*tmp.x + y*tmp.y + z*tmp.z;
                }
                inline double length(){
                        return sqrt(dot(*this));
                }
                inline Quaternion4D<double> normalize(){
                        double len = 1.0/length();
                        return Quaternion4D<double>(*this) * len;
                }
};
#endif
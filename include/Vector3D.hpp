#ifndef VECTOR3D_HPP
#define VECTOR3D_HPP

#include <cstdint>
#include <math.h>
#include <initializer_list>

template <typename T>  
class Vector3D {
        public:
                T x = 0;
                T y = 0;
                T z = 0;
                inline Vector3D<T>(){}; 
                inline Vector3D<T>(const T &x_, const T &y_, const T &z_){
                        this->x = x_;
                        this->y = y_;
                        this->z = z_;
                }
                inline friend Vector3D<T> operator + (const Vector3D<T> &v1, const Vector3D<T> &v2) {
                        return Vector3D<T>(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
                }
                inline friend Vector3D<T> operator - (const Vector3D<T> &v1, const Vector3D<T> &v2) {
                        return Vector3D<T>(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
                }
                inline friend Vector3D<T> operator * (const Vector3D<T> &v1, const Vector3D<T> &v2) {
                        return Vector3D<T>(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
                }
                inline friend Vector3D<T> operator * ( const T& tmp, const Vector3D<T> &v1) {
                        return Vector3D<T>(v1.x*tmp, v1.y*tmp, v1.z*tmp);
                }
                inline friend Vector3D<T> operator / (const Vector3D<T>& v1, const T& tmp) {
                        return Vector3D<T>(v1.x/tmp, v1.y/tmp, v1.z/tmp);
                }
                inline Vector3D<T>& operator = (const T& tmp) {
                        x = tmp;
                        y = tmp;
                        z = tmp;

                        return *this;
                }
                inline Vector3D<T>& operator = (const std::initializer_list<T>& tmp) {
                        x = *tmp.begin(); y = *(tmp.begin()+1); z = *(tmp.begin()+2);
                        return *this;
                }
                inline friend bool operator == (const Vector3D<T> &v1, const Vector3D<T> &v2) {
                        if(v1.x == v2.x && v1.y == v2.y && v1.z == v2.z) {return true;}
                        else {return false;}
                }
                inline friend bool operator == (const Vector3D<T> &v1, const T& val) {
                        if(v1.x == val && v1.y == val && v1.z == val) {return true;}
                        else {return false;}
                }
                inline friend bool operator < (const Vector3D<T> &v1, const T& val) {
                        if(v1.x < val || v1.y < val || v1.z < val) {return true;}
                        else {return false;}
                }
                // inline friend bool operator < (const Vector3D<T> &v1, const Vector3D<T> &v2) {
                //         if(v1.x < v2.x && v1.y < v2.y && v1.z < v2.z) {return true;}
                //         else {return false;}
                // }
                inline friend bool operator > (const Vector3D<T> &v1, const T& val) {
                        if(v1.x > val || v1.y > val || v1.z > val) {return true;}
                        else {return false;}
                }
                // inline friend bool operator > (const Vector3D<T> &v1, const Vector3D<T> &v2) {
                //         if(v1.x > v2.x && v1.y > v2.y && v1.z > v2.z) {return true;}
                //         else {return false;}
                // }
                inline friend bool operator != (const Vector3D<T> &v1, const Vector3D<T> &v2) {
                        if(v1.x != v2.x || v1.y != v2.y || v1.z != v2.z) {return true;}
                        else {return false;}
                }
                void operator() (const T& x_, const T& y_, const T& z_) {
                        this->x = x_;
                        this->y = y_;
                        this->z = z_;
                }
                template <typename M>
                operator Vector3D<M>() {
                        Vector3D<M> tmp;
                        tmp.x = M(this->x);
                        tmp.y = M(this->y);
                        tmp.z = M(this->z);

                        return tmp;
                }
                inline friend T operator ^ (const Vector3D<T> &v1, const Vector3D<T> &v2) {
                        return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
                }
                inline T dot(const Vector3D<T> &tmp){
                        return x*tmp.x + y*tmp.y + z*tmp.z;
                }
                inline Vector3D<T> cross(const Vector3D<T>& tmp){
                        return Vector3D<T>(this->y*tmp.z - this->z*tmp.y, this->z*tmp.x - this->x*tmp.z, this->x*tmp.y - this->y*tmp.x);
                }
                inline double length(){
                        return sqrt(dot(*this));
                }
                inline Vector3D<double> normalized(){
                        return Vector3D<double>(*this)/length();
                }

                static double getL2Norm(Vector3D<T> vec1) {
                    return sqrt(vec1.x*vec1.x+vec1.y*vec1.y+vec1.z*vec1.z);
                }


                static Vector3D<T> cross(Vector3D<T> vec1,Vector3D<T> vec2) {
                        Vector3D<T> results;
                        results.x=vec1.y*vec2.z-vec1.z*vec2.y;
                        results.y=vec1.z*vec2.x-vec1.x*vec2.z;
                        results.z=vec1.x*vec2.y-vec1.y*vec2.x;
                        return results;
                }

                static double dot(Vector3D<T> vec1,Vector3D<T> vec2) {
                    return vec1.x*vec2.x+vec1.y*vec2.y+vec1.z*vec2.z;
                }
};

#endif
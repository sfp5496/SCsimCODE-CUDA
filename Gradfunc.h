#ifndef __GRADFUNC_H__
#define __GRADFUNC_H__

double GradSum();
__device__ double GradSum_dev(int npart, double3* GU1);
double GradSumA();
__device__ double GradSumA_dev(int npart, double2* GU1A);

__global__ void Kernel(double3* r, double3* u, double2* theta, double* l, double* sigma, double3* r1, double3* r2, double3* U2, double2* U2A, double* U1, double3* GU1, double3* GU0, double2* GU1A, double2* GU0A, double* params, double3* h, double3* hA);

__global__ void GradKernel1(double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double3* U2, double2* U2A, double* U1, double* params);
__global__ void GradKernel2(double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double3* U2, double2* U2A, double* U1, double3* GU1, double3* GU0, double2* GU1A, double2* GU0A, double* params);
__global__ void GradKernel3(double3* GU1, double2* GU1A, double3* GU0, double2* GU0A, double3* U2, double2* U2A, double* U1);
//void Gradient2();
void Gradient3();

#endif

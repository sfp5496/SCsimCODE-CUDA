#ifndef __CONGRADFUNC_H__
#define __CONGRADFUNC_H__

//__global__ void ConGradKernel1(double3* r, double3* h, double3* GU1, double2* theta, double2* hA, double2* GU1A, double* params, int iter);
__global__ void ConGradKernel(double3* r, double3* h, double3* GU1, double3* GU0, double2* theta, double2* hA, double2* GU1A, double2* GU0A, double* params);
void ConGrad1();
void ConGrad2();

#endif

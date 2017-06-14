#ifndef __MISC_H__
#define __MISC_H__

double unitrand();
void initcond(double3* x, double2* ang, double* len, double* diam);
double packfrac();
void updatephi();
__global__ void updatephikernel(double* l, double* sigma, double* params, double* phi);
__global__ void variabledphi(double* params, double U);
__global__ void DeviceAdd(int i, double* a, double b);
double Pressure();
double contacts();

#endif

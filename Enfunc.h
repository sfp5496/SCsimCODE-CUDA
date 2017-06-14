#ifndef __ENFUNC_H__
#define __ENFUNC_H__

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val);
#endif

void sptoca(int i);
void ends(int i);
__host__ __device__ double3 sptocadev(double2 angle);
__global__ void sptocadev_all(double2* angle, double3* u_dev);
__device__ double3 endsdev1(double3 r, double3 u, double l);
__device__ double3 endsdev2(double3 r, double3 u, double l);
__host__ __device__ double d3SCdist(double3 ri, double3 rj, double3 ui, double3 uj, double li, double lj);
__host__ __device__ double lambda(double3 ri, double3 rj, double3 ui, double3 uj, double li);
__host__ __device__ double PotEnergydev(double3 ri, double3 rj, double3 ui, double3 uj, double li, double lj, double si, double sj, double R, double H);
__host__ __device__ double WallEnergy(double3 r1, double3 r2, double li, double si, double R, double H);
__global__ void PotEnergyKernel(int i, double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double* U, double* params);
__global__ void PotEnergyKernel2(double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double* U, double* params);
__global__ void PotEnergyKernel3(double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double* U, double* params);
double collider();
double collider2();

#endif

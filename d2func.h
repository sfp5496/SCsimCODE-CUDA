#ifndef __D2FUNC_H__
#define __D2FUNC_H__

__host__ __device__ double2 d2add(double2 a, double2 b);
__host__ __device__ double2 d2sub(double2 a, double2 b);
__host__ __device__ double2 d2multscal(double2 a, double b);
__host__ __device__ double2 d2multscal2(double2 a, double b);
__host__ __device__ double2 d2divscal(double2 a, double b);
__host__ __device__ double d2dotp(double2 a, double2 b);
__host__ __device__ double d2mag(double2 a);
__host__ __device__ double d2dist(double2 a, double2 b);
__host__ __device__ double2 d2unit(double2 a);
__host__ __device__ double2 d2null();

#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

__host__ __device__ double3 d3add(double3 a, double3 b) {
	/*
	 * Arguments: two 3d vectors
	 * Returns: a new vector that is a vector addition of the arguments
	 */
	
	double3 ret;
	ret.x=a.x+b.x;
	ret.y=a.y+b.y;
	ret.z=a.z+b.z;
	return ret;
}

__host__ __device__ double3 d3sub(double3 a, double3 b) {
	/*
	 * Arguments: two 3d vectors
	 * Returns: a new vector that is the first vector minus the second vector
	 */
	double3 ret;
	ret.x=a.x-b.x;
	ret.y=a.y-b.y;
	ret.z=a.z-b.z;
	return ret;
}

__host__ __device__ double3 d3multscal(double3 a, double b) {
	/*
	 * Arguments: one 3d vector and a scalar
	 * Returns: the vector with each component multiplied by the scalar
	 */
	double3 ret;
	ret.x=b*a.x;
	ret.y=b*a.y;
	ret.z=b*a.z;
	return ret;
}

__host__ __device__ double3 d3multscal2(double3 a, double b) {
	/*
	 * Arguments: one 3d vector and a scalar
	 * Returns: the vector with each component multiplied by the scalar squared
	 */
	double3 ret;
	ret.x=b*b*a.x;
	ret.y=b*b*a.y;
	ret.z=b*b*a.z;
	return ret;
}

__host__ __device__ double3 d3divscal(double3 a, double b) {
	/*
	 * Arguments: one 3d vector and a scalar
	 * Returns: the vector with each component divided by the scalar
	 */
	double3 ret;
	ret.x=a.x/b;
	ret.y=a.y/b;
	ret.z=a.z/b;
	return ret;
}

__host__ __device__ double d3dotp(double3 a, double3 b) {
	/*
	 * Arguments: two 3d vectors
	 * Returns: the dot product of the two vectors
	 */
	return (a.x*b.x)+(a.y*b.y)+(a.z*b.z);
}

__host__ __device__ double3 d3crossp(double3 a, double3 b) {
	/*
	 * Arguments: two 3d vectors
	 * Returns: the first vector crossed with the second vector
	 */
	double3 ret;
	ret.x=(a.y*b.z)-(a.z*b.y);
	ret.y=-(a.x*b.z)+(a.z*b.x);
	ret.z=(a.x*b.y)-(a.y*b.x);
	return ret;
}

__host__ __device__ double d3mag(double3 a) {
	/*
	 * Arguments: one 3d vector
	 * Returns: the magnitude of the vector
	 */
	return sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}

__host__ __device__ double d3dist(double3 a, double3 b) {
	/*
	 * Arguments: two 3d vectors
	 * Returns: the magnitude of the first vector minus the second
	 */
	return d3mag(d3sub(a,b));
}

__host__ __device__ double3 d3unit(double3 a) {
	/*
	 * Arguments: one 3d vector
	 * Returns: a unit vector pointing in the same direction as the argument vector
	 */
	return d3divscal(a,d3mag(a));
}

__host__ __device__ double3 d3null() {
	/*
	 * Arguments: none
	 * Returns: the null vector in 3d
	 */
	double3 a;
	a.x=0.0;
	a.y=0.0;
	a.z=0.0;
	return a;
}


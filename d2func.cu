#include <stdio.h>
#include <math.h>
#include <stdlib.h>

__host__ __device__ double2 d2add(double2 a, double2 b) {
	/*
	 * Arguments: two 2d vectors
	 * Returns: the vector addition of the two vectors
	 */
	double2 ret;
	ret.x=a.x+b.x;
	ret.y=a.y+b.y;
	return ret;
}

__host__ __device__ double2 d2sub(double2 a, double2 b) {
	/*
	 * Arguments: two 2d vectors
	 * Returns: the vector subtraction of the two vectors
	 */
	double2 ret;
	ret.x=a.x-b.x;
	ret.y=a.y-b.y;
	return ret;
}

__host__ __device__ double2 d2multscal(double2 a, double b) {
	/*
	 * Arguments: a 2d vector and a scalar
	 * Returns: the vector with each component multiplied by the scalar
	 */
	double2 ret;
	ret.x=b*a.x;
	ret.y=b*a.y;
	return ret;
}

__host__ __device__ double2 d2multscal2(double2 a, double b) {
	/*
	 * Arguments: a 2d vector and a scalar
	 * Returns: the vector with each component multiplied by the scalar squared
	 */
	double2 ret;
	ret.x=b*b*a.x;
	ret.y=b*b*a.y;
	return ret;
}

__host__ __device__ double2 d2divscal(double2 a, double b) {
	/*
	 * Arguments: a 2d vector and a scalar
	 * Returns: the vector with each component divided by the scalar
	 */
	double2 ret;
	ret.x=a.x/b;
	ret.y=a.y/b;
	return ret;
}

__host__ __device__ double d2dotp(double2 a, double2 b) {
	/*
	 * Arguments: two 2d vectors
	 * Returns: the dot product between the two vectors
	 */
	return (a.x*b.x)+(a.y*b.y);
}

__host__ __device__ double d2mag(double2 a) {
	/*
	 * Arguments: one 2d vector
	 * Returns: the magnitude of the argument vector
	 */
	return sqrt((a.x)*(a.x)+(a.y)*(a.y));
}

__host__ __device__ double d2dist(double2 a, double2 b) {
	/*
	 * Arguments: two 2d vectors
	 * Returns: the magnitude of the first vector minus the second vector
	 */
	return d2mag(d2sub(a,b));
}

__host__ __device__ double2 d2unit(double2 a) {
	/*
	 * Arguments: one 2d vector
	 * Returns: a unit vector pointing in the same direction as the argument vector
	 */ 
	return d2divscal(a, d2mag(a));
}

__host__ __device__ double2 d2null() {
	/*
	 * Arguments: none
	 * Returns: the null vector (2d)
	 */
	double2 a;
	a.x=0.0;
	a.y=0.0;
	return a;
}


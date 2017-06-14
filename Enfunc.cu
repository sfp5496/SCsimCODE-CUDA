#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "myhelpers.h"
#include "d2func.h"
#include "d3func.h"
#include "Globalvars.h"

#if __CUDA_ARCH__ < 600
/*
Got this from docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
*/
__device__ double atomicAdd(double* address, double val) {
	/*
	 * This function essentially locks a piece of memory so that only one thread may write to it
	 * Basically it fixes race conditions
	 */
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull,assumed,__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

void sptoca(int i) {
	/*
	 * Arguments: the particle id
	 * Returns: updates the orientation vector of the ith particle based on the theta and phi values
	 */
	u[i].x=sin(theta[i].y)*cos(theta[i].x);
	u[i].y=sin(theta[i].y)*sin(theta[i].x);
	u[i].z=cos(theta[i].y);
}

void ends(int i) {
	/*
	 * Arguments: the particle id
	 * Returns: calculates where the ends of the particle are, useful in calculating potential 
	 *  energy between a particle and a wall
	 */
	r1[i]=d3add(r[i],d3multscal(u[i],0.5*l[i]));
	r2[i]=d3sub(r[i],d3multscal(u[i],0.5*l[i]));
	r1[i].x=hypot(r1[i].x,r1[i].y);
	r2[i].x=hypot(r2[i].x,r2[i].y);
}

__host__ __device__ double3 sptocadev(double2 angle) {
	/*
	 * Arguments: the theta (angle.x) and phi (angle.y) values of the particle
	 * Returms: orientation vector of the particle
	 * Why: to make sptoca CUDA usable I would need to pass EVERYTHING to it, and this is
	 *      an easier function to use in CUDA
	 */
	
	double3 ret;
	ret.x=sin(angle.y)*cos(angle.x);
	ret.y=sin(angle.y)*sin(angle.x);
	ret.z=cos(angle.y);
	return ret;
}

__global__ void sptocadev_all(double2* angle, double3* u_dev) {
	/*
	 * Arguments:
	 * 	angle:an array containing the angular orientation coordinates for every particle
	 * Returns: updates the orientation vectors of every particle
	 */

	 int tx=threadIdx.x+blockIdx.x*blockDim.x;

	 u_dev[tx]=sptocadev(angle[tx]);
}

__device__ double3 endsdev1(double3 r,double3 u,double l) {
	/*
	 * Arguments: the position (r) and orientation (u) vectors and the length (l) of the particle
	 * Returns: the ends of the particle to be used in WallEnergy()
	 * Why: to make ends easier to use in CUDA, to use the other one, I'd have to start passing everything to it (basically it's just like sptocadev)
	 */

	double3 ret;
	ret=d3add(r,d3multscal(u,0.5*l));
	ret.x=hypot(ret.x,ret.y);
	return ret;
}

__device__ double3 endsdev2(double3 r,double3 u,double l) {
	/*
	 * This just calculates the end that the previous function doesn't
	 */

	 double3 ret;
	 ret=d3sub(r,d3multscal(u,0.5*l));
	 ret.x=hypot(ret.x,ret.y);
	 return ret;
}

__host__ __device__ double d3SCdist(double3 ri, double3 rj, double3 ui, double3 uj, double li, double lj) {
	/*
	 * Arguments: the position and orientation vectors of two spherolines and their corresponding
	 *  lambda values
	 * Returns: the distance between the two spherolines
	 */
	double3 dij;
	dij=d3sub(d3sub(d3add(rj,d3multscal(uj,lj)),ri),d3multscal(ui,li));
	return d3mag(dij);
	
}

__host__ __device__ double lambda(double3 ri, double3 rj, double3 ui, double3 uj, double li) {
	/*
	 * Arguments: the position and orientation of two spherolines and the length of the 
	 *  spheroline whose lambda we are attempting to find
	 * Returns: the lambda value of the ith spheroline of the ij spheroline pair
	 * What is This: the lambda values are the points on the line of the spherolines, 
	 *  where the two most closely intersect
	 */
	double retn,retd;
	retn=d3dotp(ui,d3sub(rj,ri))-d3dotp(ui,uj)*d3dotp(uj,d3sub(rj,ri));
	retd=1.0-(d3dotp(ui,uj)*d3dotp(ui,uj));
	if((retn/retd)>(li/2.0)) {
		return (li/2.0);
	}
	else if((retn/retd)<(-li/2.0)) {
		return (-li/2.0);
	}
	else {
		return (retn/retd);
	}
}

__host__ __device__ double PotEnergydev(double3 ri,double3 rj,double3 ui, double3 uj, double li, double lj, double si, double sj, double R, double H) {
	/*
	 * Arguments:
	 *    ri: position of ith particle
	 *    rj: position of jth particle
	 *    ui: orientation of ith particle
	 *    uj: orientation of jth particle
	 *    li: length of ith particle
	 *    lj: length of jth particle
	 *    si: diameter of ith particle
	 *    sj: diameter of jth particle
	 *    R: container Radius
	 *    H: container Height
	 * 
	 * Returns: the potential energy between the two particles
	 */
	double U=0.0;
	//Check to see if the particles even have a chance to be touching
	if((d3dist(ri,rj)<(li+si+lj+sj)/2.0)) {
		double lambda_i, lambda_j;
		lambda_i=lambda(ri,rj,ui,uj,li);
		lambda_j=lambda(rj,ri,uj,ui,lj);
		double d;
		d=d3SCdist(ri,rj,ui,uj,lambda_i,lambda_j);
		//Check to see if the particles are ACTUALLY in contact
		if(d<(si+sj)/2.0) {
			U=0.25*((si+sj)/2.0-d)*((si+sj)/2.0-d);
		}
	}
	return U;
}

__host__ __device__ double WallEnergy(double3 r1, double3 r2, double li, double si, double R, double H) {
	/*
	 * Arguments:
	 *    r1: position of one end of the particle
	 *    r2: position of the other end of the particle
	 *    li: length of the particle
	 *    si: diameter of the particle
	 *    R: container radius
	 *    H: container height
	 *
	 * Returns: the potential energy between the particle and the walls
	 */
	
	double U=0.0;

	//check to see if the first end is in contact with the radial wall of the container
	if(r1.x>R-(si/2.0)) {
		U+=0.5*(r1.x-(R-si/2.0))*(r1.x-(R-si/2.0));
	}
	//check to see if the second end is in contact with the radial wall of the container
	if(r2.x>R-(si/2.0)) {
		U+=0.5*(r2.x-(R-si/2.0))*(r2.x-(R-si/2.0));
	}
	//check to see if the first end is in contact with the floor
	if(r1.z-si/2.0<0.0) {
		U+=0.5*(r1.z-si/2.0)*(r1.z-si/2.0);
	}
	//check to see if the first end is in contact with the ceiling
	else if(r1.z+si/2.0>H) {
		U+=0.5*(r1.z+si/2.0-H)*(r1.z+si/2.0-H);
	}
	//check to see if the second end is in contact with the floor
	if(r2.z-si/2.0<0.0) {
		U+=0.5*(r2.z-si/2.0)*(r2.z-si/2.0);
	}
	//check to see if the second end is in contact with the ceiling
	else if(r2.z+si/2.0>H) {
		U+=0.5*(r2.z+si/2.0-H)*(r2.z+si/2.0-H);
	}
	return U;
}

//This function doesn't use the above device functions (it predates them), it also only calculates the energy of one particle
__global__ void PotEnergyKernel(int i, double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double* U, double* params) {
	int j=threadIdx.x+blockIdx.x*blockDim.x;
	
	if(i==j) {
		//check to see if the first end is in contact with the radial wall of the container
		if(r1[i].x>params[1]-(sigma[i]/2.0)) {
			atomicAdd(U,0.5*(r1[i].x-(params[1]-sigma[i]/2.0))*(r1[i].x-(params[1]-sigma[i]/2.0)));
		}
		//check to see if the second end is in contact with the radial wall of the container
		if(r2[i].x>params[1]-(sigma[i]/2.0)) {
			atomicAdd(U,0.5*(r2[i].x-(params[1]-sigma[i]/2.0))*(r2[i].x-(params[1]-sigma[i]/2.0)));
		}
		//check to see if the first end is in contact with the floor
		if(r1[i].z-sigma[i]/2.0<0.0) {
			atomicAdd(U,0.5*(r1[i].z-sigma[i]/2.0)*(r1[i].z-sigma[i]/2.0));
		}
		//check to see if the first end is in contact with the ceiling
		else if(r1[i].z+sigma[i]/2.0>params[2]) {
			atomicAdd(U,0.5*(r1[i].z+sigma[i]/2.0-params[2])*(r1[i].z+sigma[i]/2.0-params[2]));
		}
		//check to see if the second end is in contact with the floor
		if(r2[i].z-sigma[i]/2.0<0.0) {
			atomicAdd(U,0.5*(r2[i].z-sigma[i]/2.0)*(r2[i].z-sigma[i]/2.0));
		}
		//check to see if the second end is in contact with the ceiling
		else if(r2[i].z+sigma[i]/2.0>params[2]) {
			atomicAdd(U,0.5*(r2[i].z+sigma[i]/2.0-params[2])*(r2[i].z+sigma[i]/2.0-params[2]));
		}
	}
	else {
		if((d3dist(r[i],r[j])<(l[i]+sigma[i]+l[j]+sigma[j])/2.0)) {
			double lambda_i, lambda_j;
			lambda_i=lambda(r[i],r[j],u[i],u[j],l[i]);
			lambda_j=lambda(r[j],r[i],u[j],u[i],l[j]);
			double d;
			d=d3SCdist(r[i],r[j],u[i],u[j],lambda_i,lambda_j);
			//check to see if the particles are in contact
			if(d<(sigma[i]+sigma[j])/2.0) {
				atomicAdd(U,0.25*((sigma[i]+sigma[j])/2.0-d)*((sigma[i]+sigma[j])/2.0-d));
			}
		}
	}
}

//This function uses the device functions and is a reduction algorithm
__global__ void PotEnergyKernel2(double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double* U, double* params) {
	/*
	 * Arguments:
	 *    r: double3 array containing position vectors of every particle
	 *    u: double3 array containing orientaion vectors of every particle
	 *    l: double array containing particle length of every particle
	 *    sigma: double array containing particle diameters
	 *    r1: double3 array containing the position vectors for one end of particles
	 *    r2: double3 array containing the position vectors of the other end of particles
	 *    U: double array containing the potential energy contribution of each particle
	 *    params: the parameters, 0 is npart, 1 is R, 2 is H
	 *
	 * Returns: an array U of the potential energy contribution of each block of particle
	 *    pairs (each block is 32*32)
	 */

	const int n2=32*32;

	__shared__ double cache[n2];

	int tx=threadIdx.x+blockIdx.x*blockDim.x;
	int ty=threadIdx.y+blockIdx.y*blockDim.y;

	int cacheIdx=(blockDim.x)*threadIdx.x+threadIdx.y;
	int BLOCKS=((int)params[0])/blockDim.x;
	
	double3 ri=r[tx];
	double3 rj=r[ty];
	double3 ui=u[tx];
	double3 uj=u[ty];
	double li=l[tx];
	double lj=l[ty];
	double si=sigma[tx];
	double sj=sigma[ty];

	double R=params[1];
	double H=params[2];

	double3 r1i=r1[tx];
	double3 r2i=r2[tx];

	double dU;
	if(tx==ty) {
		//printf("tx: %i\n",tx);
		dU=WallEnergy(r1i,r2i,li,si,R,H);
		//printf("dU: %E\n",dU);
		cache[cacheIdx]=dU;
	}
	else {
		dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
		cache[cacheIdx]=dU;
	}
	__syncthreads();
	
	int i=blockDim.x*blockDim.y/2;
	//printf("i: %i\n",i);
	while(i!=0) {
		if(cacheIdx<i) {
			//printf("cacheIdx: %i %i\n",i,cacheIdx);
			cache[cacheIdx]+=cache[cacheIdx+i];
			//printf("cache: %E\n",cache[cacheIdx]);
			//printf("cache[%i]: %lf\n",cacheIdx,cache[cacheIdx]);
		}
		__syncthreads();
		i/=2;
	}
	if(cacheIdx==0) {
		//printf("BLOCKS: %i\n",BLOCKS);
		//printf("U_idx: %i\n",blockIdx.x+BLOCKS*blockIdx.y);
		U[blockIdx.x*BLOCKS+blockIdx.y]=cache[0];
		//U[blockIdx.x+blockDim.x*blockIdx.y]=1.0;
	}
}

//This function uses the device functions and atomicAdd (which is slower)
__global__ void PotEnergyKernel3(double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double* U, double* params) {
	/*
	 * Arguments:
	 *    r: double3 array containing position vectors of every particle
	 *    u: double3 array containing orientaion vectors of every particle
	 *    l: double array containing particle length of every particle
	 *    sigma: double array containing particle diameters
	 *    r1: double3 array containing the position vectors for one end of particles
	 *    r2: double3 array containing the position vectors of the other end of particles
	 *    U: double array containing the potential energy contribution of each particle
	 *    params: the parameters, 0 is npart, 1 is R, 2 is H
	 *
	 * Returns: an array U of the potential energy contribution of each particle
	 */

	int tx=threadIdx.x+blockIdx.x*blockDim.x;
	int ty=threadIdx.y+blockIdx.y*blockDim.y;

	double3 ri=r[tx];
	double3 rj=r[ty];
	double3 ui=u[tx];
	double3 uj=u[ty];
	double li=l[tx];
	double lj=l[ty];
	double si=sigma[tx];
	double sj=sigma[ty];

	double R=params[1];
	double H=params[2];

	double3 r1i=r1[tx];
	double3 r2i=r2[tx];

	double dU;
	if(tx==ty) {
		dU=WallEnergy(r1i,r2i,li,si,R,H);
		atomicAdd(U+tx,dU);
	}
	else {
		dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
		atomicAdd(U+tx,dU);
	}
}

//This function uses PotEnergyKernel2
double collider() {
	/*
	 * Arguments: none
	 * Returns: the total potential energy of the current packing
	 */
		
	double ret=0.0;
	double* U1;
	double* U1_dev;
	
	int N=grid.x*grid.y;
	HANDLE_ERROR(cudaMalloc(&U1_dev,N*sizeof(double)));
	U1=(double*)malloc(N*sizeof(double));

	int i;
	for(i=0;i<npart;i++) {
		sptoca(i);
		ends(i);
	}
	
	HANDLE_ERROR(cudaMemcpy(r_dev,r,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(u_dev,u,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(l_dev,l,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(sigma_dev,sigma,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(r1_dev,r1,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(r2_dev,r2,npart*sizeof(double3),cudaMemcpyHostToDevice));
	if(cudaSuccess != cudaGetLastError()) {
		printf("Error 1\n");
	}

	PotEnergyKernel2<<<grid,block>>>(r_dev,u_dev,l_dev,sigma_dev,r1_dev,r2_dev,U1_dev,params);
	if(cudaSuccess != cudaGetLastError()) {
		printf("Error 2\n");
	}
	HANDLE_ERROR(cudaMemcpy(U1,U1_dev,N*sizeof(double),cudaMemcpyDeviceToHost));
	for(i=0;i<N;i++) {
		//fprintf(stderr,"U1[i]: %E\n",U1[i]);
		ret+=U1[i];
	}
	return ret;
}

//This function uses PotEnergyKernel3
double collider2() {
	/*
	 * Arguments: none
	 * Returns: the total potential energy of the current packing
	 */
	double ret=0.0;
	double* U1;
	double* U1_dev;
	

	HANDLE_ERROR(cudaMalloc(&U1_dev,npart*sizeof(double)));
	U1=(double*)malloc(npart*sizeof(double));

	int i;
	for(i=0;i<npart;i++) {
		sptoca(i);
		ends(i);
	}
	//HANDLE_ERROR(cudaMemcpy(r_dev,r,npart*sizeof(double3),cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(u_dev,u,npart*sizeof(double3),cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(l_dev,l,npart*sizeof(double),cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(sigma_dev,sigma,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(r1_dev,r1,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(r2_dev,r2,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(U1_dev,0,npart*sizeof(double)));
	PotEnergyKernel3<<<grid,block>>>(r_dev,u_dev,l_dev,sigma_dev,r1_dev,r2_dev,U1_dev,params);
	cudaError_t error;
	error=cudaGetLastError();
	if(cudaSuccess != error) {
		printf("%s\n",cudaGetErrorString(error));
	}

	HANDLE_ERROR(cudaMemcpy(U1,U1_dev,npart*sizeof(double),cudaMemcpyDeviceToHost));
	for(i=0;i<npart;i++) {
		ret+=U1[i];
	}
	return ret;
}


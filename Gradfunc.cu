#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "myhelpers.h"
#include "d2func.h"
#include "d3func.h"
#include "Globalvars.h"
#include "Enfunc.h"

double GradSum() {
	/*
	 * Arguments: none
	 * Returns: the sum of the magnitudes of the gradient vector of the potential energy of every
	 *  particle in the cartesian directions, mostly used in calculating whether or not a given 
	 *  packing may be minimized further or not
	 */
	int i;
	double ret=0;
	for(i=0;i<npart;i++) {
		ret+=d3mag(GU1[i]);
	}
	return ret;
}

__device__ double GradSum_dev(int npart, double3* GU1) {
	int i;
	double ret=0.0;
	for(i=0;i<npart;i++) {
		ret+=d3mag(GU1[i]);
	}
	return ret;
}

double GradSumA() {
	/*
	 * Arguments: none
	 * Returns: the sum of the magnitudes of the gradient vector of the potential energy of every
	 *  particle in the theta and phi directions, mostly used in calculating whether or not a given 
	 *  packing may be minimized further or not
	 */
	int i;
	double ret=0;
	for(i=0;i<npart;i++) {
		ret+=hypot(GU1A[i].x,GU1A[i].y);
	}
	return ret;
}

__device__ double GradSumA_dev(int npart, double2* GU1A) {
	int i;
	double ret=0.0;
	for(i=0;i<npart;i++) {
		ret+=d2mag(GU1A[i]);
	}
	return ret;
}

__global__ void GradKernel1(double3* r, double3* u, double* l, double* sigma, double3* r1, double3* r2, double3* U2, double2* U2A, double* U1, double* params) {
	/*
	 * Arguments:
	 *    r: double3 array containing position vector of every particle
	 *    u:
	 *    l:
	 *    sigma:
	 *    r1:
	 *    r2:
	 *    
	 * Returns: 
	 */

	int tx=threadIdx.x+blockIdx.x*blockDim.x;
	int ty=threadIdx.y+blockIdx.y*blockDim.y;

	int i=tx/6;
	int j=ty;
	int dim=tx%6;

	const int n2=32*32;

	double3 ri=r[i];
	double3 rj=r[j];
	double3 ui=u[i];
	double3 uj=u[j];
	double li=l[i];
	double lj=l[j];
	double si=sigma[i];
	double sj=sigma[j];

	double2 theta;
	theta.x=atan2(ui.y,ui.x);
	theta.y=acos(ui.z);

	double R=params[1];
	double H=params[2];
	int npart=((int)params[0]);

	double3 r1i=r1[i];
	double3 r2i=r2[i];

	double dU;

	if(dim==0) { //x derivatives
		double dx=.0001;
		ri.x+=dx;
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			atomicAdd(&(U2[i].x),dU);
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			atomicAdd(&(U2[i].x),dU);
		}
	}
	else if(dim==1) { //y derivatives
		double dy=.0001;
		ri.y+=dy;

		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			atomicAdd(&(U2[i].y),dU);
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			atomicAdd(&(U2[i].y),dU);
		}
	}
	else if(dim==2) { //z derivatives
		double dz=.0001;
		ri.z+=dz;

		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			atomicAdd(&(U2[i].z),dU);
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			atomicAdd(&(U2[i].z),dU);
		}
	}
	else if(dim==3) { //theta derivatives
		double dt=.0001;
		theta.x+=dt;
		ui=sptocadev(theta);

		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			atomicAdd(&(U2A[i].x),dU);
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			atomicAdd(&(U2A[i].x),dU);
		}
	}
	else if(dim==4) { //phi derivatives
		double dp=.0001;
		theta.y+=dp;
		ui=sptocadev(theta);

		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			atomicAdd(&(U2A[i].y),dU);
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			atomicAdd(&(U2A[i].y),dU);
		}
	}
	else { //these threads are just to calculate U w/o anything being moved
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			atomicAdd(U1+i,dU);
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			atomicAdd(U1+i,dU);
		}
	}
}



/*__global__ void Kernel(double3* r, double3* u, double2* theta,double* l, double* sigma, double3* r1, double3* r2, double3* U2, double2* U2A, double* U1, double3* GU1, double3* GU0, double2* GU1A, double2* GU0A, double* params, double3* h, double2* hA) {
	*
	 * Arguments:
	 *    r: double3 array containing position vector of every particle
	 *    u:
	 *    l:
	 *    sigma:
	 *    r1:
	 *    r2:
	 *    
	 * Returns: This gives us the potential energies of each particle and what happens to that energy when we perturb that particle in each dimension, we can then use these to calculate the gradient vector
	 *

	double R=params[1];
	double H=params[2];
	int npart=((int)params[0]);
	
	int tx=threadIdx.x+blockIdx.x*blockDim.x;
	int ty=threadIdx.y+blockIdx.y*blockDim.y;

	int dim=blockIdx.x%6;
	int i=blockIdx.x/6;
	int j=threadIdx.x;

	int cacheIdx=j;

	const int n2=128; //IMPORTANT: this needs to be changed manually to match npart!!

	__shared__ double cache[n2];
	
	double3 ri=r[i];
	double3 rj=r[j];
	double3 ui=u[i];
	double3 uj=u[j];
	double li=l[i];
	double lj=l[j];
	double si=sigma[i];
	double sj=sigma[j];

	double2 thetai;
	thetai.x=atan2(ui.y,ui.x);
	thetai.y=acos(ui.z);

	double3 r1i;
	double3 r2i;

	double dU;

	if(dim==0) { //x derivatives
		double dx=.0001;
		ri.x+=dx;
		
		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);

		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		ri.x-=dx;
	}
	else if(dim==1) { //y derivatives
		double dy=.0001;
		ri.y+=dy;

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);

		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		ri.y-=dy;
	}
	else if(dim==2) { //z derivatives
		double dz=.0001;
		ri.z+=dz;

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		ri.z-=dz;
	}
	else if(dim==3) { //theta derivatives
		double dt=.0001;
		thetai.x+=dt;
		ui=sptocadev(thetai);

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		thetai.x-=dt;
		ui=sptocadev(theta);
	}
	else if(dim==4) { //phi derivatives
		double dp=.0001;
		thetai.y+=dp;
		ui=sptocadev(thetai);

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		thetai.y-=dp;
		ui=sptocadev(theta);
	}
	else { //these threads are just to calculate U w/o anything being moved

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
	}

	__syncthreads();

	int q=n2/2;
	while(q!=0) {
		if(cacheIdx<q) {
			cache[cacheIdx]+=cache[cacheIdx+q];
		}
		__syncthreads();
		q/=2;
	}
	if(cacheIdx==0) {
		if(dim==0) {
			atomicAdd(&(U2[i].x),cache[0]);
		}
		else if(dim==1) {
			atomicAdd(&(U2[i].y),cache[0]);
		}
		else if(dim==2) {
			atomicAdd(&(U2[i].z),cache[0]);
		}
		else if(dim==3) {
			atomicAdd(&(U2A[i].x),cache[0]);
		}
		else if(dim==4) {
			atomicAdd(&(U2A[i].y),cache[0]);
		}
		else {
			atomicAdd(U1+i,cache[0]);
		}
	}

	__syncthreads();

	double GS=0.0;
	double GSA=0.0;

	if(tx==0) {
		double d=.0001;

		int k;

		for(k=0;k<npart;k++) {

			GU0[k]=GU1[k];
			GU0A[k]=GU1A[k];
		
			GU1[k].x=(U2[k].x-U1[k])/d;
			GU1[k].y=(U2[k].y-U1[k])/d;
			GU1[k].z=(U2[k].z-U1[k])/d;
			GU1A[k].x=(U2A[k].x-U1[k])/d;
			GU1A[k].y=(U2A[k].y-U1[k])/d;

			printf("U1[%i]: %lf\n",k,U1[k]);
			printf("GU1[%i]: %lf %lf %lf\n",k,GU1[k].x,GU1[k].y,GU1[k].z);
			printf("U2[%i]: %lf %lf %lf\n",k,U2[k].x,U2[k].y,U2[k].z);
			printf("GU1A[%i]: %lf %lf\n",k,GU1A[k].x,GU1A[k].y);
			printf("U2A[%i]: %lf %lf\n",k,U2A[k].x,U2A[k].y);

			GS+=d3mag(GU1);
			GSA+=d2mag(GU1A);
		}
	}
	while(GS>1e-9 && GSA>1e-7) {
			
		if(ty==0 && tx<npart) {
			double eta=0.0;
			double etaA=0.0;
		
			eta=-fabs(GradSum_dev(npart,GU1));
			if(fabs(eta)>1.0) {
				eta=-1.0;
			}
			else if(fabs(eta)<.001) {
				eta=-.001;
			}
		
			for(i=0;i<npart;i++) {
				etaA+=d2mag(GU1A[i]);
			}
			etaA=-fabs(etaA);
			if(fabs(etaA)>1.0) {
				etaA=-1.0;
			}
			else if(fabs(etaA)<.001) {
				etaA=-.001;
			}
		
			double gamma, gammaA;
		
			if(d3dotp(GU0[tx],GU0[tx])!=0.0) {
				gamma=d3dotp(GU1[tx],GU1[tx])/d3dotp(GU0[tx],GU0[tx]);
				if(gamma>1.0) {
					gamma=1.0;
				}
				else if(gamma<-1.0) {
					gamma=-1.0;
				}
			}
			else {
				gamma=1.0;
			}
			
			h[tx]=d3add(GU1[tx],d3multscal(h[tx],gamma));
			r[tx]=d3add(r[tx],d3multscal(h[tx],eta));
			
			if(d2dotp(GU0A[tx],GU0A[tx])!=0) {
				gammaA=d2dotp(GU1A[tx],GU1A[tx])/d2dotp(GU0A[tx],GU0A[tx]);
				if(gammaA>1.0) {
					gammaA=1.0;
				}
				else if(gammaA<-1.0) {
					gammaA=-1.0;
				}
			}
			else {
				gammaA=1.0;
			}
			hA[tx]=d2add(GU1A[tx],d2multscal(hA[tx],gammaA));
			theta[tx]=d2add(theta[tx],d2multscal(hA[tx],etaA));
		
			//printf("r[%i]: %lf %lf %lf\n",tx,r[tx].x,r[tx].y,r[tx].z);
			//printf("theta[%i]: %lf %lf\n",tx,theta[tx].x,theta[tx].y);
			//printf("hA_dev[%i]: %lf %lf\n",tx,hA[tx].x,hA[tx].y);
			//printf("etaA[%i]: %lf\n",tx,etaA);
			//printf("gammaA[%i]: %lf\n",tx,gammaA);
			//printf("GU1A[%i]: %lf %lf\n",tx,GU1A[tx].x,GU1A[tx].y);
			//printf("GU0A[%i]: %lf %lf\n",tx,GU0A[tx].x,GU0A[tx].y);
		}

		ri=r[i];
		rj=r[j];
		ui=u[i];
		uj=u[j];
		li=l[i];
		lj=l[j];
		si=sigma[i];
		sj=sigma[j];
	
		theta.x=atan2(ui.y,ui.x);
		theta.y=acos(ui.z);
	
		if(dim==0) { //x derivatives
			dx=.0001;
			ri.x+=dx;
			
			r1i=endsdev1(ri,ui,li);
			r2i=endsdev2(ri,ui,li);
	
			if(i==j) {
				dU=WallEnergy(r1i,r2i,li,si,R,H);
				cache[cacheIdx]=dU;
			}
			else {
				dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
				cache[cacheIdx]=dU;
			}
			ri.x-=dx;
		}
		else if(dim==1) { //y derivatives
			dy=.0001;
			ri.y+=dy;
	
			r1i=endsdev1(ri,ui,li);
			r2i=endsdev2(ri,ui,li);
	
			if(i==j) {
				dU=WallEnergy(r1i,r2i,li,si,R,H);
				cache[cacheIdx]=dU;
			}
			else {
				dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
				cache[cacheIdx]=dU;
			}
			ri.y-=dy;
		}
		else if(dim==2) { //z derivatives
			dz=.0001;
			ri.z+=dz;
	
			r1i=endsdev1(ri,ui,li);
			r2i=endsdev2(ri,ui,li);
			
			if(i==j) {
				dU=WallEnergy(r1i,r2i,li,si,R,H);
				cache[cacheIdx]=dU;
			}
			else {
				dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
				cache[cacheIdx]=dU;
			}
			ri.z-=dz;
		}
		else if(dim==3) { //theta derivatives
			dt=.0001;
			theta.x+=dt;
			ui=sptocadev(theta);
	
			r1i=endsdev1(ri,ui,li);
			r2i=endsdev2(ri,ui,li);
			
			if(i==j) {
				dU=WallEnergy(r1i,r2i,li,si,R,H);
				cache[cacheIdx]=dU;
			}
			else {
				dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
				cache[cacheIdx]=dU;
			}
			theta.x-=dt;
			ui=sptocadev(theta);
		}
		else if(dim==4) { //phi derivatives
			dp=.0001;
			theta.y+=dp;
			ui=sptocadev(theta);
	
			r1i=endsdev1(ri,ui,li);
			r2i=endsdev2(ri,ui,li);
			
			if(i==j) {
				dU=WallEnergy(r1i,r2i,li,si,R,H);
				cache[cacheIdx]=dU;
			}
			else {
				dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
				cache[cacheIdx]=dU;
			}
			theta.y-=dp;
			ui=sptocadev(theta);
		}
		else { //these threads are just to calculate U w/o anything being moved
	
			r1i=endsdev1(ri,ui,li);
			r2i=endsdev2(ri,ui,li);
			
			if(i==j) {
				dU=WallEnergy(r1i,r2i,li,si,R,H);
				cache[cacheIdx]=dU;
			}
			else {
				dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
				cache[cacheIdx]=dU;
			}
		}
	
		__syncthreads();
	
		q=n2/2;
		while(q!=0) {
			if(cacheIdx<q) {
				cache[cacheIdx]+=cache[cacheIdx+q];
			}
			__syncthreads();
			q/=2;
		}
		if(cacheIdx==0) {
			if(dim==0) {
				atomicAdd(&(U2[i].x),cache[0]);
			}
			else if(dim==1) {
				atomicAdd(&(U2[i].y),cache[0]);
			}
			else if(dim==2) {
				atomicAdd(&(U2[i].z),cache[0]);
			}
			else if(dim==3) {
				atomicAdd(&(U2A[i].x),cache[0]);
			}
			else if(dim==4) {
				atomicAdd(&(U2A[i].y),cache[0]);
			}
			else {
				atomicAdd(U1+i,cache[0]);
			}
		}
	
		__syncthreads();
	
		GS=0.0;
		GSA=0.0;
	
		if(tx==0) {
			d=.0001;
	
			for(k=0;k<npart;k++) {
	
				GU0[k]=GU1[k];
				GU0A[k]=GU1A[k];
			
				GU1[k].x=(U2[k].x-U1[k])/d;
				GU1[k].y=(U2[k].y-U1[k])/d;
				GU1[k].z=(U2[k].z-U1[k])/d;
				GU1A[k].x=(U2A[k].x-U1[k])/d;
				GU1A[k].y=(U2A[k].y-U1[k])/d;
	
				printf("U1[%i]: %lf\n",k,U1[k]);
				printf("GU1[%i]: %lf %lf %lf\n",k,GU1[k].x,GU1[k].y,GU1[k].z);
				printf("U2[%i]: %lf %lf %lf\n",k,U2[k].x,U2[k].y,U2[k].z);
				printf("GU1A[%i]: %lf %lf\n",k,GU1A[k].x,GU1A[k].y);
				printf("U2A[%i]: %lf %lf\n",k,U2A[k].x,U2A[k].y);
	
				GS+=d3mag(GU1);
				GSA+=d2mag(GU1A);
			}
		}

	}
}
*/

__global__ void GradKernel2(double3* r, double3* u, double2* theta, double* l, double* sigma, double3* r1, double3* r2, double3* U2, double2* U2A, double* U1, double3* GU1, double3* GU0, double2* GU1A, double2* GU0A, double* params) {
	/*
	 * Arguments:
	 *    r: double3 array containing position vector of every particle
	 *    u:
	 *    l:
	 *    sigma:
	 *    r1:
	 *    r2:
	 *    
	 * Returns: This gives us the potential energies of each particle and what happens to that energy when we perturb that particle in each dimension, we can then use these to calculate the gradient vector
	 */

	double R=params[1];
	double H=params[2];
	int npart=((int)params[0]);
	
	int tx=threadIdx.x+blockIdx.x*blockDim.x;

	int dim=blockIdx.x%6;
	int i=blockIdx.x/6;
	int j=threadIdx.x;

	int cacheIdx=j;

	const int n2=128; //IMPORTANT: this needs to be changed manually to match npart!!

	__shared__ double cache[n2];
	
	double3 ri=r[i];
	double3 rj=r[j];
	double3 ui=u[i];
	double3 uj=u[j];
	double li=l[i];
	double lj=l[j];
	double si=sigma[i];
	double sj=sigma[j];

	double2 thetai=theta[i];
	//theta.x=atan2(ui.y,ui.x);
	//theta.y=acos(ui.z);
	double2 thetaj=theta[j];

	ui=sptocadev(thetai);
	uj=sptocadev(thetaj);

	double3 r1i;
	double3 r2i;

	double dU;

	if(dim==0) { //x derivatives
		double dx=.0001;
		ri.x+=dx;
		
		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);

		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		ri.x-=dx;
	}
	else if(dim==1) { //y derivatives
		double dy=.0001;
		ri.y+=dy;

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);

		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		ri.y-=dy;
	}
	else if(dim==2) { //z derivatives
		double dz=.0001;
		ri.z+=dz;

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		ri.z-=dz;
	}
	else if(dim==3) { //theta derivatives
		double dt=.0001;
		thetai.x+=dt;
		ui=sptocadev(thetai);

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		thetai.x-=dt;
		ui=sptocadev(thetai);
	}
	else if(dim==4) { //phi derivatives
		double dp=.0001;
		thetai.y+=dp;
		ui=sptocadev(thetai);

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
		thetai.y-=dp;
		ui=sptocadev(thetai);
	}
	else { //these threads are just to calculate U w/o anything being moved

		r1i=endsdev1(ri,ui,li);
		r2i=endsdev2(ri,ui,li);
		
		if(i==j) {
			dU=WallEnergy(r1i,r2i,li,si,R,H);
			cache[cacheIdx]=dU;
		}
		else {
			dU=PotEnergydev(ri,rj,ui,uj,li,lj,si,sj,R,H);
			cache[cacheIdx]=dU;
		}
	}

	__syncthreads();

	int q=n2/2;
	while(q!=0) {
		if(cacheIdx<q) {
			cache[cacheIdx]+=cache[cacheIdx+q];
		}
		__syncthreads();
		q/=2;
	}
	if(cacheIdx==0) {
		if(dim==0) {
			atomicAdd(&(U2[i].x),cache[0]);
		}
		else if(dim==1) {
			atomicAdd(&(U2[i].y),cache[0]);
		}
		else if(dim==2) {
			atomicAdd(&(U2[i].z),cache[0]);
		}
		else if(dim==3) {
			atomicAdd(&(U2A[i].x),cache[0]);
		}
		else if(dim==4) {
			atomicAdd(&(U2A[i].y),cache[0]);
		}
		else {
			atomicAdd(U1+i,cache[0]);
		}
	}
	/*
	__syncthreads();

	double d=.0001;

	if(i==0 && dim==0) {
		GU1[j].x=(U2[j].x-U1[j])/d;
		GU1[j].y=(U2[j].y-U1[j])/d;
		GU1[j].z=(U2[j].z-U1[j])/d;
		GU1A[j].x=(U2A[j].x-U1[j])/d;
		GU1A[j].y=(U2A[j].y-U1[j])/d;

		printf("U1[%i]: %lf\n",j,U1[j]);
		printf("GU1[%i]: %lf %lf %lf\n",j,GU1[j].x,GU1[j].y,GU1[j].z);
		printf("U2[%i]: %lf %lf %lf\n",j,U2[j].x,U2[j].y,U2[j].z);
		printf("GU1A[%i]: %lf %lf\n",j,GU1A[j].x,GU1A[j].y);
		printf("U2A[%i]: %lf %lf\n",j,U2A[j].x,U2A[j].y);
	}

	if(i==0) {
		double d=.0001;
		
		printf("HELLO?\n");

		if(dim==0) {
			GU0[j]=GU1[j];
			GU0A[j]=GU1A[j];
			
			GU1[j].x=(U2[j].x-U1[j]);
		}
		else if(dim==1) {
			GU1[j].y=(U2[j].y-U1[j]);
		}
		else if(dim==2) {
			GU1[j].z=(U2[j].z-U1[j]);
		}
		else if(dim==3) {
			GU1A[j].x=(U2A[j].x-U1[j]);
		}
		else if(dim==4) {
			GU1A[j].y=(U2A[j].y-U1[j]);
		}
		else {
			printf("U1[%i]: %lf\n",j,U1[j]);
			printf("GU1[%i]: %lf %lf %lf\n",j,GU1[j].x,GU1[j].y,GU1[j].z);
			printf("U2[%i]: %lf %lf %lf\n",j,U2[j].x,U2[j].y,U2[j].z);
			printf("GU1A[%i]: %lf %lf\n",j,GU1A[j].x,GU1A[j].y);
			printf("U2A[%i]: %lf %lf\n",j,U2A[j].x,U2A[j].y);
		}
	}*/
	//printf("%i %i %i\n")
/*	__syncthreads();
	if(tx==10) {
		printf("\n\n");
		int k;
		double d=.0001;
		for(k=0;k<npart;k++) {

			GU0[k]=GU1[k];
			GU0A[k]=GU1A[k];
	
			GU1[k].x=(U2[k].x-U1[k])/d;
			GU1[k].y=(U2[k].y-U1[k])/d;
			GU1[k].z=(U2[k].z-U1[k])/d;
			GU1A[k].x=(U2A[k].x-U1[k])/d;
			GU1A[k].y=(U2A[k].y-U1[k])/d;


			printf("U1[%i]: %lf\n",k,U1[k]);
			printf("GU1[%i]: %lf %lf %lf\n",k,GU1[k].x,GU1[k].y,GU1[k].z);
			printf("U2[%i]: %lf %lf %lf\n",k,U2[k].x,U2[k].y,U2[k].z);
			//printf("GU1A[%i]: %lf %lf\n",k,GU1A[k].x,GU1A[k].y);
			//printf("U2A[%i]: %lf %lf\n",k,U2A[k].x,U2A[k].y);
		}
	}*/
}

__global__ void GradKernel3(double3* GU1, double2* GU1A, double3* GU0, double2* GU0A, double3* U2, double2* U2A, double* U1) {
	/*
	 * Arguments:
	 *	GU1: double3 array which contains the gradients of the particle in x,y,z
	 *	GU1A: double2 array which contains the gradients of the particle in theta,phi
	 *	U2: double3 array with the energy when the particle is perturbed in x,y,z
	 *	U2A: double2 array with the energy when the particle is perturbed in theta,phi
	 *	U1: array with the unperturbed energies
	 *
	 * Returns: This function takes the energies calculated in GradKernel2 and uses them to make the gradient vectors
	 */
	
	int tx=threadIdx.x+blockIdx.x*blockDim.x;
	
	double d=.0001;

	GU0[tx]=GU1[tx];
	GU0A[tx]=GU1A[tx];

	GU1[tx].x=(U2[tx].x-U1[tx])/d;
	GU1[tx].y=(U2[tx].y-U1[tx])/d;
	GU1[tx].z=(U2[tx].z-U1[tx])/d;
	GU1A[tx].x=(U2A[tx].x-U1[tx])/d;
	GU1A[tx].y=(U2A[tx].y-U1[tx])/d;
	
	//printf("U1[%i]: %E\n",tx,U1[tx]);
	//printf("GU1[%i]: %E %E %E\n",tx,GU1[tx].x,GU1[tx].y,GU1[tx].z);
	//printf("U2[%i]: %lf %lf %lf\n",tx,U2[tx].x,U2[tx].y,U2[tx].z);
	//printf("GU1A[%i]: %E %E\n",tx,GU1A[tx].x,GU1A[tx].y);
	//printf("U2A[%i]: %E %E\n",tx,U2A[tx].x,U2A[tx].y);

}

/*void Gradient2() {
	dim3 block, thread;
	
	double* U1;
	double3* U2;
	double2* U2A;
	U1=(double*)malloc(npart*sizeof(double));
	U2=(double3*)malloc(npart*sizeof(double3));
	U2A=(double2*)malloc(npart*sizeof(double2));

	if(npart<1024) {
		thread.x=npart;
		block.x=6*npart;
	}
	else {
		thread.x=1024;
		block.x=6*npart*npart/1024;
	}
	
	int i;
	for(i=0;i<npart;i++) {
		sptoca(i);
	}

	HANDLE_ERROR(cudaMemcpy(r_dev,r,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(u_dev,u,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(l_dev,l,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(sigma_dev,sigma,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(r1_dev,r1,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(r2_dev,r2,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemset(U1_dev,0,npart*sizeof(double)));
	HANDLE_ERROR(cudaMemset(U2_dev,0,npart*sizeof(double3)));
	HANDLE_ERROR(cudaMemset(U2A_dev,0,npart*sizeof(double2)));
	GradKernel2<<<block,thread>>>(r_dev,u_dev,l_dev,sigma_dev,r1_dev,r2_dev,U2_dev,U2A_dev,U1_dev,params);
	cudaError_t error;
	error=cudaGetLastError();
	if(cudaSuccess != error) {
		printf("%s\n",cudaGetErrorString(error));
	}
	HANDLE_ERROR(cudaMemcpy(U1,U1_dev,npart*sizeof(double),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(U2,U2_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(U2A,U2A_dev,npart*sizeof(double2),cudaMemcpyDeviceToHost));

	double d=.0001;
	for(i=0;i<npart;i++) {
		GU1[i].x=(U2[i].x-U1[i])/d;
		GU1[i].y=(U2[i].y-U1[i])/d;
		GU1[i].z=(U2[i].z-U1[i])/d;
		GU1A[i].x=(U2A[i].x-U1[i])/d;
		GU1A[i].y=(U2A[i].y-U1[i])/d;
	}
}*/

void Gradient3() {
	dim3 block, thread;

	HANDLE_ERROR(cudaMemset(U1_dev,0,npart*sizeof(double)));
	HANDLE_ERROR(cudaMemset(U2_dev,0,npart*sizeof(double3)));
	HANDLE_ERROR(cudaMemset(U2A_dev,0,npart*sizeof(double2)));

	if(npart<1024) {
		thread.x=npart;
		block.x=6*npart;
	}
	else {
		thread.x=1024;
		block.x=6*npart*npart/1024;
	}

	//sptocadev_all<<<blocks,threads>>>(theta_dev,u_dev);
	GradKernel2<<<block,thread>>>(r_dev,u_dev,theta_dev,l_dev,sigma_dev,r1_dev,r2_dev,U2_dev,U2A_dev,U1_dev,GU1_dev,GU0_dev,GU1A_dev,GU0A_dev,params);
	//GradKernel1<<<block,thread>>>(r_dev,u_dev,l_dev,sigma_dev,r1_dev,r2_dev,U2_dev,U2A_dev,U1_dev,params);
	GradKernel3<<<blocks,threads>>>(GU1_dev,GU1A_dev,GU0_dev,GU0A_dev,U2_dev,U2A_dev,U1_dev);
	HANDLE_ERROR(cudaMemcpy(GU1,GU1_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(GU1A,GU1A_dev,npart*sizeof(double2),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(GU0,GU0_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(GU0A,GU0A_dev,npart*sizeof(double2),cudaMemcpyDeviceToHost));
}


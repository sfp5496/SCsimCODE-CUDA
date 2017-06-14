#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "myhelpers.h"
#include "Globalvars.h"
#include "d2func.h"
#include "d3func.h"
#include "Enfunc.h"
#include "Gradfunc.h"

/*
 * The Conjugate Gradient Method:
 * -------------------------------
 * eta is some scalar
 * h is a vector
 * gamma is a scalar
 * r is our displacement vector
 *--------------------------------
 * h_0 = del(U_0)
 * gamma_(i-1) = dotp(del(U_i),del(U_i))/dotp(del(U_(i-1)),del(U_(i-1)))
 * h_i = del(U_i) + gamma_(i-1)*h_(i-1)
 * r_(i+1) = r_i + eta*h_i
 */

/*__global__ void ConGradKernel(double3* r,double3* h,double3* GU1,double2* theta,double2* hA,double2* GU1A,double* params) {
	*
	 * Arguments:
	 * 	r: array of position vectors
	 *	h: some vector we can use in the conjugate gradient method
	 *	GU1: array of gradients (in the cartesian directions)
	 *	theta: array of orientation angles
	 *	hA: the angular version of h
	 *	GU1A: array of gradients (orientation angles)
	 *
	 * Returns: Does the first iteration of the conjugate gradient method
	 *
	int tx=threadIdx.x+blockIdx.x*blockDim.x;

	if(iter==0) {

		int npart=((int)params[0]);
		double eta,etaA=0.0;
		int i;
		for(i=0;i<npart;i++) {
			eta+=d3mag(GU1[i]);
		}
		eta=-fabs(eta);
		if(fabs(eta)>1.0) {
			eta=-1.0;
		}
		else if(fabs(eta)<.001) {
			eta=-.001;
		}
	
		h[tx]=GU1[tx];
	
		double3 dr;
		dr=d3multscal(h[tx],eta);
		
		if(fabs(d3mag(dr))>0.5*params[1]) {
			dr=d3multscal(dr,0.5*params[1]/d3mag(dr));
		}
	
		r[tx]=d3add(r[tx],dr);
	
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
		hA[tx]=GU1A[tx];
		theta[tx]=d2add(theta[tx],d2multscal(hA[tx],etaA));

	}
	else {
		int npart=((int)params[0]);
		double eta=0.0;
		double etaA=0.0;
		int i;
	
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

	}

	//printf("r_dev[%i]: %lf %lf %lf\n",tx,h[tx].x,h[tx].y,h[tx].z);
	//printf("r_dev[%i]: %lf %lf %lf\n",tx,r[tx].x,r[tx].y,r[tx].z);
	//printf("theta_host[%i]: %lf %lf\n",tx,theta[tx].x,theta[tx].y);
	//printf("etaA[%i]: %lf\n",tx,etaA);
	//printf("dr[%i]: %lf %lf %lf\n",tx,dr.x,dr.y,dr.z); 
}
*/

__global__ void ConGradKernel(double3* r,double3* h,double3* GU1,double3* GU0,double2* theta,double2* hA,double2* GU1A,double2* GU0A,double* params) {
	/*
	 * Arguments:
	 * 	r: array of position vectors
	 *	h: some vector we can use in the conjugate gradient method
	 *	GU1: array of gradients (in the cartesian directions)
	 *	theta: array of orientation angles
	 *	hA: the angular version of h
	 *	GU1A: array of gradients (orientation angles)
	 *
	 * Returns: Does one iteration of the conjugate gradient method (ConGradKernel1 is the first iteration though)
	 */
	int tx=threadIdx.x+blockIdx.x*blockDim.x;

	int npart=((int)params[0]);
	double eta=0.0;
	double etaA=0.0;
	int i;

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

void ConGrad1() {
	double eta,etaA;
	eta=0.0;
	etaA=0.0;
	Gradient3();
	eta=-fabs(GradSum());
	if(fabs(eta)>1.0) {
		eta=-1.0;
	}
	else if(eta<.001) {
		eta=-.001;
	}

	etaA=-fabs(GradSumA());
	if(fabs(etaA)>1.0) {
		eta=-1.0;
	}
	else if(fabs(etaA)<.001) {
		etaA=-.001;
	}

	int i;
	for(i=0;i<npart;i++) {
		h_host[i]=GU1[i];
		r[i]=d3add(r[i],d3multscal(h_host[i],eta));

		hA_host[i]=GU1A[i];
		theta[i]=d2add(theta[i],d2multscal(hA_host[i],etaA));
	}
}

void ConGrad2() {
	double eta=0.0;
	double etaA=0.0;
	double gamma;
	double gammaA;

	Gradient3();

	eta=-fabs(GradSum());
	if(fabs(eta)>1.0) {
		eta=-1.0;
	}
	else if(fabs(eta)<.001) {
		eta=-.001;
	}
	etaA=-fabs(GradSumA());
	if(fabs(etaA)>1.0) {
		eta=-1.0;
	}
	else if(etaA<.001) {
		etaA=-.001;
	}

	int i;
	for(i=0;i<npart;i++) {
		if(d3dotp(GU0[i],GU0[i])!=0.0) {
			gamma=d3dotp(GU1[i],GU1[i])/d3dotp(GU0[i],GU0[i]);
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
		
		h_host[i].x=GU1[i].x+gamma*h_host[i].x;
		h_host[i].y=GU1[i].y+gamma*h_host[i].y;
		h_host[i].z=GU1[i].z+gamma*h_host[i].z;
		double3 dr;
		dr=d3multscal(h_host[i],eta);
		if(fabs(d3mag(dr))>0.5*R) {
			dr=d3multscal(dr,0.5*R/d3mag(dr));
		}
		r[i]=d3add(r[i],dr);

		etaA=-fabs(GradSumA());
		if(fabs(etaA)>1.0) {
			etaA=-1.0;
		}
		else if(fabs(etaA)<.001) {
			etaA=-.001;
		}
		
		if(d2mag(GU0A[i])!=0) {
			gammaA=d2dotp(GU1A[i],GU1A[i])/d2dotp(GU0A[i],GU0A[i]);
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
	
		hA_host[i]=d2add(GU1A[i],d2multscal(hA_host[i],gammaA));
		theta[i]=d2add(theta[i],d2multscal(hA_host[i],etaA));
	
		//printf("theta_host[%i]: %lf %lf\n",i,theta[i].x,theta[i].y);
		//printf("hA_host[%i]: %lf %lf\n",i,hA_host[i].x,hA_host[i].y);
		//printf("etaA[%i]: %lf\n",i,etaA);
		//printf("GradSumA(): %lf\n",GradSumA());
		//printf("gammaA[%i]: %lf\n",i,gammaA);
		//printf("GU1A[%i]: %lf %lf\n",i,GU1A[i].x,GU1A[i].y);
		//printf("GU0A[%i]: %lf %lf\n",i,GU0A[i].x,GU0A[i].y);
	}
}


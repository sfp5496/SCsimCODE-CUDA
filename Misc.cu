#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "myhelpers.h"
#include "d2func.h"
#include "d3func.h"
#include "Globalvars.h"
#include "Enfunc.h"
#include "Gradfunc.h"
#include "ConGradfunc.h"

double unitrand() {
	/*
	 * Arguments: none
	 * Returns: a "random" number between 0 and 1
	 */
	return (((double)rand())/((double)(RAND_MAX)));
}

void initcond(double3 *x, double2 *ang, double *len, double *diam) {
	/*
	 * Arguments: the 3d position array, the 2d angular orientation array, the length array, and
	 *  the diameter array
	 * Returns: generates an initial random packing for our system, the position array will be in 
	 *  cylindrical coordinates and will need to be converted to cartesian later
	 */
	int i;
	srand(time(NULL));
	for(i=0;i<npart;i++) {
		x[i].x=R*unitrand();
		x[i].y=2*M_PI*unitrand();
		x[i].z=H*unitrand();
		len[i]=0.0;
		diam[i]=0.0;
		ang[i].x=unitrand()*2*M_PI;
		ang[i].y=unitrand()*2*M_PI;
	}
}

double packfrac() {
	/*
	 * Arguments: none
	 * Returns: calculates and returns the packing fraction (currently only works if all particles
	 *  are the same size, if you want to fix that uncomment the for loop and don't multiply
	 *  Vp by npart and make that "=" a "+=")
	 */
	
	double Vp=0.0;
	double Vb=M_PI*R*R*H;
	int i=0;
	//for(i=0;i<npart;i++) {
	Vp=npart*(M_PI*(sigma[i]/2.0)*(sigma[i]/2.0)*l[i]+(4.0/3.0)*M_PI*(sigma[i]/2.0)*(sigma[i]/2.0)*(sigma[i]/2.0));
	//}
	return Vp/Vb;
}

void updatephi() {
	/*
	 * Arguments: none
	 * Returns: updates our l and sigma values based on the current value of our global variable
	 *  phi, which represents our packing fraction (used to more easily iterate through packing
	 *  fractions rather than iterate through l and sigma values)
	 */
	double Vcont;
	double Vpart;
	if(CUBE==1) {
		Vcont=LENGTH*WIDTH*HEIGHT;
	}
	else {
		Vcont=M_PI*R*R*H;
	}
	Vpart=phi*Vcont;
	int i;
	for(i=0;i<npart;i++) {
		sigma[i]=pow((Vpart/npart)/(M_PI/6.0+M_PI*ALPHA/4.0),1.0/3.0);
		l[i]=ALPHA*sigma[i];
	}
}

__global__ void updatephikernel(double* l, double* sigma, double* params, double* phi) {
	int tx=threadIdx.x+blockIdx.x*blockDim.x;
	
	double Vcont, Vpart;

	int npart;
	double R,H,ALPHA;

	npart=(int)params[0];
	R=params[1];
	H=params[2];
	ALPHA=params[3];

	Vcont=M_PI*R*R*H;
	Vpart=phi[0]*Vcont;

	sigma[tx]=pow((Vpart/npart)/(M_PI/6.0+M_PI*ALPHA/4.0),1.0/3.0);
	l[tx]=ALPHA*sigma[tx];
}

__global__ void variabledphi(double* params, double U) {
	if(U<1e-9) {
		params[4]=params[4]*1.1;
	}
	else {
		params[4]=params[4]*0.9;
	}
}

__global__ void DeviceAdd(int i, double* a, double b) {
	a[i]+=b;
}

double Pressure() {
	/*
	 * Arguments: none
	 * Returns: calculates and returns the total pressure that the particles exert on the walls
	 */
	int i;
	double3 F; //F.x is the force on the radial wall, F.y is on the top wall, F.z -> bottom
	double3 P;
	F=d3null();
	P=d3null();
	for(i=0;i<npart;i++) {
		sptoca(i);
		ends(i);
		if(r1[i].x>R-(sigma[i]/2.0)) {
			F.x+=r1[i].x-(R-sigma[i]/2.0);
		}
		if(r2[i].x>R-(sigma[i]/2.0)) {
			F.x+=r2[i].x-(R-sigma[i]/2.0);
		}
		if(r1[i].z-sigma[i]/2.0<0.0) {
			F.z+=fabs(r1[i].z-sigma[i]/2.0);

		}
		if(r1[i].z+sigma[i]/2.0>H) {
			F.y+=r1[i].z+sigma[i]/2.0-H;
		}
		if(r2[i].z-sigma[i]/2.0<0.0) {
			F.z+=fabs(r2[i].z-sigma[i]/2.0);
		}
		if(r2[i].z+sigma[i]/2.0>H) {
			F.y+=r2[i].z+sigma[i]/2.0-H;
		}
	}
	P.x=F.x/(2.0*M_PI*R*H);
	P.y=F.y/(M_PI*R*R);
	P.z=F.z/(M_PI*R*R);
	return sqrt(P.x*P.x)+sqrt(P.y*P.y)+sqrt(P.z*P.z);
}

double contacts() {
	/*
	 * Arguments: none
	 * Returns: the average number of other particles a particle is currently in contact with, this
	 *  value doesn't really mean much anymore
	 */
	int i;
	int j;
	double ret=0.0;
	for(i=0;i<npart;i++) {
		for(j=0;j<npart;j++) {
			sptoca(i);
			ends(i);
			if((i!=j) && (d3dist(r[i],r[j])<(l[i]+sigma[i]+l[j]+sigma[j])/2.0)) {
				double lambda_i, lambda_j;
				lambda_i=lambda(r[i],r[j],u[i],u[j],l[i]);
				lambda_j=lambda(r[j],r[i],u[j],u[i],l[j]);
				double d;
				d=d3SCdist(r[i],r[j],u[i],u[j],lambda_i,lambda_j);
				if(d<(sigma[i]+sigma[j])/2.0) {
					ret+=1.0;
				}
			}
		}
		sptoca(i);
		ends(i);
		if(r1[i].x>R-(sigma[i]/2.0)) {
			ret+=1.0;
		}
		if(r2[i].x>R-(sigma[i]/2.0)) {
			ret+=1.0;
		}
		if(r1[i].z-sigma[i]/2.0<0.0) {
			ret+=1.0;
		}
		if(r1[i].z+sigma[i]/2.0>H) {
			ret+=1.0;
		}
		if(r2[i].z-sigma[i]/2.0<0.0) {
			ret+=1.0;
		}
		if(r2[i].z+sigma[i]/2.0>H) {
			ret+=1.0;
		}
	}
	double U=collider();
	if(sqrt((U/npart)*2.0)/(sigma[0]/2.0)<.05) {
		return 0;
	}
	else {
		return ret;
	}
}



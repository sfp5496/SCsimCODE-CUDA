/*
 * Author: Sean Peterson
 * Advisor: Dr. Scott Franklin
 * Program: SCsim.c
 * What it does: randomly generates npart spherolines within a cylindrical container then proceeds
 *  to grow them from volume zero until they fill their container
 *
 *  Inputs:
 *  argv[1]: name of the directory where you want to store packing files
 *  argv[2]: parameter file
 *  argv[3]: file which contains an initial packing
 *
 *  Parameter Files:
 *  the files should include the following values in order
 *  npart R H ETA DPHI PHI ALPHA CUBE LENGTH WIDTH HEIGHT
 *  
 *  npart: number of particles
 *  R: radius of container, which is a cylinder
 *  H: height of container, which is a cylinder
 *  ETA: used to be a scalar used in the conjugate gradient method but isn't used anymore
 *  DPHI: the increase in our packing fraction each iteration
 *  PHI: the packing fraction that the simulation ends at
 *  ALPHA: the aspect ratio of our particles
 *  CUBE: if set to 1, our container is a cube instead of a cylinder (WIP)
 *  LENGTH: length of the cube (WIP)
 *  WIDTH: width of the cube (WIP)
 *  HEIGHT: height of the cube (WIP)
 *
 *  Packing Files:
 *  iter pid r[i].x r[i].y r[i].z u[i].x u[i].y u[i].z
 *  iter: the current iteration number of the program
 *  pid: particle ID, just tells you where its data is in the arrays
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "myhelpers.h"
#include "Globalvars.h"
#include "d2func.h"
#include "d3func.h"
#include "Enfunc.h"
#include "Gradfunc.h"
#include "ConGradfunc.h"
#include "Misc.h"

void start() {
	/*
	 * Arguments: none
	 * Returns: allocates the necessary memory for our arrays and uses initcond() for to generate
	 *  the initial packing and converts it to cartesian coordinates
	 */
	
	cudaSetDevice(1);

	blocks.x=npart/32; //number of blocks when I do a 1D CUDA call
	threads.x=32; //number of threads per block when I do a 1D CUDA call

	grid.x=npart/32; //block dimension in x when I do a 2D CUDA call
	grid.y=npart/32; //block dimension in y when I do a 2D CUDA call
	grid.z=1; //block dimension in z when I do a 2D CUDA call
	block.x=32; //threads per block in x
	block.y=32; //threads per block in y
	block.z=1; //threads per block in z

	r=(double3*)malloc(npart*sizeof(double3));
	HANDLE_ERROR(cudaMalloc(&r_dev,npart*sizeof(double3)));
	r1=(double3*)malloc(npart*sizeof(double3));
	HANDLE_ERROR(cudaMalloc(&r1_dev,npart*sizeof(double3)));
	r2=(double3*)malloc(npart*sizeof(double3));
	HANDLE_ERROR(cudaMalloc(&r2_dev,npart*sizeof(double3)));
	u=(double3*)malloc(npart*sizeof(double3));
	HANDLE_ERROR(cudaMalloc(&u_dev,npart*sizeof(double3)));
	theta=(double2*)malloc(npart*sizeof(double2));
	HANDLE_ERROR(cudaMalloc(&theta_dev,npart*sizeof(double2)));
	l=(double*)malloc(npart*sizeof(double));
	HANDLE_ERROR(cudaMalloc(&l_dev,npart*sizeof(double)));
	sigma=(double*)malloc(npart*sizeof(double));
	HANDLE_ERROR(cudaMalloc(&sigma_dev,npart*sizeof(double)));
	GU1=(double3*)malloc(npart*sizeof(double3));
	GU1A=(double2*)malloc(npart*sizeof(double2));
	GU0=(double3*)malloc(npart*sizeof(double3));
	GU0A=(double2*)malloc(npart*sizeof(double2));
	rc=(double3*)malloc(npart*sizeof(double3));
	initcond(rc,theta,l,sigma);
	int i;
	for(i=0;i<npart;i++) {
		r[i].x=rc[i].x*cos(rc[i].y);
		r[i].y=rc[i].x*sin(rc[i].y);
		r[i].z=rc[i].z;
		sptoca(i);
	}

	HANDLE_ERROR(cudaMemcpy(r_dev,r,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(theta_dev,theta,npart*sizeof(double2),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(l_dev,l,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(sigma_dev,sigma,npart*sizeof(double),cudaMemcpyHostToDevice));

	h_host=(double3*)malloc(npart*sizeof(double3));
	hA_host=(double2*)malloc(npart*sizeof(double2));

	HANDLE_ERROR(cudaMalloc(&phi_dev,sizeof(double)));

	phi=packfrac();

	HANDLE_ERROR(cudaMalloc(&U_dev,sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&params,6*sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&U2_dev,npart*sizeof(double3)));
	HANDLE_ERROR(cudaMalloc(&U2A_dev,npart*sizeof(double2)));
	HANDLE_ERROR(cudaMalloc(&U1_dev,npart*sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&GU1_dev,npart*sizeof(double3)));
	HANDLE_ERROR(cudaMalloc(&GU1A_dev,npart*sizeof(double2)));
	HANDLE_ERROR(cudaMalloc(&GU0_dev,npart*sizeof(double3)));
	HANDLE_ERROR(cudaMalloc(&GU0A_dev,npart*sizeof(double2)));
	double a;
	a=(double)npart;
	HANDLE_ERROR(cudaMemcpy(params,&a,sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(params+1,&R,sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(params+2,&H,sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(params+3,&ALPHA,sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(params+4,&DPHI,sizeof(double),cudaMemcpyHostToDevice));
}

void doAnIter2() {
	double3* h;
	double2* hA;
	HANDLE_ERROR(cudaMalloc(&h,npart*sizeof(double3)));
	HANDLE_ERROR(cudaMalloc(&hA,npart*sizeof(double2)));

	Gradient3();

	int iter=0;
	int s=0;
	double GS=GradSum();
	//ConGradKernel1<<<blocks,threads>>>(r_dev,h,GU1_dev,theta_dev,hA,GU1A_dev,params);
	//cudaError_t error;
	//error=cudaGetLastError();
	//if(cudaSuccess != error) {
	//	printf("%s\n",cudaGetErrorString(error));
	//}

	//ConGrad1(h_host,hA_host);
	//HANDLE_ERROR(cudaMemcpy(r_dev,r,npart*sizeof(double3),cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(u_dev,u,npart*sizeof(double3),cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(theta_dev,theta,npart*sizeof(double2),cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy(r,r_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(u,u_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(theta,theta_dev,npart*sizeof(double2),cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(h_host,h,npart*sizeof(double3),cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaMemcpy(hA_host,hA,npart*sizeof(double2),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemset(h,0,npart*sizeof(double3)));
	HANDLE_ERROR(cudaMemset(hA,0,npart*sizeof(double2)));
	HANDLE_ERROR(cudaMemset(GU0_dev,0,npart*sizeof(double3)));
	HANDLE_ERROR(cudaMemset(GU0A_dev,0,npart*sizeof(double2)));
	/*int i;
	for(i=0;i<npart;i++) {
		fprintf(stderr,"r[%i]: %lf %lf %lf\n",i,r[i].x,r[i].y,r[i].z);
	}*/
	//int q;
	//for(q=0;q<0;q++) {
	//while((GradSum()>1e-10) && (GradSumA()>1e-10)) {
	//while((GradSum()!=0.0) && (GradSumA()!=0.0)) {
	while((GradSum()>1e-9) && (GradSumA()>1e-7)) {
		//fprintf(stderr,"GradSum(): %E\n",GradSum());
		//fprintf(stderr,"GradSumA(): %E\n",GradSumA());
		iter++;
		//HANDLE_ERROR(cudaMemcpy(r,r_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
		//HANDLE_ERROR(cudaMemcpy(u,u_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
		
		ConGradKernel<<<blocks,threads>>>(r_dev,h,GU1_dev,GU0_dev,theta_dev,hA,GU1A_dev,GU0A_dev,params);
		//cudaError_t error;
		//error=cudaGetLastError();
		//if(cudaSuccess != error) {
		//printf("%s\n",cudaGetErrorString(error));
		//}
		Gradient3();
		
		//ConGrad2();
		
		//HANDLE_ERROR(cudaMemcpy(r_dev,r,npart*sizeof(double3),cudaMemcpyHostToDevice));
		//HANDLE_ERROR(cudaMemcpy(u_dev,u,npart*sizeof(double3),cudaMemcpyHostToDevice));
		//HANDLE_ERROR(cudaMemcpy(theta_dev,theta,npart*sizeof(double2),cudaMemcpyHostToDevice));
		if(iter>1000000) {
			fprintf(stderr,"BREAK(1)!\n");
			break;
		}
		if(fabs(GS-GradSum())<1e-7) {
			s++;
		}
		else {
			s=0;
		}
		
		if(s>10000) {
			fprintf(stderr,"BREAK(2)!\n");
			break;
		}
		GS=GradSum();

	}
	fprintf(stderr,"Iterations: %i\n",iter);
	fprintf(stderr,"GradSum(): %E\n",GradSum());
	fprintf(stderr,"GradSumA(): %E\n",GradSumA());
	HANDLE_ERROR(cudaMemcpy(r,r_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(u,u_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
	//double c=contacts();
	double p=packfrac();
	double U=collider2();
	//double P=Pressure();
	fprintf(stderr,"Rod Radius: %lf\n",sigma[0]/2.0);
	fprintf(stderr,"Rod Length: %lf\n",l[0]);
	fprintf(stderr,"Packing Fraction: %lf\n",p);
	//fprintf(stderr,"Average Number of Contacts: %lf\n",c/npart);
	fprintf(stderr,"Potential Energy: %E\n",U);
	fprintf(stderr,"Average Overlap: %lf\n",sqrt((U/npart)*2.0)/(sigma[0]/2.0));
	//fprintf(stderr,"Pressure on Walls: %lf\n",P);
	fprintf(stdout,"%E	%E	%E\n",p,U,sqrt((U/npart)*2.0)/(sigma[0]/2.0));
	phi+=DPHI; //update phi
	//variabledphi<<<1,1>>>(params,collider2());
	//DeviceAdd<<<1,1>>>(0,phi_dev,params[4]);
	updatephi(); //use the new phi value to get a new sigma and l
	//updatephikernel<<<blocks,threads>>>(l_dev,sigma_dev,params,phi_dev);
	HANDLE_ERROR(cudaMemcpy(l_dev,l,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(sigma_dev,sigma,npart*sizeof(double),cudaMemcpyHostToDevice));
}

void DEMend() {
	/*
	 * Arguments: none
	 * Returns: doesn't really return anything, just frees all the memory that we used during the
	 *  simulation
	 */
	free(r);
	free(u);
	free(theta);
	free(rc);
	free(GU1);
	free(GU1A);
	free(r1);
	free(r2);
	free(sigma);
	free(l);
}

int main(int argc, char* argv[]) {
	if(argc>2) { //check for a parameter file
		const char* infile;
		inputs=fopen(argv[2],"r");
		fscanf(inputs,"%d %lf %lf %lf %lf %lf %lf %d %lf %lf %lf",&npart,&R,&H,&ETA,&DPHI,&PHI,&ALPHA,&CUBE,&LENGTH,&WIDTH,&HEIGHT);
	}
	else {
		fprintf(stderr,"No Parameter File given.\n");
		return 1;
	}
	if(argc>1) { //check for a directory to put configuration files
		struct stat st={0};

		if(stat(argv[1],&st)==-1) {
			fprintf(stderr,"Creating Directory %s...\n",argv[1]);
			mkdir(argv[1],0777);
		}
		else {
			fprintf(stderr,"Directory %s Already Exists.\n",argv[1]);
		}
	}
	else {
		fprintf(stderr,"No directory given, if you want to print here use './'.\n");
		return 1;
	}
	char* filename;
	filename=(char*)malloc(20*sizeof(char));
	int j=0;
	start(); //allocate memory and generate an initial packing
	if(argc>3) { //Check for an initial packing file
		inputs=fopen(argv[3],"r");
		int k;
		int iter,pid; //just places to send the variables I don't need
		fprintf(stderr,"Loading initial packing.\n");
		for(k=0;k<npart;k++) {
			fscanf(inputs,"%d %d %lf %lf %lf %lf %lf %lf %lf %lf",&iter,&pid,sigma+k,l+k,&(r[k].x),&(r[k].y),&(r[k].z),&(u[k].x),&(u[k].y),&(u[k].z));
			theta[k].x=atan2(u[k].y,u[k].x);
			theta[k].y=atan(u[k].z);
		}

		HANDLE_ERROR(cudaMemcpy(r_dev,r,npart*sizeof(double3),cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(theta_dev,theta,npart*sizeof(double2),cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(l_dev,l,npart*sizeof(double),cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(sigma_dev,sigma,npart*sizeof(double),cudaMemcpyHostToDevice));
		//clock_t t0,t1;
		//t0=clock();
		double X=collider();
		double Y=collider2();
		fprintf(stderr,"POTENTIAL ENERGY: %E %E\n",X,Y);
		//Gradient();
		//fprintf(stderr,"GradSum(): %E\n",GradSum());
		//fprintf(stderr,"GradSumA(): %E\n",GradSumA());
		//Gradient3();
		//HANDLE_ERROR(cudaMemcpy(GU1,GU1_dev,npart*sizeof(double3),cudaMemcpyDeviceToHost));
		//HANDLE_ERROR(cudaMemcpy(GU1A,GU1A_dev,npart*sizeof(double2),cudaMemcpyDeviceToHost));
		//for(f=0;f<npart;f++) {
			//printf("GU1[%i]: %E %E %E\n",f,GU1[f].x,GU1[f].y,GU1[f].z);
		//}
		//fprintf(stderr,"GradSum(): %E\n",GradSum());
		//fprintf(stderr,"GradSumA(): %E\n",GradSumA());
		//phi=packfrac();
		//t1=clock();
		//fprintf(stderr,"total time: %E\n",t1);
	}
	else {
		fprintf(stderr,"No initial packing given...\n");
		fprintf(stderr,"Using randomly generated packing.\n");
	}
	//Gradient3();
	HANDLE_ERROR(cudaMemcpy(r_dev,r,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(u_dev,u,npart*sizeof(double3),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(l_dev,l,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(sigma_dev,sigma,npart*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(theta_dev,theta,npart*sizeof(double2),cudaMemcpyHostToDevice));

	while(phi<PHI) {
	//int m;
	//for(m=0;m<1;m++) {
		doAnIter2();
		int q;
		fprintf(stderr,"PRINT CHECK: %i\n",(int)(phi*1000)%10);
		if(((((int)(phi/DPHI))%(int)(.01/DPHI)))==0) {
			sprintf(filename,"%s/%03d.dat",argv[1],(int)(phi*100));
			fp=fopen((const char*)filename,"w");
			for(q=0;q<npart;q++) {
				fprintf(fp,"%i %i %lf %lf %lf %lf %lf %lf %lf %lf\n",j,q,sigma[q],l[q],r[q].x,r[q].y,r[q].z,u[q].x,u[q].y,u[q].z);
			}
			j++;
			fclose(fp);
		}
	}
	DEMend();
	return 0;
}

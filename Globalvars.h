#ifndef __GLOBALVARS_H__
#define __GLOBALVARS_H__

extern double3* r; //array of displacement vectors
extern double3* rc; //array of displacement vectors in cylindrical coords, used in initcond() only
extern double3* r1; //array of displacement vectors of one end of particles
extern double3* r2; //array of displacement vectors of the other end of particles
extern double3* u; //array of orientation vectors
extern double2* theta; //array of theta and phi orientation values
extern double* l; //array of particle lengths
extern double* sigma; //array of particle diameters
extern double3* GU1; //array of cartesian gradient vectors (of U)
extern double2* GU1A; //array of theta and phi gradients (of U)
extern double3* GU0;
extern double2* GU0A;
extern double phi; //packing fraction

extern double3* h_host;
extern double2* hA_host;

extern FILE* fp;
extern FILE* inputs;

extern dim3 blocks, threads;
extern dim3 grid, block;

//THESE VALUES ARE IMPORTED FROM A FILE AS THE PARAMETERS OF THE SIMULATION//

extern int npart;
extern double R;
extern double H;
extern double ETA;
extern double DPHI;
extern double PHI;
extern double ALPHA;
extern int CUBE;
extern double LENGTH;
extern double WIDTH;
extern double HEIGHT;

//THESE VALUES ARE SIMPLY COPIES OF SOME OF THE ABOVE VALUES TO BE STORED IN DEVICE MEMORY

extern double3* r_dev;
extern double3* u_dev;
extern double2* theta_dev;
extern double* l_dev;
extern double* sigma_dev;
extern double3* r1_dev;
extern double3* r2_dev;
extern double* params;
extern double* U_dev;
extern double* U1_dev;
extern double3* U2_dev;
extern double2* U2A_dev;
extern double3* GU1_dev;
extern double2* GU1A_dev;
extern double3* GU0_dev;
extern double2* GU0A_dev;

extern double* phi_dev;

#endif

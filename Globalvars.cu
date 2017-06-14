#include <stdlib.h>
#include <stdio.h>

double3* r; //array of displacement vectors
double3* rc; //array of displacement vectors in cylindrical coords, used in initcond() only
double3* r1; //array of displacement vectors of one end of particles
double3* r2; //array of displacement vectors of the other end of particles
double3* u; //array of orientation vectors
double2* theta; //array of theta and phi orientation values
double* l; //array of particle lengths
double* sigma; //array of particle diameters
double3* GU1; //array of cartesian gradient vectors (of U)
double2* GU1A; //array of theta and phi gradients (of U)
double3* GU0;
double2* GU0A;
double phi; //packing fraction

double3* h_host;
double2* hA_host;

FILE* fp;
FILE* inputs;

dim3 blocks, threads;
dim3 grid, block;

//THESE VALUES ARE IMPORTED FROM A FILE AS THE PARAMETERS OF THE SIMULATION//

int npart;
double R;
double H;
double ETA;
double DPHI;
double PHI;
double ALPHA;
int CUBE;
double LENGTH;
double WIDTH;
double HEIGHT;

//THESE VALUES ARE SIMPLY COPIES OF SOME OF THE ABOVE VALUES TO BE STORED IN DEVICE MEMORY

double3* r_dev;
double3* u_dev;
double2* theta_dev;
double* l_dev;
double* sigma_dev;
double3* r1_dev;
double3* r2_dev;
double* params;
double* U_dev;
double* U1_dev;
double3* U2_dev;
double2* U2A_dev;
double3* GU1_dev;
double2* GU1A_dev;
double3* GU0_dev;
double2* GU0A_dev;

double* phi_dev;

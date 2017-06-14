#ifndef __MYHELPERS_H__
#define __MYHELPERS_H__
#include <stdio.h>

//Handle Error stolen from CUDA By Example

static void HandleError( cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s:%d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif

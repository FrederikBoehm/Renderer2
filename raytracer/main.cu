#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void add(int a, int b, int* c) {
	*c = a + b;
}

void main() {
	int h_c;
	int* d_c;

	cudaMalloc((void**)&d_c, sizeof(int));

	add << <1, 1 >> > (2, 7, d_c);

	cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("2 + 7 = %d\n", h_c);

	cudaFree(d_c);

}
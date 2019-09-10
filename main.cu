#include "functions.cuh"
#include "ksvd.cuh"

#include <fstream>
#include <iostream>
#include <CImg.h>

using namespace std;
using namespace cimg_library;

__global__
void display_dictionary(data_t * I, data_t * D, lenght_t m, lenght_t n, lenght_t p)
{
	index_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < m * n)
	{
		index_t x = (i % n) * p;
		index_t y = (i / n) * p;
		
		D = & D[i * p * p];
		index_t k = 0;
		for(index_t a = x; a < x + p; a++)
		for(index_t b = y; b < y + p; b++)
			I[a + b * n] = D[k++];
	}
}

void test_image_denoising(char * image_file, lenght_t p, lenght_t m, lenght_t L)
{
	CImg<data_t> image(image_file);
	image.resize(128, 128);

	lenght_t rows = image.width();
	lenght_t columns = image.height();
	
	CImg<data_t> image_out(rows, columns);
	
	lenght_t T = rows * columns;
	lenght_t M = (rows - p + 1) * (columns - p + 1);
	lenght_t n = p * p;		//patch size
	
	data_t * h_I = image.data();
	data_t * h_IR = image_out.data();
	data_t * h_D = new data_t[n * m];
	
	data_t * d_D;
	data_t * d_I;
	data_t * d_IR;
	data_t * d_X;
	data_t * d_XR;
	data_t * d_alpha;
	index_t * d_aux;
	
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	cout<<"aprox mem: "<<((n * m + 2 * T + 2 * n * M + m * M + T) * sizeof(data_t)) / 1e6 <<" Mb"<<endl;
	cudaMalloc(& d_D, n * m * sizeof(data_t));
	cudaMalloc(& d_I, T * sizeof(data_t));
	cudaMalloc(& d_IR, T * sizeof(data_t));
	cudaMalloc(& d_X, n * M * sizeof(data_t));
	cudaMalloc(& d_XR, n * M * sizeof(data_t));
	cudaMalloc(& d_alpha, m * M * sizeof(data_t));
	cudaMalloc(& d_aux, T * sizeof(lenght_t));
	
	fill_matrix(h_D, n * m);
	
	cudaMemcpy(d_D, h_D, n * m * sizeof(data_t), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_I, h_I, T * sizeof(data_t), cudaMemcpyHostToDevice);

	divide_image<<< D_NB(dim3(rows, columns)), D_NT >>>(d_I, d_X, rows, columns, n, p);
	cudaDeviceSynchronize();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;	
	cudaEventRecord(start);
	
	KSVD(d_X, d_D, d_alpha, n, m, M, L, L);

	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<milliseconds/1000<<" s"<<endl;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	MP(handle, d_alpha, d_D, d_X, n, m, M, L);
	
	gpu_blas_mmul(handle, d_D, d_alpha, d_XR, n, m, M);
	
	cudaMemset(d_IR, 0, T * sizeof(data_t));
	cudaMemset(d_aux, 0, T * sizeof(index_t));
	
	sum_image_patches<<<D_NB(dim3(rows, columns)), D_NT >>>(d_IR, d_XR, d_aux, rows, columns, n, p);
	cudaDeviceSynchronize();
	
	divide_matrix<<< NB(T), NT >>>(d_IR, d_aux, T);
	cudaMemcpy(h_IR, d_IR, T * sizeof(data_t), cudaMemcpyDeviceToHost);

	//print_matrix<data_t>(h_IR, t, t);

	data_t * d_DI;
	cudaMalloc(& d_DI, m * n * sizeof(data_t));
	CImg<data_t> dictionary(16 * p, 16 * p);
	display_dictionary<<<NB(m), NT>>>(d_DI, d_D, 16, 16, p);
	cudaMemcpy(dictionary.data(), d_DI, m * n * sizeof(data_t), cudaMemcpyDeviceToHost);

	CImg<data_t> diff = (image - image_out).abs();
	(image, image_out, diff, dictionary).display();

	cudaFree(d_D);
	cudaFree(d_I);
	cudaFree(d_IR);
	cudaFree(d_X);
	cudaFree(d_XR);
	cudaFree(d_alpha);
	cudaFree(d_aux);
	
	// Destroy the handle
	cublasDestroy(handle);
	
	delete [] h_D;
}

int main(int nargs, char ** args)
{
	test_image_denoising(args[1], 8, 256, 10);
	return 0;
}

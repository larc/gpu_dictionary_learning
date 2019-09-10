#include "functions.cuh"

#include <iostream>
#include <fstream>

using namespace std;

__global__
void substract_matrix(data_t * A, data_t * B, data_t * C, lenght_t n)
{
	index_t id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < n)
		C[id] = A[id] - B[id];
}

__global__
void divide_matrix(data_t * A, int * B, lenght_t n)
{
	index_t id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < n)
		A[id] = A[id] / B[id];
}

__global__
void divide_image(data_t * I, data_t * X, lenght_t rows, lenght_t colums, lenght_t n, lenght_t p)
{
	index_t x = blockDim.x * blockIdx.x + threadIdx.x;
	index_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x < rows - p + 1 && y < colums - p + 1)
	{
		index_t i = x + y * (rows - p + 1);
		index_t k = 0;
		X = & X[i * n];

		for (index_t b = y; b < y + p; ++b)
		{
			for (index_t a = x; a < x + p; ++a)
			{
				X[k] = I[a + b * rows];
				k++;
			}
		}
	}
}

__global__
void sum_image_patches(data_t * I, data_t * X, index_t * C, lenght_t rows, lenght_t colums, lenght_t n, lenght_t p)
{
	index_t x = blockDim.x * blockIdx.x + threadIdx.x;
	index_t y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x < rows && y < colums)
	{
		index_t idx = x + y * rows;
		index_t a = x - p + 1;
		index_t b = y - p + 1;
	
		index_t px, py, pp, pi;

		px = p;
		for(index_t i = a; i < a + p; i++)
		{
			px = px - 1;
			py = p;
		
			for(index_t j = b; j < b + p; j++)
			{
				py = py - 1;
				if(i >= 0 && j >= 0 && i < (rows - p + 1) && j < (colums - p + 1))
				{
					pp = i + j * (rows - p + 1);
					pi = px + py * p;
					I[idx] += X[pi + pp * n];
					C[idx] ++;
				}
			}
		}
	}
}

void gpu_blas_mmul(cublasHandle_t & handle, data_t * A, data_t * B, data_t * C, int m, int k, int n, bool transpose)
{	
	data_t alf = 1;
	data_t bet = 0;
	data_t * alpha = &alf;
	data_t * beta = &bet;
	
	// Do the actual multiplication
	if(transpose)
		cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, A, k, B, k, beta, C, m);
	else
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, m, B, k, beta, C, m);
}

void fill_matrix(data_t * A, lenght_t n)
{
	srand (time(NULL));
	
	for(index_t i = 0; i < n; i++)
	{
		A[i] = rand() % int(1E6);
		A[i] /= 1E6;
	}
}

void read_image(data_t * A, const char * file, lenght_t nr_rows_A, lenght_t nr_cols_A)
{
	ifstream lee(file);

	for(index_t i = 0; i < nr_rows_A; ++i)
		for(index_t j = 0; j < nr_cols_A; ++j)
			lee>>A[i + j * nr_rows_A];

	lee.close();
}


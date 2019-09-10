#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH

#define NT 16
#define NB(d) ( (d + NT - 1) / NT )
#define D_NT dim3(NT, NT)
#define D_NB(d) dim3( (d.x + D_NT.x - 1) / D_NT.x, (d.y + D_NT.y - 1) / D_NT.y )

#include <cublas_v2.h>
#include <iostream>

typedef int index_t;
typedef int lenght_t;
typedef float data_t;

__global__
void substract_matrix(data_t * A, data_t * B, data_t * C, lenght_t n);

__global__
void divide_matrix(data_t * A, int * B, lenght_t n);

__global__
void divide_image(data_t * I, data_t * X, lenght_t rows, lenght_t colums, lenght_t n, lenght_t p);

__global__
void sum_image_patches(data_t * I, data_t * X, lenght_t * aux, lenght_t rows, lenght_t colums, lenght_t n, lenght_t p);

void gpu_blas_mmul(cublasHandle_t & handle, data_t * A, data_t * B, data_t * C, int m, int k, int n, bool transpose = false);

void fill_matrix(data_t * A, lenght_t n);

void read_image(data_t * A, const char * file, lenght_t nr_rows_A, lenght_t nr_cols_A);

template <typename D>
void print_matrix(D * A, lenght_t nr_rows_A, lenght_t nr_cols_A)
{
	std::cout.precision(16);
	for(index_t i = 0; i < nr_rows_A; ++i)
	{
		for(index_t j = 0; j < nr_cols_A; ++j)
			std::cout << A[i + j * nr_rows_A] << " ";
		std::cout << std::endl;
	}
	
	std::cout << std::endl;
}

#endif


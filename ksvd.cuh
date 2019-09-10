#ifndef KSVD_CUH
#define KSVD_CUH

#include "omp.cuh"

#include <cstdio>
#include <cusolverDn.h>

// sum: E_{j_0} =  Y - \sum_{j \not = j_0} a_j * x^T_j; E (n \times M)
__global__
void sum(float * A, float * X, float * E, int n, int m, int M, int j0)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int i = x + y * n;

	if(x < n && y < M)
		E[i] = E[i] - A[x + j0 * n] * X[j0 + y * m];
}

__global__
void select_omega(bool * O, data_t * alpha, lenght_t m, lenght_t M, lenght_t j, lenght_t * p)
{
	lenght_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < M && abs(alpha[j + i * m]) > 1E-5)
	{
		O[i] = true;
		atomicAdd(p, 1);
	}
}

__global__
void select_columns(data_t * ER, data_t * E, bool * O, lenght_t n, lenght_t M, lenght_t p)
{
	lenght_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < n)
		for(lenght_t j = 0, k = 0; k < p && j < M; j++)
			if(O[j])
			{
				ER[i + k * n] = E[i + j * n];
				k++;
			}
}

// data Y_{n \times M}
// dictionary A_{n \times m}(normalized) random... (initial) input and output
// K: number of iterations
// L: parameter OMP,... sparse length
void KSVD(data_t * d_X, data_t * d_D, data_t * d_alpha, lenght_t n, lenght_t m, lenght_t M, lenght_t K, lenght_t L)
{
	bool * d_O;			//omega, selected columns
	data_t * d_E;		// E_{j_o}
	data_t * d_S;		// result of sum_{j \not= j_0} a_j * x^T_j
	data_t * d_s;
	data_t * d_U;
	data_t * d_V;
	data_t * d_ER;

	cudaMalloc(&d_O, M * sizeof(bool));
	cudaMalloc(&d_E, n * M * sizeof(data_t));
	cudaMalloc(&d_S, n * M * sizeof(data_t));
	cudaMalloc(&d_s, n * sizeof(data_t));			//*****
	cudaMalloc(&d_U, n * n * sizeof(data_t));		// SVD	
	cudaMalloc(&d_V, n * n * sizeof(data_t));		//*****
	cudaMalloc(&d_ER, n * n * sizeof(data_t));

	lenght_t p;				// size of V (p * p)
	lenght_t * d_p;
	cudaMalloc(&d_p, sizeof(lenght_t));

	cusolverDnHandle_t cusolver_handle;
	cusolverDnCreate(&cusolver_handle);

	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	data_t milliseconds = 0;

	data_t * h_D = new data_t[n * m];
	data_t * h_S = new data_t[n * M];
	data_t * h_alpha = new data_t[m * M];

	while(K--)
	{
		MP(cublas_handle, d_alpha, d_D, d_X, n, m, M, L);
	
		for(lenght_t j = 0; j < m; j++)
		{
			// omega_{j_0} = {i | 1 <= i <= M, alpha[j_0, i] \not = 0}
			cudaMemset(d_O, 0, M * sizeof(bool));
			cudaMemset(d_p, 0, sizeof(lenght_t));

			select_omega<<< NB(M), NT >>>(d_O, d_alpha, m, M, j, d_p);
			cudaMemcpy(&p, d_p, sizeof(lenght_t), cudaMemcpyDeviceToHost);

			// sum: E_{j_0} =  Y - \sum_{j \not = j_0} a_j * x^T_j; E (n \times M)
			gpu_blas_mmul(cublas_handle, d_D, d_alpha, d_S, n, m, M);

			sum<<< D_NB(dim3(n, M)), D_NT >>>(d_D, d_alpha, d_S, n, m, M, j);
			cudaDeviceSynchronize();	

			substract_matrix<<< NB(n * M), NT >>>(d_X, d_S, d_E, n * M);
			cudaDeviceSynchronize();

			p = p > n ? n : p;
			
			if(p)
			{
				// E_{j_0}^R = UsV^T
				// ER = select \omega form E
				select_columns<<< NB(n), NT >>>(d_ER, d_E, d_O, n, M, p);
				cudaDeviceSynchronize();

				//SVD
				float * buffer = NULL;
				int * devInfo = NULL;
				int Lwork = 0;

				cusolverDnSgesvd_bufferSize(cusolver_handle, n, p, &Lwork );

				cudaMalloc(&buffer, sizeof(data_t) * Lwork);
				cudaMalloc(&devInfo, sizeof(lenght_t));
				
				cusolverDnSgesvd(cusolver_handle, 'A', 'A', n, p, d_ER, n, d_S, d_U, n, d_V, p, buffer, Lwork, buffer, devInfo);	

				cudaFree(buffer);
				cudaFree(devInfo);
				//copy colum U_0 to D_j0
				cudaMemcpy(& d_D[j * n], d_U, n * sizeof(data_t), cudaMemcpyDeviceToDevice);
				//cudaMemcpy(& d_X[j0 * n], d_V, n * sizeof(data_t), cudaMemcpyDeviceToDevice);
			}
		}
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	//cout<<"Time: "<<milliseconds<<endl;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cusolverDnDestroy(cusolver_handle);
	cublasDestroy(cublas_handle);
	
	cudaFree(d_O);
	cudaFree(d_E);
	cudaFree(d_S);
	cudaFree(d_s);
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(d_ER);
}


#endif


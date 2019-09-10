#ifndef OMP_CUH
#define OMP_CUH

#include "functions.cuh"

__global__
void MP_patch(data_t * D, data_t * DtX, data_t * R, data_t * alpha, index_t * selected_atom, lenght_t n, lenght_t m, lenght_t M, index_t L, index_t l)
{
	lenght_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < M)
	{
		DtX = & DtX[i * m];			// m x 1
		R = & R[i * n];				// n x 1
		alpha = & alpha[i * m];		// m x 1
		selected_atom = & selected_atom[i * L];

		lenght_t max = 0;
		for(lenght_t j = 1; j < m; j++)
		{
			if(abs(DtX[j]) > abs(DtX[max]))
				max = j;
		}

		selected_atom[l] = max;
		data_t * atom;
		for(index_t a = 0; a < l; a++)
		{
			atom = & D[selected_atom[a] * n];

			for(index_t k = 0; k < n; k++)
				R[k] -= D[k] * DtX[max];
		}
		alpha[max] += DtX[max];
	}
}

__global__
void MP_patch(data_t * D, data_t * DtX, data_t * R, data_t * alpha, lenght_t n, lenght_t m, lenght_t M, index_t l)
{
	lenght_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < M)
	{
		DtX = & DtX[i * m];			// m x 1
		R = & R[i * n];				// n x 1
		alpha = & alpha[i * m];		// m x 1
		
		lenght_t max = 0;
		for(lenght_t j = 1; j < m; j++)
		{
			if(abs(DtX[j]) > abs(DtX[max]))
				max = j;
		}
		
		D = & D[max * n];			// n x 1

		for(index_t k = 0; k < n; k++)
			R[k] -= D[k] * DtX[max];

		alpha[max] += DtX[max];
	}
}

void MP(cublasHandle_t & handle, data_t * d_alpha, data_t * d_D, data_t * d_X, lenght_t n, lenght_t m, lenght_t M, lenght_t L)
{
	data_t * d_DtX;		// m x M (like d_alpha)
	data_t * d_R;		// n x M (like d_X)

	cudaMalloc(& d_DtX, m * M * sizeof(data_t));
	cudaMalloc(& d_R, n * M * sizeof(data_t));

	cudaMemcpy(d_R, d_X, n * M * sizeof(data_t), cudaMemcpyDeviceToDevice);
	cudaMemset(d_alpha, 0, m * M * sizeof(data_t));

	for(lenght_t l = 0; l < L; l++)
	{
		gpu_blas_mmul(handle, d_D, d_R, d_DtX, m, n, M, true);
		MP_patch<<< NB(M), NT >>>(d_D, d_DtX, d_R, d_alpha, n, m, M, l);
		cudaDeviceSynchronize();
	}

	cudaFree(d_DtX);
	cudaFree(d_R);
}

void OMP(cublasHandle_t & handle, data_t * d_alpha, data_t * d_D, data_t * d_X, lenght_t n, lenght_t m, lenght_t M, lenght_t L)
{
	data_t * d_DtX;		// m x M (like d_alpha)
	data_t * d_R;		// n x M (like d_X)
	index_t * selected_atom;	//L x M

	cudaMalloc(& d_DtX, m * M * sizeof(data_t));
	cudaMalloc(& d_R, n * M * sizeof(data_t));
	cudaMalloc(& selected_atom, L * M * sizeof(index_t));

	cudaMemcpy(d_R, d_X, n * M * sizeof(data_t), cudaMemcpyDeviceToDevice);
	cudaMemset(d_alpha, 0, m * M * sizeof(data_t));

	for(lenght_t l = 0; l < L; l++)
	{
		gpu_blas_mmul(handle, d_D, d_R, d_DtX, m, n, M);
		MP_patch<<< NB(M), NT >>>(d_D, d_DtX, d_R, d_alpha, selected_atom, n, m, M, L, l);
		cudaDeviceSynchronize();
	}

	cudaFree(d_DtX);
	cudaFree(d_R);
	cudaFree(selected_atom);
}

#endif


// CUDA codes for freq multiplication

#ifndef __FREQ_MUL_CU
#define __FREQ_MUL_CU

#include <stdio.h>
#include <stdlib.h>

#include <cutil_inline.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>

#include <cufft.h>

#define BLOCK_2DIM_X 32
#define BLOCK_2DIM_Y 32
 
#define PRECISION 1e-5f

__global__ void freq_mul_kernel( cufftComplex *, float*, dim3 ); 


extern "C" 
void freq_mul_wrapper( cufftComplex * d_phase_complex, float* d_filter, 
		       int xm_padding, int ym_padding ){

  // perform filtering on the Fourier data
  int blockWidth  = BLOCK_2DIM_X;
  int blockHeight = BLOCK_2DIM_Y;   

  int nBlockX = (int)ceil((float) (xm_padding) / (float)blockWidth);   
  int nBlockY = (int)ceil((float) (ym_padding) / (float)blockHeight);

  dim3 dimGrid(nBlockX, nBlockY);                // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight);
  dim3 imagedim( xm_padding, ym_padding );

  freq_mul_kernel<<< dimGrid, dimBlock >>>( d_phase_complex, d_filter, imagedim );
  CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void freq_mul_kernel( cufftComplex * d_phase_complex, float* d_filter, dim3 imagedim){

  uint idx, idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  // shared memory
  
  __shared__ float shmem_phase_x[BLOCK_2DIM_X][BLOCK_2DIM_Y];
  __shared__ float shmem_phase_y[BLOCK_2DIM_X][BLOCK_2DIM_Y];
  __shared__ float shmem_filter[BLOCK_2DIM_X][BLOCK_2DIM_Y];
 
  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;

  // load shared memory
  // if(idx < imagedim.x && idy < imagedim.y) // It seems that this line makes a difference, maybe
                                            // because the image is of irregular size (1025 * 1501)
                                            // Pad to (2048 * 2048) if problem still happens
  {

    shmem_phase_x[ cidx.x ][ cidx.y ] = d_phase_complex[ idy * imagedim.x + idx ].x;
    shmem_phase_y[ cidx.x ][ cidx.y ] = d_phase_complex[ idy * imagedim.x + idx ].y;
    shmem_filter[ cidx.x ][ cidx.y ] = d_filter[ idy * imagedim.x + idx ]; 
       
  }

  __syncthreads();

  // 2. apply kernel and store the result

  d_phase_complex[ idy * imagedim.x + idx ].x =  shmem_phase_x[ cidx.x ][ cidx.y ] * shmem_filter[ cidx.x ][ cidx.y ];
  d_phase_complex[ idy * imagedim.x + idx ].y =  shmem_phase_y[ cidx.x ][ cidx.y ] * shmem_filter[ cidx.x ][ cidx.y ];

}

#endif // __FREQ_MUL_CU

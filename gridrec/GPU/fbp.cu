// CUDA codes for FBP 

#ifndef __FBP_CU
#define __FBP_CU

#include <stdio.h>
#include <stdlib.h>

#include <cutil_inline.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>

#include <cufft.h>
 
#include "tomo_recon.h"

#define PRECISION 1e-5f

__global__ void freq_mul_kernel( cufftComplex *, float*, dim3 ); 

__global__ void backproj_kernel( float*, dim3, float, float ); 

texture<float, 2, cudaReadModeElementType> texSinoIfft;

extern "C" 
void fbp_wrapper( float* d_sinogram, float* d_filter_lut, cudaArray* d_sino_ifft, 
		  cufftComplex * d_sinogram_fourier, 
		  float* d_recon, 
		  int num_ray, int num_proj, float start_rot, float inv_rot ){

  // perform 1D fft on sinogram data
  cufftHandle plan;
  
  cufftPlan1d( &plan, num_ray, CUFFT_R2C, num_proj );
  cufftExecR2C( plan, d_sinogram, d_sinogram_fourier );

  // perform filtering on the Fourier data
  int blockWidth  = BLOCK_2DIM_X;
  int blockHeight = BLOCK_2DIM_Y;   

  int nBlockX = (int)ceil((float) (num_ray/2 + 1) / (float)blockWidth);   
  int nBlockY = (int)ceil((float) num_proj / (float)blockHeight);

  dim3 dimGrid(nBlockX, nBlockY);                // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight);
  dim3 projdim(num_ray/2+1, num_proj);

  freq_mul_kernel<<< dimGrid, dimBlock >>>( d_sinogram_fourier, d_filter_lut, projdim );

  // perform 1D ifft on filtered Fourier data
  cufftHandle plan2;

  cufftPlan1d( &plan2, num_ray, CUFFT_C2R, num_proj );
  cufftExecC2R( plan2, d_sinogram_fourier, d_sinogram );

  // perform back projection
  int nBlockX2 = (int)ceil((float)num_ray / (float)blockWidth);
  int nBlockY2 = (int)ceil((float)num_ray / (float)blockHeight);

  dim3 dimGrid2(nBlockX2, nBlockY2);    
  dim3 dimBlock2(blockWidth, blockHeight); 
  dim3 projdim2(num_ray, num_proj);

  // set texture parameters
  texSinoIfft.normalized = false;                      // access with normalized texture coordinates
  texSinoIfft.filterMode = cudaFilterModeLinear;      // linear interpolation
  texSinoIfft.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates
  texSinoIfft.addressMode[1] = cudaAddressModeClamp;

  cudaMemcpyToArray( d_sino_ifft, 0, 0, d_sinogram, sizeof(float) * num_ray * num_proj, cudaMemcpyDeviceToDevice );

  cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  CUDA_SAFE_CALL(cudaBindTextureToArray(texSinoIfft, d_sino_ifft, float1Desc));

  backproj_kernel<<< dimGrid2, dimBlock2 >>> ( d_recon, projdim2, start_rot, inv_rot );

  CUDA_SAFE_CALL( cudaUnbindTexture( texSinoIfft ) );

  CUT_CHECK_ERROR("Kernel execution failed");

  // 
  cufftDestroy(plan);
  cufftDestroy(plan2);
}

__global__ void freq_mul_kernel( cufftComplex * d_sinogram_fourier, float* d_filter_lut, dim3 projdim){

  uint idx, idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  // shared memory
  
  __shared__ float shmem_sino_x[BLOCK_2DIM_X][BLOCK_2DIM_Y];
  __shared__ float shmem_sino_y[BLOCK_2DIM_X][BLOCK_2DIM_Y];
  __shared__ float shmem_filter_lut[BLOCK_2DIM_X];
 
  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;

  // load shared memory
  // if(idx < projdim.x && idy < projdim.y) // It seems that this line makes a difference, maybe
                                            // because the image is of irregular size (1025 * 1501)
                                            // Pad to (2048 * 2048) if problem still happens
  {

    shmem_sino_x[ cidx.x ][ cidx.y ] = d_sinogram_fourier[ idy * projdim.x + idx ].x;
    shmem_sino_y[ cidx.x ][ cidx.y ] = d_sinogram_fourier[ idy * projdim.x + idx ].y;
    shmem_filter_lut[ cidx.x ] = d_filter_lut[ idx ]; 
       
  }

  __syncthreads();

  // 2. apply kernel and store the result

  d_sinogram_fourier[ idy * projdim.x + idx ].x =  shmem_sino_x[ cidx.x ][ cidx.y ] * shmem_filter_lut[ cidx.x ];

  d_sinogram_fourier[ idy * projdim.x + idx ].y =  shmem_sino_y[ cidx.x ][ cidx.y ] * shmem_filter_lut[ cidx.x ];

}

__global__ void backproj_kernel( float* d_recon, dim3 projdim, float start_rot, float inv_rot){

  uint idx, idy;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;

  // 
  float res = 0.0f;
  float angle; 
  float pos; 
  float i;
  float half_dimx = projdim.x / 2.0f;
  if(idx < projdim.x && idy < projdim.x){  // assume num_height = num_width = num_ray
    
    for( i = 0.0f; i < 1.0f * projdim.y; i++ ){
    // i = 0.0f; 

      angle = (start_rot + i * inv_rot) * 3.1416f / 180.0f;

      pos = (1.0f * idx - half_dimx) * cosf( angle ) + (1.0f * idy - half_dimx) * sinf( angle ) + half_dimx;

      pos = floor( pos );   // trick from FBP codes. Theoretically unnecessary, even inappropriate

      if( pos >= 0.0f && pos < 1.0f * projdim.x )
	res += tex2D( texSinoIfft, pos + 0.5f, i + 0.5f );

    }
  }

  d_recon[ idy * projdim.x + idx ] = res;

}




#endif // __FBP_CU

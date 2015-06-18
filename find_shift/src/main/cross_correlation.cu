// CUDA codes for FBP 

#ifndef __CROSS_CORRELATION_CU__
#define __CROSS_CORRELATION_CU__

#define BLOCK_2DIM_X 32
#define BLOCK_2DIM_Y 32

#include <stdio.h>
#include <stdlib.h>

#include <cutil_inline.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>

__global__ void data_match_avg_kernel( float* d_DataMatch, 
				       float dTransX, float dTransY, 
				       int nROILowerX, int nROIUpperX, 
				       int nROILowerY, int nROIUpperY,
				       dim3 imagedim );

__global__ void cross_correlation_kernel (float* d_DataTemplate, 
					  float* d_DataCrossCor, float* d_DataMatchSq,
					  float dTransX, float dTransY, 
					  float dTemplateAvg, float dMatchAvg,
				          int nROILowerX, int nROIUpperX, 
				          int nROILowerY, int nROIUpperY,
					  dim3 imagedim);

texture<float, 2, cudaReadModeElementType> texDataMatch;

extern "C"
void cross_correlation_wrapper (float* d_DataTemplate, cudaArray* d_DataMatch2,
			        double* score_match,
			        float dTransStartX, float dTransStartY,
			        float dTransInvX, float dTransInvY,
			        int numTransX, int numTransY,  
				float dTemplateAvg, float dTemplateRMS,
 			        int nROILowerX, int nROIUpperX, int nROILowerY, int nROIUpperY,
				int nWidth, int nHeight) {

  // this simiplifed implementation works for an image of dimensions to be power of 2

  int blockWidth  = BLOCK_2DIM_X;
  int blockHeight = BLOCK_2DIM_Y;   

  int nBlockX = (int)ceil( (float) nWidth / (float)blockWidth);   
  int nBlockY = (int)ceil( (float) nHeight / (float)blockHeight);

  dim3 dimGrid(nBlockX, nBlockY);                // 3D grid is not supported on G80
  dim3 dimBlock(blockWidth, blockHeight);
  dim3 imagedim(nWidth, nHeight);

  // memory for reduction results
  float* dataCrossCor = new float[ nBlockX * nBlockY ];    // CPU
  float* dataMatchSq =  new float[ nBlockX * nBlockY ];

  float* d_DataCrossCor = NULL;                            // GPU
  float* d_DataMatchSq = NULL;
  CUDA_SAFE_CALL( cudaMalloc ( (void**) &d_DataCrossCor, nBlockX * nBlockY * sizeof(float) ) );
  CUDA_SAFE_CALL( cudaMalloc ( (void**) &d_DataMatchSq, nBlockX * nBlockY * sizeof(float) ) );

  // set texture parameters
  texDataMatch.normalized = false;                      // access with normalized texture coordinates
  texDataMatch.filterMode = cudaFilterModeLinear;      // linear interpolation
  texDataMatch.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates
  texDataMatch.addressMode[1] = cudaAddressModeClamp;

  cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // float
  CUDA_SAFE_CALL(cudaBindTextureToArray(texDataMatch, d_DataMatch2, float1Desc));

  int nTransIndX, nTransIndY;
  float sumCrossCor, sumMatchSq;
  float dMatchAvg;
  float dTransX, dTransY;
  for( nTransIndY = 0; nTransIndY < numTransY; nTransIndY++ ){  // for each translation in Y
    for( nTransIndX = 0; nTransIndX < numTransX; nTransIndX++ ){  // for each translation in X

      dTransX = dTransStartX + nTransIndX * dTransInvX;
      dTransY = dTransStartY + nTransIndY * dTransInvY;


      // 
      data_match_avg_kernel<<< dimGrid, dimBlock >>>( d_DataMatchSq, 
						      dTransX, dTransY, 
 			                              nROILowerX, nROIUpperX, 
						      nROILowerY, nROIUpperY,
						      imagedim );

      cutilSafeCall( cudaMemcpy(dataMatchSq, d_DataMatchSq, 
				nBlockX * nBlockY * sizeof(float), 
				cudaMemcpyDeviceToHost) ); 

      dMatchAvg = 0.0f;
      for( int i = 0; i < nBlockX * nBlockY; i++ ){
	dMatchAvg += dataMatchSq[ i ];
      }
      dMatchAvg /= (nROIUpperX - nROILowerX + 1) * (nROIUpperY - nROILowerY + 1);

      cross_correlation_kernel<<< dimGrid, dimBlock >>>( d_DataTemplate, 
							 d_DataCrossCor, d_DataMatchSq,
							 dTransX, dTransY,
							 dTemplateAvg, dMatchAvg, 
 			                                 nROILowerX, nROIUpperX, 
						         nROILowerY, nROIUpperY,
							 imagedim );

      cutilSafeCall( cudaMemcpy(dataCrossCor, d_DataCrossCor, 
				nBlockX * nBlockY * sizeof(float), cudaMemcpyDeviceToHost) ); 
      cutilSafeCall( cudaMemcpy(dataMatchSq, d_DataMatchSq, 
				nBlockX * nBlockY * sizeof(float), cudaMemcpyDeviceToHost) ); 
      sumCrossCor = 0.0f;
      sumMatchSq = 0.0f;
      for( int i = 0; i < nBlockX * nBlockY; i++ ){
	sumCrossCor += dataCrossCor[ i ];
	sumMatchSq += dataMatchSq[ i ];
      }

      if( sqrt( sumMatchSq ) > 1e-5f && dTemplateRMS > 1e-5f ){
	score_match[ nTransIndY * numTransX + nTransIndX ] = sumCrossCor / sqrt( sumMatchSq ) / dTemplateRMS;
      }
      else{
	score_match[ nTransIndY * numTransX + nTransIndX ] = 0.0f;
      }

    }
  }

  CUDA_SAFE_CALL( cudaUnbindTexture( texDataMatch ) );

  cudaFree( d_DataCrossCor );
  cudaFree( d_DataMatchSq );
  delete [] dataCrossCor;
  delete [] dataMatchSq;

}

__global__ void data_match_avg_kernel( float* d_DataMatch, 
				       float dTransX, float dTransY, 
				       int nROILowerX, int nROIUpperX, 
				       int nROILowerY, int nROIUpperY,
				       dim3 imagedim ){

  // the following codes are customized for BLOCK_2DIM_X = 32 (BLOCK_2DIM_Y = 32)

  __shared__ float sdataMatch[ BLOCK_2DIM_Y ][ BLOCK_2DIM_X ];  // 1024

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tidx = threadIdx.x;
  unsigned int tidy = threadIdx.y;

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  float tmpMatch = 0.0f;
  float x, y;

  int blockSize = blockDim.x * blockDim.y;

  // reduction kernel 5
  if ( idx < 1.0f * imagedim.x && idy < 1.0f * imagedim.y ){
    x = 1.0f * idx - dTransX;
    y = 1.0f * idy - dTransY; 
    if( x >= 1.0f * nROILowerX && x <= 1.0f * nROIUpperX 
	&& y >= 1.0f * nROILowerY && y <= 1.0f * nROIUpperY ){
      tmpMatch = tex2D( texDataMatch, x + 0.5f, y + 0.5f );
    }
    else{
      tmpMatch = 0.0f;
    }      
  }
  
  // reduction kernel 6
  // we reduce multiple elements per thread.  The number is determined by the 
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  // while (i < n){

  //   mySum += fabsf( d_idata1[i+blockSize] - d_idata2[i+blockSize] );  
  //   // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
  //   if (nIsPow2 == 1 || i + blockSize < n) 
  //     mySum += fabsf( d_idata1[i+blockSize] - d_idata2[i+blockSize] );  
  //   i += gridSize;
  // } 

  // each thread puts its local sum into shared memory 
  sdataMatch[tidy][tidx] = tmpMatch;

  __syncthreads();

  unsigned int tid = tidy * blockDim.x + tidx;
  // do reduction in shared mem
  if (blockSize >= 1024) { 
    if (tid < 512) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy + 512/BLOCK_2DIM_X ][tidx]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 512) { 
    if (tid < 256) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy + 256/BLOCK_2DIM_X ][tidx]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 256) { 
    if (tid < 128) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy + 128/BLOCK_2DIM_X ][tidx]; 
    } 
    __syncthreads(); 
  }
  
  if (blockSize >= 128) { 
    if (tid <  64) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy + 64/BLOCK_2DIM_X ][tidx]; 
    } 
    __syncthreads(); 
  }

  if (tid < 32){
    
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    // volatile float* smem = sdataCrossCor;
    // if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
    // if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
    // if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
    // if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
    // if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
    // if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
  
    if (blockSize >=  64) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy + 32/BLOCK_2DIM_X ][tidx]; 
    }
    if (blockSize >=  32) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy ][tidx + 16]; 
    }
    if (blockSize >=  16) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy][tidx + 8]; 
    }
    if (blockSize >=   8) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy][tidx + 4]; 
    }
    if (blockSize >=   4) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy][tidx + 2]; 
    }
    if (blockSize >=   2) { 
      sdataMatch[tidy][tidx] += sdataMatch[tidy][tidx + 1]; 
    }
  }
    
  // write result for this block to global mem 
  if (tid == 0) {
    d_DataMatch[blockIdx.y * gridDim.x + blockIdx.x] = sdataMatch[0][0];
  }

}


__global__ void cross_correlation_kernel( float* d_DataTemplate,
					  float* d_DataCrossCor, float* d_DataMatchSq, 
				          float dTransX, float dTransY, 
					  float dTemplateAvg, float dMatchAvg,
				          int nROILowerX, int nROIUpperX, 
				          int nROILowerY, int nROIUpperY,
					  dim3 imagedim ){

  // the following codes are customized for BLOCK_2DIM_X = 32 (BLOCK_2DIM_Y = 32)

  __shared__ float sdataCrossCor[ BLOCK_2DIM_Y ][ BLOCK_2DIM_X ];  // 1024
  __shared__ float sdataMatchSq[ BLOCK_2DIM_Y ][ BLOCK_2DIM_X ];  // 1024

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tidx = threadIdx.x;
  unsigned int tidy = threadIdx.y;

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  float tmpCrossCor = 0.0f;
  float tmpMatchSq = 0.0f;
  float x, y;

  int blockSize = blockDim.x * blockDim.y;

  // reduction kernel 5
  if ( idx < 1.0f * imagedim.x && idy < 1.0f * imagedim.y ){
    x = 1.0f * idx - dTransX;
    y = 1.0f * idy - dTransY; 

    if( idx >= 1.0f * nROILowerX && idx <= 1.0f * nROIUpperX 
	&& idy >= 1.0f * nROILowerY && idy <= 1.0f * nROIUpperY
        && x >= 0.0f && x <= 1.0f * imagedim.x 
	&& y >= 0.0f && y <= 1.0f * imagedim.y ){
      
      tmpMatchSq = tex2D( texDataMatch, x + 0.5f, y + 0.5f ) - dMatchAvg;
      tmpCrossCor = ( d_DataTemplate[idy * imagedim.x + idx] - dTemplateAvg ) * tmpMatchSq;  
      tmpMatchSq *= tmpMatchSq;
    }
    else{
      tmpCrossCor = 0.0f;
      tmpMatchSq = 0.0f;
    }
  }
  
  // reduction kernel 6
  // we reduce multiple elements per thread.  The number is determined by the 
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  // while (i < n){

  //   mySum += fabsf( d_idata1[i+blockSize] - d_idata2[i+blockSize] );  
  //   // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
  //   if (nIsPow2 == 1 || i + blockSize < n) 
  //     mySum += fabsf( d_idata1[i+blockSize] - d_idata2[i+blockSize] );  
  //   i += gridSize;
  // } 

  // each thread puts its local sum into shared memory 
  sdataCrossCor[tidy][tidx] = tmpCrossCor;
  sdataMatchSq[tidy][tidx] = tmpMatchSq;

  __syncthreads();

  unsigned int tid = tidy * blockDim.x + tidx;
  // do reduction in shared mem
  if (blockSize >= 1024) { 
    if (tid < 512) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy + 512/BLOCK_2DIM_X ][tidx]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy + 512/BLOCK_2DIM_X ][tidx]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 512) { 
    if (tid < 256) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy + 256/BLOCK_2DIM_X ][tidx]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy + 256/BLOCK_2DIM_X ][tidx]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 256) { 
    if (tid < 128) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy + 128/BLOCK_2DIM_X ][tidx]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy + 128/BLOCK_2DIM_X ][tidx]; 
    } 
    __syncthreads(); 
  }
  
  if (blockSize >= 128) { 
    if (tid <  64) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy + 64/BLOCK_2DIM_X ][tidx]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy + 64/BLOCK_2DIM_X ][tidx]; 
    } 
    __syncthreads(); 
  }

  if (tid < 32){
    
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    // volatile float* smem = sdataCrossCor;
    // if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
    // if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
    // if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
    // if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
    // if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
    // if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
  
    if (blockSize >=  64) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy + 32/BLOCK_2DIM_X ][tidx]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy + 32/BLOCK_2DIM_X ][tidx]; 
    }
    if (blockSize >=  32) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy ][tidx + 16]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy ][tidx + 16 ]; 
    }
    if (blockSize >=  16) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy][tidx + 8]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy][tidx + 8]; 
    }
    if (blockSize >=   8) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy][tidx + 4]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy][tidx + 4]; 
    }
    if (blockSize >=   4) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy][tidx + 2]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy][tidx + 2]; 
    }
    if (blockSize >=   2) { 
      sdataCrossCor[tidy][tidx] += sdataCrossCor[tidy][tidx + 1]; 
      sdataMatchSq[tidy][tidx] += sdataMatchSq[tidy][tidx + 1]; 
    }
  }
    
  // write result for this block to global mem 
  if (tid == 0) {
    d_DataCrossCor[blockIdx.y * gridDim.x + blockIdx.x] = sdataCrossCor[0][0];
    d_DataMatchSq[blockIdx.y * gridDim.x + blockIdx.x] = sdataMatchSq[0][0];
  }

}

#endif //  __CROSS_CORRELATION_CU__

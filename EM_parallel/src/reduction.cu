// codes for reduction

#ifndef __REDUCTION_CU
#define __REDUCTION_CU

#include <stdio.h>
#include <stdlib.h>

#include <cutil_inline.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>

#include "ParallelBeamEM.h" 

__global__ void reduction_kernel( float*, float*, int, int, int );

__global__ void reduction2_kernel( float*, float*, float*, int, int, int );

__global__ void reduction2_kernel_2D( float* , float* , float* , 
				      int , int , int );

// dynamic shared memory allocation does not work well yet. Y. Pan, 5/11/2011

// // Utility class used to avoid linker errors with extern
// // unsized shared memory arrays with templated type
// template<class T>
// struct SharedMemory
// {
//     __device__ inline operator       T*()
//     {
//         extern __shared__ float __smem[];
//         return (T*)__smem;
//     }

//     __device__ inline operator const T*() const
//     {
//         extern __shared__ float __smem[];
//         return (T*)__smem;
//     }
// };

// // specialize for double to avoid unaligned memory 
// // access compile errors
// template<>
// struct SharedMemory<double>
// {
//     __device__ inline operator       double*()
//     {
//         extern __shared__ double __smem_d[];
//         return (double*)__smem_d;
//     }

//     __device__ inline operator const double*() const
//     {
//         extern __shared__ double __smem_d[];
//         return (double*)__smem_d;
//     }
// };


extern "C"
int nextPow2( int x ) {  // x should better be "unsigned int"
  if( x < 0 ){  
    return 0;
  }
 
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  
  return ++x;
}


extern "C"
int isPow2(int x) {  // x should better be "unsigned int"

  if( x < 0 ){
    return 0;
  }

  if( (x&(x-1))==0 )
    return 1;
  else
    return 0;

}


extern "C" 
float reduction_wrapper( float* d_in, float* d_sub, int nx, int ny, int nz ){

  int maxThreads;
  int nVolume = nx * ny * nz; 

  if(ny == 1 || nz == 1) {
    maxThreads = BLOCK_2DIM_X * BLOCK_2DIM_Y * BLOCK_2DIM_Z;       
  }
  else {                  
    maxThreads = BLOCK_3DIM_X * BLOCK_3DIM_Y * BLOCK_3DIM_Z;       
  }

  int nThreads = (nVolume < maxThreads * 2) ? nextPow2((nVolume + 1)/ 2) : maxThreads;
  int nBlocks = (nVolume + (nThreads * 2 - 1) ) / (nThreads * 2);

  dim3 dimGrid(nBlocks, 1);                // 3D grid is not supported on G80
  dim3 dimBlock(nThreads, 1, 1); 

  int nIsPow2 = isPow2( nVolume );
  reduction_kernel<<< dimGrid, dimBlock >>>( d_in, d_sub, nVolume, nThreads, nIsPow2 );

  nVolume = nBlocks;
  while ( nVolume > 1 ){
    nThreads = (nVolume < maxThreads * 2) ? nextPow2((nVolume + 1)/ 2) : maxThreads;
    nBlocks = (nVolume + (nThreads * 2 - 1) ) / (nThreads * 2);

    // execute the kernel
    dim3 dimGrid(nBlocks, 1);                // 3D grid is not supported on G80
    dim3 dimBlock(nThreads, 1, 1); 

    nIsPow2 = isPow2( nVolume );
    reduction_kernel<<< dimGrid, dimBlock >>>( d_sub, d_sub, nVolume, nThreads, nIsPow2);
    CUT_CHECK_ERROR("Kernel execution failed");


    nVolume = (nVolume + (nThreads*2 - 1)) / (nThreads * 2);
  }

  // extract reduction results
  float out = 0.0f; 
  cutilSafeCall( cudaMemcpy( &out, d_sub, sizeof(float), cudaMemcpyDeviceToHost) ); 

  return out; 
}

__global__ void reduction_kernel( float* d_idata, float* d_odata, int n, 
				  int blockSize, int nIsPow2 ){

  __shared__ float sdata[ THREADS_MAX ];  // 1024

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  // unsigned int gridSize = blockSize*2*gridDim.x;
    
  float mySum = 0.0;

  // reduction kernel 5
  if (i < n){
    mySum += fabsf( d_idata[i] );  
  }
   
  if (i + blockSize < n) {
    mySum += fabsf( d_idata[i+blockSize] );  
  } 
  
  // reduction kernel 6
  // we reduce multiple elements per thread.  The number is determined by the 
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  // while (i < n){

  //   mySum += fabsf( d_idata[i] ); // d_idata[i] * d_idata[i];
  //   // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
  //   if (nIsPow2 == 1 || i + blockSize < n) 
  //     mySum += fabsf( d_idata[i+blockSize] ); // d_idata[i+blockSize] * d_idata[i+blockSize];  
  //   i += gridSize;
  // } 

  // each thread puts its local sum into shared memory 
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 1024) { 
    if (tid < 512) { 
      // sdata[tid] = mySum = mySum + sdata[tid + 512]; 
      sdata[tid] += sdata[tid + 512]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 512) { 
    if (tid < 256) { 
      // sdata[tid] = mySum = mySum + sdata[tid + 256]; 
      sdata[tid] += sdata[tid + 256]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 256) { 
    if (tid < 128) { 
      // sdata[tid] = mySum = mySum + sdata[tid + 128]; 
      sdata[tid] += sdata[tid + 128]; 
    } 
    __syncthreads(); 
  }
  
  if (blockSize >= 128) { 
    if (tid <  64) { 
      // sdata[tid] = mySum = mySum + sdata[tid +  64]; 
      sdata[tid] += sdata[tid +  64]; 
    } 
    __syncthreads(); 
  }

  if (tid < 32){
    
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    // volatile float* smem = sdata;
    if (blockSize >=  64) { 
      // smem[tid] = mySum = mySum + smem[tid + 32]; 
      sdata[tid] += sdata[tid + 32]; 
    }
    if (blockSize >=  32) { 
      // smem[tid] = mySum = mySum + smem[tid + 16]; 
      sdata[tid] += sdata[tid + 16]; 
    }
    if (blockSize >=  16) { 
      // smem[tid] = mySum = mySum + smem[tid +  8]; 
      sdata[tid] += sdata[tid + 8]; 
    }
    if (blockSize >=   8) { 
      // smem[tid] = mySum = mySum + smem[tid +  4]; 
      sdata[tid] += sdata[tid + 4]; 
    }
    if (blockSize >=   4) { 
      // smem[tid] = mySum = mySum + smem[tid +  2]; 
      sdata[tid] += sdata[tid + 2]; 
    }
    if (blockSize >=   2) { 
      // smem[tid] = mySum = mySum + smem[tid +  1]; 
      sdata[tid] += sdata[tid + 1]; 
    }
  }
    
  // write result for this block to global mem 
  if (tid == 0) 
    d_odata[blockIdx.x] = sdata[0];
}



// -----------------------------------------------------------------------------------------

extern "C" 
float reduction2_wrapper( float* d_in1, float* d_in2, float* d_sub, int nx, int ny, int nz ){

  int maxThreads = THREADS_MAX;
  int nVolume = nx * ny * nz; 

  // 1D reduction kernel

  int nThreads = (nVolume < maxThreads * 2) ? nextPow2((nVolume + 1)/ 2) : maxThreads;
  int nBlocks = (nVolume + (nThreads * 2 - 1) ) / (nThreads * 2);

  // int nThreads = (nVolume < maxThreads) ? nextPow2(nVolume) : maxThreads;
  // int nBlocks = (nVolume + (nThreads - 1) ) / (nThreads);

  // Note that the dimension of grid is limited to 65535 * 65535 * 1
  dim3 dimGrid(nBlocks, 1);               
  dim3 dimBlock(nThreads, 1, 1); 

  int nIsPow2 = isPow2( nVolume );
  reduction2_kernel<<< dimGrid, dimBlock >>>( d_in1, d_in2, d_sub, 
  					      nVolume, nThreads, nIsPow2);

  // // for debug
  // float* tmp = (float*) malloc( nBlocks * sizeof( float ) );

  // cudaMemcpy( tmp, d_sub, sizeof(float) * nBlocks, cudaMemcpyDeviceToHost );
  // int i; 
  // float total = 0.0f;
  // for( i = 0; i < nBlocks; i++ ){
  //   total += tmp[i];
  //   if( isnan( total ) ){
  //     total -= tmp[i];
  //   }
  // }

  nVolume = nBlocks;
  while ( nVolume > 1 ){

    nThreads = (nVolume < maxThreads * 2) ? nextPow2((nVolume + 1)/ 2) : maxThreads;
    nBlocks = (nVolume + (nThreads * 2 - 1) ) / (nThreads * 2);

    // execute the kernel
    dim3 dimGrid(nBlocks, 1);                // 3D grid is not supported on G80
    dim3 dimBlock(nThreads, 1, 1); 

    int nIsPow2 = isPow2( nVolume );
    reduction_kernel<<< dimGrid, dimBlock >>>( d_sub, d_sub, nVolume, nThreads, nIsPow2);
    CUT_CHECK_ERROR("Kernel execution failed");

    nVolume = (nVolume + (nThreads*2 - 1)) / (nThreads * 2);
  }

  // extract reduction results
  float out = 0.0f; 
  cutilSafeCall( cudaMemcpy( &out, d_sub, sizeof(float), cudaMemcpyDeviceToHost) ); 

  return out; 

  // Weird: somehow only part of blocks are computed, corresponding to the first 
  //        dimension in dimGrid. For example, only 256 blocks for dimGrid(256, 256). 
  // Yongsheng Pan, 5/12/2011

  // 2D reduction kernel. Kept for future use. 
  
  // int blockWidth = BLOCK_2DIM_X; // 16
  // int blockHeight = BLOCK_2DIM_Y; // 16

  // int nBlockX = (int)ceil( 1.0 * nx * ny / blockWidth );
  // int nBlockY = (int)ceil( 1.0 * nz / blockHeight );

  // dim3 dimGrid(nBlockX, nBlockY, 1);
  // dim3 dimBlock(blockWidth, blockHeight);

  // int nBlocks = nBlockX * nBlockY;
  // int nThreads = blockWidth * blockHeight;

  // reduction2_kernel_2D<<< dimGrid, dimBlock >>>( d_in1, d_in2, d_sub, 
  // 					      nVolume, nx * ny, 1);

  // CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void reduction2_kernel( float* d_idata1, float* d_idata2, float* d_odata, 
				   int n, int blockSize, int nIsPow2 ){

  __shared__ float sdata[ THREADS_MAX ];  // 1024

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  // unsigned int gridSize = blockSize*2*gridDim.x;

  float mySum = 0.0f;

  // reduction kernel 5
  if (i < n ){
    mySum = fabsf( d_idata1[i] - d_idata2[i] );  
  }
 
  if (1.0f * i + blockSize < 1.0f * n - 0.5f) {
    mySum += fabsf( d_idata1[i+blockSize] - d_idata2[i+blockSize] );  
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
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 1024) { 
    if (tid < 512) { 
      // sdata[tid] = mySum = mySum + sdata[tid + 512]; 
      sdata[tid] += sdata[tid + 512]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 512) { 
    if (tid < 256) { 
      // sdata[tid] = mySum = mySum + sdata[tid + 256]; 
      sdata[tid] += sdata[tid + 256]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 256) { 
    if (tid < 128) { 
      // sdata[tid] = mySum = mySum + sdata[tid + 128]; 
      sdata[tid] += sdata[tid + 128]; 
    } 
    __syncthreads(); 
  }
  
  if (blockSize >= 128) { 
    if (tid <  64) { 
      //  sdata[tid] = mySum = mySum + sdata[tid +  64]; 
      sdata[tid] += sdata[tid +  64]; 
    } 
    __syncthreads(); 
  }

  if (tid < 32){
    
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    // volatile float* smem = sdata;
    // if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
    // if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
    // if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
    // if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
    // if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
    // if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
  
    if (blockSize >=  64) { 
      // smem[tid] = mySum = mySum + smem[tid + 32]; 
      sdata[tid] += sdata[tid + 32]; 
    }
    if (blockSize >=  32) { 
      // smem[tid] = mySum = mySum + smem[tid + 16]; 
      sdata[tid] += sdata[tid + 16]; 
    }
    if (blockSize >=  16) { 
      // smem[tid] = mySum = mySum + smem[tid +  8]; 
      sdata[tid] += sdata[tid + 8]; 
    }
    if (blockSize >=   8) { 
      // smem[tid] = mySum = mySum + smem[tid +  4]; 
      sdata[tid] += sdata[tid + 4]; 
    }
    if (blockSize >=   4) { 
      // smem[tid] = mySum = mySum + smem[tid +  2]; 
      sdata[tid] += sdata[tid + 2]; 
    }
    if (blockSize >=   2) { 
      // smem[tid] = mySum = mySum + smem[tid +  1]; 
      sdata[tid] += sdata[tid + 1]; 
    }
  }
    
    
  // write result for this block to global mem 
  if (tid == 0) 
    d_odata[blockIdx.x] = sdata[0];

}

__global__ void reduction2_kernel_2D( float* d_idata1, float* d_idata2, float* d_odata, 
				      int n, int nx, int nIsPow2 ){

  // Note: these codes are customized for BLOCK_2DIM_X = 16 (for 2D reduction)

  __shared__ float sdata[ BLOCK_2DIM_Y ][ BLOCK_2DIM_X ];  // 1024

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tidx = threadIdx.x;
  unsigned int tidy = threadIdx.y;

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  float mySum = 0.0f;

  int blockSize = blockDim.x * blockDim.y;

  // reduction kernel 5
  if ( idy * nx + idx < n ){
    mySum = fabsf( d_idata1[idy * nx + idx] - d_idata2[idy * nx + idx] );  
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
  sdata[tidy][tidx] = mySum;
  __syncthreads();

  unsigned int tid = tidy * blockDim.x + tidx;
  // do reduction in shared mem
  if (blockSize >= 1024) { 
    if (tid < 512) { 
      sdata[tidy][tidx] += sdata[tidy + 32][tidx]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 512) { 
    if (tid < 256) { 
      sdata[tidy][tidx] += sdata[tidy + 16][tidx]; 
    }
    __syncthreads(); 
  }

  if (blockSize >= 256) { 
    if (tid < 128) { 
      sdata[tidy][tidx] += sdata[tidy + 8][tidx]; 
    } 
    __syncthreads(); 
  }
  
  if (blockSize >= 128) { 
    if (tid <  64) { 
      sdata[tidy][tidx] += sdata[tidy + 4][tidx]; 
    } 
    __syncthreads(); 
  }

  if (tid < 32){
    
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    // volatile float* smem = sdata;
    // if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
    // if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
    // if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
    // if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
    // if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
    // if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
  
    if (blockSize >=  64) { 
      sdata[tidy][tidx] += sdata[tidy + 2][tidx]; 
    }
    if (blockSize >=  32) { 
      sdata[tidy][tidx] += sdata[tidy + 1][tidx]; 
    }
    if (blockSize >=  16) { 
      sdata[tidy][tidx] += sdata[tidy][tidx + 8]; 
    }
    if (blockSize >=   8) { 
      sdata[tidy][tidx] += sdata[tidy][tidx + 4]; 
    }
    if (blockSize >=   4) { 
      sdata[tidy][tidx] += sdata[tidy][tidx + 2]; 
    }
    if (blockSize >=   2) { 
      sdata[tidy][tidx] += sdata[tidy][tidx + 1]; 
    }
  }
    
  // write result for this block to global mem 
  if (tid == 0) 
    d_odata[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0][0];

}


#endif // __REDUCTION_CU

// CUDA codes for Projection calculation

#ifndef __TV3D_CAL_CU
#define __TV3D_CAL_CU

#include <stdio.h>
#include <stdlib.h>

#include <cutil_inline.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define BLOCK_DIM_Z 4

#define PI 3.1415926

__global__ void tv3d_cal_kernel_old( float*, 
	                         float *, float*, 
                                 dim3, dim3, int, int, int, float);


extern "C" 
void tv3d_cal_wrapper_old( float * d_image,  
                       float* d_image_update, float* d_image_grad,
		       int num_width, int num_height, int num_depth, 
		       float epsilon){

    // setup execution parameters
    int blockWidth, blockHeight, blockDepth, nBlockX, nBlockY, nBlockZ;
    
    // Setting block size
    if(num_depth == 1 ) {
      blockWidth  = 16;
      blockHeight = 16;   
      blockDepth  = 1;       
    }
    else {                  

      blockWidth  = BLOCK_DIM_X;
      blockHeight = BLOCK_DIM_Y;
      blockDepth  = BLOCK_DIM_Z;     
    }

   // compute how many blocks are needed
    nBlockX = ceil((float)num_width / (float)blockWidth);
    nBlockY = ceil((float)num_height / (float)blockHeight);
    nBlockZ = ceil((float)num_depth / (float)blockDepth);
   
    dim3 dimGrid(nBlockX, nBlockY*nBlockZ);                // 3D grid is not supported on G80
    dim3 dimBlock(blockWidth, blockHeight, blockDepth);     
    dim3 griddim(nBlockX, nBlockY, nBlockZ); 
    dim3 imagedim( num_width, num_height, num_depth);
   
    // execute the kernel
    tv3d_cal_kernel_old<<< dimGrid, dimBlock >>>( d_image,
                                              d_image_update, d_image_grad,
					      griddim, imagedim, 
    		       			      num_width, num_height, num_depth, epsilon);

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void 
tv3d_cal_kernel_old( float* d_image, 
		 float* d_image_update, float* d_image_grad,
		 dim3 griddim, dim3 imagedim, 
		 int num_width, int num_height, int num_depth, float epsilon){

  uint idx, idy, idz, tid, tidx, tidy, tidz;

  idx = blockIdx.x;
  idz = blockIdx.y / griddim.y;
  idy = blockIdx.y - idz*griddim.y;

  idx = idx*blockDim.x + threadIdx.x;
  idy = idy*blockDim.y + threadIdx.y;
  idz = idz*blockDim.z + threadIdx.z;
  
  __shared__ float shmem_image[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size : 8x8x8

  
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    dim3 cidx;
    cidx.x = threadIdx.x + 1;
    cidx.y = threadIdx.y + 1;
    cidx.z = threadIdx.z + 1;

    //
    // load shared memory
    //
    shmem_image[cidx.x][cidx.y][cidx.z] = d_image[(idz*imagedim.y + (idy))*imagedim.x + idx];

    // loading neighbors : Neumann boundary condition
    if( threadIdx.x == 0 && threadIdx.y == 0 ){
      tidx = max(idx, 1) - 1;
      tidy = max(idy, 1) - 1;
      shmem_image[0][0][cidx.z] = d_image[ (idz * imagedim.y + tidy ) * imagedim.x + tidx];
    }

    if( threadIdx.x == 0 && threadIdx.z == 0 ){
      tidx = max(idx, 1) - 1;
      tidz = max(idz, 1) - 1;
      shmem_image[0][cidx.y][0] = d_image[ (tidz * imagedim.y + idy ) * imagedim.x + tidx];
    }

    if( threadIdx.y == 0 && threadIdx.z == 0 ){
      tidy = max(idy, 1) - 1;
      tidz = max(idz, 1) - 1;
      shmem_image[cidx.x][0][0] = d_image[ (tidz * imagedim.y + tidy ) * imagedim.x + idx];
    }

    //
    
    if( threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 ){
      tidx = max(idx, 1) - 1;
      tidy = min(idy + 1, imagedim.y - 1);
      shmem_image[0][blockDim.y + 1][cidx.z] = d_image[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
    }

    if( threadIdx.x == 0 && threadIdx.z == blockDim.z - 1 ){
      tidx = max(idx, 1) - 1;
      tidz = min(idz + 1, imagedim.z - 1);
      shmem_image[0][cidx.y][blockDim.z + 1] = d_image[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
    }

    if( threadIdx.y == 0 && threadIdx.x == blockDim.x - 1 ){
      tidx = min(idx + 1, imagedim.x - 1);
      tidy = max(idy, 1) - 1; 
      shmem_image[blockDim.x + 1][0][cidx.z] = d_image[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
    }

    if( threadIdx.y == 0 && threadIdx.z == blockDim.z - 1 ){
      tidy = max(idy, 1) - 1; 
      tidz = min(idz + 1, imagedim.z - 1);
      shmem_image[cidx.x][0][blockDim.z + 1] = d_image[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
    }
 
    if( threadIdx.z == 0 && threadIdx.x == blockDim.x - 1 ){
      tidx = min(idx + 1, imagedim.x - 1);
      tidz = max(idz, 1) - 1; 
      shmem_image[blockDim.x + 1][cidx.y][0] = d_image[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
    }

    if( threadIdx.z == 0 && threadIdx.y == blockDim.y - 1 ){
      tidy = min(idy + 1, imagedim.y - 1);
      tidz = max(idz, 1) - 1; 
      shmem_image[cidx.x][blockDim.y + 1][0] = d_image[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
    }

    //

    if( threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 ){
      tidx = min(idx + 1, imagedim.x - 1);
      tidy = min(idy + 1, imagedim.y - 1);
      shmem_image[blockDim.x+1][blockDim.y+1][cidx.z] = d_image[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
    }

    if( threadIdx.x == blockDim.x - 1 && threadIdx.z == blockDim.z - 1 ){
      tidx = min(idx + 1, imagedim.x - 1);
      tidz = min(idz + 1, imagedim.z - 1);
      shmem_image[blockDim.x+1][cidx.y][blockDim.z+1] = d_image[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
    }

    if( threadIdx.y == blockDim.y - 1 && threadIdx.z == blockDim.z - 1 ){
      tidy = min(idy + 1, imagedim.y - 1);
      tidz = min(idz + 1, imagedim.z - 1);
      shmem_image[cidx.x][blockDim.y+1][blockDim.z+1] = d_image[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
    }

    ////

    if(threadIdx.x == 0)
    {
      tid = max(idx,1) - 1;
      shmem_image[0][cidx.y][cidx.z] = d_image[((idz)*imagedim.y + (idy))*imagedim.x + tid];
      
    }

    if(threadIdx.x == blockDim.x-1)
    {
      tid = min(idx+1,imagedim.x-1);
      shmem_image[blockDim.x + 1][cidx.y][cidx.z] = d_image[((idz)*imagedim.y + (idy))*imagedim.x + tid];

    }
    
    if(threadIdx.y == 0)
    {
      tid = max(idy,1) - 1;
      shmem_image[cidx.x][0][cidx.z] = d_image[((idz)*imagedim.y + (tid))*imagedim.x + idx];
      
    }

    if(threadIdx.y == blockDim.y-1)
    {
      tid = min(idy+1,imagedim.y-1);
      shmem_image[cidx.x][blockDim.y + 1][cidx.z] = d_image[((idz)*imagedim.y + (tid))*imagedim.x + idx];

     }
      
    if(threadIdx.z == 0)
    {
      tid = max(idz,1) - 1;
      shmem_image[cidx.x][cidx.y][0] = d_image[((tid)*imagedim.y + (idy))*imagedim.x + idx];

    }

    if(threadIdx.z == blockDim.z-1)
    {
      tid = min(idz+1,imagedim.z-1);
      shmem_image[cidx.x][cidx.y][blockDim.z + 1] = d_image[((tid)*imagedim.y + (idy))*imagedim.x + idx];
      
    }
       
    __syncthreads();


    // prepare for diffusion


    float grad_x, grad_y, grad_z, grad_xx, grad_yy, grad_zz, grad_xy, grad_xz, grad_yz, curv, grad_mag;
  
    grad_x = ( shmem_image[cidx.x + 1][cidx.y][cidx.z] - shmem_image[cidx.x - 1][cidx.y][cidx.z] ) / 2.0;
    grad_y = ( shmem_image[cidx.x][cidx.y + 1][cidx.z] - shmem_image[cidx.x][cidx.y - 1][cidx.z] ) / 2.0;
    grad_z = ( shmem_image[cidx.x][cidx.y][cidx.z + 1] - shmem_image[cidx.x][cidx.y][cidx.z - 1] ) / 2.0;

    grad_xx = shmem_image[cidx.x + 1][cidx.y][cidx.z] + shmem_image[cidx.x - 1][cidx.y][cidx.z]	
            - 2 * shmem_image[cidx.x][cidx.y][cidx.z];

    grad_yy = shmem_image[cidx.x][cidx.y + 1][cidx.z] + shmem_image[cidx.x][cidx.y - 1][cidx.z]	
            - 2 * shmem_image[cidx.x][cidx.y][cidx.z];

    grad_zz = shmem_image[cidx.x][cidx.y][cidx.z + 1] + shmem_image[cidx.x][cidx.y][cidx.z - 1]	
            - 2 * shmem_image[cidx.x][cidx.y][cidx.z];
	    
    grad_xy = (shmem_image[cidx.x + 1][cidx.y + 1][cidx.z] + shmem_image[cidx.x - 1][cidx.y - 1][cidx.z]	
            - shmem_image[cidx.x + 1][cidx.y - 1][cidx.z] - shmem_image[cidx.x - 1][cidx.y + 1][cidx.z])/4.0f;

    grad_xz = (shmem_image[cidx.x + 1][cidx.y][cidx.z + 1] + shmem_image[cidx.x - 1][cidx.y][cidx.z - 1]	
            - shmem_image[cidx.x + 1][cidx.y][cidx.z - 1] - shmem_image[cidx.x - 1][cidx.y][cidx.z + 1])/4.0f;

    grad_yz = (shmem_image[cidx.x][cidx.y + 1][cidx.z + 1] + shmem_image[cidx.x][cidx.y - 1][cidx.z - 1]	
            - shmem_image[cidx.x][cidx.y + 1][cidx.z - 1] - shmem_image[cidx.x][cidx.y - 1][cidx.z + 1])/4.0f;

    curv = grad_xx * (grad_y * grad_y + grad_z * grad_z) + grad_yy * (grad_x * grad_x + grad_z * grad_z)
	 + grad_zz * (grad_x * grad_x + grad_y * grad_y);
    curv -= 2 * ( grad_xy * grad_x * grad_y + grad_xz * grad_x * grad_z + grad_yz * grad_y * grad_z );

    curv /= ( ( grad_x * grad_x + grad_y * grad_y + grad_z * grad_z + epsilon ) );

    grad_mag = sqrt( grad_x * grad_x + grad_y * grad_y + grad_z * grad_z );
 
    __syncthreads();

       
    d_image_update[((idz)*imagedim.y + (idy))*imagedim.x + idx] = curv;
    d_image_grad[ ((idz)*imagedim.y + (idy))*imagedim.x + idx] = grad_mag;

    // for debug
    // d_image_update[((idz)*imagedim.y + (idy))*imagedim.x + idx] = shmem_image[cidx.x + 1][cidx.y - 1][cidx.z];  
    // d_image_grad[ ((idz)*imagedim.y + (idy))*imagedim.x + idx] = shmem_image[cidx.x - 1][cidx.y + 1][cidx.z];
                 
  }

 
}

__global__ void tv3d_cal_kernel( float*, float*, float*, 
	                         float *,  
                                 dim3, dim3, int, int, int, 
				 float, float, float);

extern "C" 
void tv3d_cal_wrapper( float * d_image_prev,  float* d_image_const_int,
                       float* d_image_curr,
		       int num_width, int num_height, int num_depth, 
		       int iter, float weight, float epsilon, float timestep){

    // setup execution parameters
    int blockWidth, blockHeight, blockDepth, nBlockX, nBlockY, nBlockZ;
    
    // Setting block size
    if(num_depth == 1 ) {
      blockWidth  = 16;
      blockHeight = 16;   
      blockDepth  = 1;       
    }
    else {                  

      blockWidth  = BLOCK_DIM_X;
      blockHeight = BLOCK_DIM_Y;
      blockDepth  = BLOCK_DIM_Z;     
    }

    // compute how many blocks are needed
    nBlockX = ceil((float)num_width / (float)blockWidth);
    nBlockY = ceil((float)num_height / (float)blockHeight);
    nBlockZ = ceil((float)num_depth / (float)blockDepth);
   
    dim3 dimGrid(nBlockX, nBlockY*nBlockZ);                // 3D grid is not supported on G80
    dim3 dimBlock(blockWidth, blockHeight, blockDepth);     
    dim3 griddim(nBlockX, nBlockY, nBlockZ); 
    dim3 imagedim( num_width, num_height, num_depth);
   
    float * d_image_org;

    cudaMalloc((void**) &d_image_org,  sizeof(float) * num_width * num_height * num_depth);           
    CUT_CHECK_ERROR("Memory creation failed");
  
    cudaMemcpy( d_image_org, d_image_prev, sizeof(float) * num_width * num_height * num_depth, cudaMemcpyDeviceToDevice );	
    CUT_CHECK_ERROR("Memory copy failed");

    for( int i = 0; i < iter; i++ ){
        
        // execute the kernel
        tv3d_cal_kernel<<< dimGrid, dimBlock >>>( d_image_prev, d_image_const_int, d_image_org,
                                                  d_image_curr, 
					          griddim, imagedim, 
    		       			          num_width, num_height, num_depth, 
						  weight, epsilon, timestep);

        cudaMemcpy( d_image_prev, d_image_curr, sizeof(float) * num_width * num_height * num_depth, cudaMemcpyDeviceToDevice );
	CUT_CHECK_ERROR("Memory copy failed");
    }

    cudaFree((void*)d_image_org);        
    
    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void 
tv3d_cal_kernel( float* d_image_prev, float * d_image_const_int, float* d_image_org,  
		 float* d_image_curr, 
		 dim3 griddim, dim3 imagedim, 
		 int num_width, int num_height, int num_depth, 
		 float weight, float epsilon, float timestep){

  uint idx, idy, idz, tid, tidx, tidy, tidz;

  idx = blockIdx.x;
  idz = blockIdx.y / griddim.y;
  idy = blockIdx.y - idz*griddim.y;

  idx = idx*blockDim.x + threadIdx.x;
  idy = idy*blockDim.y + threadIdx.y;
  idz = idz*blockDim.z + threadIdx.z;
  
  __shared__ float shmem_image[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size : 8x8x8
  __shared__ float shmem_image_org[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size : 8x8x8	
  __shared__ float shmem_image_const[BLOCK_DIM_X+2][BLOCK_DIM_Y+2][BLOCK_DIM_Z+2];   // pre-defined 3D block size : 8x8x8

  
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {
    dim3 cidx;
    cidx.x = threadIdx.x + 1;
    cidx.y = threadIdx.y + 1;
    cidx.z = threadIdx.z + 1;

    //
    // load shared memory
    //
    shmem_image[cidx.x][cidx.y][cidx.z] = d_image_prev[((idz)*imagedim.y + (idy))*imagedim.x + idx];
    shmem_image_org[cidx.x][cidx.y][cidx.z] = d_image_org[((idz)*imagedim.y + (idy))*imagedim.x + idx];
    shmem_image_const[cidx.x][cidx.y][cidx.z] = d_image_const_int[((idz)*imagedim.y + (idy))*imagedim.x + idx];	

    // loading neighbors : Neumann boundary condition
    if( threadIdx.x == 0 && threadIdx.y == 0 ){
      tidx = max(idx, 1) - 1;
      tidy = max(idy, 1) - 1;
      shmem_image[0][0][cidx.z] = d_image_prev[ (idz * imagedim.y + tidy ) * imagedim.x + tidx];
      shmem_image_org[0][0][cidx.z] = d_image_org[ (idz * imagedim.y + tidy ) * imagedim.x + tidx];
      shmem_image_const[0][0][cidx.z] = d_image_const_int[ (idz * imagedim.y + tidy ) * imagedim.x + tidx];
    }

    if( threadIdx.x == 0 && threadIdx.z == 0 ){
      tidx = max(idx, 1) - 1;
      tidz = max(idz, 1) - 1;
      shmem_image[0][cidx.y][0] = d_image_prev[ (tidz * imagedim.y + idy ) * imagedim.x + tidx];
      shmem_image_org[0][cidx.y][0] = d_image_org[ (tidz * imagedim.y + idy ) * imagedim.x + tidx];
      shmem_image_const[0][cidx.y][0] = d_image_const_int[ (tidz * imagedim.y + idy ) * imagedim.x + tidx];
    }

    if( threadIdx.y == 0 && threadIdx.z == 0 ){
      tidy = max(idy, 1) - 1;
      tidz = max(idz, 1) - 1;
      shmem_image[cidx.x][0][0] = d_image_prev[ (tidz * imagedim.y + tidy ) * imagedim.x + idx];
      shmem_image_org[cidx.x][0][0] = d_image_org[ (tidz * imagedim.y + tidy ) * imagedim.x + idx];
      shmem_image_const[cidx.x][0][0] = d_image_const_int[ (tidz * imagedim.y + tidy ) * imagedim.x + idx];
    }

    //
    
    if( threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 ){
      tidx = max(idx, 1) - 1;
      tidy = min(idy + 1, imagedim.y - 1);
      shmem_image[0][blockDim.y + 1][cidx.z] = d_image_prev[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
      shmem_image_org[0][blockDim.y + 1][cidx.z] = d_image_org[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
      shmem_image_const[0][blockDim.y + 1][cidx.z] = d_image_const_int[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
    }

    if( threadIdx.x == 0 && threadIdx.z == blockDim.z - 1 ){
      tidx = max(idx, 1) - 1;
      tidz = min(idz + 1, imagedim.z - 1);
      shmem_image[0][cidx.y][blockDim.z + 1] = d_image_prev[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
      shmem_image_org[0][cidx.y][blockDim.z + 1] = d_image_org[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
      shmem_image_const[0][cidx.y][blockDim.z + 1] = d_image_const_int[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
    }

    if( threadIdx.y == 0 && threadIdx.x == blockDim.x - 1 ){
      tidx = min(idx + 1, imagedim.x - 1);
      tidy = max(idy, 1) - 1; 
      shmem_image[blockDim.x + 1][0][cidx.z] = d_image_prev[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
      shmem_image_org[blockDim.x + 1][0][cidx.z] = d_image_org[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
      shmem_image_const[blockDim.x + 1][0][cidx.z] = d_image_const_int[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
    }

    if( threadIdx.y == 0 && threadIdx.z == blockDim.z - 1 ){
      tidy = max(idy, 1) - 1; 
      tidz = min(idz + 1, imagedim.z - 1);
      shmem_image[cidx.x][0][blockDim.z + 1] = d_image_prev[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
      shmem_image_org[cidx.x][0][blockDim.z + 1] = d_image_org[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
      shmem_image_const[cidx.x][0][blockDim.z + 1] = d_image_const_int[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
    }
 
    if( threadIdx.z == 0 && threadIdx.x == blockDim.x - 1 ){
      tidx = min(idx + 1, imagedim.x - 1);
      tidz = max(idz, 1) - 1; 
      shmem_image[blockDim.x + 1][cidx.y][0] = d_image_prev[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
      shmem_image_org[blockDim.x + 1][cidx.y][0] = d_image_org[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
      shmem_image_const[blockDim.x + 1][cidx.y][0] = d_image_const_int[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
    }

    if( threadIdx.z == 0 && threadIdx.y == blockDim.y - 1 ){
      tidy = min(idy + 1, imagedim.y - 1);
      tidz = max(idz, 1) - 1; 
      shmem_image[cidx.x][blockDim.y + 1][0] = d_image_prev[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
      shmem_image_org[cidx.x][blockDim.y + 1][0] = d_image_org[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
      shmem_image_const[cidx.x][blockDim.y + 1][0] = d_image_const_int[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];

    }

    //

    if( threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 ){
      tidx = min(idx + 1, imagedim.x - 1);
      tidy = min(idy + 1, imagedim.y - 1);
      shmem_image[blockDim.x+1][blockDim.y+1][cidx.z] = d_image_prev[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
      shmem_image_org[blockDim.x+1][blockDim.y+1][cidx.z] = d_image_org[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
      shmem_image_const[blockDim.x+1][blockDim.y+1][cidx.z] = d_image_const_int[ (idz * imagedim.y + tidy ) * imagedim.x + tidx ];
    }

    if( threadIdx.x == blockDim.x - 1 && threadIdx.z == blockDim.z - 1 ){
      tidx = min(idx + 1, imagedim.x - 1);
      tidz = min(idz + 1, imagedim.z - 1);
      shmem_image[blockDim.x+1][cidx.y][blockDim.z+1] = d_image_prev[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
      shmem_image_org[blockDim.x+1][cidx.y][blockDim.z+1] = d_image_org[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
      shmem_image_const[blockDim.x+1][cidx.y][blockDim.z+1] = d_image_const_int[ (tidz * imagedim.y + idy ) * imagedim.x + tidx ];
    }

    if( threadIdx.y == blockDim.y - 1 && threadIdx.z == blockDim.z - 1 ){
      tidy = min(idy + 1, imagedim.y - 1);
      tidz = min(idz + 1, imagedim.z - 1);
      shmem_image[cidx.x][blockDim.y+1][blockDim.z+1] = d_image_prev[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
      shmem_image_org[cidx.x][blockDim.y+1][blockDim.z+1] = d_image_org[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
      shmem_image_const[cidx.x][blockDim.y+1][blockDim.z+1] = d_image_const_int[ (tidz * imagedim.y + tidy ) * imagedim.x + idx ];
    }

    //
    if(threadIdx.x == 0)
    {
      tid = max(idx,1) - 1;
      shmem_image[0][cidx.y][cidx.z] = d_image_prev[((idz)*imagedim.y + (idy))*imagedim.x + tid];
      shmem_image_org[0][cidx.y][cidx.z] = d_image_org[((idz)*imagedim.y + (idy))*imagedim.x + tid];
      shmem_image_const[0][cidx.y][cidx.z] = d_image_const_int[((idz)*imagedim.y + (idy))*imagedim.x + tid]; 
      
    }

    if(threadIdx.x == blockDim.x-1)
    {
      tid = min(idx+1,imagedim.x-1);
      shmem_image[blockDim.x + 1][cidx.y][cidx.z] = d_image_prev[((idz)*imagedim.y + (idy))*imagedim.x + tid];
      shmem_image_org[blockDim.x + 1][cidx.y][cidx.z] = d_image_org[((idz)*imagedim.y + (idy))*imagedim.x + tid]; 
      shmem_image_const[blockDim.x + 1][cidx.y][cidx.z] = d_image_const_int[((idz)*imagedim.y + (idy))*imagedim.x + tid];

    }
    
    if(threadIdx.y == 0)
    {
      tid = max(idy,1) - 1;
      shmem_image[cidx.x][0][cidx.z] = d_image_prev[((idz)*imagedim.y + (tid))*imagedim.x + idx];
      shmem_image_org[cidx.x][0][cidx.z] = d_image_org[((idz)*imagedim.y + (tid))*imagedim.x + idx];
      shmem_image_const[cidx.x][0][cidx.z] = d_image_const_int[((idz)*imagedim.y + (tid))*imagedim.x + idx];
    }

    if(threadIdx.y == blockDim.y-1)
    {
      tid = min(idy+1,imagedim.y-1);
      shmem_image[cidx.x][blockDim.y + 1][cidx.z] = d_image_prev[((idz)*imagedim.y + (tid))*imagedim.x + idx];
      shmem_image_org[cidx.x][blockDim.y + 1][cidx.z] = d_image_org[((idz)*imagedim.y + (tid))*imagedim.x + idx];
      shmem_image_const[cidx.x][blockDim.y + 1][cidx.z] = d_image_const_int[((idz)*imagedim.y + (tid))*imagedim.x + idx];
     }
      
    if(threadIdx.z == 0)
    {
      tid = max(idz,1) - 1;
      shmem_image[cidx.x][cidx.y][0] = d_image_prev[((tid)*imagedim.y + (idy))*imagedim.x + idx];
      shmem_image_org[cidx.x][cidx.y][0] = d_image_org[((tid)*imagedim.y + (idy))*imagedim.x + idx];
      shmem_image_const[cidx.x][cidx.y][0] = d_image_const_int[((tid)*imagedim.y + (idy))*imagedim.x + idx];
    }

    if(threadIdx.z == blockDim.z-1)
    {
      tid = min(idz+1,imagedim.z-1);
      shmem_image[cidx.x][cidx.y][blockDim.z + 1] = d_image_prev[((tid)*imagedim.y + (idy))*imagedim.x + idx];
      shmem_image_org[cidx.x][cidx.y][blockDim.z + 1] = d_image_org[((tid)*imagedim.y + (idy))*imagedim.x + idx];
      shmem_image_const[cidx.x][cidx.y][blockDim.z + 1] = d_image_const_int[((tid)*imagedim.y + (idy))*imagedim.x + idx];      
    }
       
    __syncthreads();


    // prepare for diffusion

    float grad_x, grad_y, grad_z, grad_xx, grad_yy, grad_zz, grad_xy, grad_xz, grad_yz, curv, pixel;
    float image_const = 1.0f;
  
    grad_x = ( shmem_image[cidx.x + 1][cidx.y][cidx.z] - shmem_image[cidx.x - 1][cidx.y][cidx.z] ) / 2.0;
    grad_y = ( shmem_image[cidx.x][cidx.y + 1][cidx.z] - shmem_image[cidx.x][cidx.y - 1][cidx.z] ) / 2.0;
    grad_z = ( shmem_image[cidx.x][cidx.y][cidx.z + 1] - shmem_image[cidx.x][cidx.y][cidx.z - 1] ) / 2.0;

    grad_xx = shmem_image[cidx.x + 1][cidx.y][cidx.z] + shmem_image[cidx.x - 1][cidx.y][cidx.z]	
            - 2 * shmem_image[cidx.x][cidx.y][cidx.z];

    grad_yy = shmem_image[cidx.x][cidx.y + 1][cidx.z] + shmem_image[cidx.x][cidx.y - 1][cidx.z]	
            - 2 * shmem_image[cidx.x][cidx.y][cidx.z];

    grad_zz = shmem_image[cidx.x][cidx.y][cidx.z + 1] + shmem_image[cidx.x][cidx.y][cidx.z - 1]	
            - 2 * shmem_image[cidx.x][cidx.y][cidx.z];
	    
    grad_xy = (shmem_image[cidx.x + 1][cidx.y + 1][cidx.z] + shmem_image[cidx.x - 1][cidx.y - 1][cidx.z]	
            - shmem_image[cidx.x + 1][cidx.y - 1][cidx.z] - shmem_image[cidx.x - 1][cidx.y + 1][cidx.z])/4.0f;

    grad_xz = (shmem_image[cidx.x + 1][cidx.y][cidx.z + 1] + shmem_image[cidx.x - 1][cidx.y][cidx.z - 1]	
            - shmem_image[cidx.x + 1][cidx.y][cidx.z - 1] - shmem_image[cidx.x - 1][cidx.y][cidx.z + 1])/4.0f;

    grad_yz = (shmem_image[cidx.x][cidx.y + 1][cidx.z + 1] + shmem_image[cidx.x][cidx.y - 1][cidx.z - 1]	
            - shmem_image[cidx.x][cidx.y + 1][cidx.z - 1] - shmem_image[cidx.x][cidx.y - 1][cidx.z + 1])/4.0f;

    curv = grad_xx * (grad_y * grad_y + grad_z * grad_z) + grad_yy * (grad_x * grad_x + grad_z * grad_z)
	 + grad_zz * (grad_x * grad_x + grad_y * grad_y);
    curv -= 2 * ( grad_xy * grad_x * grad_y + grad_xz * grad_x * grad_z + grad_yz * grad_y * grad_z );

    curv /= ( ( grad_x * grad_x + grad_y * grad_y + grad_z * grad_z + epsilon ) );
  
    image_const = shmem_image_const[ cidx.x ][ cidx.y ][ cidx.z ]; 
    image_const *= shmem_image_const[ cidx.x + 1 ][ cidx.y ][ cidx.z ];
    image_const *= shmem_image_const[ cidx.x - 1 ][ cidx.y ][ cidx.z ];
    image_const *= shmem_image_const[ cidx.x ][ cidx.y + 1 ][ cidx.z ];
    image_const *= shmem_image_const[ cidx.x ][ cidx.y - 1 ][ cidx.z ];
    image_const *= shmem_image_const[ cidx.x ][ cidx.y ][ cidx.z + 1 ]; 
    image_const *= shmem_image_const[ cidx.x ][ cidx.y ][ cidx.z - 1 ]; 
    image_const *= shmem_image_const[ cidx.x + 1 ][ cidx.y ][ cidx.z + 1 ]; 
    image_const *= shmem_image_const[ cidx.x + 1 ][ cidx.y ][ cidx.z - 1 ]; 
    image_const *= shmem_image_const[ cidx.x - 1 ][ cidx.y ][ cidx.z + 1 ];
    image_const *= shmem_image_const[ cidx.x - 1 ][ cidx.y ][ cidx.z - 1 ];
    image_const *= shmem_image_const[ cidx.x ][ cidx.y + 1 ][ cidx.z + 1 ];
    image_const *= shmem_image_const[ cidx.x ][ cidx.y - 1 ][ cidx.z + 1 ]; 
    image_const *= shmem_image_const[ cidx.x ][ cidx.y + 1 ][ cidx.z - 1 ];
    image_const *= shmem_image_const[ cidx.x ][ cidx.y - 1 ][ cidx.z - 1 ]; 
    image_const *= shmem_image_const[ cidx.x + 1 ][ cidx.y + 1 ][ cidx.z ]; 
    image_const *= shmem_image_const[ cidx.x - 1 ][ cidx.y + 1 ][ cidx.z ]; 
    image_const *= shmem_image_const[ cidx.x + 1 ][ cidx.y - 1 ][ cidx.z ]; 
    image_const *= shmem_image_const[ cidx.x - 1 ][ cidx.y - 1 ][ cidx.z ];
 	
    pixel = shmem_image[cidx.x][cidx.y][cidx.z];
		
    pixel += image_const * timestep * ( - weight * (pixel - shmem_image_org[cidx.x][cidx.y][cidx.z]) + curv );

    // pixel += image_const * timestep * ( - weight * (pixel - shmem_image_org[cidx.x][cidx.y][cidx.z]) );

    // pixel += image_const * timestep * ( curv );

    // pixel = image_const * timestep * curv; // test
	     
    if( pixel < 0.0f )
       pixel = 0.0f;

    __syncthreads();

    d_image_curr[ ((idz)*imagedim.y + (idy))*imagedim.x + idx] = pixel;
                 
  }
 
}

#endif // TV3D_CAL_CU
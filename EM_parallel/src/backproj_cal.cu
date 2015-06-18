
// CUDA codes for Projection calculation

#ifndef __BACKPROJ_CAL_CU
#define __BACKPROJ_CAL_CU

#include <stdio.h>
#include <stdlib.h>

#include <cutil_inline.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>

#include "ParallelBeamEM.h"

texture<float, 3, cudaReadModeElementType> tex_proj;  // 3D texture

__global__ void backproj_cal_kernel_2D( float*, float*, 
 			             float* ,                       
			             dim3, dim3, dim3, 
				     float, float, float, float, float,
				     float, float, float, float, float);

__global__ void backproj_cal_kernel_3D( float*, float*, 
 			             float* ,                       
			             dim3, dim3, dim3, 
				     float, float, float, float, float,
				     float, float, float, float, float);

extern "C"
void backproj_cal_wrapper( cudaArray*  d_array_proj, float* d_image_prev,  float* d_image_iszero, 
			   float* d_image_curr,									  // output
			   int num_depth, int num_height, int num_width,                                          // parameters
			   int num_proj, int num_elevation, int num_ray, 
			   float lambda, float voxel_size, float proj_pixel_size, 
                           float SOD, float inv_rot, float xoffset, float yoffset, 
			   float zoffset, float start_rot, float end_rot){

    // setup execution parameters
    int blockWidth, blockHeight, blockDepth, nBlockX, nBlockY, nBlockZ;
    
    // Setting block size
    if(num_depth == 1 ) {
      blockWidth  = BLOCK_2DIM_X;
      blockHeight = BLOCK_2DIM_Y;
      blockDepth  = BLOCK_2DIM_Z;     

    }
    else {                  

      blockWidth  = BLOCK_3DIM_X;
      blockHeight = BLOCK_3DIM_Y;
      blockDepth  = BLOCK_3DIM_Z;     
    }

   // compute how many blocks are needed
    nBlockX = (int) ceil((float)num_width / (float)blockWidth);
    nBlockY = (int) ceil((float)num_height / (float)blockHeight);
    nBlockZ = (int) ceil((float)num_depth / (float)blockDepth);
   
    dim3 dimGrid(nBlockX, nBlockY*nBlockZ);                // 3D grid is not supported on G80
    dim3 dimBlock(blockWidth, blockHeight, blockDepth); 
    dim3 projdim(num_ray, num_elevation, num_proj);
    dim3 griddim(nBlockX, nBlockY, nBlockZ); 
    dim3 imagedim( num_width, num_height, num_depth);
   
    // set texture parameters
    tex_proj.normalized = false;                      // access with normalized texture coordinates
    tex_proj.filterMode = cudaFilterModeLinear;       // linear interpolation
    tex_proj.addressMode[0] = cudaAddressModeClamp;    // wrap texture coordinates
    tex_proj.addressMode[1] = cudaAddressModeClamp;
    tex_proj.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // float
    CUDA_SAFE_CALL(cudaBindTextureToArray(tex_proj, d_array_proj, float1Desc));

    // execute the kernel
    if( num_depth == 1 ){
      backproj_cal_kernel_2D<<< dimGrid, dimBlock >>>( d_image_prev, d_image_iszero,                   // input
						       d_image_curr,                                   // output
						       projdim, griddim, imagedim, 
						       lambda, voxel_size, proj_pixel_size, SOD, 
						       inv_rot, xoffset, yoffset, zoffset, 
						       start_rot, end_rot);      // parameters
    }
    else{
      backproj_cal_kernel_3D<<< dimGrid, dimBlock >>>( d_image_prev, d_image_iszero,                   // input
						       d_image_curr,                                   // output
						       projdim, griddim, imagedim, 
						       lambda, voxel_size, proj_pixel_size, SOD, 
						       inv_rot, xoffset, yoffset, zoffset, 
						       start_rot, end_rot);      // parameters
    }

    CUDA_SAFE_CALL( cudaUnbindTexture( tex_proj ) );

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");
}

__global__ void backproj_cal_kernel_2D( float* d_image_prev, float* d_image_iszero,		   
 			             float* d_image_curr,                             
			             dim3 projdim, dim3 griddim, dim3 imagedim, float lambda,
				     float voxel_size, float proj_pixel_size, float SOD,
	                             float inv_rot,  float xoffset, float yoffset, 
				     float zoffset, float start_rot, float end_rot){

  // 1. initialize shared memory
  
  dim3 bidx;
  uint idx, idy, idz;

  bidx.x = blockIdx.x;   
  bidx.z = blockIdx.y / griddim.y;
  bidx.y = blockIdx.y - bidx.z*griddim.y;   

  idx = bidx.x * blockDim.x + threadIdx.x;
  idy = bidx.y * blockDim.y + threadIdx.y;
  idz = bidx.z * blockDim.z + threadIdx.z;

  // shared memory
  __shared__ float shmem_d_image_prev[BLOCK_2DIM_X][BLOCK_2DIM_Y][BLOCK_2DIM_Z];

#ifndef WEIGHT_CAL
  __shared__ float shmem_d_image_iszero[BLOCK_2DIM_X][BLOCK_2DIM_Y][BLOCK_2DIM_Z];
#endif 

  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;
  cidx.z = threadIdx.z;

  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {    
    // load shared memory
    shmem_d_image_prev[cidx.x][cidx.y][cidx.z] = d_image_prev[((idz)*imagedim.y + (idy))*imagedim.x + idx];

#ifndef WEIGHT_CAL
    shmem_d_image_iszero[cidx.x][cidx.y][cidx.z] = d_image_iszero[((idz)*imagedim.y + (idy))*imagedim.x + idx];
#endif

  }

  __syncthreads();
  
  // 2. apply kernel
   
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {

    float pixel = 0.0f;
                
    float i;
    float x, y, z, t, u, v, angle;
    float weight_voxel = 0.0f;
     
    x = (1.0f * idx - 1.0f * imagedim.x / 2.0f) * voxel_size;	
    y = (1.0f * idy - 1.0f * imagedim.y / 2.0f) * voxel_size;
    z = (1.0f * idz - 1.0f * imagedim.z / 2.0f) * voxel_size;

    // i = projdim.z/2;
    for( i = 0.0f; i < 1.0f * projdim.z; i = i + 1.0f )             
    {

	// rotation angle (Parallel beam)
#ifdef ROTATION_CLOCKWISE
      // angle = -PI * i / projdim.z;
      angle = -(start_rot + i * inv_rot) * PI / 180.0f;
#else
      // angle = PI * i / projdim.z + PI / 2.0f;
      angle = (start_rot + i * inv_rot) * PI / 180.0f + PI / 2.0f;
#endif

	u =  -x * sinf( angle ) + y * cosf( angle ) ;
	v = z;

	u = u / proj_pixel_size + 1.0f * projdim.x / 2.0f;
	v = v / proj_pixel_size + 1.0f * projdim.y / 2.0f;

	if( u >= 0.0f && u < 1.0f * projdim.x && v >= 0.0f && v < 1.0f * projdim.y ){

	  pixel += tex3D( tex_proj, u + 0.5f, v + 0.5f, 1.0f * i + 0.5f );

	  float pixel_tmp1 = tex3D( tex_proj, floorf( u ) + 0.5f, floorf( v ) + 0.5f, 1.0f * i + 0.5f );
	  float pixel_tmp2 = tex3D( tex_proj, floorf( u ) + 1.0f + 0.5f, floorf( v ) + 0.5f, 1.0f * i + 0.5f );

	  // interpolation in v direction is not required for parallel beam
	  if( fabsf( pixel_tmp1 ) > 1e-5f )
	    weight_voxel += 1.0f - (u - floorf( u )) ;

	  if( fabsf( pixel_tmp2 ) > 1e-5f )
	    weight_voxel += u - floorf( u );
           
	}
    }

    if( weight_voxel > 0.0f )
      pixel = shmem_d_image_prev[cidx.x][cidx.y][cidx.z] * pixel / weight_voxel;
    else
      pixel = 0.0; 

#ifndef WEIGHT_CAL
    pixel *= shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];
#endif

#ifdef STEVE_DATA
    // thresholding
    if (pixel > 0.3f)
      pixel = 0.3f;
#endif

    __syncthreads();
        
    // store the result
    uint outidx = ((idz)* imagedim.y + (idy))*imagedim.x + idx;

    d_image_curr[ outidx ] =  pixel;

  } 
}	


__global__ void backproj_cal_kernel_3D( float* d_image_prev, float* d_image_iszero,		   
 			             float* d_image_curr,                             
			             dim3 projdim, dim3 griddim, dim3 imagedim, float lambda,
				     float voxel_size, float proj_pixel_size, float SOD,
	                             float inv_rot,  float xoffset, float yoffset, 
				     float zoffset, float start_rot, float end_rot){

  // 1. initialize shared memory
  
  dim3 bidx;
  uint idx, idy, idz;

  bidx.x = blockIdx.x;   
  bidx.z = blockIdx.y / griddim.y;
  bidx.y = blockIdx.y - bidx.z*griddim.y;   

  idx = bidx.x * blockDim.x + threadIdx.x;
  idy = bidx.y * blockDim.y + threadIdx.y;
  idz = bidx.z * blockDim.z + threadIdx.z;

  // shared memory
  __shared__ float shmem_d_image_prev[BLOCK_3DIM_X][BLOCK_3DIM_Y][BLOCK_3DIM_Z];

#ifndef WEIGHT_CAL
  __shared__ float shmem_d_image_iszero[BLOCK_3DIM_X][BLOCK_3DIM_Y][BLOCK_3DIM_Z];
#endif 

  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;
  cidx.z = threadIdx.z;

  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {    
    // load shared memory
    shmem_d_image_prev[cidx.x][cidx.y][cidx.z] = d_image_prev[((idz)*imagedim.y + (idy))*imagedim.x + idx];

#ifndef WEIGHT_CAL
    shmem_d_image_iszero[cidx.x][cidx.y][cidx.z] = d_image_iszero[((idz)*imagedim.y + (idy))*imagedim.x + idx];
#endif

  }

  __syncthreads();
  
  // 2. apply kernel
   
  if(idx < imagedim.x && idy < imagedim.y && idz < imagedim.z)
  {

    float pixel = 0.0f;
                
    float i;
    float x, y, z, t, u, v, angle;
    float weight_voxel = 0.0f;
     
    x = (1.0f * idx - 1.0f * imagedim.x / 2.0f) * voxel_size;	
    y = (1.0f * idy - 1.0f * imagedim.y / 2.0f) * voxel_size;
    z = (1.0f * idz - 1.0f * imagedim.z / 2.0f) * voxel_size;

    // i = projdim.z/2;
    for( i = 0.0f; i < 1.0f * projdim.z; i = i + 1.0f )             
    {

	// rotation angle (Parallel beam)
#ifdef ROTATION_CLOCKWISE
      // angle = -PI * i / projdim.z;
      angle = -(start_rot + i * inv_rot) * PI / 180.0f;
#else
      // angle = PI * i / projdim.z + PI / 2.0f;
      angle = (start_rot + i * inv_rot) * PI / 180.0f + PI / 2.0f;
#endif

	u =  -x * sinf( angle ) + y * cosf( angle ) ;
	v = z;

	u = u / proj_pixel_size + 1.0f * projdim.x / 2.0f;
	v = v / proj_pixel_size + 1.0f * projdim.y / 2.0f;

	if( u >= 0.0f && u < 1.0f * projdim.x && v >= 0.0f && v < 1.0f * projdim.y ){

	  pixel += tex3D( tex_proj, u + 0.5f, v + 0.5f, 1.0f * i + 0.5f );

	  float pixel_tmp1 = tex3D( tex_proj, floorf( u ) + 0.5f, floorf( v ) + 0.5f, 1.0f * i + 0.5f );
	  float pixel_tmp2 = tex3D( tex_proj, floorf( u ) + 1.0f + 0.5f, floorf( v ) + 0.5f, 1.0f * i + 0.5f );

	  // interpolation in v direction is not required for parallel beam
	  if( fabsf( pixel_tmp1 ) > 1e-5f )
	    weight_voxel += 1.0f - (u - floorf( u )) ;

	  if( fabsf( pixel_tmp2 ) > 1e-5f )
	    weight_voxel += u - floorf( u );
           
	}
    }

    if( weight_voxel > 0.0f )
      pixel = shmem_d_image_prev[cidx.x][cidx.y][cidx.z] * pixel / weight_voxel;
    else
      pixel = 0.0; 

#ifndef WEIGHT_CAL
    pixel *= shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];
#endif

#ifdef STEVE_DATA
    // thresholding
    if (pixel > 0.3f)
      pixel = 0.3f;
#endif


#ifdef PETER_DATA
    // thresholding
    if (pixel > 0.01f)
      pixel = 0.01f;
#endif


    __syncthreads();
        
    // store the result
    uint outidx = ((idz)* imagedim.y + (idy))*imagedim.x + idx;

    d_image_curr[ outidx ] =  pixel;

  } 
}	

#endif // __BACKPROJ_CAL_CU

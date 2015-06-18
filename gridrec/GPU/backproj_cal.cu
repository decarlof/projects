
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

#include "tomo_recon.h"

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
void backproj_cal_wrapper( cudaArray*  d_array_proj, float* d_image_prev,  float* d_image_iszero,      // input
			   float* d_image_curr,							       // output
			   int num_depth, int num_height, int num_width,                               // parameters
			   int num_proj, int num_elevation, int num_ray, 
			   float lambda, float voxel_size, float proj_pixel_size, 
                           float SOD, float inv_rot,
			   float xoffset, float yoffset, float zoffset, float start_rot, float end_rot){

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
    if( num_depth == 1 )
      backproj_cal_kernel_2D<<< dimGrid, dimBlock >>>( d_image_prev, d_image_iszero,                        // input
			          		  d_image_curr,                                        // output
  						  projdim, griddim, imagedim, 
						  lambda, voxel_size, proj_pixel_size, SOD, inv_rot,   // parameters
						  xoffset, yoffset, zoffset, start_rot, end_rot );
    else
      backproj_cal_kernel_3D<<< dimGrid, dimBlock >>>( d_image_prev, d_image_iszero,                        // input
			          		  d_image_curr,                                        // output
  						  projdim, griddim, imagedim, 
						  lambda, voxel_size, proj_pixel_size, SOD, inv_rot,   // parameters
						  xoffset, yoffset, zoffset, start_rot, end_rot );

    CUDA_SAFE_CALL( cudaUnbindTexture( tex_proj ) );

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");
}

__global__ void backproj_cal_kernel_2D( float* d_image_prev, float* d_image_iszero,		   
 			             float* d_image_curr,                             
			             dim3 projdim, dim3 griddim, dim3 imagedim, float lambda,
				     float voxel_size, float proj_pixel_size, float SOD, float inv_rot,
				     float xoffset, float yoffset, float zoffset, float start_rot, float end_rot){

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
    float x, y, z, u, v, angle;
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
 
    // if( weight_voxel > 0.0f )

#ifndef WEIGHT_CAL

    if( weight_voxel > 0.1f )
        pixel = ( lambda * pixel / weight_voxel + shmem_d_image_prev[cidx.x][cidx.y][cidx.z] ) 
	       *  shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];
    else
        pixel =  shmem_d_image_prev[cidx.x][cidx.y][cidx.z] 
	       *  shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];

    if( pixel < 0.0f )
      pixel = 0.0f;

    if( isnan( pixel ) )
      pixel = shmem_d_image_prev[cidx.x][cidx.y][cidx.z] * shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];; 

#else

    if( weight_voxel > 0.1f )
        pixel = ( lambda * pixel / weight_voxel + shmem_d_image_prev[cidx.x][cidx.y][cidx.z] );
    else
        pixel =  shmem_d_image_prev[cidx.x][cidx.y][cidx.z];

    if( pixel < 0.0f )
      pixel = 0.0f;

    if( isnan( pixel ) )
      pixel = shmem_d_image_prev[cidx.x][cidx.y][cidx.z]; 

#endif // WEIGHT_CAL

    // debug
    // pixel = u;
    // pixel = tex3D( tex_proj, 9.0f + 0.5f, 189.0f + 0.5f, 0.5f );
    // pixel =  1.0f * shmem_d_image_iszero[cidx.x][cidx.y][cidx.z]; 

    __syncthreads();
        
    // store the result
    uint outidx = ((idz)* imagedim.y + (idy))*imagedim.x + idx;

    d_image_curr[ outidx ] =  pixel;

  } 
}	

__global__ void backproj_cal_kernel_3D( float* d_image_prev, float* d_image_iszero,		   
 			             float* d_image_curr,                             
			             dim3 projdim, dim3 griddim, dim3 imagedim, float lambda,
				     float voxel_size, float proj_pixel_size, float SOD, float inv_rot,
				     float xoffset, float yoffset, float zoffset, float start_rot, float end_rot){

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
    float x, y, z, u, v, angle;
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
 
    // if( weight_voxel > 0.0f )

#ifndef WEIGHT_CAL

    if( weight_voxel > 0.1f )
        pixel = ( lambda * pixel / weight_voxel + shmem_d_image_prev[cidx.x][cidx.y][cidx.z] ) 
	       *  shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];
    else
        pixel =  shmem_d_image_prev[cidx.x][cidx.y][cidx.z] 
	       *  shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];

    if( pixel < 0.0f )
      pixel = 0.0f;

    if( isnan( pixel ) )
      pixel = shmem_d_image_prev[cidx.x][cidx.y][cidx.z] * shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];; 

#else

    if( weight_voxel > 0.1f )
        pixel = ( lambda * pixel / weight_voxel + shmem_d_image_prev[cidx.x][cidx.y][cidx.z] );
    else
        pixel =  shmem_d_image_prev[cidx.x][cidx.y][cidx.z];

    if( pixel < 0.0f )
      pixel = 0.0f;

    if( isnan( pixel ) )
      pixel = shmem_d_image_prev[cidx.x][cidx.y][cidx.z]; 

#endif // WEIGHT_CAL

    // debug
    // pixel = u;
    // pixel = tex3D( tex_proj, 9.0f + 0.5f, 189.0f + 0.5f, 0.5f );
    // pixel =  1.0f * shmem_d_image_iszero[cidx.x][cidx.y][cidx.z]; 

    __syncthreads();
        
    // store the result
    uint outidx = ((idz)* imagedim.y + (idy))*imagedim.x + idx;

    d_image_curr[ outidx ] =  pixel;

  } 
}	

////////////////////////////////////////////////////////////////////////////////////////////////////////
// implementations using shared memory instead of texture memory

__global__ void backproj_cal_kernel_sharemem_2D( float*, float*, float*, float* ,                       
						 dim3, dim3, dim3, 
						 float, float, float, float, float,
						 unsigned int);

__global__ void backproj_cal_kernel_sharemem_3D( float*, float*, float*, float* ,                       
						 dim3, dim3, dim3, 
						 float, float, float, float, float,
						 unsigned int);
extern "C"
void backproj_cal_wrapper_sharemem( float* d_proj_cur, float* d_image_prev,  float* d_image_iszero,  // input
				    float* d_image_curr,			      		     // output
				    int num_depth, int num_height, int num_width,                    // parameters
				    int num_proj, int num_elevation, int num_ray, 
				    float lambda, float voxel_size, float proj_pixel_size, 
				    float SOD, float inv_rot,
				    unsigned int deviceShareMemSizePerBlock){

    // setup execution parameters
    int blockWidth, blockHeight, blockDepth, nBlockX, nBlockY, nBlockZ;
    
    // Setting block size
    if( num_depth == 1 ) {
      blockWidth  = BLOCK_2DIM_X;
      blockHeight = BLOCK_2DIM_Y;   

      // compute how many blocks are needed
      nBlockX = (int) ceil((float)num_width / (float)blockWidth);
      nBlockY = (int) ceil((float)num_height / (float)blockHeight);
   
      dim3 dimGrid(nBlockX, nBlockY);                // 3D grid is not supported on G80
      dim3 dimBlock(blockWidth, blockHeight, 1); 
      dim3 projdim(num_ray, num_proj, 1);
      dim3 griddim(nBlockX, nBlockY, 1); 
      dim3 imagedim( num_width, num_height, 1);
   
      // execute the kernel
      backproj_cal_kernel_sharemem_2D<<< dimGrid, dimBlock >>>( d_proj_cur, d_image_prev, d_image_iszero, // input
								d_image_curr,                             // output
								projdim, griddim, imagedim, 
								lambda, voxel_size, proj_pixel_size, SOD, 
								inv_rot, deviceShareMemSizePerBlock);  // parameters
    }
    else {                  

      blockWidth  = BLOCK_3DIM_X;
      blockHeight = BLOCK_3DIM_Y;
      blockDepth  = BLOCK_3DIM_Z;     

      // compute how many blocks are needed
      nBlockX = (int) ceil((float)num_width / (float)blockWidth);
      nBlockY = (int) ceil((float)num_height / (float)blockHeight);
      nBlockZ = (int) ceil((float)num_depth / (float)blockDepth);
   
      dim3 dimGrid(nBlockX, nBlockY*nBlockZ);                // 3D grid is not supported on G80
      dim3 dimBlock(blockWidth, blockHeight, blockDepth); 
      dim3 projdim(num_ray, num_elevation, num_proj);
      dim3 griddim(nBlockX, nBlockY, nBlockZ); 
      dim3 imagedim( num_width, num_height, num_depth);
   
      // execute the kernel
      backproj_cal_kernel_sharemem_3D<<< dimGrid, dimBlock >>>( d_proj_cur, d_image_prev, d_image_iszero,  // input
								d_image_curr,                              // output
								projdim, griddim, imagedim, 
								lambda, voxel_size, proj_pixel_size, SOD, 
								inv_rot, deviceShareMemSizePerBlock);  // parameters
    
    }

    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");
}

__global__ void backproj_cal_kernel_sharemem_2D( float* d_proj_cur, float* d_image_prev, float* d_image_iszero,	 
						 float* d_image_curr,                             
						 dim3 projdim, dim3 griddim, dim3 imagedim, float lambda,
						 float voxel_size, float proj_pixel_size, float SOD,
						 float inv_rot, unsigned int deviceShareMemSizePerBlock){

  // 1. initialize shared memory

  uint idx, idy, idx_c, idy_c;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  idx_c = blockIdx.x * blockDim.x + blockDim.x/2;
  idy_c = blockIdx.y * blockDim.y + blockDim.y/2;

  // shared memory
  __shared__ float shmem_d_image_prev[BLOCK_2DIM_X][BLOCK_2DIM_Y];
  __shared__ float shmem_proj[BLOCK_2DIM_X];         // 
  // __shared__ float shmem_proj[2* BLOCK_2DIM_X + 2* BLOCK_2DIM_Y];         // 

#ifndef WEIGHT_CAL
  __shared__ float shmem_d_image_iszero[BLOCK_2DIM_X][BLOCK_2DIM_Y];
#endif 

  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;

  if(idx < imagedim.x && idy < imagedim.y)
  {    
    // load shared memory
    
    shmem_d_image_prev[cidx.x][cidx.y] = d_image_prev[idy*imagedim.x + idx];

#ifndef WEIGHT_CAL
    shmem_d_image_iszero[cidx.x][cidx.y] = d_image_iszero[idy*imagedim.x + idx];
#endif
          
  }

  __syncthreads();
  
  // 2. apply kernel
   
  if(idx < imagedim.x && idy < imagedim.y)
  {

    float pixel = 0.0f;
    float x, y, u, angle;
    float weight_voxel = 0.0f;
    int   i, ub_c, ub_s; 

    for( i = 0; i < 1.0f * projdim.y; i = i + 1 )             
    {
    
	// rotation angle (Parallel beam)
#ifdef ROTATION_CLOCKWISE
	angle = -PI * i / projdim.y;
#else
	angle = PI * i / projdim.y + PI / 2.0f;
#endif

	// find the center of the shared memory block
	x = (1.0f * idx_c - 1.0f * imagedim.x / 2.0f) * voxel_size;	
	y = (1.0f * idy_c - 1.0f * imagedim.y / 2.0f) * voxel_size;

	u =  -x * sinf( angle ) + y * cosf( angle ) ;
	u = u / proj_pixel_size + 1.0f * projdim.x / 2.0f;

	ub_c = (int) round( u ); 

	// load data to shared memory
	ub_s = ub_c - (BLOCK_2DIM_X + BLOCK_2DIM_Y); 

	if( ub_s + cidx.x >= 0.0f && ub_s + cidx.x < projdim.x )
	  shmem_proj[ cidx.x ] = d_proj_cur[ i * projdim.x + ub_s + cidx.x ];

	__syncthreads();	  

	ub_s = ub_c - BLOCK_2DIM_Y; 

	if( ub_s + cidx.y >= 0.0f && ub_s + cidx.y < projdim.x )
	  shmem_proj[ cidx.x + BLOCK_2DIM_X ] = d_proj_cur[ i * projdim.x + ub_s + cidx.y ];

	__syncthreads();	  

	ub_s = ub_c;

	if( ub_s + cidx.x >= 0.0f && ub_s + cidx.x < projdim.x )
	  shmem_proj[ cidx.x + BLOCK_2DIM_X  + BLOCK_2DIM_Y ] = d_proj_cur[ i * projdim.x + ub_s + cidx.x ];

	__syncthreads();	  

	ub_s = ub_c + BLOCK_2DIM_X;

	if( ub_s + cidx.y >= 0.0f && ub_s + cidx.y < projdim.x )
	  shmem_proj[ cidx.y  + 2 * BLOCK_2DIM_X  + BLOCK_2DIM_Y ] = d_proj_cur[ i * projdim.x + ub_s + cidx.y ];

	__syncthreads();	  


	// perform backprojection
	x = (1.0f * idx - 1.0f * imagedim.x / 2.0f) * voxel_size;	
	y = (1.0f * idy - 1.0f * imagedim.y / 2.0f) * voxel_size;

	u =  -x * sinf( angle ) + y * cosf( angle ) ;

	u = u / proj_pixel_size + 1.0f * projdim.x / 2.0f;

	int nu0, nu1;
	float pixel_tmp1 = 0.0f, pixel_tmp2 = 0.0f;
	
	if( u >= -1.0f && u < 1.0f * projdim.x ){

	  nu0 = (int)floor(u);
	  nu1 = nu0 + 1;

	  if( nu0 >= 0.0f && nu0 <= 1.0f * projdim.x - 1.0f)
	    pixel_tmp1 = shmem_proj[nu0 - ub_s];
	  if( nu1 >= 0.0f && nu1 <= 1.0f * projdim.x - 1.0f)
	    pixel_tmp2 = shmem_proj[nu1 - ub_s];

	  pixel += pixel_tmp1 * (1.0f - (u - floorf( u ))) + pixel_tmp2 * (u - floorf( u ));

	  // interpolation in v direction is not required for parallel beam
	  if( fabsf( pixel_tmp1 ) > 1e-5f )
	    weight_voxel += 1.0f - (u - floorf( u )) ;

	  if( fabsf( pixel_tmp2 ) > 1e-5f )
	    weight_voxel += u - floorf( u );
           
	}
    }
 
#ifndef WEIGHT_CAL

    if( weight_voxel > 0.1f )
        pixel = ( lambda * pixel / weight_voxel + shmem_d_image_prev[cidx.x][cidx.y] ) 
	       *  shmem_d_image_iszero[cidx.x][cidx.y];
    else
        pixel =  shmem_d_image_prev[cidx.x][cidx.y] * shmem_d_image_iszero[cidx.x][cidx.y];

    if( pixel < 0.0f )
      pixel = 0.0f;

    if( isnan( pixel ) )
      pixel = shmem_d_image_prev[cidx.x][cidx.y] * shmem_d_image_iszero[cidx.x][cidx.y]; 

#else

    if( weight_voxel > 0.1f )
        pixel = ( lambda * pixel / weight_voxel + shmem_d_image_prev[cidx.x][cidx.y] );
    else
        pixel =  shmem_d_image_prev[cidx.x][cidx.y];

    if( pixel < 0.0f )
      pixel = 0.0f;

    if( isnan( pixel ) )
      pixel = shmem_d_image_prev[cidx.x][cidx.y]; 

#endif // WEIGHT_CAL

    // debug
    // pixel = u;
    // pixel = tex3D( tex_proj, 9.0f + 0.5f, 189.0f + 0.5f, 0.5f );
    // pixel =  1.0f * shmem_d_image_iszero[cidx.x][cidx.y][cidx.z]; 

    __syncthreads();
        
    // store the result
    uint outidx = idy*imagedim.x + idx;
    d_image_curr[ outidx ] =  pixel;

  } 
}	

__global__ void backproj_cal_kernel_sharemem_3D( float* d_proj_cur, float* d_image_prev, float* d_image_iszero,	  
						 float* d_image_curr,                             
						 dim3 projdim, dim3 griddim, dim3 imagedim, float lambda,
						 float voxel_size, float proj_pixel_size, float SOD,
						 float inv_rot, unsigned int deviceShareMemSizePerBlock){

  // 1. initialize shared memory
  
  dim3 bidx;
  uint idx, idy, idz, idx_c, idy_c;

  bidx.x = blockIdx.x;   
  bidx.z = blockIdx.y / griddim.y;
  bidx.y = blockIdx.y - bidx.z*griddim.y;   

  idx = bidx.x * blockDim.x + threadIdx.x;
  idy = bidx.y * blockDim.y + threadIdx.y;
  idz = bidx.z * blockDim.z + threadIdx.z;

  idx_c = bidx.x * blockDim.x + blockDim.x / 2;
  idy_c = bidx.y * blockDim.y + blockDim.y / 2;

  // shared memory
  __shared__ float shmem_d_image_prev[BLOCK_3DIM_X][BLOCK_3DIM_Y][BLOCK_3DIM_Z];
  __shared__ float shmem_proj[BLOCK_3DIM_X + BLOCK_3DIM_Y][BLOCK_3DIM_Z];         // 

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
                
    int i;
    float x, y, z, u, v, angle;
    float weight_voxel = 0.0f;
    int   ub_c, ub_s; 
     
    int   vb_s = bidx.z * blockDim.z;

    for( i = 0; i < 1.0f * projdim.z; i = i + 1 )             
    {
    
	// rotation angle (Parallel beam)
#ifdef ROTATION_CLOCKWISE
	angle = -PI * i / projdim.z;
#else
	angle = PI * i / projdim.z + PI / 2.0f;
#endif

	// find the center of the shared memory block
	x = (1.0f * idx_c - 1.0f * imagedim.x / 2.0f) * voxel_size;	
	y = (1.0f * idy_c - 1.0f * imagedim.y / 2.0f) * voxel_size;

	u =  -x * sinf( angle ) + y * cosf( angle ) ;
	u = u / proj_pixel_size + 1.0f * projdim.x / 2.0f;

	ub_c = (int) round( u ); 

	// load data to shared memory. Note the coordinate difference between object space and projection space
        // cidx.z here corresponds to elevation in the projection data and depth in the object space
	shmem_proj[cidx.x + cidx.y][cidx.z] = d_proj_cur[ (i * projdim.y + idz) * projdim.x + ub_s + cidx.x + cidx.y ];

	__syncthreads();	  

	// load data to shared memory
	ub_s = ub_c - (BLOCK_2DIM_X + BLOCK_2DIM_Y); 

	if( ub_s + cidx.x >= 0.0f && ub_s + cidx.x < projdim.x )
	  shmem_proj[ cidx.x ][cidx.z] = d_proj_cur[ (i * projdim.y + idz) * projdim.x + ub_s + cidx.x ];

	__syncthreads();	  

	ub_s = ub_c - BLOCK_2DIM_Y; 

	if( ub_s + cidx.y >= 0.0f && ub_s + cidx.y < projdim.x )
	  shmem_proj[ cidx.x + BLOCK_2DIM_X ][cidx.z] = d_proj_cur[ (i * projdim.y + idz) * projdim.x + ub_s + cidx.y ];

	__syncthreads();	  

	ub_s = ub_c;

	if( ub_s + cidx.x >= 0.0f && ub_s + cidx.x < projdim.x )
	  shmem_proj[ cidx.x + BLOCK_2DIM_X  + BLOCK_2DIM_Y ][cidx.z] = d_proj_cur[ (i * projdim.y + idz) * projdim.x + ub_s + cidx.x ];

	__syncthreads();	  

	ub_s = ub_c + BLOCK_2DIM_X;

	if( ub_s + cidx.y >= 0.0f && ub_s + cidx.y < projdim.x )
	  shmem_proj[ cidx.y  + 2 * BLOCK_2DIM_X  + BLOCK_2DIM_Y ][cidx.z] = d_proj_cur[ (i * projdim.y + idz) * projdim.x + ub_s + cidx.y ];

	__syncthreads();	  

	// perform backprojection
	x = (1.0f * idx - 1.0f * imagedim.x / 2.0f) * voxel_size;	
	y = (1.0f * idy - 1.0f * imagedim.y / 2.0f) * voxel_size;
	z = (1.0f * idz - 1.0f * imagedim.z / 2.0f) * voxel_size;

	u =  -x * sinf( angle ) + y * cosf( angle ) ;
	v = z;

	u = u / proj_pixel_size + 1.0f * projdim.x / 2.0f;
	v = v / proj_pixel_size + 1.0f * projdim.y / 2.0f;

	float pixel_tmp00 = 0.0f, pixel_tmp01 = 0.0f, pixel_tmp10 = 0.0f, pixel_tmp11 = 0.0f;
	int nu0, nu1, nv0, nv1; 

	if( u >= -1.0f && u < 1.0f * projdim.x && v >= 0.0f && v < 1.0f * projdim.y ){

	  nu0 = (int)floor(u);
	  nv0 = (int)floor(v);
	  nu1 = nu0 + 1;
	  nv1 = nv0 + 1; 

	  if( nu0 >= 0.0f && nu0 <= 1.0f * imagedim.x - 1.0f && nv0 >= 0.0f && nv0 <= 1.0f * imagedim.y - 1.0f )
	    pixel_tmp00 = shmem_proj[nu0 - ub_s][nv0 - vb_s];
	  if( nu1 >= 0.0f && nu1 <= 1.0f * imagedim.x - 1.0f && nv0 >= 0.0f && nv0 <= 1.0f * imagedim.y - 1.0f )
	    pixel_tmp01 = shmem_proj[nu1 - ub_s][nv0 - vb_s];
	  if( nu0 >= 0.0f && nu0 <= 1.0f * imagedim.x - 1.0f && nv1 >= 0.0f && nv1 <= 1.0f * imagedim.y - 1.0f )
	    pixel_tmp10 = shmem_proj[nu0 - ub_s][nv1 - vb_s];
	  if( nu1 >= 0.0f && nu1 <= 1.0f * imagedim.x - 1.0f && nv1 >= 0.0f && nv1 <= 1.0f * imagedim.y - 1.0f )
	    pixel_tmp11 = shmem_proj[nu1 - ub_s][nv1 - vb_s];

	  pixel += pixel_tmp00 * (1.0f - (u - floorf( u ))) * (1.0f - (v - floorf( v ))) ;
	  pixel += pixel_tmp01 * (u - floorf( u )) * (1.0f - (v - floorf( v ))) ;
	  pixel += pixel_tmp10 * (1.0f - (u - floorf( u ))) * (v - floorf( v )) ;
	  pixel += pixel_tmp11 * (u - floorf( u )) * (v - floorf( v )) ;

	  // interpolation in v direction is not required for parallel beam
	  if( fabsf( pixel_tmp00 ) > 1e-5f )
	    weight_voxel += ( 1.0f - (u - floorf( u ))) * (1.0f - (v - floorf( v ))) ;
	  if( fabsf( pixel_tmp01 ) > 1e-5f )
	    weight_voxel += ( u - floorf( u )) * (1.0f - (v - floorf( v ))) ;
	  if( fabsf( pixel_tmp10 ) > 1e-5f )
	    weight_voxel += ( 1.0f - (u - floorf( u ))) * (v - floorf( v )) ;
	  if( fabsf( pixel_tmp00 ) > 1e-5f )
	    weight_voxel += ( u - floorf( u )) * (v - floorf( v )) ;
           
	}
    }
 
#ifndef WEIGHT_CAL

    if( weight_voxel > 0.1f )
        pixel = ( lambda * pixel / weight_voxel + shmem_d_image_prev[cidx.x][cidx.y][cidx.z] ) 
	       *  shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];
    else
        pixel =  shmem_d_image_prev[cidx.x][cidx.y][cidx.z] 
	       *  shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];

    if( pixel < 0.0f )
      pixel = 0.0f;

    if( isnan( pixel ) )
      pixel = shmem_d_image_prev[cidx.x][cidx.y][cidx.z] * shmem_d_image_iszero[cidx.x][cidx.y][cidx.z];; 

#else

    if( weight_voxel > 0.1f )
        pixel = ( lambda * pixel / weight_voxel + shmem_d_image_prev[cidx.x][cidx.y][cidx.z] );
    else
        pixel =  shmem_d_image_prev[cidx.x][cidx.y][cidx.z];

    if( pixel < 0.0f )
      pixel = 0.0f;

    if( isnan( pixel ) )
      pixel = shmem_d_image_prev[cidx.x][cidx.y][cidx.z]; 

#endif // WEIGHT_CAL

    // debug
    // pixel = tex3D( tex_proj, 9.0f + 0.5f, 189.0f + 0.5f, 0.5f );
    // pixel =  1.0f * shmem_d_image_iszero[cidx.x][cidx.y][cidx.z]; 

    __syncthreads();
        
    // store the result
    uint outidx = ((idz)* imagedim.y + (idy))*imagedim.x + idx;

    d_image_curr[ outidx ] =  pixel;

  } 
}	


#endif // __BACKPROJ_CAL_CU

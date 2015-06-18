// CUDA codes for Projection calculation

#ifndef __PROJ_CAL_CU
#define __PROJ_CAL_CU

#include <stdio.h>
#include <stdlib.h>

#include <cutil_inline.h>
#include <cutil.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>
 
#include "ParallelBeamEM.h"

texture<float, 3, cudaReadModeElementType> tex_voxel;  // 3D texture

#define PRECISION 1e-5f
#define PRECISION_WEIGHT 1e-1f

__global__ void proj_cal_kernel_2D( float*, float*, int, float*, dim3, dim3, dim3, float, float, float, 
				    float, float, float, float, float, float, float);

__global__ void proj_cal_kernel_3D( float*, float*, int, float*, dim3, dim3, dim3, float, float, float, 
				    float, float, float, float, float, float, float);

extern "C" 
void proj_cal_wrapper( cudaArray* d_array_voxel, float* d_proj, int n_proj_mask_image, float* d_proj_mask_data,                // input
                       float* d_proj_cur, 
		       int num_depth, int num_height, int num_width, 
		       int num_proj, int num_elevation, int num_ray,
		       float spacing, float voxel_size, float proj_pixel_size,
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
    nBlockX = (int)ceil((float)num_ray / (float)blockWidth);
    nBlockY = (int)ceil((float)num_elevation / (float)blockHeight);
    nBlockZ = (int)ceil((float)num_proj / (float)blockDepth);
   
    dim3 dimGrid(nBlockX, nBlockY*nBlockZ);                // 3D grid is not supported on G80
    dim3 dimBlock(blockWidth, blockHeight, blockDepth); 
    dim3 projdim(num_ray, num_elevation, num_proj);
    dim3 griddim(nBlockX, nBlockY, nBlockZ); 
    dim3 imagedim( num_width, num_height, num_depth);
   
    // set texture parameters
    tex_voxel.normalized = false;                      // access with normalized texture coordinates
    tex_voxel.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex_voxel.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates
    tex_voxel.addressMode[1] = cudaAddressModeClamp;
    tex_voxel.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // float
    CUDA_SAFE_CALL(cudaBindTextureToArray(tex_voxel, d_array_voxel, float1Desc));

    // execute the kernel
    if(num_depth == 1 ) {
      proj_cal_kernel_2D<<< dimGrid, dimBlock >>>( d_proj, d_proj_cur, n_proj_mask_image, d_proj_mask_data,   
						   projdim, griddim, imagedim, 
						   spacing, voxel_size, proj_pixel_size, SOD, 
						   inv_rot, xoffset, yoffset, zoffset, start_rot, end_rot);
    }
    else{
      proj_cal_kernel_3D<<< dimGrid, dimBlock >>>( d_proj, d_proj_cur, n_proj_mask_image, d_proj_mask_data,   
						   projdim, griddim, imagedim, 
						   spacing, voxel_size, proj_pixel_size, SOD, 
						   inv_rot, xoffset, yoffset, zoffset, start_rot, end_rot);
    }

    CUDA_SAFE_CALL( cudaUnbindTexture( tex_voxel ) );
    
    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void 
proj_cal_kernel_2D( float* proj, float* proj_cur, int n_proj_mask_image, float* proj_mask, 
		    dim3 projdim, dim3 griddim, dim3 imagedim, 
		    float spacing, float voxel_size, float proj_pixel_size, float SOD, 
		    float inv_rot,  float xoffset, float yoffset, float zoffset, 
		    float start_rot, float end_rot){

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
  
  __shared__ float shmem_proj[BLOCK_2DIM_X][BLOCK_2DIM_Y][BLOCK_2DIM_Z];
  __shared__ float shmem_proj_mask[BLOCK_2DIM_X][BLOCK_2DIM_Y]; 

  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;
  cidx.z = threadIdx.z;

  // load shared memory

  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {
  
    shmem_proj[cidx.x][cidx.y][cidx.z] = proj[((idz)*projdim.y + (idy))*projdim.x + idx];

    if( n_proj_mask_image == 1 ){
      shmem_proj_mask[cidx.x][cidx.y] = proj_mask[ idy*projdim.x + idx ];
    }
           
  }

  __syncthreads();

  
  // 2. apply kernel
   
  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {

    float proj_res = 0.0f;

    float len = 0.0f;
        
    // rotation angle

#ifdef ROTATION_CLOCKWISE
    // float angle = -PI * idz / projdim.z; 
    float angle = -(start_rot + idz * inv_rot) * PI / 180.0f;
#else
    // float angle = PI * idz / projdim.z + PI / 2.0f; 
    float angle = (start_rot + idz * inv_rot) * PI / 180.0f + PI / 2.0f;
#endif

    float xs = SOD * cosf( angle ) + (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * sinf( -angle ); 
    float ys = SOD * sinf( angle ) + (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * cosf( angle );
    float zs = (1.0f * idy - projdim.y / 2.0f ) * proj_pixel_size ; 


    // detector pixel position (-SDD, (idx - projdim.x / 2.0f ) * proj_pixel_size, 
    //                                (idy - projdim.y / 2.0f ) * proj_pixel_size ) 

    // SDD should be zero here to utilize the virtual detector
    // float xp = -SDD * cosf( angle ) + (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * sinf( -angle );
    // float yp = -SDD * sinf( angle ) + (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * cosf( angle );
  
    float xp = (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * sinf( -angle );
    float yp = (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * cosf( angle );
    float zp = (1.0f * idy - projdim.y / 2.0f ) * proj_pixel_size ;

    // vector direction
    len = sqrtf( (xp - xs) * (xp - xs) + (yp - ys) * (yp - ys) + (zp - zs) * (zp - zs) );
    float dx = (xp - xs) * spacing / len;   
    float dy = (yp - ys) * spacing / len;
    float dz = (zp - zs) * spacing / len;

    // determine intersection points
    // The bounded volumes is specified by  (-imagedim.x/2.0f, -imagedim.y/2.0f, -imagedim.z/2.0f)
    //                                  and (imagedim.x/2.0f, imagedim.y/2.0f, imagedim.z/2.0f)

    float t_near = -100.0f;
    float t_far = 100.0f;

    // calculate t_near
    if( fabsf( xp - xs ) > PRECISION ){

    	if( dx < 0.0f &&  t_near < ( 1.0f * imagedim.x/2 - 1.0f  - xs ) / ( xp - xs ) )
	    t_near = ( 1.0f * imagedim.x/2 - 1.0f   - xs ) / ( xp - xs );
	
        if( dx > 0.0f &&  t_near < ( -1.0f * imagedim.x/2 - xs ) / ( xp - xs ) )
	    t_near = ( -1.0f * imagedim.x/2  - xs ) / ( xp - xs );
    }

    if( fabsf( yp - ys ) > PRECISION ){

        if( dy < 0.0f && t_near < ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_near = ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys );

        if( dy > 0.0f && t_near < ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys ) )
    	    t_near = ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys );
    }

    if( fabsf( zp - zs ) > PRECISION ){
        if( dz < 0.0f && t_near < ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs ) )
    	    t_near = ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs );

        if( dz > 0.0f && t_near < ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs ) )
    	    t_near = ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs );
    }

    // calculate t_far
    if( fabsf( xp - xs ) > PRECISION ){
        if( dx < 0.0f && t_far >  ( -1.0f *imagedim.x/2 - xs ) / ( xp - xs ) )
            t_far = ( -1.0f * imagedim.x/2 - xs ) / ( xp - xs );

        if( dx > 0.0f && t_far > ( 1.0f * imagedim.x/2 - 1.0f   - xs ) / ( xp - xs ) )
            t_far = ( 1.0f * imagedim.x/2 - 1.0f  - xs ) / ( xp - xs );

    }

    if( fabsf( yp - ys ) > PRECISION ){
        if( dy < 0.0f && t_far > ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys ) )
    	    t_far = ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys );

        if( dy > 0.0f && t_far > ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_far = ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys );
    }

    if( fabsf( zp - zs ) > PRECISION ){
        if( dz < 0.0f && t_far > ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs ) )
      	    t_far = ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs );

        if( dz > 0.0f && t_far > ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs ) )
      	    t_far = ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs );
    }

    //
    if( fabsf(imagedim.x - imagedim.x / 2 - 1 ) < PRECISION ){
      if(  fabsf( xp - xs ) < PRECISION && ( xp < -1.0f * imagedim.x / 2.0f || xp > 1.0f * imagedim.x / 2.0f ) ){
    	t_near = t_far + 1.0f;
      }  
    }
    else {
      if(  fabsf( xp - xs ) < PRECISION && ( xp < -1.0f * imagedim.x / 2.0f || xp > 1.0f * imagedim.x / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      }  
    }

    if( fabsf(imagedim.y - imagedim.y / 2 - 1 ) < PRECISION ){
      if(  fabsf( yp - ys ) < PRECISION && ( yp < -1.0f * imagedim.y / 2.0f || yp > 1.0f * imagedim.y / 2.0f ) ){
    	t_near = t_far + 1.0f;
      }
    }
    else {
      if(  fabsf( yp - ys ) < PRECISION && ( yp < -1.0f * imagedim.y / 2.0f || yp > 1.0f * imagedim.y / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      }
    }

    if( fabsf(imagedim.z - imagedim.z / 2 - 1 ) < PRECISION ){
      if(  fabsf( zp - zs ) < PRECISION && ( zp < -1.0f * imagedim.z / 2.0f || zp > 1.0f * imagedim.z / 2.0f ) ){
    	t_near = t_far + 1.0f;
      } 
    }
    else {
      if(  fabsf( zp - zs ) < PRECISION && ( zp < -1.0f * imagedim.z / 2.0f || zp > 1.0f * imagedim.z / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      } 
    }

    if( t_near > t_far ) // no intersection
        proj_res = 0.0f;

    else{
	len = (t_far - t_near) * len + 1.0f;

	int num_steps = (int) (len / spacing ) ;
	float i;
		
	float x, y, z;

	for( i = 0; i <= num_steps; i++ ){
	     x = xs + (xp - xs) * t_near + i * dx;
	     y = ys + (yp - ys) * t_near + i * dy;
	     z = zs + (zp - zs) * t_near + i * dz;

	     x = x / voxel_size + 1.0f * imagedim.x / 2;
	     y = y / voxel_size + 1.0f * imagedim.y / 2;
	     z = z / voxel_size + 1.0f * imagedim.z / 2;

             proj_res += tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f) * spacing; 
	}
	
	x = xs + (xp - xs) * t_far;
        y = ys + (yp - ys) * t_far;
	z = zs + (zp - zs) * t_far;

        x = x / voxel_size + 1.0f * imagedim.x / 2;
	y = y / voxel_size + 1.0f * imagedim.y / 2;
	z = z / voxel_size + 1.0f * imagedim.z / 2;

        proj_res += tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f) * (len - num_steps * spacing);
        	
	if( proj_res > PRECISION_WEIGHT )	
	  proj_res =  shmem_proj[cidx.x][cidx.y][cidx.z] / proj_res ; 
	else
	  proj_res = 0.0f;

        // proj_res = 1.0 * t_near;
	// proj_res = weight_len; 

    }

    if( n_proj_mask_image == 1 && shmem_proj_mask[cidx.x][cidx.y] > 0.5f ){
      proj_res = 0.0f;
    }
     
    
    __syncthreads();
  
    // store the result
    uint outidx = ((idz)* projdim.y + (idy))*projdim.x + idx;
 
    proj_cur[ outidx ] =  proj_res;
 
  }
 
}


__global__ void 
proj_cal_kernel_3D( float* proj, float* proj_cur, int n_proj_mask_image, float* proj_mask,
		    dim3 projdim, dim3 griddim, dim3 imagedim, 
		    float spacing, float voxel_size, float proj_pixel_size, float SOD, 
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
  
  __shared__ float shmem_proj[BLOCK_3DIM_X][BLOCK_3DIM_Y][BLOCK_3DIM_Z];
  __shared__ float shmem_proj_mask[BLOCK_3DIM_X][BLOCK_3DIM_Y]; 
 
  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;
  cidx.z = threadIdx.z;

  // load shared memory

  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {
  
    shmem_proj[cidx.x][cidx.y][cidx.z] = proj[((idz)*projdim.y + (idy))*projdim.x + idx];

    if( n_proj_mask_image == 1 ){
      shmem_proj_mask[cidx.x][cidx.y] = proj_mask[ idy*projdim.x + idx ];
    }
                      
  }

  __syncthreads();

  
  // 2. apply kernel
   
  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {

    float proj_res = 0.0f;

    float len = 0.0f;
        
    // rotation angle
#ifdef ROTATION_CLOCKWISE
    // float angle = -PI * idz / projdim.z; 
    float angle = -(start_rot + idz * inv_rot) * PI / 180.0f;
#else
    // float angle = PI * idz / projdim.z + PI / 2.0f; 
    float angle = (start_rot + idz * inv_rot) * PI / 180.0f + PI / 2.0f;
#endif

    float xs = SOD * cosf( angle ) + (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * sinf( -angle ); 
    float ys = SOD * sinf( angle ) + (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * cosf( angle );
    float zs = (1.0f * idy - projdim.y / 2.0f ) * proj_pixel_size ; 


    // detector pixel position (-SDD, (idx - projdim.x / 2.0f ) * proj_pixel_size, 
    //                                (idy - projdim.y / 2.0f ) * proj_pixel_size ) 

    // SDD should be zero here to utilize the virtual detector
    // float xp = -SDD * cosf( angle ) + (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * sinf( -angle );
    // float yp = -SDD * sinf( angle ) + (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * cosf( angle );
  
    float xp = (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * sinf( -angle );
    float yp = (1.0f * idx - projdim.x / 2.0f ) * proj_pixel_size * cosf( angle );
    float zp = (1.0f * idy - projdim.y / 2.0f ) * proj_pixel_size ;

    // vector direction
    len = sqrtf( (xp - xs) * (xp - xs) + (yp - ys) * (yp - ys) + (zp - zs) * (zp - zs) );
    float dx = (xp - xs) * spacing / len;   
    float dy = (yp - ys) * spacing / len;
    float dz = (zp - zs) * spacing / len;

    // determine intersection points
    // The bounded volumes is specified by  (-imagedim.x/2.0f, -imagedim.y/2.0f, -imagedim.z/2.0f)
    //                                  and (imagedim.x/2.0f, imagedim.y/2.0f, imagedim.z/2.0f)

    float t_near = -100.0f;
    float t_far = 100.0f;

    // calculate t_near
    if( fabsf( xp - xs ) > PRECISION ){

    	if( dx < 0.0f &&  t_near < ( 1.0f * imagedim.x/2 - 1.0f  - xs ) / ( xp - xs ) )
	    t_near = ( 1.0f * imagedim.x/2 - 1.0f   - xs ) / ( xp - xs );
	
        if( dx > 0.0f &&  t_near < ( -1.0f * imagedim.x/2 - xs ) / ( xp - xs ) )
	    t_near = ( -1.0f * imagedim.x/2  - xs ) / ( xp - xs );
    }

    if( fabsf( yp - ys ) > PRECISION ){

        if( dy < 0.0f && t_near < ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_near = ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys );

        if( dy > 0.0f && t_near < ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys ) )
    	    t_near = ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys );
    }

    if( fabsf( zp - zs ) > PRECISION ){
        if( dz < 0.0f && t_near < ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs ) )
    	    t_near = ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs );

        if( dz > 0.0f && t_near < ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs ) )
    	    t_near = ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs );
    }

    // calculate t_far
    if( fabsf( xp - xs ) > PRECISION ){
        if( dx < 0.0f && t_far >  ( -1.0f *imagedim.x/2 - xs ) / ( xp - xs ) )
            t_far = ( -1.0f * imagedim.x/2 - xs ) / ( xp - xs );

        if( dx > 0.0f && t_far > ( 1.0f * imagedim.x/2 - 1.0f   - xs ) / ( xp - xs ) )
            t_far = ( 1.0f * imagedim.x/2 - 1.0f  - xs ) / ( xp - xs );

    }

    if( fabsf( yp - ys ) > PRECISION ){
        if( dy < 0.0f && t_far > ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys ) )
    	    t_far = ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys );

        if( dy > 0.0f && t_far > ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_far = ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys );
    }

    if( fabsf( zp - zs ) > PRECISION ){
        if( dz < 0.0f && t_far > ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs ) )
      	    t_far = ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs );

        if( dz > 0.0f && t_far > ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs ) )
      	    t_far = ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs );
    }

    //
    if( fabsf(imagedim.x - imagedim.x / 2 - 1 ) < PRECISION ){
      if(  fabsf( xp - xs ) < PRECISION && ( xp < -1.0f * imagedim.x / 2.0f || xp > 1.0f * imagedim.x / 2.0f ) ){
    	t_near = t_far + 1.0f;
      }  
    }
    else {
      if(  fabsf( xp - xs ) < PRECISION && ( xp < -1.0f * imagedim.x / 2.0f || xp > 1.0f * imagedim.x / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      }  
    }

    if( fabsf(imagedim.y - imagedim.y / 2 - 1 ) < PRECISION ){
      if(  fabsf( yp - ys ) < PRECISION && ( yp < -1.0f * imagedim.y / 2.0f || yp > 1.0f * imagedim.y / 2.0f ) ){
    	t_near = t_far + 1.0f;
      }
    }
    else {
      if(  fabsf( yp - ys ) < PRECISION && ( yp < -1.0f * imagedim.y / 2.0f || yp > 1.0f * imagedim.y / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      }
    }

    if( fabsf(imagedim.z - imagedim.z / 2 - 1 ) < PRECISION ){
      if(  fabsf( zp - zs ) < PRECISION && ( zp < -1.0f * imagedim.z / 2.0f || zp > 1.0f * imagedim.z / 2.0f ) ){
    	t_near = t_far + 1.0f;
      } 
    }
    else {
      if(  fabsf( zp - zs ) < PRECISION && ( zp < -1.0f * imagedim.z / 2.0f || zp > 1.0f * imagedim.z / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      } 
    }

    if( t_near > t_far ) // no intersection
        proj_res = 0.0f;

    else{
	len = (t_far - t_near) * len + 1.0f;

	int num_steps = (int) (len / spacing ) ;
	float i;
		
	float x, y, z;

	for( i = 0; i <= num_steps; i++ ){
	     x = xs + (xp - xs) * t_near + i * dx;
	     y = ys + (yp - ys) * t_near + i * dy;
	     z = zs + (zp - zs) * t_near + i * dz;

	     x = x / voxel_size + 1.0f * imagedim.x / 2;
	     y = y / voxel_size + 1.0f * imagedim.y / 2;
	     z = z / voxel_size + 1.0f * imagedim.z / 2;

             proj_res += tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f) * spacing; 
	}
	
	x = xs + (xp - xs) * t_far;
        y = ys + (yp - ys) * t_far;
	z = zs + (zp - zs) * t_far;

        x = x / voxel_size + 1.0f * imagedim.x / 2;
	y = y / voxel_size + 1.0f * imagedim.y / 2;
	z = z / voxel_size + 1.0f * imagedim.z / 2;

        proj_res += tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f) * (len - num_steps * spacing);
        	
	if( proj_res > PRECISION_WEIGHT )	
	  proj_res =  shmem_proj[cidx.x][cidx.y][cidx.z] / proj_res ; 
	else
	  proj_res = 0.0f;

        // proj_res = 1.0 * t_near;
	// proj_res = weight_len; 

    }

    // remove the effects of pixel intensities in corrupted regionss
    if( n_proj_mask_image == 1 && shmem_proj_mask[cidx.x][cidx.y] > 0.5f ){
      proj_res = 0.0f;
    }
    
    __syncthreads();
  
    // store the result
    uint outidx = ((idz)* projdim.y + (idy))*projdim.x + idx;
 
    proj_cur[ outidx ] =  proj_res;
 
  }
 
}


///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void proj_gen_kernel( float*, dim3, dim3, dim3, float, float, float, 
	                         float, float, float, float, float, 
	                         float, float);

extern "C" 
void proj_gen_wrapper( cudaArray* d_array_voxel,
                       float* d_proj_cur, 
		       int num_depth, int num_height, int num_width, 
		       int num_proj, int num_elevation, int num_ray,
		       float spacing, float voxel_size, float proj_pixel_size,
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
    nBlockX = (int)ceil((float)num_ray / (float)blockWidth);
    nBlockY = (int)ceil((float)num_elevation / (float)blockHeight);
    nBlockZ = (int)ceil((float)num_proj / (float)blockDepth);
   
    dim3 dimGrid(nBlockX, nBlockY*nBlockZ);                // 3D grid is not supported on G80
    dim3 dimBlock(blockWidth, blockHeight, blockDepth); 
    dim3 projdim(num_ray, num_elevation, num_proj);
    dim3 griddim(nBlockX, nBlockY, nBlockZ); 
    dim3 imagedim( num_width, num_height, num_depth);
   
    // set texture parameters
    tex_voxel.normalized = false;                      // access with normalized texture coordinates
    tex_voxel.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex_voxel.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates
    tex_voxel.addressMode[1] = cudaAddressModeClamp;
    tex_voxel.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // float
    CUDA_SAFE_CALL(cudaBindTextureToArray(tex_voxel, d_array_voxel, float1Desc));

    // execute the kernel
    proj_gen_kernel<<< dimGrid, dimBlock >>>( d_proj_cur, projdim, griddim, imagedim, 
    		       			      spacing, voxel_size, proj_pixel_size, SOD, 
					      inv_rot, xoffset, yoffset, zoffset, start_rot, end_rot);

    CUDA_SAFE_CALL( cudaUnbindTexture( tex_voxel ) );
    
    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void 
proj_gen_kernel( float* proj_cur, dim3 projdim, dim3 griddim, dim3 imagedim, 
		 float spacing, float voxel_size, float proj_pixel_size, float SOD, 
                 float inv_rot, float xoffset, float yoffset, float zoffset, 
		 float start_rot, float end_rot){

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
  
  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;
  cidx.z = threadIdx.z;

  // 2. apply kernel
   
  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {

    float proj_res = 0.0f;

    float len = 0.0f;
        
    // rotation angle

#ifdef ROTATION_CLOCKWISE
    float angle = -PI * idz / projdim.z; 
#else
    float angle = PI * idz / projdim.z + PI / 2.0f; 
#endif

    float xs = SOD * cosf( angle ) + (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
    float ys = SOD * sinf( angle ) + (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * cosf( angle );
    float zs = (1.0f * idy - projdim.y / 2 ) * proj_pixel_size ; 


    // detector pixel position (-SDD, (idx - projdim.x / 2 ) * proj_pixel_size, 
    //                                (idy - projdim.y / 2 ) * proj_pixel_size ) 

    // SDD should be zero here to utilize the virtual detector
    // float xp = -SDD * cosf( angle ) + (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    // float yp = -SDD * sinf( angle ) + (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * cosf( angle );
  
    float xp = (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    float yp = (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * cosf( angle );
    float zp = (1.0f * idy - projdim.y / 2 ) * proj_pixel_size ;

    // vector direction
    len = sqrtf( (xp - xs) * (xp - xs) + (yp - ys) * (yp - ys) + (zp - zs) * (zp - zs) );
    float dx = (xp - xs) * spacing / len;   
    float dy = (yp - ys) * spacing / len;
    float dz = (zp - zs) * spacing / len;

    // determine intersection points
    // The bounded volumes is specified by  (-imagedim.x/2, -imagedim.y/2, -imagedim.z/2)
    //                                  and (imagedim.x/2, imagedim.y/2, imagedim.z/2)

    float t_near = -100.0f;
    float t_far = 100.0f;

    // calculate t_near
    if( fabsf( xp - xs ) > PRECISION ){

    	if( dx < 0.0f &&  t_near < ( 1.0f * imagedim.x/2 - 1.0f  - xs ) / ( xp - xs ) )
	    t_near = ( 1.0f * imagedim.x/2 - 1.0f   - xs ) / ( xp - xs );
	
        if( dx > 0.0f &&  t_near < ( -1.0f * imagedim.x/2 - xs ) / ( xp - xs ) )
	    t_near = ( -1.0f * imagedim.x/2  - xs ) / ( xp - xs );
    }

    if( fabsf( yp - ys ) > PRECISION ){

        if( dy < 0.0f && t_near < ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_near = ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys );

        if( dy > 0.0f && t_near < ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys ) )
    	    t_near = ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys );
    }

    if( fabsf( zp - zs ) > PRECISION ){
        if( dz < 0.0f && t_near < ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs ) )
    	    t_near = ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs );

        if( dz > 0.0f && t_near < ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs ) )
    	    t_near = ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs );
    }

    // calculate t_far
    if( fabsf( xp - xs ) > PRECISION ){
        if( dx < 0.0f && t_far >  ( -1.0f *imagedim.x/2 - xs ) / ( xp - xs ) )
            t_far = ( -1.0f * imagedim.x/2 - xs ) / ( xp - xs );

        if( dx > 0.0f && t_far > ( 1.0f * imagedim.x/2 - 1.0f   - xs ) / ( xp - xs ) )
            t_far = ( 1.0f * imagedim.x/2 - 1.0f  - xs ) / ( xp - xs );

    }

    if( fabsf( yp - ys ) > PRECISION ){
        if( dy < 0.0f && t_far > ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys ) )
    	    t_far = ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys );

        if( dy > 0.0f && t_far > ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_far = ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys );
    }

    if( fabsf( zp - zs ) > PRECISION ){
        if( dz < 0.0f && t_far > ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs ) )
      	    t_far = ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs );

        if( dz > 0.0f && t_far > ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs ) )
      	    t_far = ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs );
    }

    //
    if( fabsf(imagedim.x - imagedim.x / 2 - 1 ) < PRECISION ){
      if(  fabsf( xp - xs ) < PRECISION && ( xp < -1.0f * imagedim.x / 2.0f || xp > 1.0f * imagedim.x / 2.0f ) ){
    	t_near = t_far + 1.0f;
      }  
    }
    else {
      if(  fabsf( xp - xs ) < PRECISION && ( xp < -1.0f * imagedim.x / 2.0f || xp > 1.0f * imagedim.x / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      }  
    }

    if( fabsf(imagedim.y - imagedim.y / 2 - 1 ) < PRECISION ){
      if(  fabsf( yp - ys ) < PRECISION && ( yp < -1.0f * imagedim.y / 2.0f || yp > 1.0f * imagedim.y / 2.0f ) ){
    	t_near = t_far + 1.0f;
      }
    }
    else {
      if(  fabsf( yp - ys ) < PRECISION && ( yp < -1.0f * imagedim.y / 2.0f || yp > 1.0f * imagedim.y / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      }
    }

    if( fabsf(imagedim.z - imagedim.z / 2 - 1 ) < PRECISION ){
      if(  fabsf( zp - zs ) < PRECISION && ( zp < -1.0f * imagedim.z / 2.0f || zp > 1.0f * imagedim.z / 2.0f ) ){
    	t_near = t_far + 1.0f;
      } 
    }
    else {
      if(  fabsf( zp - zs ) < PRECISION && ( zp < -1.0f * imagedim.z / 2.0f || zp > 1.0f * imagedim.z / 2.0f - 1.0f ) ){
    	t_near = t_far + 1.0f;
      } 
    }

    if( t_near > t_far ) // no intersection
        proj_res = 0.0f;

    else{
	len = (t_far - t_near) * len + 1.0f;

	int num_steps = (int) (len / spacing ) ;
	float i, x, y, z;

	for( i = 0; i <= num_steps; i++ ){
	     x = xs + (xp - xs) * t_near + i * dx;
	     y = ys + (yp - ys) * t_near + i * dy;
	     z = zs + (zp - zs) * t_near + i * dz;

	     x = x / voxel_size + 1.0f * imagedim.x / 2;
	     y = y / voxel_size + 1.0f * imagedim.y / 2;
	     z = z / voxel_size + 1.0f * imagedim.z / 2;

             proj_res += tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f) * spacing;

	}
	
	x = xs + (xp - xs) * t_far;
        y = ys + (yp - ys) * t_far;
	z = zs + (zp - zs) * t_far;

        x = x / voxel_size + 1.0f * imagedim.x / 2;
	y = y / voxel_size + 1.0f * imagedim.y / 2;
	z = z / voxel_size + 1.0f * imagedim.z / 2;

        proj_res += tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f) * spacing;

    }
    
    __syncthreads();
  
    // store the result
    uint outidx = ((idz)* projdim.y + (idy))*projdim.x + idx;
 
    proj_cur[ outidx ] =  proj_res;
 
  }
}

#endif // __PROJ_CAL_CU

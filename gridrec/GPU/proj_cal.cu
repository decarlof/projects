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
 
#include "tomo_recon.h"

texture<float, 3, cudaReadModeElementType> tex_voxel;  // 3D texture

#define PRECISION 1e-5f
#define PRECISION_WEIGHT 1e-1f

#define BLOCK_SHMEM_2DIM_X 48  // 48   // customized for NVIDIA Tesla C2050: 48KB shared memory per block
#define BLOCK_SHMEM_2DIM_Y 48  // 48   // customized for 2048*2048*1 CT reconstruction too
                                // need to be multiples of BLOCK_2DIM_X/Y for current version

#define BLOCK_SHMEM_3DIM_X 16  // 24   // customized for NVIDIA Tesla C2050: 48KB shared memory per block
#define BLOCK_SHMEM_3DIM_Y 16  // 24   // customized for 2048*2048*1 CT reconstruction too
                                // need to be multiples of BLOCK_3DIM_X/Y for current version
                                // 3D: BLOCK_3DIM_Y = 4

__global__ void proj_cal_kernel_2D( float*, float*, dim3, dim3, dim3, float, float, float, 
	                         float, float, float , float , float , float , float);

__global__ void proj_cal_kernel_3D( float*, float*, dim3, dim3, dim3, float, float, float, 
	                         float, float, float , float , float , float , float);


extern "C" 
void proj_cal_wrapper( cudaArray* d_array_voxel, float* d_proj, 
                       float* d_proj_cur, 
		       int num_depth, int num_height, int num_width, 
		       int num_proj, int num_elevation, int num_ray,
		       float spacing, float voxel_size, float proj_pixel_size,
	   	       float SOD, float inv_rot,
		       float xoffset, float yoffset, float zoffset, float start_rot, float end_rot){

    // setup execution parameters
    int blockWidth, blockHeight, blockDepth, nBlockX, nBlockY, nBlockZ;
    
    // Setting block size. 3D projection data only.
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
    if( num_depth == 1 )
      proj_cal_kernel_2D<<< dimGrid, dimBlock >>>( d_proj, d_proj_cur, projdim, griddim, imagedim, 
    		       			      spacing, voxel_size, proj_pixel_size, SOD, inv_rot,
					      xoffset, yoffset, zoffset, start_rot, end_rot);
    else
      proj_cal_kernel_3D<<< dimGrid, dimBlock >>>( d_proj, d_proj_cur, projdim, griddim, imagedim, 
    		       			      spacing, voxel_size, proj_pixel_size, SOD, inv_rot,
					      xoffset, yoffset, zoffset, start_rot, end_rot);

    CUDA_SAFE_CALL( cudaUnbindTexture( tex_voxel ) );
    
    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void 
proj_cal_kernel_2D( float* proj, float* proj_cur, dim3 projdim, dim3 griddim, dim3 imagedim, 
		 float spacing, float voxel_size, float proj_pixel_size, float SOD, float inv_rot,
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
  
  __shared__ float shmem_proj[BLOCK_2DIM_X][BLOCK_2DIM_Y][BLOCK_2DIM_Z];
 
  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;
  cidx.z = threadIdx.z;

  // load shared memory

  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {
  
    shmem_proj[cidx.x][cidx.y][cidx.z] = proj[((idz)*projdim.y + (idy))*projdim.x + idx];
           
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
    float angle = - (start_rot + idz * inv_rot) * PI / 180.0f; 
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
	float i;
		
	float x, y, z;

        float weight_len = 0.0f;
	float tmp;
	 
	for( i = 0; i <= num_steps; i++ ){
	     x = xs + (xp - xs) * t_near + i * dx;
	     y = ys + (yp - ys) * t_near + i * dy;
	     z = zs + (zp - zs) * t_near + i * dz;

	     x = x / voxel_size + 1.0f * imagedim.x / 2;
	     y = y / voxel_size + 1.0f * imagedim.y / 2;
	     z = z / voxel_size + 1.0f * imagedim.z / 2;

             tmp = tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f);
	     // tmp = 0.01f;   // test
	     if( tmp > PRECISION ){
	         weight_len = weight_len + 1.0f;
  	         proj_res += tmp * spacing;	     
             }

	}
	
	x = xs + (xp - xs) * t_far;
        y = ys + (yp - ys) * t_far;
	z = zs + (zp - zs) * t_far;

        x = x / voxel_size + 1.0f * imagedim.x / 2;
	y = y / voxel_size + 1.0f * imagedim.y / 2;
	z = z / voxel_size + 1.0f * imagedim.z / 2;

        tmp = tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f);
	// tmp = 0.01f;        // test
	if( tmp > PRECISION ){
	  weight_len = weight_len + 1.0f;
	  proj_res += tmp * (len - num_steps * spacing); 
        }
        	
	if( weight_len > PRECISION_WEIGHT )	
	  proj_res = ( shmem_proj[cidx.x][cidx.y][cidx.z] - proj_res ) / weight_len; 
	else
	  proj_res = 0.0f;

        // proj_res = 1.0 * t_near;
	// proj_res = weight_len; 

    }
    
    __syncthreads();
  
    // store the result
    uint outidx = ((idz)* projdim.y + (idy))*projdim.x + idx;
 
    proj_cur[ outidx ] =  proj_res;
 
  }
 
}

__global__ void 
proj_cal_kernel_3D( float* proj, float* proj_cur, dim3 projdim, dim3 griddim, dim3 imagedim, 
		 float spacing, float voxel_size, float proj_pixel_size, float SOD, float inv_rot,
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
  
  __shared__ float shmem_proj[BLOCK_3DIM_X][BLOCK_3DIM_Y][BLOCK_3DIM_Z];
 
  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;
  cidx.z = threadIdx.z;

  // load shared memory

  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {
  
    shmem_proj[cidx.x][cidx.y][cidx.z] = proj[((idz)*projdim.y + (idy))*projdim.x + idx];
           
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
    float angle = - (start_rot + idz * inv_rot) * PI / 180.0f; 
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
	float i;
		
	float x, y, z;

        float weight_len = 0.0f;
	float tmp;
	 
	for( i = 0; i <= num_steps; i++ ){
	     x = xs + (xp - xs) * t_near + i * dx;
	     y = ys + (yp - ys) * t_near + i * dy;
	     z = zs + (zp - zs) * t_near + i * dz;

	     x = x / voxel_size + 1.0f * imagedim.x / 2;
	     y = y / voxel_size + 1.0f * imagedim.y / 2;
	     z = z / voxel_size + 1.0f * imagedim.z / 2;

             tmp = tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f);
	     // tmp = 0.01f;   // test
	     if( tmp > PRECISION ){
	         weight_len = weight_len + 1.0f;
  	         proj_res += tmp * spacing;	     
             }

	}
	
	x = xs + (xp - xs) * t_far;
        y = ys + (yp - ys) * t_far;
	z = zs + (zp - zs) * t_far;

        x = x / voxel_size + 1.0f * imagedim.x / 2;
	y = y / voxel_size + 1.0f * imagedim.y / 2;
	z = z / voxel_size + 1.0f * imagedim.z / 2;

        tmp = tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f);
	// tmp = 0.01f;        // test
	if( tmp > PRECISION ){
	  weight_len = weight_len + 1.0f;
	  proj_res += tmp * (len - num_steps * spacing); 
        }
        	
	if( weight_len > PRECISION_WEIGHT )	
	  proj_res = ( shmem_proj[cidx.x][cidx.y][cidx.z] - proj_res ) / weight_len; 
	else
	  proj_res = 0.0f;

        // proj_res = 1.0 * t_near;
	// proj_res = weight_len; 

    }
    
    __syncthreads();
  
    // store the result
    uint outidx = ((idz)* projdim.y + (idy))*projdim.x + idx;
 
    proj_cur[ outidx ] =  proj_res;
 
  }
 
}


///////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void proj_gen_kernel( float*, dim3, dim3, dim3, float, float, float, 
	                         float, float);

extern "C" 
void proj_gen_wrapper( cudaArray* d_array_voxel,
                       float* d_proj_cur, 
		       int num_depth, int num_height, int num_width, 
		       int num_proj, int num_elevation, int num_ray,
		       float spacing, float voxel_size, float proj_pixel_size,
	   	       float SOD, float inv_rot){

    // setup execution parameters
    int blockWidth, blockHeight, blockDepth, nBlockX, nBlockY, nBlockZ;
    
    // Setting block size. 3D projection data only. 
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
					      inv_rot);

    CUDA_SAFE_CALL( cudaUnbindTexture( tex_voxel ) );
    
    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void 
proj_gen_kernel( float* proj_cur, dim3 projdim, dim3 griddim, dim3 imagedim, 
		 float spacing, float voxel_size, float proj_pixel_size, float SOD, 
                 float inv_rot){


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

//////////////////////////////////////////////////////////////////////////////////////////
// implementations using shared memory instead of texture memory

__global__ void proj_cal_kernel_sharemem_2D( float*, float*, float*, dim3, dim3, dim3, float, float, float, 
					     float, float, unsigned int);

__global__ void proj_cal_kernel_sharemem_3D( float*, float*, float*, dim3, dim3, dim3, float, float, float, 
					     float, float, unsigned int);

extern "C" 
void proj_cal_wrapper_sharemem( float* d_image_prev, float* d_proj, 
				float* d_proj_cur, 
				int num_depth, int num_height, int num_width, 
				int num_proj, int num_elevation, int num_ray,
				float spacing, float voxel_size, float proj_pixel_size,
				float SOD, float inv_rot, 
				unsigned int deviceShareMemSizePerBlock){

  // The original objective for this function is to avoid data transfer from GPU
  // global memory to texture memory. However, this alternative requires huge 
  // data transfer from GPU global memory to GPU shared memory, which is time
  // consuming too, even much more time required. 

  // Yongsheng Pan             4/11/2011

  // setup execution parameters
  int blockWidth, blockHeight, blockDepth, nBlockX, nBlockY, nBlockZ;
    
  // Setting block size
  if(num_elevation == 1 ) {  // i.e., num_depth == 1 for parallel beam

    blockWidth  = BLOCK_2DIM_X;
    blockHeight = BLOCK_2DIM_Y;   

    // compute how many blocks are needed
    nBlockX = (int)ceil((float)num_ray / (float)blockWidth);
    nBlockY = (int)ceil((float)num_proj / (float)blockHeight);
   
    dim3 dimGrid(nBlockX, nBlockY);                // 3D grid is not supported on G80
    dim3 dimBlock(blockWidth, blockHeight, 1); 
    dim3 projdim(num_ray, num_proj, 1);
    dim3 griddim(nBlockX, nBlockY, 1); 
    dim3 imagedim( num_width, num_height, 1);
   
    // execute the kernel
    proj_cal_kernel_sharemem_2D<<< dimGrid, dimBlock >>>( d_image_prev, d_proj, d_proj_cur, projdim, griddim, 
							  imagedim, spacing, voxel_size, proj_pixel_size, SOD, 
							  inv_rot, deviceShareMemSizePerBlock);
    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

  }
  else {                  
    blockWidth  = BLOCK_3DIM_X;
    blockHeight = BLOCK_3DIM_Y;
    blockDepth  = BLOCK_3DIM_Z;     

    // compute how many blocks are needed
    nBlockX = (int)ceil((float)num_ray / (float)blockWidth);
    nBlockY = (int)ceil((float)num_elevation / (float)blockHeight);
    nBlockZ = (int)ceil((float)num_proj / (float)blockDepth);
   
    dim3 dimGrid(nBlockX, nBlockY*nBlockZ);                // 3D grid is not supported on G80
    dim3 dimBlock(blockWidth, blockHeight, blockDepth); 
    dim3 projdim(num_ray, num_elevation, num_proj);
    dim3 griddim(nBlockX, nBlockY, nBlockZ); 
    dim3 imagedim( num_width, num_height, num_depth);
   
    // execute the kernel
    proj_cal_kernel_sharemem_3D<<< dimGrid, dimBlock >>>( d_image_prev, d_proj, d_proj_cur, projdim, griddim, 
							  imagedim, spacing, voxel_size, proj_pixel_size, SOD, 
							  inv_rot, deviceShareMemSizePerBlock);
    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

  }    

}

__global__ void 
proj_cal_kernel_sharemem_2D( float* d_image_prev, float* proj, float* proj_cur, 
			     dim3 projdim, dim3 griddim, dim3 imagedim, 
			     float spacing, float voxel_size, float proj_pixel_size, float SOD, 
			     float inv_rot, unsigned int deviceShareMemSizePerBlock){

  // 1. initialize shared memory
  
  uint idx, idy, idx_s, idx_c, idx_e;

  idx = blockIdx.x * blockDim.x + threadIdx.x;
  idy = blockIdx.y * blockDim.y + threadIdx.y;

  idx_s = blockIdx.x * blockDim.x;                    // starting ray in the current block
  idx_c = blockIdx.x * blockDim.x + blockDim.x / 2;   // central ray in the current block
  idx_e = blockIdx.x * blockDim.x + blockDim.x - 1;   // ending ray in the current block

  dim3 cidx;

  // shared memory

  __shared__ float shmem_proj[BLOCK_2DIM_X][BLOCK_2DIM_Y];
  __shared__ float shmem_image[BLOCK_SHMEM_2DIM_X][BLOCK_SHMEM_2DIM_Y];  // Note the dimension

  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;

  if(idx < projdim.x && idy < projdim.y){
    
    shmem_proj[cidx.x][cidx.y] = proj[idy * projdim.x + idx];
  }

  __syncthreads();

  // apply kernel

  float proj_res = 0.0f;
  float weight_len = 0.0f;
  float len = 0.0f;
   
  if(idx < projdim.x && idy < projdim.y)
  {

    // rotation angle

#ifdef ROTATION_CLOCKWISE
    float angle = -PI * idy / projdim.y; 
#else
    float angle = PI * idy / projdim.y + PI / 2.0f; 
#endif

    // determine central ray in the current block
    float xs_c = SOD * cosf( angle ) + (1.0f * idx_c - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
    float ys_c = SOD * sinf( angle ) + (1.0f * idx_c - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    // detector pixel position (-SDD = 0, (idx - projdim.x / 2 ) * proj_pixel_size)
    float xp_c = (1.0f * idx_c - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    float yp_c = (1.0f * idx_c - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    // determine the intersection point from (0, 0) perpendicular to the central ray 
    // determined by (xs_c, ys_c) and (xp_c, yp_c)
    float xc = 0.0f;
    float yc = 0.0f;
    float tc = 0.0f;

    if( fabsf( xs_c * yp_c - xp_c * ys_c ) > PRECISION ){  // (0, 0) is not on the central ray
      tc = -( xs_c * (xp_c - xs_c) + ys_c * (yp_c - ys_c) );
      tc /= (yp_c - ys_c) * (yp_c - ys_c) + (xp_c - xs_c) * (xp_c - xs_c);

      xc = xs_c + (xp_c - xs_c) * tc;
      yc = ys_c + (yp_c - ys_c) * tc;
    }
    else{
       xc = 0.0f;
       yc = 0.0f; 
       tc = -xs_c / (xp_c - xs_c);
    }

    proj_res = xc;  // test
    
    // vector direction
    len = sqrtf( (xp_c - xs_c) * (xp_c - xs_c) + (yp_c - ys_c) * (yp_c - ys_c) );
    float dx = (xp_c - xs_c) * spacing / len;   
    float dy = (yp_c - ys_c) * spacing / len;

    // determine intersection points of the starting ray in the current block
    // The bounded volumes is specified by  (xc-BLOCK_SHMEM_2DIM_X/2, yc-BLOCK_SHMEM_2DIM_Y/2)
    //                                  and (xc+BLOCK_SHMEM_2DIM_X/2, yc+BLOCK_SHMEM_2DIM_Y/2)

    float xs = SOD * cosf( angle ) + (1.0f * idx_s - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
    float ys = SOD * sinf( angle ) + (1.0f * idx_s - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    // detector pixel position (-SDD = 0, (idx - projdim.x / 2 ) * proj_pixel_size)

    float xp = (1.0f * idx_s - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    float yp = (1.0f * idx_s - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    float t_near_s = -100.0f;
    float t_far_s = 100.0f;

    // calculate t_near_s
    if( fabsf( xp - xs ) > PRECISION ){

    	if( dx < 0.0f &&  t_near_s < ( xc + 1.0f * BLOCK_SHMEM_2DIM_X/2 - 1.0f  - xs ) / ( xp - xs ) )
    	    t_near_s = ( xc + 1.0f * BLOCK_SHMEM_2DIM_X/2 - 1.0f - xs ) / ( xp - xs );
	
        if( dx > 0.0f &&  t_near_s < ( xc -1.0f * BLOCK_SHMEM_2DIM_X/2 - xs ) / ( xp - xs ) )
    	    t_near_s = ( xc - 1.0f * BLOCK_SHMEM_2DIM_X/2  - xs ) / ( xp - xs );
    }

    if( fabsf( yp - ys ) > PRECISION ){

        if( dy < 0.0f && t_near_s < ( yc + 1.0f * BLOCK_SHMEM_2DIM_Y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_near_s = ( yc + 1.0f * BLOCK_SHMEM_2DIM_Y/2 - 1.0f  - ys ) / ( yp - ys );

        if( dy > 0.0f && t_near_s < ( yc - 1.0f * BLOCK_SHMEM_2DIM_Y/2 - ys ) / ( yp - ys ) )
    	    t_near_s = ( yc - 1.0f * BLOCK_SHMEM_2DIM_Y/2 - ys ) / ( yp - ys );
    }

    // calculate t_far_s
    if( fabsf( xp - xs ) > PRECISION ){
        if( dx < 0.0f && t_far_s >  ( xc - 1.0f *BLOCK_SHMEM_2DIM_X/2 - xs ) / ( xp - xs ) )
            t_far_s = ( xc - 1.0f * BLOCK_SHMEM_2DIM_X/2 - xs ) / ( xp - xs );

        if( dx > 0.0f && t_far_s > ( xc + 1.0f * BLOCK_SHMEM_2DIM_X/2 - 1.0f   - xs ) / ( xp - xs ) )
            t_far_s = ( xc + 1.0f * BLOCK_SHMEM_2DIM_X/2 - 1.0f  - xs ) / ( xp - xs );

    }

    if( fabsf( yp - ys ) > PRECISION ){
        if( dy < 0.0f && t_far_s > ( yc - 1.0f * BLOCK_SHMEM_2DIM_Y/2 - ys ) / ( yp - ys ) )
    	    t_far_s = ( yc - 1.0f * BLOCK_SHMEM_2DIM_Y/2 - ys ) / ( yp - ys );

        if( dy > 0.0f && t_far_s > ( yc + 1.0f * BLOCK_SHMEM_2DIM_Y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_far_s = ( yc + 1.0f * BLOCK_SHMEM_2DIM_Y/2 - 1.0f  - ys ) / ( yp - ys );
    }

    //
    if( fabsf(BLOCK_SHMEM_2DIM_X - BLOCK_SHMEM_2DIM_X / 2 - 1 ) < PRECISION ){
      if(  fabsf( xp - xs ) < PRECISION && ( xp < xc - 1.0f * BLOCK_SHMEM_2DIM_X / 2.0f 
    					     || xp > xc + 1.0f * BLOCK_SHMEM_2DIM_X / 2.0f ) ){
    	t_near_s = t_far_s + 1.0f;
      }  
    }
    else {
      if(  fabsf( xp - xs ) < PRECISION && ( xp < xc - 1.0f * BLOCK_SHMEM_2DIM_X / 2.0f 
    					     || xp > xc + 1.0f * BLOCK_SHMEM_2DIM_X / 2.0f - 1.0f ) ){
    	t_near_s = t_far_s + 1.0f;
      }  
    }

    if( fabsf(BLOCK_SHMEM_2DIM_Y - BLOCK_SHMEM_2DIM_Y / 2 - 1 ) < PRECISION ){
      if(  fabsf( yp - ys ) < PRECISION && ( yp < yc - 1.0f * BLOCK_SHMEM_2DIM_Y / 2.0f 
    					     || yp > yc + 1.0f * BLOCK_SHMEM_2DIM_Y / 2.0f ) ){
    	t_near_s = t_far_s + 1.0f;
      }
    }
    else {
      if(  fabsf( yp - ys ) < PRECISION && ( yp < yc - 1.0f * BLOCK_SHMEM_2DIM_Y / 2.0f 
    					     || yp > yc + 1.0f * BLOCK_SHMEM_2DIM_Y / 2.0f - 1.0f ) ){
    	t_near_s = t_far_s + 1.0f;
      }
    }

    // determine intersection points of the ending ray in the current block
    // The bounded volumes is specified by  (xc-BLOCK_SHMEM_2DIM_X/2, yc-BLOCK_SHMEM_2DIM_Y/2)
    //                                  and (xc+BLOCK_SHMEM_2DIM_X/2, yc+BLOCK_SHMEM_2DIM_Y/2)

    xs = SOD * cosf( angle ) + (1.0f * idx_e - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
    ys = SOD * sinf( angle ) + (1.0f * idx_e - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    // detector pixel position (-SDD = 0, (idx - projdim.x / 2 ) * proj_pixel_size)

    xp = (1.0f * idx_e - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    yp = (1.0f * idx_e - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    float t_near_e = -100.0f;
    float t_far_e = 100.0f;

    // calculate t_near_e
    if( fabsf( xp - xs ) > PRECISION ){

    	if( dx < 0.0f &&  t_near_e < ( xc + 1.0f * BLOCK_SHMEM_2DIM_X/2 - 1.0f  - xs ) / ( xp - xs ) )
    	    t_near_e = ( xc + 1.0f * BLOCK_SHMEM_2DIM_X/2 - 1.0f   - xs ) / ( xp - xs );
	
        if( dx > 0.0f &&  t_near_e < ( xc - 1.0f * BLOCK_SHMEM_2DIM_X/2 - xs ) / ( xp - xs ) )
    	    t_near_e = ( xc - 1.0f * BLOCK_SHMEM_2DIM_X/2  - xs ) / ( xp - xs );
    }

    if( fabsf( yp - ys ) > PRECISION ){

        if( dy < 0.0f && t_near_e < ( yc + 1.0f * BLOCK_SHMEM_2DIM_Y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_near_e = ( yc + 1.0f * BLOCK_SHMEM_2DIM_Y/2 - 1.0f  - ys ) / ( yp - ys );

        if( dy > 0.0f && t_near_e < ( yc - 1.0f * BLOCK_SHMEM_2DIM_Y/2 - ys ) / ( yp - ys ) )
    	    t_near_e = ( yc - 1.0f * BLOCK_SHMEM_2DIM_Y/2 - ys ) / ( yp - ys );
    }

    // calculate t_far_e
    if( fabsf( xp - xs ) > PRECISION ){
        if( dx < 0.0f && t_far_e >  ( xc - 1.0f *BLOCK_SHMEM_2DIM_X/2 - xs ) / ( xp - xs ) )
            t_far_e = ( xc - 1.0f * BLOCK_SHMEM_2DIM_X/2 - xs ) / ( xp - xs );

        if( dx > 0.0f && t_far_e > ( xc + 1.0f * BLOCK_SHMEM_2DIM_X/2 - 1.0f   - xs ) / ( xp - xs ) )
            t_far_e = ( xc + 1.0f * BLOCK_SHMEM_2DIM_X/2 - 1.0f  - xs ) / ( xp - xs );

    }

    if( fabsf( yp - ys ) > PRECISION ){
        if( dy < 0.0f && t_far_e > ( yc - 1.0f * BLOCK_SHMEM_2DIM_Y/2 - ys ) / ( yp - ys ) )
    	    t_far_e = ( yc - 1.0f * BLOCK_SHMEM_2DIM_Y/2 - ys ) / ( yp - ys );

        if( dy > 0.0f && t_far_e > ( yc + 1.0f * BLOCK_SHMEM_2DIM_Y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_far_e = ( yc + 1.0f * BLOCK_SHMEM_2DIM_Y/2 - 1.0f  - ys ) / ( yp - ys );
    }

    //
    if( fabsf(BLOCK_SHMEM_2DIM_X - BLOCK_SHMEM_2DIM_X / 2 - 1 ) < PRECISION ){
      if(  fabsf( xp - xs ) < PRECISION && ( xp < xc - 1.0f * BLOCK_SHMEM_2DIM_X / 2.0f 
    					     || xp > xc + 1.0f * BLOCK_SHMEM_2DIM_X / 2.0f ) ){
    	t_near_e = t_far_e + 1.0f;
      }  
    }
    else {
      if(  fabsf( xp - xs ) < PRECISION && ( xp < xc - 1.0f * BLOCK_SHMEM_2DIM_X / 2.0f 
    					     || xp > xc + 1.0f * BLOCK_SHMEM_2DIM_X / 2.0f - 1.0f ) ){
    	t_near_e = t_far_e + 1.0f;
      }  
    }

    if( fabsf(BLOCK_SHMEM_2DIM_Y - BLOCK_SHMEM_2DIM_Y / 2 - 1 ) < PRECISION ){
      if(  fabsf( yp - ys ) < PRECISION && ( yp < yc - 1.0f * BLOCK_SHMEM_2DIM_Y / 2.0f 
    					     || yp > yc + 1.0f * BLOCK_SHMEM_2DIM_Y / 2.0f ) ){
    	t_near_e = t_far_e + 1.0f;
      }
    }
    else {
      if(  fabsf( yp - ys ) < PRECISION && ( yp < yc - 1.0f * BLOCK_SHMEM_2DIM_Y / 2.0f 
    					     || yp > yc + 1.0f * BLOCK_SHMEM_2DIM_Y / 2.0f - 1.0f ) ){
    	t_near_e = t_far_e + 1.0f;
      }
    }

    // A good point for parallel beam is that t_near_s, t_near_e (for the starting ray), 
    // t_far_s and t_far_e (for the ending ray) are exactly same as the t_near and t_far 
    // for the central ray, because of the parallel feature of the beam. This property 
    // makes it much easier to determine the accumulating portions inside each shared 
    // memory block. 

    float t_block = 0.0f; 
    if( t_near_s > t_far_s || t_near_e > t_far_e  ) // no intersection, meaning the shared memory block is too small. 
        proj_res = 0.0f;
    else{

      t_block = t_far_e;
      if( t_block > t_far_s )     // smaller for t_far
    	t_block = t_far_s;
    
      if( t_near_e < t_near_s )   // larger for t_near
    	t_block -= t_near_s;
      else
    	t_block -= t_near_e;

      t_block = t_block * 0.9;    // shrink for redundency so that shmem_image is large enough

      float delta_t = sqrtf( 1.0f * imagedim.x * imagedim.x + 1.0f * imagedim.y * imagedim.y ) / len / 2.0f; 

      float t;
      // sampling (interval [tc - delta_t, tc + delta_t]
      for( t = tc - delta_t; t <= tc + delta_t + t_block; t = t + t_block ){

    	// t + t_block / 2.0 corresponds to the center of the shared memory block
    	// load image data into the shared memory 

    	int xb_c = (int) round( ( xs_c + (xp_c - xs_c) * (t + t_block / 2.0f) ) / voxel_size + imagedim.x / 2);
    	int yb_c = (int) round( ( ys_c + (yp_c - ys_c) * (t + t_block / 2.0f) ) / voxel_size + imagedim.y / 2);

    	int xb_s = xb_c - BLOCK_SHMEM_2DIM_X / 2;
    	int yb_s = yb_c - BLOCK_SHMEM_2DIM_Y / 2;

    	// here we assume BLOCK_SHMEM_2DIM_X/Y are multiples of BLOCK_2DIM_X/Y. Be careful to set these parameters
    	unsigned int nx, ny;
    	int xm, ym;
    	for( ny = 0; ny < BLOCK_SHMEM_2DIM_Y / BLOCK_2DIM_Y; ny++  ){
    	  for( nx = 0; nx < BLOCK_SHMEM_2DIM_X / BLOCK_2DIM_X; nx++ ){
	    
    	    xm = xb_s + nx * BLOCK_2DIM_X + cidx.x; 
    	    ym = yb_s + ny * BLOCK_2DIM_Y + cidx.y;

    	    if(  xm >= 0.0f && xm < 1.0f * imagedim.x && ym >= 0.0f && ym < 1.0f * imagedim.y ){
    	      shmem_image[nx * BLOCK_2DIM_X + cidx.x][ ny * BLOCK_2DIM_Y + cidx.y] = d_image_prev[ ym * imagedim.x + xm ];
    	    }
    	    else{
    	      shmem_image[nx * BLOCK_2DIM_X + cidx.x][ ny * BLOCK_2DIM_Y + cidx.y] = 0.0f;
    	    }
    	    __syncthreads();
    	  }
    	}

    	xs = SOD * cosf( angle ) + (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
    	ys = SOD * sinf( angle ) + (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * cosf( angle );
      
    	xp = (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    	yp = (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * cosf( angle );
    
    	// calculate projection for each ray between [t, t + t_block]
    	int num_steps = (int) (t_block * len / spacing ) ;
    	float i, x, y;
    	int nx0, ny0, nx1, ny1; 

    	float tmp;
    	for( i = 0; i <= num_steps; i++ ){
    	  x = xs + (xp - xs) * t + i * dx;
    	  y = ys + (yp - ys) * t + i * dy;

    	  x = x / voxel_size + 1.0f * imagedim.x / 2;
    	  y = y / voxel_size + 1.0f * imagedim.y / 2;

    	  tmp = 0.0f; 
    	  if( x >= -1.0f && x < 1.0f * imagedim.x && y >= -1.0f && y < 1.0f * imagedim.y ){
    	    nx0 = (int)floor(x);
    	    ny0 = (int)floor(y);
    	    nx1 = nx0 + 1;
    	    ny1 = ny0 + 1; 

    	    // Need to make sure that all required values are stored in shmem_image[][]
    	    if( nx0 >= 0.0f && nx0 <= 1.0f * imagedim.x - 1.0f && ny0 >= 0.0f && ny0 <= 1.0f * imagedim.y - 1.0f )
    	      tmp += shmem_image[nx0 - xb_s][ny0 - yb_s] * (1 - (x - nx0)) * (1 - (y - ny0));
    	    if( nx1 >= 0.0f && nx1 <= 1.0f * imagedim.x - 1.0f && ny0 >= 0.0f && ny0 <= 1.0f * imagedim.y - 1.0f )
    	      tmp += shmem_image[nx1 - xb_s][ny0 - yb_s] * (x - nx0) * (1 - (y - ny0));
    	    if( nx0 >= 0.0f && nx0 <= 1.0f * imagedim.x - 1.0f && ny1 >= 0.0f && ny1 <= 1.0f * imagedim.y - 1.0f )
    	      tmp += shmem_image[nx0 - xb_s][ny1 - yb_s] * (1 - (x - nx0)) * (y - ny0);
    	    if( nx1 >= 0.0f && nx1 <= 1.0f * imagedim.x - 1.0f && ny1 >= 0.0f && ny1 <= 1.0f * imagedim.y - 1.0f )
    	      tmp += shmem_image[nx1 - xb_s][ny1 - yb_s] * (x - nx0) * (y - ny0);		 
    	  }

    	  if( tmp > PRECISION ){
    	    weight_len = weight_len + 1.0f;
    	    proj_res += tmp * spacing;	     
    	  }
    	}
	
    	x = xs + (xp - xs) * (t + t_block);
        y = ys + (yp - ys) * (t + t_block);

        x = x / voxel_size + 1.0f * imagedim.x / 2;
    	y = y / voxel_size + 1.0f * imagedim.y / 2;

    	tmp = 0.0f; 
    	if( x >= -1.0f && x <= 1.0f * imagedim.x - 1.0f && y >= -1.0f && y <= 1.0f * imagedim.y - 1.0f ){
    	  nx0 = (int)floor(x);
    	  ny0 = (int)floor(y);
    	  nx1 = nx0 + 1;
    	  ny1 = ny0 + 1; 

    	  // Need to make sure that all required values are stored in shmem_image[][]
    	  tmp += shmem_image[nx0 - xb_s][ny0 - yb_s] * (1 - (x - nx0)) * (1 - (y - ny0));
    	  tmp += shmem_image[nx1 - xb_s][ny0 - yb_s] * (x - nx0) * (1 - (y - ny0));
    	  tmp += shmem_image[nx0 - xb_s][ny1 - yb_s] * (1 - (x - nx0)) * (y - ny0);
    	  tmp += shmem_image[nx1 - xb_s][ny1 - yb_s] * (x - nx0) * (y - ny0);		 
    	}

    	if( tmp > PRECISION ){
    	  weight_len = weight_len + 1.0f;
    	  proj_res += tmp * (t_block * len - num_steps * spacing); 
        }
      }
    
      __syncthreads();

    }
  
  }
   
  if( weight_len > PRECISION_WEIGHT )	
    proj_res = ( shmem_proj[cidx.x][cidx.y] - proj_res ) / weight_len; 
  else
    proj_res = 0.0f;

  __syncthreads();
  
  // store the result
  uint outidx = idy*projdim.x + idx;
 
  proj_cur[ outidx ] =  proj_res;

}

__global__ void 
proj_cal_kernel_sharemem_3D( float* d_image_prev, float* proj, float* proj_cur, 
			     dim3 projdim, dim3 griddim, dim3 imagedim, 
			     float spacing, float voxel_size, float proj_pixel_size, float SOD, 
			     float inv_rot, unsigned int deviceShareMemSizePerBlock){

  // 1. initialize shared memory
  
  dim3 bidx;
  uint idx, idy, idz, idx_s, idx_c, idx_e;;

  bidx.x = blockIdx.x;   
  bidx.z = blockIdx.y / griddim.y;
  bidx.y = blockIdx.y - bidx.z*griddim.y;   // 0 when num_depth = 1

  idx = bidx.x * blockDim.x + threadIdx.x;
  idy = bidx.y * blockDim.y + threadIdx.y;
  idz = bidx.z * blockDim.z + threadIdx.z;

  idx_s = bidx.x * blockDim.x;                    // starting ray in the current block
  idx_c = bidx.x * blockDim.x + blockDim.x / 2;   // central ray in the current block
  idx_e = bidx.x * blockDim.x + blockDim.x - 1;   // ending ray in the current block

  // shared memory
  __shared__ float shmem_proj[BLOCK_3DIM_X][BLOCK_3DIM_Y][BLOCK_3DIM_Z];
  __shared__ float shmem_image[BLOCK_SHMEM_3DIM_X][BLOCK_SHMEM_3DIM_Y][BLOCK_3DIM_Y];  // Note the dimension
                                 // num_ray            num_proj        num_elevation                     

  dim3 cidx;
  cidx.x = threadIdx.x;
  cidx.y = threadIdx.y;
  cidx.z = threadIdx.z;

  // load shared memory

  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {
    shmem_proj[cidx.x][cidx.y][cidx.z] = proj[((idz)*projdim.y + (idy))*projdim.x + idx];
  }

  __syncthreads();

  // 2. apply kernel
  float proj_res = 0.0f;
  float weight_len = 0.0f;
  float len = 0.0f;
   
  if(idx < projdim.x && idy < projdim.y && idz < projdim.z)
  {

    // rotation angle

#ifdef ROTATION_CLOCKWISE
    float angle = -PI * idz / projdim.z; 
#else
    float angle = PI * idz / projdim.z + PI / 2.0f; 
#endif

    // determine central ray in the current block
    float xs_c = SOD * cosf( angle ) + (1.0f * idx_c - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
    float ys_c = SOD * sinf( angle ) + (1.0f * idx_c - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    // detector pixel position (-SDD = 0, (idx - projdim.x / 2 ) * proj_pixel_size)
    float xp_c = (1.0f * idx_c - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    float yp_c = (1.0f * idx_c - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    // determine the intersection point from (0, 0) perpendicular to the central ray 
    // determined by (xs_c, ys_c) and (xp_c, yp_c)
    float xc = 0.0f;
    float yc = 0.0f;
    float tc = 0.0f;

    if( fabsf( xs_c * yp_c - xp_c * ys_c ) > PRECISION ){  // (0, 0) is not on the central ray
      tc = -( xs_c * (xp_c - xs_c) + ys_c * (yp_c - ys_c) );
      tc /= (yp_c - ys_c) * (yp_c - ys_c) + (xp_c - xs_c) * (xp_c - xs_c);

      xc = xs_c + (xp_c - xs_c) * tc;
      yc = ys_c + (yp_c - ys_c) * tc;
    }
    else{
       xc = 0.0f;
       yc = 0.0f; 
       tc = -xs_c / (xp_c - xs_c);
    }
    
    // vector direction (zs_c = zp_c for parallel beam)
    len = sqrtf( (xp_c - xs_c) * (xp_c - xs_c) + (yp_c - ys_c) * (yp_c - ys_c) );
    float dx = (xp_c - xs_c) * spacing / len;   
    float dy = (yp_c - ys_c) * spacing / len;

    // determine intersection points of the starting ray in the current block
    // The bounded volumes is specified by  (xc-BLOCK_SHMEM_3DIM_X/2, yc-BLOCK_SHMEM_3DIM_Y/2)
    //                                  and (xc+BLOCK_SHMEM_3DIM_X/2, yc+BLOCK_SHMEM_3DIM_Y/2)

    float xs = SOD * cosf( angle ) + (1.0f * idx_s - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
    float ys = SOD * sinf( angle ) + (1.0f * idx_s - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    // detector pixel position (-SDD = 0, (idx - projdim.x / 2 ) * proj_pixel_size)

    float xp = (1.0f * idx_s - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    float yp = (1.0f * idx_s - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    float t_near_s = -100.0f;
    float t_far_s = 100.0f;

    // calculate t_near_s
    if( fabsf( xp - xs ) > PRECISION ){

    	if( dx < 0.0f &&  t_near_s < ( xc + 1.0f * BLOCK_SHMEM_3DIM_X/2 - 1.0f  - xs ) / ( xp - xs ) )
	    t_near_s = ( xc + 1.0f * BLOCK_SHMEM_3DIM_X/2 - 1.0f - xs ) / ( xp - xs );
	
        if( dx > 0.0f &&  t_near_s < ( xc -1.0f * BLOCK_SHMEM_3DIM_X/2 - xs ) / ( xp - xs ) )
	    t_near_s = ( xc - 1.0f * BLOCK_SHMEM_3DIM_X/2  - xs ) / ( xp - xs );
    }

    if( fabsf( yp - ys ) > PRECISION ){

        if( dy < 0.0f && t_near_s < ( yc + 1.0f * BLOCK_SHMEM_3DIM_Y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_near_s = ( yc + 1.0f * BLOCK_SHMEM_3DIM_Y/2 - 1.0f  - ys ) / ( yp - ys );

        if( dy > 0.0f && t_near_s < ( yc - 1.0f * BLOCK_SHMEM_3DIM_Y/2 - ys ) / ( yp - ys ) )
    	    t_near_s = ( yc - 1.0f * BLOCK_SHMEM_3DIM_Y/2 - ys ) / ( yp - ys );
    }

    // calculate t_far_s
    if( fabsf( xp - xs ) > PRECISION ){
        if( dx < 0.0f && t_far_s >  ( xc - 1.0f *BLOCK_SHMEM_3DIM_X/2 - xs ) / ( xp - xs ) )
            t_far_s = ( xc - 1.0f * BLOCK_SHMEM_3DIM_X/2 - xs ) / ( xp - xs );

        if( dx > 0.0f && t_far_s > ( xc + 1.0f * BLOCK_SHMEM_3DIM_X/2 - 1.0f   - xs ) / ( xp - xs ) )
            t_far_s = ( xc + 1.0f * BLOCK_SHMEM_3DIM_X/2 - 1.0f  - xs ) / ( xp - xs );

    }

    if( fabsf( yp - ys ) > PRECISION ){
        if( dy < 0.0f && t_far_s > ( yc - 1.0f * BLOCK_SHMEM_3DIM_Y/2 - ys ) / ( yp - ys ) )
    	    t_far_s = ( yc - 1.0f * BLOCK_SHMEM_3DIM_Y/2 - ys ) / ( yp - ys );

        if( dy > 0.0f && t_far_s > ( yc + 1.0f * BLOCK_SHMEM_3DIM_Y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_far_s = ( yc + 1.0f * BLOCK_SHMEM_3DIM_Y/2 - 1.0f  - ys ) / ( yp - ys );
    }

    //
    if( fabsf(BLOCK_SHMEM_3DIM_X - BLOCK_SHMEM_3DIM_X / 2 - 1 ) < PRECISION ){
      if(  fabsf( xp - xs ) < PRECISION && ( xp < xc - 1.0f * BLOCK_SHMEM_3DIM_X / 2.0f 
					     || xp > xc + 1.0f * BLOCK_SHMEM_3DIM_X / 2.0f ) ){
    	t_near_s = t_far_s + 1.0f;
      }  
    }
    else {
      if(  fabsf( xp - xs ) < PRECISION && ( xp < xc - 1.0f * BLOCK_SHMEM_3DIM_X / 2.0f 
					     || xp > xc + 1.0f * BLOCK_SHMEM_3DIM_X / 2.0f - 1.0f ) ){
    	t_near_s = t_far_s + 1.0f;
      }  
    }

    if( fabsf(BLOCK_SHMEM_3DIM_Y - BLOCK_SHMEM_3DIM_Y / 2 - 1 ) < PRECISION ){
      if(  fabsf( yp - ys ) < PRECISION && ( yp < yc - 1.0f * BLOCK_SHMEM_3DIM_Y / 2.0f 
					     || yp > yc + 1.0f * BLOCK_SHMEM_3DIM_Y / 2.0f ) ){
    	t_near_s = t_far_s + 1.0f;
      }
    }
    else {
      if(  fabsf( yp - ys ) < PRECISION && ( yp < yc - 1.0f * BLOCK_SHMEM_3DIM_Y / 2.0f 
					     || yp > yc + 1.0f * BLOCK_SHMEM_3DIM_Y / 2.0f - 1.0f ) ){
    	t_near_s = t_far_s + 1.0f;
      }
    }

    // determine intersection points of the ending ray in the current block
    // The bounded volumes is specified by  (xc-BLOCK_SHMEM_3DIM_X/2, yc-BLOCK_SHMEM_3DIM_Y/2)
    //                                  and (xc+BLOCK_SHMEM_3DIM_X/2, yc+BLOCK_SHMEM_3DIM_Y/2)

    xs = SOD * cosf( angle ) + (1.0f * idx_e - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
    ys = SOD * sinf( angle ) + (1.0f * idx_e - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    // detector pixel position (-SDD = 0, (idx - projdim.x / 2 ) * proj_pixel_size)

    xp = (1.0f * idx_e - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
    yp = (1.0f * idx_e - projdim.x / 2 ) * proj_pixel_size * cosf( angle );

    float t_near_e = -100.0f;
    float t_far_e = 100.0f;

    // calculate t_near_e
    if( fabsf( xp - xs ) > PRECISION ){

    	if( dx < 0.0f &&  t_near_e < ( xc + 1.0f * BLOCK_SHMEM_3DIM_X/2 - 1.0f  - xs ) / ( xp - xs ) )
	    t_near_e = ( xc + 1.0f * BLOCK_SHMEM_3DIM_X/2 - 1.0f   - xs ) / ( xp - xs );
	
        if( dx > 0.0f &&  t_near_e < ( xc - 1.0f * BLOCK_SHMEM_3DIM_X/2 - xs ) / ( xp - xs ) )
	    t_near_e = ( xc - 1.0f * BLOCK_SHMEM_3DIM_X/2  - xs ) / ( xp - xs );
    }

    if( fabsf( yp - ys ) > PRECISION ){

        if( dy < 0.0f && t_near_e < ( yc + 1.0f * BLOCK_SHMEM_3DIM_Y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_near_e = ( yc + 1.0f * BLOCK_SHMEM_3DIM_Y/2 - 1.0f  - ys ) / ( yp - ys );

        if( dy > 0.0f && t_near_e < ( yc - 1.0f * BLOCK_SHMEM_3DIM_Y/2 - ys ) / ( yp - ys ) )
    	    t_near_e = ( yc - 1.0f * BLOCK_SHMEM_3DIM_Y/2 - ys ) / ( yp - ys );
    }

    // calculate t_far_e
    if( fabsf( xp - xs ) > PRECISION ){
        if( dx < 0.0f && t_far_e >  ( xc - 1.0f *BLOCK_SHMEM_3DIM_X/2 - xs ) / ( xp - xs ) )
            t_far_e = ( xc - 1.0f * BLOCK_SHMEM_3DIM_X/2 - xs ) / ( xp - xs );

        if( dx > 0.0f && t_far_e > ( xc + 1.0f * BLOCK_SHMEM_3DIM_X/2 - 1.0f   - xs ) / ( xp - xs ) )
            t_far_e = ( xc + 1.0f * BLOCK_SHMEM_3DIM_X/2 - 1.0f  - xs ) / ( xp - xs );

    }

    if( fabsf( yp - ys ) > PRECISION ){
        if( dy < 0.0f && t_far_e > ( yc - 1.0f * BLOCK_SHMEM_3DIM_Y/2 - ys ) / ( yp - ys ) )
    	    t_far_e = ( yc - 1.0f * BLOCK_SHMEM_3DIM_Y/2 - ys ) / ( yp - ys );

        if( dy > 0.0f && t_far_e > ( yc + 1.0f * BLOCK_SHMEM_3DIM_Y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_far_e = ( yc + 1.0f * BLOCK_SHMEM_3DIM_Y/2 - 1.0f  - ys ) / ( yp - ys );
    }

    //
    if( fabsf(BLOCK_SHMEM_3DIM_X - BLOCK_SHMEM_3DIM_X / 2 - 1 ) < PRECISION ){
      if(  fabsf( xp - xs ) < PRECISION && ( xp < xc - 1.0f * BLOCK_SHMEM_3DIM_X / 2.0f 
					     || xp > xc + 1.0f * BLOCK_SHMEM_3DIM_X / 2.0f ) ){
    	t_near_e = t_far_e + 1.0f;
      }  
    }
    else {
      if(  fabsf( xp - xs ) < PRECISION && ( xp < xc - 1.0f * BLOCK_SHMEM_3DIM_X / 2.0f 
					     || xp > xc + 1.0f * BLOCK_SHMEM_3DIM_X / 2.0f - 1.0f ) ){
    	t_near_e = t_far_e + 1.0f;
      }  
    }

    if( fabsf(BLOCK_SHMEM_3DIM_Y - BLOCK_SHMEM_3DIM_Y / 2 - 1 ) < PRECISION ){
      if(  fabsf( yp - ys ) < PRECISION && ( yp < yc - 1.0f * BLOCK_SHMEM_3DIM_Y / 2.0f 
					     || yp > yc + 1.0f * BLOCK_SHMEM_3DIM_Y / 2.0f ) ){
    	t_near_e = t_far_e + 1.0f;
      }
    }
    else {
      if(  fabsf( yp - ys ) < PRECISION && ( yp < yc - 1.0f * BLOCK_SHMEM_3DIM_Y / 2.0f 
					     || yp > yc + 1.0f * BLOCK_SHMEM_3DIM_Y / 2.0f - 1.0f ) ){
    	t_near_e = t_far_e + 1.0f;
      }
    }

    // A good point for parallel beam is that t_near_s, t_near_e (for the starting ray), 
    // t_far_s and t_far_e (for the ending ray) are exactly same as the t_near and t_far 
    // for the central ray, because of the parallel feature of the beam. This property 
    // makes it much easier to determine the accumulating portions inside each shared 
    // memory block. 

    float t_block = 0.0f; 
    if( t_near_s > t_far_s || t_near_e > t_far_e  ) //no intersection, meaning the shared memory block is too small. 
        proj_res = 0.0f;
    else{

      t_block = t_far_e;
      if( t_block > t_far_s )     // smaller for t_far
	t_block = t_far_s;
    
      if( t_near_e < t_near_s )   // larger for t_near
	t_block -= t_near_s;
      else
	t_block -= t_near_e;

      t_block = t_block * 0.9;    // shrink for redundency so that shmem_image is large enough

      float delta_t = sqrtf( 1.0f * imagedim.x * imagedim.x + 1.0f * imagedim.y * imagedim.y + 1.0f * imagedim.z * imagedim.z) / len / 2.0f; 

      float t;
      // sampling (interval [tc - delta_t, tc + delta_t]
      for( t = tc - delta_t; t <= tc + delta_t + t_block; t = t + t_block ){

	// t + t_block / 2.0 corresponds to the center of the shared memory block
	// load image data into the shared memory 

	int xb_c = (int) round( ( xs_c + (xp_c - xs_c) * (t + t_block / 2.0f) ) / voxel_size + imagedim.x / 2);
	int yb_c = (int) round( ( ys_c + (yp_c - ys_c) * (t + t_block / 2.0f) ) / voxel_size + imagedim.y / 2);

	int xb_s = xb_c - BLOCK_SHMEM_3DIM_X / 2;
	int yb_s = yb_c - BLOCK_SHMEM_3DIM_Y / 2;

	// here we assume BLOCK_SHMEM_3DIM_X/Y are multiples of BLOCK_2DIM_X/Y. Be careful to set these parameters
	unsigned int nx, ny;
	int xm, ym;
	for( ny = 0; ny < BLOCK_SHMEM_3DIM_Y / BLOCK_2DIM_Y; ny++  ){
	  for( nx = 0; nx < BLOCK_SHMEM_3DIM_X / BLOCK_2DIM_X; nx++ ){
	    
	    xm = xb_s + nx * BLOCK_2DIM_X + cidx.x; 
	    ym = yb_s + ny * BLOCK_2DIM_Y + cidx.z;  // note the dimension

	    if(  xm >= 0.0f && xm < 1.0f * imagedim.x && ym >= 0.0f && ym < 1.0f * imagedim.y ){
	      shmem_image[nx * BLOCK_2DIM_X + cidx.x][ ny * BLOCK_2DIM_Y + cidx.z][cidx.y] = d_image_prev[ (idy * imagedim.y + ym) * imagedim.x + xm ];   // note the dimension
	    }
	    else{
	      shmem_image[nx * BLOCK_2DIM_X + cidx.x][ ny * BLOCK_2DIM_Y + cidx.z][cidx.y] = 0.0f;
	    }
	    __syncthreads();
	  }
	}

	xs = SOD * cosf( angle ) + (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * sinf( -angle ); 
	ys = SOD * sinf( angle ) + (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * cosf( angle );
      
	xp = (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * sinf( -angle );
	yp = (1.0f * idx - projdim.x / 2 ) * proj_pixel_size * cosf( angle );
    
	// calculate projection for each ray between [t, t + t_block]
	int num_steps = (int) (t_block * len / spacing ) ;
	float i, x, y;
	int nx0, ny0, nx1, ny1; 

	float tmp;
	for( i = 0; i <= num_steps; i++ ){
	  x = xs + (xp - xs) * t + i * dx;
	  y = ys + (yp - ys) * t + i * dy;

	  x = x / voxel_size + 1.0f * imagedim.x / 2;
	  y = y / voxel_size + 1.0f * imagedim.y / 2;

	  tmp = 0.0f; 
	  if( x >= -1.0f && x < 1.0f * imagedim.x && y >= -1.0f && y < 1.0f * imagedim.y ){
	    nx0 = (int)floor(x);
	    ny0 = (int)floor(y);
	    nx1 = nx0 + 1;
	    ny1 = ny0 + 1; 

	    // Need to make sure that all required values are stored in shmem_image[][]
	    if( nx0 >= 0.0f && nx0 <= 1.0f * imagedim.x - 1.0f && ny0 >= 0.0f && ny0 <= 1.0f * imagedim.y - 1.0f )
	      tmp += shmem_image[nx0 - xb_s][ny0 - yb_s][cidx.y] * (1 - (x - nx0)) * (1 - (y - ny0));
	    if( nx1 >= 0.0f && nx1 <= 1.0f * imagedim.x - 1.0f && ny0 >= 0.0f && ny0 <= 1.0f * imagedim.y - 1.0f )
	      tmp += shmem_image[nx1 - xb_s][ny0 - yb_s][cidx.y] * (x - nx0) * (1 - (y - ny0));
	    if( nx0 >= 0.0f && nx0 <= 1.0f * imagedim.x - 1.0f && ny1 >= 0.0f && ny1 <= 1.0f * imagedim.y - 1.0f )
	      tmp += shmem_image[nx0 - xb_s][ny1 - yb_s][cidx.y] * (1 - (x - nx0)) * (y - ny0);
	    if( nx1 >= 0.0f && nx1 <= 1.0f * imagedim.x - 1.0f && ny1 >= 0.0f && ny1 <= 1.0f * imagedim.y - 1.0f )
	      tmp += shmem_image[nx1 - xb_s][ny1 - yb_s][cidx.y] * (x - nx0) * (y - ny0);		 
	  }

	  if( tmp > PRECISION ){
	    weight_len = weight_len + 1.0f;
	    proj_res += tmp * spacing;	     
	  }
	}
	
	x = xs + (xp - xs) * (t + t_block);
        y = ys + (yp - ys) * (t + t_block);

        x = x / voxel_size + 1.0f * imagedim.x / 2;
	y = y / voxel_size + 1.0f * imagedim.y / 2;

	tmp = 0.0f; 
	if( x >= -1.0f && x <= 1.0f * imagedim.x - 1.0f && y >= -1.0f && y <= 1.0f * imagedim.y - 1.0f ){
	  nx0 = (int)floor(x);
	  ny0 = (int)floor(y);
	  nx1 = nx0 + 1;
	  ny1 = ny0 + 1; 

	  // Need to make sure that all required values are stored in shmem_image[][]
	  tmp += shmem_image[nx0 - xb_s][ny0 - yb_s][cidx.y] * (1 - (x - nx0)) * (1 - (y - ny0));
	  tmp += shmem_image[nx1 - xb_s][ny0 - yb_s][cidx.y] * (x - nx0) * (1 - (y - ny0));
	  tmp += shmem_image[nx0 - xb_s][ny1 - yb_s][cidx.y] * (1 - (x - nx0)) * (y - ny0);
	  tmp += shmem_image[nx1 - xb_s][ny1 - yb_s][cidx.y] * (x - nx0) * (y - ny0);		 
	}

	if( tmp > PRECISION ){
	  weight_len = weight_len + 1.0f;
	  proj_res += tmp * (t_block * len - num_steps * spacing); 
        }
      }
    
      __syncthreads();

    }
  }
   
  if( weight_len > PRECISION_WEIGHT )	
    proj_res = ( shmem_proj[cidx.x][cidx.y][cidx.z] - proj_res ) / weight_len; 
  else
    proj_res = 0.0f;

  __syncthreads();
  
  // store the result
  uint outidx = (idz * projdim.y + idy)*projdim.x + idx;
 
  proj_cur[ outidx ] =  proj_res;

}

#endif // __PROJ_CAL_CU

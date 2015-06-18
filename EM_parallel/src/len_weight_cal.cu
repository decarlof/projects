// CUDA codes for Projection calculation

#ifndef __LEN_WEIGHT_CAL_CU
#define __LEN_WEIGHT_CAL_CU

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

texture<float, 3, cudaReadModeElementType> tex_voxel;  // 3D texture

#define PI 3.1415926


// structure for SART3D GPU implementation
typedef struct{
  int x;            // corresponds to the location of image samples
  int y;
  int z;
  float wa;        // corresponds to a_{ij}
  float wt;        // corresponds to t_{ij}
} WeightSample;


__global__ void len_weight_cal_kernel( float*, dim3, dim3, dim3, float, float, float, float);


extern "C" 
void len_weight_cal_wrapper( cudaArray* d_array_voxel,  
                       float* d_proj_cur, 
		       int num_depth, int num_height, int num_width, 
		       int num_proj, int num_elevation, int num_ray,
		       float spacing, float voxel_size, float proj_pixel_size,
		       float SOD){

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
    nBlockX = ceil((float)num_ray / (float)blockWidth);
    nBlockY = ceil((float)num_elevation / (float)blockHeight);
    nBlockZ = ceil((float)num_proj / (float)blockDepth);
   
    dim3 dimGrid(nBlockX, nBlockY*nBlockZ);                // 3D grid is not supported on G80
    dim3 dimBlock(blockWidth, blockHeight, blockDepth); 
    dim3 projdim(num_ray, num_elevation, num_proj);
    dim3 griddim(nBlockX, nBlockY, nBlockZ); 
    dim3 imagedim( num_width, num_height, num_depth);
   
    // set texture parameters
    tex_voxel.normalized = false;                      // access with normalized texture coordinates
    tex_voxel.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex_voxel.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
    tex_voxel.addressMode[1] = cudaAddressModeClamp;
    tex_voxel.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // float
    CUDA_SAFE_CALL(cudaBindTextureToArray(tex_voxel, d_array_voxel, float1Desc));

    // execute the kernel
    len_weight_cal_kernel<<< dimGrid, dimBlock >>>( d_proj_cur, projdim, griddim, imagedim, 
    		       			      spacing, voxel_size, proj_pixel_size, SOD);

    CUDA_SAFE_CALL( cudaUnbindTexture( tex_voxel ) );
    
    // check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");

}

__global__ void 
len_weight_cal_kernel( float* proj_cur, dim3 projdim, dim3 griddim, dim3 imagedim, 
		 float spacing, float voxel_size, float proj_pixel_size, float SOD){


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
    float angle = 2 * PI * idz / projdim.z + PI / 2;

    // source position
    float xs = SOD * cosf( angle );
    float ys = SOD * sinf( angle );
    float zs = 0.0f;

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

    if( fabsf( xp - xs ) > 1e-7f ){

    	if( dx < 0.0f &&  t_near < ( 1.0f * imagedim.x/2 - 1.0f  - xs ) / ( xp - xs ) )
	    t_near = ( 1.0f * imagedim.x/2 - 1.0f   - xs ) / ( xp - xs );
	
        if( dx > 0.0f &&  t_near < ( -1.0f * imagedim.x/2 - xs ) / ( xp - xs ) )
	    t_near = ( -1.0f * imagedim.x/2  - xs ) / ( xp - xs );
    }

    if( fabsf( yp - ys ) > 1e-7f ){

        if( dy < 0.0f && t_near < ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_near = ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys );

        if( dy > 0.0f && t_near < ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys ) )
    	    t_near = ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys );
    }

    if( fabsf( zp - zs ) > 1e-7f ){
        if( dz < 0.0f && t_near < ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs ) )
    	    t_near = ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs );

        if( dz > 0.0f && t_near < ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs ) )
    	    t_near = ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs );
    }

    // calculate t_far
    if( fabsf( xp - xs ) > 1e-7f ){
        if( dx < 0.0f && t_far >  ( -1.0f *imagedim.x/2 - xs ) / ( xp - xs ) )
            t_far = ( -1.0f * imagedim.x/2 - xs ) / ( xp - xs );

        if( dx > 0.0f && t_far > ( 1.0f * imagedim.x/2 - 1.0f   - xs ) / ( xp - xs ) )
            t_far = ( 1.0f * imagedim.x/2 - 1.0f  - xs ) / ( xp - xs );

    }

    if( fabsf( yp - ys ) > 1e-7f ){
        if( dy < 0.0f && t_far > ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys ) )
    	    t_far = ( -1.0f * imagedim.y/2 - ys ) / ( yp - ys );

        if( dy > 0.0f && t_far > ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys ) )
    	    t_far = ( 1.0f * imagedim.y/2 - 1.0f  - ys ) / ( yp - ys );
    }

    if( fabsf( zp - zs ) > 1e-7f ){
        if( dz < 0.0f && t_far > ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs ) )
      	    t_far = ( -1.0f * imagedim.z/2 - zs ) / ( zp - zs );

        if( dz > 0.0f && t_far > ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs ) )
      	    t_far = ( 1.0f * imagedim.z/2 - 1.0f  - zs ) / ( zp - zs );
    }


    if(  fabsf( xp - xs ) < 1e-7f && ( xp < 0.0f || xp > 1.0f * imagedim.x / 2 - 1.0f ) ){
    	t_near = t_far + 1.0f;
    }  

    if(  fabsf( yp - ys ) < 1e-7f && ( yp < 0.0f || yp > 1.0f * imagedim.y / 2 - 1.0f ) ){
    	t_near = t_far + 1.0f;
    }

    if(  fabsf( zp - zs ) < 1e-7f && ( zp < 0.0f || zp > 1.0f * imagedim.z / 2 - 1.0f ) ){
    	t_near = t_far + 1.0f;
    }    

    if( t_near > t_far ) // no intersection
        proj_res = 0.0f;

    else{
	len = (t_far - t_near) * len + 1.0f;

	int num_steps = (int) (len / spacing );
	int i;
		
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
	
	if( len - num_steps * spacing > 1e-7f ){
	    x = xs + (xp - xs) * t_far;
	    y = ys + (yp - ys) * t_far;
	    z = zs + (zp - zs) * t_far;

            x = x / voxel_size + 1.0f * imagedim.x / 2;
	    y = y / voxel_size + 1.0f * imagedim.y / 2;
	    z = z / voxel_size + 1.0f * imagedim.z / 2;

	    proj_res +=  tex3D( tex_voxel, x + 0.5f, y + 0.5f, z + 0.5f) * (len - num_steps * spacing);		
        }	

    } 

    // proj_res = t_near;
    // proj_res = t_far;
    // proj_res = dz;
    // proj_res = tex3D( tex_voxel, 1.0f * idx, 1.0f * idy, 1.0f * idz);

    __syncthreads();
    

    // store the result
    uint outidx = ((idz)* projdim.y + (idy))*projdim.x + idx;
 
    proj_cur[ outidx ] =  proj_res;
 
  }
 
}

#endif // __LEN_WEIGHT_CAL_CU
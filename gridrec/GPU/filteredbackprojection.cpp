//---------------------------------------------------------------------------
#include <stdlib.h>

#pragma hdrstop

#include "filteredbackprojection.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//Filtered Back Projection
//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

FBP::FBP (void)
{
  num_sinograms_needed = 1;

  fft_proj = NULL;
  sines = NULL;
  cosines = NULL;
  filter_lut = NULL;
  filtered_proj = NULL;
	
  sinogram1 = (float *) malloc (sizeof (float *));
  sinogram2 = NULL;
  reconstruction1 = (float *) malloc (sizeof (float *));
  reconstruction1 = NULL;
}

//---------------------------------------------------------------------------

// void FBP::acknowledgements (LogFileClass *acknowledge_file)
// {
//   acknowledge_file->Message ("__________________________________________________________________\n");
//   acknowledge_file->Message ("FBP class");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("Class for performing Filtered Back Projection Reconstructions.");
//   acknowledge_file->Message ("Origional source code developed in C by:");
//   acknowledge_file->Message ("       Steve Wang ");
//   acknowledge_file->Message ("CPP Class Developed and Maintained by:");
//   acknowledge_file->Message ("       Brian Tieman & Francesco DeCarlo");
//   acknowledge_file->Message ("       Argonne National Laboratory");
//   acknowledge_file->Message ("       tieman@aps.anl.gov");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  First version with acknowledgements");
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  Ported C code to a CPP object structure");
//   acknowledge_file->Message ("8/27/2003  V1.0   BT  Fixed a bug in type conversion that was");
//   acknowledge_file->Message ("                      only allowing the bottom right quandrant to");
//   acknowledge_file->Message ("                      be reconstructed.");
//   acknowledge_file->Message ("8/27/2003  V1.0   BT  Cleaned up some memory usage--got rid of variable");
//   acknowledge_file->Message ("                      t which was only a placeholder for calculations");
//   acknowledge_file->Message ("                      but was sinogram_x*sizeof(float) in size.");
//   acknowledge_file->Message ("8/27/2003  V1.0   BT  Converted to use same FFT routines as everyone else.");
//   acknowledge_file->Message ("                      Previously this routine was using an FFT routine");
//   acknowledge_file->Message ("                      written by Steve Wang that required the data as");
//   acknowledge_file->Message ("                      double and had a distinctly different calling");
//   acknowledge_file->Message ("                      convention as the numerical recipies routines--which");
//   acknowledge_file->Message ("                      are now the standard FFT calling convention throughout");
//   acknowledge_file->Message ("                      TomoMPI.");
//   acknowledge_file->Message ("                      In addition, this changed allowed for the removal of");
//   acknowledge_file->Message ("                      the bit_swap and lut arrays--which were each fairly");
//   acknowledge_file->Message ("                      large.");
//   acknowledge_file->Message ("9/7/2003  V1.3   BT   Re-optimized FBP1.  Reconstruct times are now 25%");
//   acknowledge_file->Message ("9/7/2003  V1.3   BT   faster.");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("__________________________________________________________________");
// }

//-------------------------------------------------------------------------------------

void FBP::init (void)
{
  int     loop;
  float   temp;

  /*  The FBP algorythm calls Sin() and Cos() which expect angles in radians  */
  for(loop=0;loop<theta_list_size;loop++)
    theta_list[loop] = (theta_list[loop]*PI)/180.0;
    
  fft_proj = (float *) malloc ((2*sinogram_x_dim+1)*sizeof(float));
  sines = (float *) malloc (sinogram_y_dim*sizeof(float));
  cosines = (float *) malloc (sinogram_y_dim*sizeof(float));
  filter_lut = (float *) malloc (sinogram_x_dim*sizeof(float));
  filtered_proj = (float *) malloc ((sinogram_y_dim*sinogram_x_dim)*sizeof(float));
	      
  for (loop=0;loop<sinogram_x_dim;loop++)
    {
      temp = (float) (loop - (float) sinogram_x_dim/2)/((float) (sinogram_x_dim/2));
      filter_lut[loop] = fabs(temp)*exp(-1.0*temp*temp/0.25);
    }
		
  for (loop=0;loop<sinogram_y_dim;loop++)
    {
      sines[loop] = sin(theta_list[loop]);
      cosines[loop] = cos(theta_list[loop]);
    }
}

//-------------------------------------------------------------------------------------

void FBP::reconstruct (void)
{
  int     loop,
    loop2,
    loop3,
    n;
  double  x,
    y,
    r,
    temp_sum;
  float   *sinogram,
    *reconstruction,
    *recon_offset,
    sinxdiv2;
			  
			  
  sinogram = sinogram1;
  reconstruction = reconstruction1;
			      
  if (reconstruction == NULL)
    {
      printf ("reconstruction memory not allocated \n");
      return;
    }
  else
    for (loop=0;loop<sinogram_x_dim*sinogram_x_dim;loop++)
      reconstruction[loop] = 0.0;
				
  for (loop=0;loop<sinogram_y_dim;loop++)
    {
      for (loop2=0;loop2<sinogram_x_dim;loop2++)
	{
	  fft_proj[2*loop2+1] = sinogram[loop*sinogram_x_dim+loop2];
	  fft_proj[2*loop2+2] = 0.0;
	}
				      
      four1 ((float *) fft_proj, sinogram_x_dim, 1);

      for (loop2=0;loop2<2*sinogram_x_dim;loop2++)
	fft_proj[loop2+1] = fft_proj[loop2+1]*filter_lut[loop2/2];

      four1 ((float *) fft_proj, sinogram_x_dim, -1);

      for (loop2=0;loop2<sinogram_x_dim;loop2++)
	filtered_proj[loop*sinogram_x_dim+loop2] = fft_proj[2*loop2+1];
    }
				  
  sinxdiv2 = (float) sinogram_x_dim/2.0;
  recon_offset = reconstruction;
  for (loop=0;loop<sinogram_x_dim;loop++)
    {
      y = ((float) (loop - sinxdiv2))/sinxdiv2;
					    
      for (loop2=0;loop2<sinogram_x_dim;loop2++)
	{
	  x = ((float) (loop2 - sinxdiv2))/sinxdiv2;
	  for (loop3=0;loop3<sinogram_y_dim;loop3++)
	    {
	      r = x*cosines[loop3] + y*sines[loop3];
	      if ((r >= -1) && (r < 1))
		*recon_offset += (float) filtered_proj[loop3*sinogram_x_dim + (int) ((r+1)*sinxdiv2)];
	    }
	  recon_offset++;
	}
    }
}

//-------------------------------------------------------------------------------------

void FBP::destroy (void)
{
  if (fft_proj != NULL)
    free(fft_proj);
  if (sines != NULL)
    free(sines);
  if (cosines != NULL)
    free(cosines);
  if (filter_lut != NULL)
    free(filter_lut);
  if (filtered_proj != NULL)
    free(filtered_proj);
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//Optimized Filtered Back Projection
//-------------------------------------------------------------------------------------
//---------------------------------------------------------------------------

// void OptimizedFBP::acknowledgements (LogFileClass *acknowledge_file)
// {
//   acknowledge_file->Message ("__________________________________________________________________");
//   acknowledge_file->Message ("OptimizedFBP class");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("Class for performing a code optimized Filtered Back Projection Reconstructions.");
//   acknowledge_file->Message ("Origional source code developed in C by:");
//   acknowledge_file->Message ("       Steve Wang ");
//   acknowledge_file->Message ("CPP Class Developed and Maintained by:");
//   acknowledge_file->Message ("       Brian Tieman & Francesco DeCarlo");
//   acknowledge_file->Message ("       Argonne National Laboratory");
//   acknowledge_file->Message ("       tieman@aps.anl.gov");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  First version with acknowledgements");
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  Ported C code to a CPP object structure");
//   acknowledge_file->Message ("8/27/2003  V1.0   BT  Fixed a bug in type conversion that was");
//   acknowledge_file->Message ("                      only allowing the bottom right quandrant to");
//   acknowledge_file->Message ("                      be reconstructed.");
//   acknowledge_file->Message ("8/27/2003  V1.0   BT  Cleaned up some memory usage--got rid of variable");
//   acknowledge_file->Message ("                      t which was only a placeholder for calculations");
//   acknowledge_file->Message ("                      but was sinogram_x*sizeof(float) in size.");
//   acknowledge_file->Message ("                      Got rid of lut array which wasn't used.");
//   acknowledge_file->Message ("9/7/2003  V1.3   BT   Re-optimized FBP1.  Reconstruct times are now 15%");
//   acknowledge_file->Message ("9/7/2003  V1.3   BT   faster.");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("__________________________________________________________________");
// }

//-------------------------------------------------------------------------------------

void OptimizedFBP::init (void)
{
  int     loop;
  float   temp;

  /*  The FBP algorythm calls Sin() and Cos() which expect angles in radians  */
  for(loop=0;loop<theta_list_size;loop++)
    theta_list[loop] = (theta_list[loop]*PI)/180.0;
    
  fft_proj = (float *) malloc ((2*sinogram_x_dim+1)*sizeof(float));
  sines = (float *) malloc (sinogram_y_dim*sizeof(float));
  cosines = (float *) malloc (sinogram_y_dim*sizeof(float));
  filter_lut = (float *) malloc (sinogram_x_dim*sizeof(float));
  filtered_proj = (float*) malloc ((sinogram_y_dim*sinogram_x_dim)*sizeof(float));
	      
  for (loop=0;loop<sinogram_x_dim;loop++)
    {
      temp =(float)(loop - (float) (sinogram_x_dim/2))/((float) (sinogram_x_dim/2));
      filter_lut[loop] = 1-sqrt(fabs(temp));
    }
		
  for (loop=0;loop<sinogram_y_dim;loop++)
    {
      sines[loop] = sin(theta_list[loop]);
      cosines[loop] = cos(theta_list[loop]);
    }
		  
}

//-------------------------------------------------------------------------------------

void OptimizedFBP::reconstruct (void)
{
  int     loop,
    loop2,
    loop3,
    n;
  float   x,
    y,
    r,
    pos_gate=5.0,
    neg_gate=-5.0,
    *sinogram,
    *reconstruction,
    *recon_offset,
    sinxdiv2;
			
  sinogram = sinogram1;
  reconstruction = reconstruction1;
			    
  if (reconstruction == NULL)
    {
      printf("reconstruction memory not allocated \n");
      return;
    }
  else
    for (loop=0;loop<sinogram_x_dim*sinogram_x_dim;loop++)
      reconstruction[loop] = 0.0;
			      
  for (loop=0;loop<sinogram_y_dim;loop++)
    {
      for (loop2=0;loop2<sinogram_x_dim;loop2++)
	{
	  if ((sinogram[loop*sinogram_x_dim+loop2] > neg_gate) && (sinogram[loop*sinogram_x_dim+loop2] < pos_gate))
	    fft_proj[2*loop2+1] = sinogram[loop*sinogram_x_dim+loop2];
	  else
	    fft_proj[2*loop2+1] = 0.0;
	  fft_proj[2*loop2+2] = 0.0;
	}
				    
      four1 ((float *) fft_proj, sinogram_x_dim, 1);

      for (loop2=0;loop2<2*sinogram_x_dim;loop2++)
	fft_proj[loop2+1] = fft_proj[loop2+1]*filter_lut[loop2/2];


      four1 ((float *) fft_proj, sinogram_x_dim, -1);

      for (loop2=0;loop2<sinogram_x_dim;loop2++)
	filtered_proj[loop*sinogram_x_dim+loop2] = fft_proj[2*loop2+1];
    }
				
  sinxdiv2 = (float) sinogram_x_dim/2.0;
  recon_offset = reconstruction;
  for (loop=0;loop<sinogram_x_dim;loop++)
    {
      y = ((float) (loop - sinxdiv2))/sinxdiv2;
					  
      for (loop2=0;loop2<sinogram_x_dim;loop2++)
	{
	  x = ((float) (loop2 - sinxdiv2))/sinxdiv2;
						
	  for (loop3=0;loop3<sinogram_y_dim;loop3++)
	    {
	      r = x*cosines[loop3] + y*sines[loop3];
	      if ((r >= -1) && (r < 1))
		*recon_offset += (float) filtered_proj[loop3*sinogram_x_dim + (int) ((r+1)*sinxdiv2)];
	    }
	  recon_offset++;
	}
    }
}

//-------------------------------------------------------------------------------------

void OptimizedFBP::destroy (void)
{
  if (fft_proj != NULL)
    free(fft_proj);
  if (sines != NULL)
    free(sines);
  if (cosines != NULL)
    free(cosines);
  if (filter_lut != NULL)
    free(filter_lut);
  if (filtered_proj != NULL)
    free(filtered_proj);
}
	    
//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//Circle Filtered Back Projection
//-------------------------------------------------------------------------------------
//---------------------------------------------------------------------------

// void CircleFBP::acknowledgements (LogFileClass *acknowledge_file)
// {
//   acknowledge_file->Message ("__________________________________________________________________");
//   acknowledge_file->Message ("CircleFBP class");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("Class for performing optimized Filtered Back Projection Reconstructions.");
//   acknowledge_file->Message ("Optimazation is done by restricting calculations to circular area around data.");
//   acknowledge_file->Message ("Origional source code developed in C by:");
//   acknowledge_file->Message ("       Steve Wang ");
//   acknowledge_file->Message ("CPP Class Developed and Maintained by:");
//   acknowledge_file->Message ("       Brian Tieman & Francesco DeCarlo");
//   acknowledge_file->Message ("       Argonne National Laboratory");
//   acknowledge_file->Message ("       tieman@aps.anl.gov");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  First version with acknowledgements");
//   acknowledge_file->Message ("8/20/2003  V1.0   BT  Ported C code to a CPP object structure");
//   acknowledge_file->Message ("8/27/2003  V1.0   BT  Cleaned up some memory usage--got rid of variable");
//   acknowledge_file->Message ("                      t which was only a placeholder for calculations");
//   acknowledge_file->Message ("                      but was sinogram_x*sizeof(float) in size.");
//   acknowledge_file->Message ("                      Got rid of lut array which wasn't used.");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("");
//   acknowledge_file->Message ("__________________________________________________________________");
// }

//-------------------------------------------------------------------------------------

void CircleFBP::init (void)
{
  int     loop;
  float   temp;

  /*  The FBP algorythm calls Sin() and Cos() which expect angles in radians  */
  for(loop=0;loop<theta_list_size;loop++)
    theta_list[loop] = (theta_list[loop]*PI)/180.0;
    
  fft_proj = (float *) malloc ((2*sinogram_x_dim+1)*sizeof(float));
  sines = (long *) malloc (sinogram_y_dim*sizeof(unsigned long));
  cosines = (long *) malloc (sinogram_y_dim*sizeof(unsigned long));
  filter_lut = (float *) malloc (sinogram_x_dim*sizeof(float));
  filtered_proj = (float*) malloc ((sinogram_y_dim*sinogram_x_dim)*sizeof(float));
	      
  for (loop=0;loop<sinogram_x_dim;loop++)
    {
      temp = (float)(loop - (float) (sinogram_x_dim/2))/((float) (sinogram_x_dim/2));
      filter_lut[loop] = 1-sqrt(fabs(temp));
    }
		
  for (loop=0;loop<sinogram_y_dim;loop++)
    {
      sines[loop] = (long) (sin(theta_list[loop])*131072);
      cosines[loop] = (long) (cos(theta_list[loop])*131072);
    }
		  
}

//-------------------------------------------------------------------------------------

void CircleFBP::reconstruct (void)
{
  //FBP with additional filtering to do only a "circle" of the slice
  int     loop,
    loop2,
    loop3,
    r,
    sinogram_x_dim_div2;
  float   pos_gate=5.0,
    neg_gate=-5.0,
    *sinogram,
    *reconstruction,
    *filtered_offset_1,
    *recon_offset_1a,
    *recon_offset_1b,
    *filtered_offset;
  int     lookup_index;
			  
  sinogram = sinogram1;
  reconstruction = reconstruction1;
			      
  sinogram_x_dim_div2 = sinogram_x_dim/2;
  r = sinogram_x_dim*sinogram_x_dim/4;
				  
  if (reconstruction == NULL)
    {
      printf("reconstruction memory not allocated \n");
      return;
    }
  else
    for (loop=0;loop<sinogram_x_dim*sinogram_x_dim;loop++)
      {
	reconstruction[loop] = 0.0;
      }
				    
  for (loop=0;loop<sinogram_y_dim;loop++)
    {
      for (loop2=0;loop2<sinogram_x_dim;loop2++)
	{
	  if ((sinogram[loop*sinogram_x_dim+loop2] > neg_gate) && (sinogram[loop*sinogram_x_dim+loop2] < pos_gate))
	    fft_proj[2*loop2+1] = sinogram[loop*sinogram_x_dim+loop2];
	  else
	    fft_proj[2*loop2+1] = 0.0;
	  fft_proj[2*loop2+2] = 0.0;
	}
					  
      four1 (fft_proj, sinogram_x_dim, 1);
      for (loop2=0;loop2<sinogram_x_dim*2;loop2++)
	fft_proj[loop2+1] = fft_proj[loop2+1]*filter_lut[loop2/2];
      four1(fft_proj, sinogram_x_dim, -1);
      for (loop2=0;loop2<sinogram_x_dim*2;loop2++)
	filtered_proj[loop*sinogram_x_dim+loop2] = fft_proj[2*loop2+1];
    }
				      
  recon_offset_1a = reconstruction;
  for (loop=-sinogram_x_dim_div2;loop<sinogram_x_dim_div2;loop++)
    {
      recon_offset_1a = &reconstruction[(loop+sinogram_x_dim_div2)*sinogram_x_dim];
      for (loop2=-sinogram_x_dim_div2;loop2<sinogram_x_dim_div2;loop2++)
	{
	  if ((loop*loop+loop2*loop2) <= r)
	    {
	      recon_offset_1b = &recon_offset_1a[loop2+sinogram_x_dim_div2];
							
	      filtered_offset = filtered_proj + sinogram_x_dim_div2;
							  
	      for (loop3=0;loop3<sinogram_y_dim;loop3+=2)
		{
		  *recon_offset_1b += filtered_offset[(loop2*cosines[loop3] + loop*sines[loop3])>>17];
		  filtered_offset += sinogram_x_dim;
								  
		  *recon_offset_1b += filtered_offset[(loop2*cosines[loop3+1] + loop*sines[loop3+1])>>17];
		  filtered_offset += sinogram_x_dim;
		}
	    }
	}
    }
					  
  /*
    int     loop,
    loop2,
    loop3,
    n,
    r,
    sinogram_x_dim2,
    sinogram_x_dimX2;
    float   temp_sum,
    pos_gate=5.0,
    neg_gate=-5.0,
    *sinogram,
    *reconstruction,
    *filtered_offset;
					    
    sinogram = sinogram1;
    reconstruction = reconstruction1;
					    
    sinogram_x_dim2 = sinogram_x_dim/2;
    sinogram_x_dimX2 = sinogram_x_dim*2;
    r = sinogram_x_dim*sinogram_x_dim/4;
					    
    if (reconstruction == NULL)
    {
    printf("reconstruction memory not allocated \n");
    return;
    }
    else
    for (loop=0;loop<sinogram_x_dim*sinogram_x_dim;loop++)
    reconstruction[loop] = 0.0;
					    
    for (loop=0;loop<sinogram_y_dim;loop++)
    {
    for (loop2=0;loop2<sinogram_x_dim;loop2++)
    {
    if ((sinogram[loop*sinogram_x_dim+loop2] > neg_gate) && (sinogram[loop*sinogram_x_dim+loop2] < pos_gate))
    fft_proj[2*loop2+1] = sinogram[loop*sinogram_x_dim+loop2];
    else
    fft_proj[2*loop2+1] = 0.0;
    fft_proj[2*loop2+2] = 0.0;
    }
					    
    four1 (fft_proj, sinogram_x_dim, 1);
    for (loop2=0;loop2<sinogram_x_dimX2;loop2++)
    fft_proj[loop2+1] = fft_proj[loop2+1]*filter_lut[loop2/2];
    four1(fft_proj, sinogram_x_dim, -1);
    for (loop2=0;loop2<sinogram_x_dimX2;loop2++)
    filtered_proj[loop*sinogram_x_dimX2+loop2] = fft_proj[loop2+1];
    }
					    
    for (loop=-sinogram_x_dim2;loop<sinogram_x_dim2;loop++)
    {
    for (loop2=-sinogram_x_dim2;loop2<sinogram_x_dim2;loop2++)
    {
    if ((loop*loop+loop2*loop2) <= r)
    {
    temp_sum = 0.0;
    filtered_offset = filtered_proj + sinogram_x_dim;
    for (loop3=0;loop3<sinogram_y_dim;loop3++)
    {
    n = 2*((loop2*cosines[loop3] + loop*sines[loop3])/GRID_PRECISION);
    temp_sum += filtered_offset[n];
    filtered_offset += sinogram_x_dimX2;
    }
					    
    reconstruction[(loop+sinogram_x_dim2)*sinogram_x_dim+(loop2+sinogram_x_dim2)] = temp_sum;
    }
    }
    }
  */
}

//-------------------------------------------------------------------------------------

void CircleFBP::destroy (void)
{
  if (fft_proj != NULL)
    free(fft_proj);
  if (sines != NULL)
    free(sines);
  if (cosines != NULL)
    free(cosines);
  if (filter_lut != NULL)
    free(filter_lut);
  if (filtered_proj != NULL)
    free(filtered_proj);
}
						      
//-------------------------------------------------------------------------------------



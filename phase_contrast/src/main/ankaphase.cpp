// phase retrieval for data exchange file hdf5

#include "tinyxml.h"

#include <iostream>   
#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>

#include "fftw3.h"
#include "nexusbox.h"

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include "phase_contrast_config.h" 

#ifdef USE_FFT_GPU

#include <cutil.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_math.h>

extern "C"
void freq_mul_wrapper( cufftComplex*, float*, int, int );

#endif // USE_FFT_GPU


using std::cout;
using std::endl;
using std::fstream;
using std::ifstream;
using std::string;
using std::ios;

#define HDF5_DATATYPE unsigned short int  // unit16 for hdf4 

typedef itk::Image< float, 2> FloatImageType;

enum{ RECON_IMAGE_FORMAT_NRRD, RECON_IMAGE_FORMAT_BIN,
      RECON_IMAGE_FORMAT_TIF,  RECON_IMAGE_FORMAT_HDF5 };

enum{ CALCULATE_PROJ_THICKNESS, CALCULATE_PHASE_MAP };

extern "C"
void Hdf5SerialGetDim(const char* filename, const char* datasetname, 
		      int * xm, int* ym, int* zm );

extern "C"
void Hdf5SerialReadY(const char* filename, const char* datasetname, int nYSliceStart, int numYSlice,  
		     HDF5_DATATYPE * data); 

extern "C"
void Hdf5SerialReadZ(const char* filename, const char* datasetname, int nYSliceStart, int numYSlice,  
		     HDF5_DATATYPE * data); 


unsigned int next_power2( unsigned int v );
string num2str( int num );

// #define DEBUG

#define PI 3.1416

int main(int argc, char**  argv){ 

  if( argc != 2 ){   

    cout << "Usage: ankaphase  params.xml" << endl;
#ifdef USE_FFT_GPU
    cout << "Note: GPU used" << endl;
#endif
    exit( 1 );
  }

  // step 1: read the xml parameter file for detailed parameters
  string str_exp_file_directory, str_exp_file_name;
  char char_exp_file_directory[256];
  char char_exp_file_name[256];
  string str_phase_image_base_name;
  int n_phase_image_index_start, n_phase_image_index_end, n_phase_image_index_inv = 1; 

  int n_recon_image_type;
  string str_recon_image_directory, str_recon_image_base_name;
  // double d_recon_threshold_lower_value, d_recon_threshold_upper_value;
  string str_hdf_dataset_name;

  double d_delta, d_beta, d_dist_z_mm, d_energy_kev, d_pixel_size;
  int n_map_calculate = 0;


  TiXmlDocument doc( argv[1] );
  bool loadOkay = doc.LoadFile();

  if ( !loadOkay ){
    cout << "Could not load test file " << argv[1] << " Error= " << doc.ErrorDesc() << " Exiting.  " << endl;
    exit( 1 );
  }

  TiXmlNode* node = 0;
  TiXmlElement* paramsElement = 0;

  node = doc.FirstChild( "Params" );
  paramsElement = node->ToElement();

  // parameters for projection image
  node = paramsElement->FirstChild("EXP_FILE_DIRECTORY_NAME");
  if( node == NULL ){
    cout << "EXP_FILE_DIRECTORY_NAME does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    strcpy( char_exp_file_directory, node->Value() );
    str_exp_file_directory = string( char_exp_file_directory );
  }

  // 
  node = paramsElement->FirstChild("EXP_FILE_NAME");
  if( node == NULL ){
    cout << "EXP_FILE_NAME does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();

    strcpy( char_exp_file_name, node->Value() );
    str_exp_file_name = string( char_exp_file_name );
  }

  // Use names from experimental instead
//   node = paramsElement->FirstChild("PHASE_IMAGE_BASE_NAME");
//   if( node == NULL ){
//     cout << "PHASE_IMAGE_BASE_NAME does not exist in XML file. Abort! " << endl;
//     exit(1);
//   }
//   else{
//     node = node->FirstChild();
//     str_phase_image_base_name = string( node->Value() );
//   }

  //
  node = paramsElement->FirstChild("PHASE_IMAGE_INDEX_START");
  if( node == NULL ){
    cout << "PHASE_IMAGE_INDEX_START does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_phase_image_index_start = atoi( node->Value() );
  }

  //
  node = paramsElement->FirstChild("PHASE_IMAGE_INDEX_END");
  if( node == NULL ){
    cout << "PHASE_IMAGE_INDEX_END does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_phase_image_index_end = atoi( node->Value() );
  }

  //
  node = paramsElement->FirstChild("PHASE_IMAGE_INDEX_INV");
  if( node == NULL ){
    cout << "PHASE_IMAGE_INDEX_INV does not exist in XML file. Use default (1)! " << endl;
    n_phase_image_index_inv = 1;
  }
  else{
    node = node->FirstChild();
    n_phase_image_index_inv = atoi( node->Value() );
  }

  //
  node = paramsElement->FirstChild("RECON_IMAGE_TYPE");
  if( node == NULL ){
    cout << "RECON_IMAGE_TYPE does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    n_recon_image_type = atoi( node->Value() );
  }

  //
  node = paramsElement->FirstChild("RECON_IMAGE_DIRECTORY_NAME");
  if( node == NULL ){
    cout << "RECON_IMAGE_DIRECTORY_NAME does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    str_recon_image_directory = string( node->Value() );
  }
  
  //
  node = paramsElement->FirstChild("RECON_IMAGE_BASE_NAME");
  if( node == NULL ){
    cout << "RECON_IMAGE_BASE_NAME does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    str_recon_image_base_name = string( node->Value() );
  }
  
  //
//   node = paramsElement->FirstChild("RECON_THRESHOLD_LOWER_VALUE");
//   if( node == NULL ){
//     cout << "RECON_THRESHOLD_LOWER_VALUE does not exist in XML file. Abort! " << endl;
//     exit(1);
//   }
//   else{
//     node = node->FirstChild();
//     d_recon_threshold_lower_value = atof( node->Value() );
//   }
  
//   //
//   node = paramsElement->FirstChild("RECON_THRESHOLD_UPPER_VALUE");
//   if( node == NULL ){
//     cout << "RECON_THRESHOLD_UPPER_VALUE does not exist in XML file. Abort! " << endl;
//     exit(1);
//   }
//   else{
//     node = node->FirstChild();
//     d_recon_threshold_upper_value = atof( node->Value() );
//   }
  

//   node = paramsElement->FirstChild("RECON_HDF_DATASET_NAME");
//   if( node == NULL ){
//     cout << "RECON_HDF_DATASET_NAME does not exist in XML file. Use default (/data)! " << endl;
//     str_hdf_dataset_name = string( "/data" );
//   }
//   else{
//     node = node->FirstChild();
//     str_hdf_dataset_name = string( node->Value() );
//   }
  
  // 
  node = paramsElement->FirstChild("DELTA");
  if( node == NULL ){
    cout << "DELTA does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    d_delta = atof( node->Value() );
  }

  node = paramsElement->FirstChild("BETA");
  if( node == NULL ){
    cout << "BETA does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    d_beta = atof( node->Value() );
  }

  node = paramsElement->FirstChild("DIST_Z");
  if( node == NULL ){
    cout << "DIST_Z does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    d_dist_z_mm = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ENERGY");
  if( node == NULL ){
    cout << "ENERGY does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    d_energy_kev = atof( node->Value() );
  }

  node = paramsElement->FirstChild("PIXEL_SIZE");
  if( node == NULL ){
    cout << "PIXEL_SIZE does not exist in XML file. Abort! " << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    d_pixel_size = atof( node->Value() );
  }

  //
  node = paramsElement->FirstChild("MAP_CAL");
  if( node == NULL ){
    cout << "MAP_CAL does not exist in XML file. Use default(thickness map)! " << endl;
    n_map_calculate = 0;
  }
  else{
    node = node->FirstChild();
    n_map_calculate = atoi( node->Value() );
  }

    
  // check the data type of the data exchange file: hdf5
  string str_posix = str_exp_file_name.substr( str_exp_file_name.length() - 4, 4 );

  if( str_posix.compare( string( "hdf5" ) ) ){
    cout << "This version support hdf5 only!" << endl;
    exit(1);
  }

  // step 2: retrieve information from the hdf5 file
  int um, num_proj, vm; 

  string str_exchange_file_name = str_exp_file_directory;
  str_exchange_file_name.append( str_exp_file_name );
  Hdf5SerialGetDim( str_exchange_file_name.c_str(), "/entry/exchange/data", 
		    &um, &num_proj, &vm ); 

  int volsize = um * vm;
  int xm = um;
  int ym = vm; 

  HDF5_DATATYPE* data_white_field = new HDF5_DATATYPE[ 2 * volsize ];  
  HDF5_DATATYPE* data_dark_field = new HDF5_DATATYPE[ volsize ];  
  HDF5_DATATYPE* data_tmp = new HDF5_DATATYPE[ volsize ];  
  if( !data_white_field || !data_dark_field || !data_tmp ){
    cout << "Error allocating memory at " << __LINE__ << endl;
    exit(1);
  }	   

  Hdf5SerialReadZ( str_exchange_file_name.c_str(), "/entry/exchange/dark_data", 0, 1, data_dark_field );  
  Hdf5SerialReadZ( str_exchange_file_name.c_str(), "/entry/exchange/white_data", 0, 2, data_white_field );  

  // step 3: perform flat-field correction and phase calculation

  double d_lambda_wavelength = 12.398424 / d_energy_kev;  // * 10^{-10}
  double d_margin = 3 * d_lambda_wavelength * d_dist_z_mm / (d_pixel_size * d_pixel_size) / 10;   // 13.10
  int n_margin = (int)(double)ceil( d_margin );

  unsigned int xm_padding = next_power2( (unsigned int ) (xm + n_margin ) );  // find the next highest power of 2
  unsigned int ym_padding = next_power2( (unsigned int ) (ym + n_margin ) );  // find the next highest power of 2

  double* data_phase_norm_padding = new double[ xm_padding * ym_padding ];
  double* data_phase_res = new double[ xm_padding * ym_padding ];
  if( !data_phase_norm_padding || !data_phase_res ){
    cout << "Error allocating memory at " << __LINE__ << " in " << __FILE__ << endl;
    exit(1);
  }	   

  int movement_x = (xm_padding - xm) / 2;
  int movement_y = (ym_padding - ym) / 2;

  fftw_complex *dft_phase = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * ym_padding * (xm_padding/2 + 1) );
  fftw_complex *dft_product = (fftw_complex *) fftw_malloc( sizeof(fftw_complex) * ym_padding * (xm_padding/2+1) );

  double* filter = new double[ ym_padding * (xm_padding / 2 + 1 ) ];
  if( !filter ){
    cout << "Error allocating memory at " << __LINE__ << " in " << __FILE__ << endl;
    exit(1);
  }	   

  // 
  double freq_x = 4 * PI * PI / (xm_padding * xm_padding * d_pixel_size * d_pixel_size );  // * 10^12
  double freq_y = 4 * PI * PI / (ym_padding * ym_padding * d_pixel_size * d_pixel_size );  // * 10^12

  // calculate filter for phase map phi(x, y)
  double beta_over_delta = d_beta / d_delta * 1e-3; 
  double lambda_z = d_lambda_wavelength * d_dist_z_mm / ( 4 * PI ); // * 10^{-13}

  // calculate filter for thickness t(x, y)
  double mu = 4 * PI * d_beta / d_lambda_wavelength * 10; 
  double z_delta_mu_minus = d_dist_z_mm * d_delta / mu; // * 10^{-9}

  double freq_sum;
  for( int ny = 0; ny < ym_padding; ny++ ){
    for( int nx = 0; nx < xm_padding/2+1; nx++ ){

      if( nx < xm_padding / 2 && ny < ym_padding / 2 ){
	freq_sum = freq_x * nx * nx ;
	freq_sum += freq_y* ny * ny ;
      }
      else if( nx < xm_padding / 2 && ny >= ym_padding / 2) {
	freq_sum = freq_x * nx * nx ;
	freq_sum += freq_y* (ym_padding - 1 - ny ) * (ym_padding - 1 - ny ) ;
      
      }
      else if( nx >= xm_padding / 2 && ny < ym_padding / 2) {
	freq_sum = freq_x * (xm_padding - 1 - nx) * (xm_padding - 1 - nx);
	freq_sum += freq_y*  ny * ny ;
      
      }
      else if( nx >= xm_padding / 2 && ny >= ym_padding / 2) {
	freq_sum = freq_x * (xm_padding - 1 - nx ) * (xm_padding - 1 - nx );
	freq_sum += freq_y* (ym_padding - 1 - ny ) * (ym_padding - 1 - ny );
      
      }

      if( n_map_calculate == CALCULATE_PHASE_MAP )
	filter[ ny * (xm_padding/2+1) + nx] = 1.0 / (beta_over_delta + lambda_z * 1e-1 * freq_sum );

      if( n_map_calculate == CALCULATE_PROJ_THICKNESS )
	filter[ ny * (xm_padding/2+1) + nx] = 1.0 / (1.0 + z_delta_mu_minus * 1e3 * freq_sum );

    }
  }

  // prepare for FFT GPU
#ifdef USE_FFT_GPU

  float* f_filter = new float[ ym_padding * xm_padding ];
  if( !f_filter ){
    cout << "Error allocating memory at " << __LINE__ << " in " << __FILE__ << endl;
    exit(1);
  }	   

  for( int ny = 0; ny < ym_padding; ny++ ){
    for( int nx = 0; nx < xm_padding; nx++ ){

      if( nx < xm_padding / 2 && ny < ym_padding / 2 ){
	freq_sum = freq_x * nx * nx ;
	freq_sum += freq_y* ny * ny ;
      }
      else if( nx < xm_padding / 2 && ny >= ym_padding / 2) {
	freq_sum = freq_x * nx * nx ;
	freq_sum += freq_y* (ym_padding - 1 - ny ) * (ym_padding - 1 - ny ) ;
      
      }
      else if( nx >= xm_padding / 2 && ny < ym_padding / 2) {
	freq_sum = freq_x * (xm_padding - 1 - nx) * (xm_padding - 1 - nx);
	freq_sum += freq_y*  ny * ny ;
      
      }
      else if( nx >= xm_padding / 2 && ny >= ym_padding / 2) {
	freq_sum = freq_x * (xm_padding - 1 - nx ) * (xm_padding - 1 - nx );
	freq_sum += freq_y* (ym_padding - 1 - ny ) * (ym_padding - 1 - ny );
      
      }

      if( n_map_calculate == CALCULATE_PHASE_MAP )
	f_filter[ ny * xm_padding + nx] = (float)(double)1.0 / (beta_over_delta + lambda_z * 1e-1 * freq_sum );

      if( n_map_calculate == CALCULATE_PROJ_THICKNESS )
	f_filter[ ny * xm_padding + nx] = (float)(double)1.0 / (1.0 + z_delta_mu_minus * 1e3 * freq_sum );

    }
  }

  float* d_filter;
  cudaMalloc( (void**)&d_filter,  sizeof( float ) * xm_padding * ym_padding );
  cudaMemcpy( d_filter, f_filter, sizeof( float ) * xm_padding * ym_padding, cudaMemcpyHostToDevice );
  
  cufftHandle   plan_ifft, plan_fft;
  cufftComplex* data_phase_complex;
  cufftComplex* d_data_phase_complex;
  cufftComplex* d_dft_phase_complex;

  data_phase_complex = new cufftComplex[ xm_padding * ym_padding ];
  cudaMalloc( (void**)&d_data_phase_complex,  sizeof( cufftComplex ) * xm_padding * ym_padding );
  cudaMalloc( (void**)&d_dft_phase_complex,   sizeof( cufftComplex ) * xm_padding * ym_padding );

  cufftResult res = cufftPlan2d( &plan_fft, xm_padding, ym_padding, CUFFT_C2C );
  if( res != CUFFT_SUCCESS )   
    printf("cufftPlan2d for plan_fft failed\n "); 

  res = cufftPlan2d( &plan_ifft, xm_padding, ym_padding, CUFFT_C2C );
  if( res != CUFFT_SUCCESS )   
    printf("cufftPlan2d for plan_ifft failed\n "); 

#endif // USE_FFT_GPU

  // perform phase retrieval
  for( int phase_index = n_phase_image_index_start; phase_index <= n_phase_image_index_end;
       phase_index += n_phase_image_index_inv ){

    Hdf5SerialReadY( str_exchange_file_name.c_str(), "/entry/exchange/data", phase_index, 1, data_tmp);

    // perform flat-field correction

    for( int ny = 0; ny < ym; ny++ ){
      for( int nx = 0; nx < xm; nx++ ){
	if( data_white_field[ ny * xm + nx ] > data_dark_field[ ny * xm + nx ] ){

	  data_phase_norm_padding[ (ny + movement_y)  * xm_padding + nx + movement_x ] = 1.0 * (data_tmp[ ny * xm + nx ] - data_dark_field[ ny * xm + nx ] ) / (data_white_field[ ny * xm + nx ] - data_dark_field[ ny * xm + nx ]);

	}
	else{
	  data_phase_norm_padding[ (ny + movement_y)  * xm_padding + nx + movement_x ] = 0.0;
	}
      }
    }

    // padding corner 
    for( int ny = 0; ny < movement_y; ny++ ){
      for( int nx = 0; nx < movement_x; nx++ ){
	data_phase_norm_padding[ ny * xm_padding + nx ] = data_phase_norm_padding[ movement_y  * xm_padding + movement_x ]; 
      }
    }

    // padding corner 
    for( int ny = 0; ny < movement_y; ny++ ){
      for( int nx = 0; nx < movement_x; nx++ ){
	data_phase_norm_padding[ ny * xm_padding + nx + xm + movement_x] = data_phase_norm_padding[ movement_y  * xm_padding + xm - 1 + movement_x ]; 
      }
    }

    // padding corner 
    for( int ny = 0; ny < movement_y; ny++ ){
      for( int nx = 0; nx < movement_x; nx++ ){
	data_phase_norm_padding[ (ym + movement_y + ny) * xm_padding + nx ] = data_phase_norm_padding[ (ym - 1 + movement_y)  * xm_padding + movement_x ]; 
      }
    }

    // padding corner 
    for( int ny = 0; ny < movement_y; ny++ ){
      for( int nx = 0; nx < movement_x; nx++ ){
	data_phase_norm_padding[ (ym + movement_y + ny) * xm_padding + xm + movement_x + nx ] = data_phase_norm_padding[ (ym - 1 + movement_y)  * xm_padding + xm - 1 + movement_x ]; 
      }
    }
    
    // padding boundary
    for( int ny = 0; ny < ym; ny++ ){
      for( int nx = 0; nx < movement_x; nx++ ){
	data_phase_norm_padding[ (movement_y + ny) * xm_padding + nx ] = data_phase_norm_padding[ (ny + movement_y)  * xm_padding + movement_x ]; 
      }
    }

    for( int ny = 0; ny < ym; ny++ ){
      for( int nx = 0; nx < movement_x; nx++ ){
	data_phase_norm_padding[ (movement_y + ny) * xm_padding + xm + movement_x + nx ] = data_phase_norm_padding[ (ny + movement_y)  * xm_padding + xm - 1 + movement_x ]; 
      }
    }
    
    for( int ny = 0; ny < movement_y; ny++ ){
      for( int nx = 0; nx < xm; nx++ ){
	data_phase_norm_padding[ ny * xm_padding + movement_x + nx ] = data_phase_norm_padding[ movement_y * xm_padding + movement_x + nx ]; 
      }
    }

    for( int ny = 0; ny < movement_y; ny++ ){
      for( int nx = 0; nx < xm; nx++ ){
	data_phase_norm_padding[ (ym + movement_y + ny) * xm_padding + movement_x + nx ] = data_phase_norm_padding[ (ym - 1 + movement_y) * xm_padding + movement_x + nx ]; 
      }
    }


    // FFT
#ifdef USE_FFT_GPU

    for( int ny = 0; ny < ym_padding; ny++ ){
      for( int nx = 0; nx < xm_padding; nx++ ){
	data_phase_complex[ ny * xm_padding + nx ].x = (float)(double)data_phase_norm_padding[ ny * xm_padding + nx ];
	data_phase_complex[ ny * xm_padding + nx ].y = 0.0f;
      }
    }

    cudaMemcpy( d_data_phase_complex, data_phase_complex, sizeof( cufftComplex ) * xm_padding * ym_padding, 
		cudaMemcpyHostToDevice );

    // forward FFT
    cufftResult res = cufftExecC2C( plan_fft, d_data_phase_complex, d_data_phase_complex, CUFFT_FORWARD );
    if( res != CUFFT_SUCCESS )
      printf("cufftExecC2C for forward FFT failed\n "); 

    // filtering
    freq_mul_wrapper( d_data_phase_complex, d_filter, xm_padding, ym_padding );

    // backward FFT
    res = cufftExecC2C( plan_ifft, d_data_phase_complex, d_data_phase_complex, CUFFT_INVERSE );
    if( res != CUFFT_SUCCESS )
      printf("cufftExecC2C for backward FFT failed\n "); 

    cudaMemcpy( data_phase_complex, d_data_phase_complex, sizeof( cufftComplex ) * xm_padding * ym_padding, 
		cudaMemcpyDeviceToHost );

    for( int ny = 0; ny < ym_padding; ny++ ){
      for( int nx = 0; nx < xm_padding; nx++ ){
	data_phase_res[ ny * xm_padding + nx ] = data_phase_complex[ ny * xm_padding + nx ].x;
      }
    }

#else // CPU

    fftw_plan planForward = fftw_plan_dft_r2c_2d(ym_padding, xm_padding,
						 data_phase_norm_padding,
						 dft_phase,
						 FFTW_ESTIMATE);

    fftw_execute(planForward);


    // inverse FFT
    for( int ny = 0; ny < ym_padding; ny++ ){
      for( int nx = 0; nx < xm_padding/2+1; nx++ ){

	dft_product[ny * (xm_padding/2+1) + nx][0] = dft_phase[ny * (xm_padding/2+1) + nx][0] * filter[ ny * (xm_padding/2+1) + nx];
	dft_product[ny * (xm_padding/2+1) + nx][1] = dft_phase[ny * (xm_padding/2+1) + nx][1] * filter[ ny * (xm_padding/2+1) + nx];
      }
    }

    fftw_plan planBackward = fftw_plan_dft_c2r_2d(ym_padding, xm_padding,
						  dft_product, 
						  data_phase_res,
						  FFTW_ESTIMATE);

    fftw_execute(planBackward);

    fftw_destroy_plan(planForward);
    fftw_destroy_plan(planBackward);

#endif // USE_FFT_GPU

    // output
    if( n_recon_image_type == RECON_IMAGE_FORMAT_NRRD){

      FloatImageType::Pointer phase_recon = FloatImageType::New();

      FloatImageType::SizeType size;
      FloatImageType::RegionType region;

      size[0] = xm;
      size[1] = ym;

      region.SetSize( size );

      phase_recon->SetRegions( region );
      phase_recon->Allocate(); 

      // Note that FFTW performs un-normalized computation
      // That is, an image of size N * N after forward and backward FFT will be scaled by N * N. 

      FloatImageType::IndexType index;
      for( int ny = 0; ny < ym; ny++ ){
	for( int nx = 0; nx < xm; nx++ ){
				
	  index[ 0 ] = nx;
	  index[ 1 ] = ny;

	  if( n_map_calculate == CALCULATE_PHASE_MAP )
	    phase_recon->SetPixel( index, (float)(double) 0.5 * log( data_phase_res[ (ny + movement_y) * xm_padding + nx + movement_x ] / xm_padding / ym_padding ) );

	  if( n_map_calculate == CALCULATE_PROJ_THICKNESS )
	    phase_recon->SetPixel( index, (float)(double) -1.0 / mu * log( data_phase_res[ (ny + movement_y) * xm_padding + nx + movement_x ] / xm_padding / ym_padding) );

	}
      }

      itk::ImageFileWriter<FloatImageType>::Pointer ImageReconWriter;
      ImageReconWriter = itk::ImageFileWriter<FloatImageType>::New();
      ImageReconWriter->SetInput( phase_recon );

      string strPosix = ".nrrd"; 

      string strNrrdOutput = str_recon_image_directory;
      strNrrdOutput.append( str_recon_image_base_name );

      char buf[256];
      sprintf(buf, "_s%d", phase_index); 

      string str_end(buf);

      int pos = strNrrdOutput.length() - strPosix.length();
      strNrrdOutput.insert( pos,  str_end );

      cout << "    Writing file " << strNrrdOutput << endl;  // test

      ImageReconWriter->SetFileName( strNrrdOutput );
      ImageReconWriter->Update();  

    }
    else if ( n_recon_image_type == RECON_IMAGE_FORMAT_BIN){

      string strPosix = string(".bin");

      string strIndex = num2str( phase_index );
      string strCurrBinName = str_recon_image_directory;
      strCurrBinName.append( str_recon_image_base_name );

      int pos = strCurrBinName.length() - strPosix.length() - strIndex.length();
      strCurrBinName.replace(pos, strIndex.length(), strIndex); 

      fstream datafile( strCurrBinName.c_str(), ios::out | ios::binary );

      if( !datafile.is_open() ){
	cout << "    Error writing file " << strCurrBinName << endl;
	continue; 
      }
      else{
	
	cout << "    Writing file " << strCurrBinName << endl;  // test
	
	float pixel_bin;
	for( int ny = 0; ny < ym; ny++ ){
	  for( int nx = 0; nx < xm; nx++ ){  

	    if( n_map_calculate == CALCULATE_PHASE_MAP )
	      pixel_bin = (float)(double)0.5 * log( data_phase_res[ (ny + movement_y) * xm_padding + nx + movement_x ] / xm_padding / ym_padding );

	    if( n_map_calculate == CALCULATE_PROJ_THICKNESS )
	      pixel_bin = -1.0 / mu * log( data_phase_res[ (ny + movement_y) * xm_padding + nx + movement_x ] / xm_padding / ym_padding);

	    datafile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );

	  }
	}

	datafile.close();
      	
      }

    }
    else if ( n_recon_image_type == RECON_IMAGE_FORMAT_TIF){

    }
    else if ( n_recon_image_type == RECON_IMAGE_FORMAT_HDF5){

    }
    else{
      cout << "The specified format is not supported yet " << endl;
      exit(1);
    }
        
  }

#ifdef FFT_GPU

  delete [] f_filter;
  delete [] data_phase_complex;

  cufftDestroy( plan_fft );
  cufftDestroy( plan_ifft );

  cudaFree ( d_data_phase_complex );
  cudaFree ( d_dft_phase_complex );
  cudaFree ( d_filter );
  cudaFree ( d_dft_product_complex );

#endif // FFT_GPU

  fftw_free( dft_phase ); 
  fftw_free( dft_product );

  delete [] data_white_field; 
  delete [] data_dark_field; 
  delete [] data_tmp; 
  delete [] data_phase_norm_padding;
  delete [] filter; 
  delete [] data_phase_res;

} // main

unsigned int next_power2( unsigned int v ){

  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}


string num2str( int num ){

  char c_iter = '0';
  string str_ind[10];
  for( int i = 0; i < 10; i++){
    str_ind[i] = string(1, c_iter);
    c_iter = c_iter + 1;
  }

  int iter = num; 
  int res;

  string strNum = "";
  while( iter / 10 != 0 ){
    res = iter % 10;
    iter = iter / 10; 
    strNum.insert( 0, str_ind[res] );
  }
  strNum.insert( 0, str_ind[iter] );

  return strNum;
}

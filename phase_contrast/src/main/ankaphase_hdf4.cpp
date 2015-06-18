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

using std::cout;
using std::endl;
using std::fstream;
using std::ifstream;
using std::string;
using std::ios;

#define HDF4_DATATYPE unsigned short int  // unit16 for hdf4 

typedef itk::Image< HDF4_DATATYPE, 2> Uint16ImageType;
typedef itk::Image< float, 2> FloatImageType;

enum{ PHASE_IMAGE_FORMAT_TIF, PHASE_IMAGE_FORMAT_HDF };
enum{ RECON_IMAGE_FORMAT_NRRD, RECON_IMAGE_FORMAT_BIN,
      RECON_IMAGE_FORMAT_TIF, RECON_IMAGE_FORMAT_HDF4, 
      RECON_IMAGE_FORMAT_HDF5 };

enum{ CALCULATE_PROJ_THICKNESS, CALCULATE_PHASE_MAP };

void get_dim_hdf( char*, char*, char*, int* xm, int* ym );
void read_hdf( char*, char*, char*, int xm, int ym, HDF4_DATATYPE* data );
void get_dim_tif( char*, char*, int* xm, int* ym );
void read_tif( char*, char*, int xm, int ym, HDF4_DATATYPE* data );

unsigned int next_power2( unsigned int v );
string num2str( int num );

// #define DEBUG

#define PI 3.1416

int main(int argc, char**  argv){ 

  if( argc != 2 ){   

    cout << "Usage: ankaphase  params.xml" << endl;
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

    
  // step 2: retrieve information from the experimental file
  NexusBoxClass exp_file;
  
  exp_file.SetReadScheme (ENTIRE_CONTENTS); 
  exp_file.ReadAll( char_exp_file_directory, 
		    char_exp_file_name ); 

  
  int rank, dims[2], type; 

  // get the name for the image data set
  char index_data_group[256];

  strcpy (index_data_group, ";experiment;reconstruction;cluster_config;data_group_index"); 
  if (!exp_file.IndexExists(index_data_group)) { 
    cout << "Index " << index_data_group << " does not exist. Exit." << endl; 
    exit(1);
  } 

  int dims_index_data_group; 
  exp_file.GetDatumInfo (index_data_group, &rank, &dims_index_data_group, &type); 

  // Note that the entry for index_data_group is one-dimension only (rank = 1) . 
  if( rank != 1 ){
    cout << "The entry " << index_data_group << " should be one-dimensional "  << endl; 
    exit(1); 
  }

  char index_data[256]; 
  exp_file.GetDatum (index_data_group, index_data); 

  index_data[ dims_index_data_group ] = '\0';   // This step is very important. 

  cout << "The index for the image data file is " << index_data << endl; 


  //White Field File Names 
  char index_white_name[256];
  strcpy (index_white_name, ";experiment;acquisition;white_field;names"); 
  if (!exp_file.IndexExists(index_white_name)) { 
    cout << "Index " << index_white_name << " does not exist." << endl;
    exit(1);
  } 
  exp_file.GetDatumInfo (index_white_name, &rank, dims, &type); 

  char* white_file_list = (char *) new char[ dims[0]*dims[1] ]; 
  exp_file.GetDatum (index_white_name, white_file_list); 
 
  char file_name_white_0[256];  
  char file_name_white_180[256];  

  int n_white_field = dims[ 0 ];

  if( n_white_field == 0 || n_white_field > 2 ){
    cout << "The current program supports one or two white field images only! Exiting..." << endl;
    exit(1);
  }

  strncpy(file_name_white_0, white_file_list, dims[1]); 
  file_name_white_0[ dims[1] ] = '\0'; 

  // check the data type of the dark field image: hdf or tif
  string str_file_name_white_0 = string( file_name_white_0 );
  string str_posix = str_file_name_white_0.substr( str_file_name_white_0.length() - 3, 3 );
  
  int n_phase_image_format;
  if( !str_posix.compare( string( "tif" ) ) ){
    n_phase_image_format = PHASE_IMAGE_FORMAT_TIF;
  }
  else if( !str_posix.compare( string( "hdf" ) ) ){
    n_phase_image_format = PHASE_IMAGE_FORMAT_HDF;
  }
  else{
    cout << "Current version support hdf/tif only!" << endl;
    exit(1);
  }

#ifdef DEBUG
  n_phase_image_format = PHASE_IMAGE_FORMAT_TIF;  // test
#endif // DEBUG

  //  
  int xm, ym;

  char char_phase_directory[ 256 ];
  strcpy( char_phase_directory, char_exp_file_directory );
  strcat( char_phase_directory, "raw/" );
  // string str_phase_directory = str_exp_file_directory;
  // str_phase_directory.append( "raw/");

  if( n_phase_image_format == PHASE_IMAGE_FORMAT_TIF ){
    // get_dim_tif( str_phase_directory, str_file_name_white_0, &xm, &ym );

#ifdef DEBUG
    get_dim_tif( char_phase_directory, "lichen_10x_50mm_1_01503.tif", &xm, &ym );
#else
    get_dim_tif( char_phase_directory, file_name_white_0, &xm, &ym );
#endif

  }
  else if( n_phase_image_format == PHASE_IMAGE_FORMAT_HDF ){
    // get_dim_hdf( str_phase_directory, str_file_name_white_0, index_data, &xm, &ym ); 
    get_dim_hdf( char_phase_directory, file_name_white_0, index_data, &xm, &ym ); 
  }

  int volsize = xm * ym;

  HDF4_DATATYPE* data_white_field = new HDF4_DATATYPE[ volsize ];  
  HDF4_DATATYPE* data_dark_field = new HDF4_DATATYPE[ volsize ];  
  HDF4_DATATYPE* data_tmp = new HDF4_DATATYPE[ volsize ];  
  if( !data_white_field || !data_dark_field || !data_tmp ){
    cout << "Error allocating memory at " << __LINE__ << endl;
    exit(1);
  }	   

  if( n_phase_image_format == PHASE_IMAGE_FORMAT_TIF ){
    // read_tif( str_phase_directory, str_file_name_white_0, xm, ym, data_white_field );
#ifdef DEBUG
    read_tif( char_phase_directory, "lichen_10x_50mm_1_01503.tif", xm, ym, data_white_field );
#else
    read_tif( char_phase_directory, file_name_white_0, xm, ym, data_white_field );
#endif

  }
  else if( n_phase_image_format == PHASE_IMAGE_FORMAT_HDF ){
    // read_hdf( str_phase_directory, str_file_name_white_0, index_data, xm, ym, data_white_field ); 
    read_hdf( char_phase_directory, file_name_white_0, index_data, xm, ym, data_white_field ); 
  }
 

  // 
  if( n_white_field == 2 ){

    strncpy(file_name_white_180, &white_file_list[ (dims[0]-1) * dims[1] ], dims[1]);
    file_name_white_180[ dims[1] ] = '\0'; 

    // string str_file_name_white_180 = string( file_name_white_180 );

    if( n_phase_image_format == PHASE_IMAGE_FORMAT_TIF ){
      // read_tif( str_phase_directory, str_file_name_white_180, xm, ym, data_tmp );
#ifdef DEBUG
      read_tif( char_phase_directory, "lichen_10x_50mm_1_01503.tif", xm, ym, data_tmp );
#else
      read_tif( char_phase_directory, file_name_white_180, xm, ym, data_tmp );
#endif
    }
    else if( n_phase_image_format == PHASE_IMAGE_FORMAT_HDF ){
      // read_hdf( str_phase_directory, str_file_name_white_180,  index_data, xm, ym, data_tmp ); 
      read_hdf( char_phase_directory, file_name_white_180,  index_data, xm, ym, data_tmp ); 
    }

    for( int ny = 0; ny < ym; ny++ ){
      for( int nx = 0; nx < xm; nx++ ){
 
	data_white_field[ ny * xm + nx ] = (data_white_field[ ny * xm + nx ] + data_tmp[ ny * xm + nx ]) / 2;
      }
    }
  }

  delete [] white_file_list; 

  cout << "The white field file name is " << file_name_white_0 << endl; 
  cout << "The white field file name is " << file_name_white_180 << endl; 

  //Dark Field File Names 
  char index_dark_name[256];
  strcpy (index_dark_name, ";experiment;acquisition;black_field;names"); 
  if (!exp_file.IndexExists(index_dark_name)) {
    cout << "Index " << index_dark_name << " does not exist." << endl;
    exit(1);
  } 
  exp_file.GetDatumInfo (index_dark_name, &rank, dims, &type); 
  char* dark_file_list = (char *) new char[ dims[0]*dims[1] ]; 
  exp_file.GetDatum (index_dark_name, dark_file_list); 
 
  if( dims[0] != 1 ){
    cout << "The current version supports one dark field image only! Exiting..." << endl;
    exit(1);
  }

  char file_name_dark[256];
  strncpy(file_name_dark, dark_file_list, dims[1]); 
  file_name_dark[dims[1]] = '\0'; 

  string str_file_name_dark = string( file_name_dark );
  if( n_phase_image_format == PHASE_IMAGE_FORMAT_TIF ){
    // read_tif( str_phase_directory, str_file_name_dark, xm, ym, data_dark_field );
#ifdef DEBUG
    read_tif( char_phase_directory, "lichen_10x_50mm_1_01504.tif", xm, ym, data_dark_field );
#else
    read_tif( char_phase_directory, file_name_dark, xm, ym, data_dark_field );
#endif
  }
  else if( n_phase_image_format == PHASE_IMAGE_FORMAT_HDF ){
    // read_hdf( str_phase_directory, str_file_name_dark,  index_data, xm, ym, data_dark_field ); 
    read_hdf( char_phase_directory, file_name_dark,  index_data, xm, ym, data_dark_field ); 
  }

  cout << "The dark field file name is " << file_name_dark << endl; 

  //Projection File Names 
  char index_proj_name[256];
  strcpy (index_proj_name, ";experiment;acquisition;projections;names"); 
  if (!exp_file.IndexExists(index_proj_name)) {
    cout << "Index " << index_proj_name << " does not exist." << endl;
    exit(1);
  } 
  exp_file.GetDatumInfo (index_proj_name, &rank, dims, &type); 
 
  char* proj_file_list = (char *) new char[ dims[0]*dims[1] ]; 
  if (proj_file_list == NULL) {
    cout << "Could not allocat memory for file_list." << endl;
    exit(1);
  } 
  exp_file.GetDatum (index_proj_name, proj_file_list); 

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

#ifdef DEBUG

    FloatImageType::Pointer img_filter = FloatImageType::New();

    FloatImageType::SizeType size;
    FloatImageType::RegionType region;

    size[0] = xm_padding/2 + 1;
    size[1] = ym_padding;

    region.SetSize( size );

    img_filter->SetRegions( region );
    img_filter->Allocate(); 

    FloatImageType::IndexType index_tmp;
    for( int ny = 0; ny < ym_padding; ny++ ){
      for( int nx = 0; nx < xm_padding / 2 + 1; nx++ ){
				
	index_tmp[ 0 ] = nx;
	index_tmp[ 1 ] = ny;
	  
	img_filter->SetPixel( index_tmp, filter[ ny * (xm_padding/2+1) + nx] );

      }
    }

    itk::ImageFileWriter<FloatImageType>::Pointer ImageReconWriter;
    ImageReconWriter = itk::ImageFileWriter<FloatImageType>::New();
    ImageReconWriter->SetInput( img_filter );
    ImageReconWriter->SetFileName( "phase_filter.nrrd" );
    ImageReconWriter->Update();  

#endif // DEBUG


  for( int phase_index = n_phase_image_index_start; phase_index <= n_phase_image_index_end;
       phase_index += n_phase_image_index_inv ){

    // 
    char phase_file_name[256];  
    strncpy(phase_file_name, &proj_file_list[ (phase_index - n_phase_image_index_start) * dims[1] ], dims[1]);
    phase_file_name[dims[1]] = '\0'; 

    // string str_phase_file_name = string( phase_file_name );
    if( n_phase_image_format == PHASE_IMAGE_FORMAT_TIF ){
      // read_tif( str_phase_directory, str_phase_file_name, xm, ym, data_tmp );

#ifdef DEBUG
      char phase_file_name[ 256 ];
      sprintf( phase_file_name, "lichen_10x_50mm_1_0000%d.tif", phase_index );
      read_tif( char_phase_directory, phase_file_name, xm, ym, data_tmp );
#else
      read_tif( char_phase_directory, phase_file_name, xm, ym, data_tmp );
#endif

    }
    else if( n_phase_image_format == PHASE_IMAGE_FORMAT_HDF ){
      // read_hdf( str_phase_directory, str_phase_file_name, index_data, xm, ym, data_tmp ); 
      read_hdf( char_phase_directory, phase_file_name, index_data, xm, ym, data_tmp ); 
    }

    // perform flat-field correction

    for( int ny = 0; ny < ym; ny++ ){
      for( int nx = 0; nx < xm; nx++ ){
	if( data_white_field[ ny * xm + nx ] > data_dark_field[ ny * xm + nx ] ){

	  data_phase_norm_padding[ (ny + movement_y)  * xm_padding + nx + movement_x ] = 1.0 * (data_tmp[ ny * xm + nx ] - data_dark_field[ ny * xm + nx ] ) / (data_white_field[ ny * xm + nx ] - data_dark_field[ ny * xm + nx ]);

	  // this processing seems not necessary
// 	  if( data_phase_norm_padding[ (ny + movement_y)  * xm_padding + nx + movement_x ] > 1.0 )
// 	    data_phase_norm_padding[ (ny + movement_y)  * xm_padding + nx + movement_x ] = 1.0;
// 	  if( data_phase_norm_padding[ (ny + movement_y)  * xm_padding + nx + movement_x ] < 0.0 )
// 	    data_phase_norm_padding[ (ny + movement_y)  * xm_padding + nx + movement_x ] = 0.0;

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

// #ifdef DEBUG

//     FloatImageType::Pointer phase_recon = FloatImageType::New();

//     FloatImageType::SizeType size;
//     FloatImageType::RegionType region;

//     size[0] = xm_padding;
//     size[1] = ym_padding;

//     region.SetSize( size );

//     phase_recon->SetRegions( region );
//     phase_recon->Allocate(); 

//     FloatImageType::IndexType index;
//     for( int ny = 0; ny < ym_padding; ny++ ){
//       for( int nx = 0; nx < xm_padding; nx++ ){
				
// 	index[ 0 ] = nx;
// 	index[ 1 ] = ny;
	  
// 	phase_recon->SetPixel( index, data_phase_norm_padding[ ny * xm_padding + nx ] );

//       }
//     }

//     itk::ImageFileWriter<FloatImageType>::Pointer ImageReconWriter;
//     ImageReconWriter = itk::ImageFileWriter<FloatImageType>::New();
//     ImageReconWriter->SetInput( phase_recon );
//     ImageReconWriter->SetFileName( "phase_padding.nrrd" );
//     ImageReconWriter->Update();  

// #endif // DEBUG

    // FFT
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
    else if ( n_recon_image_type == RECON_IMAGE_FORMAT_HDF4){

    }
    else if ( n_recon_image_type == RECON_IMAGE_FORMAT_HDF5){

    }
    else{
      cout << "The specified format is not supported yet " << endl;
      exit(1);
    }
        


  }

  fftw_free( dft_phase ); 
  fftw_free( dft_product );

  delete [] proj_file_list;  
  delete [] data_white_field; 
  delete [] data_dark_field; 
  delete [] data_tmp; 
  delete [] data_phase_norm_padding;
  delete [] filter; 
  delete [] data_phase_res;

} // main


void get_dim_hdf( char* dir, char* file, char* index_data, int* xm, int* ym ){

  NexusBoxClass nexus_file;
  nexus_file.SetReadScheme (ENTIRE_CONTENTS); 
  nexus_file.ReadAll( dir, file );  
  if (!nexus_file.IndexExists(index_data)) {
    cout << "Error: index " << index_data << " does not exist." << endl; 
    exit(1);
  } 

  int rank, dims[2], type;

  nexus_file.GetDatumInfo (index_data, &rank, dims, &type); 

  *ym = dims[0];
  *xm = dims[1];

}

void read_hdf( char* dir, char* file, char* index_data, int xm, int ym, HDF4_DATATYPE* data ){

  NexusBoxClass nexus_file;
  nexus_file.SetReadScheme (ENTIRE_CONTENTS); 
  nexus_file.ReadAll( dir, file );  
  if (!nexus_file.IndexExists(index_data)) {
    cout << "Error: index " << index_data << " does not exist." << endl; 
    exit(1);
  } 

  int rank, dims[2], type;
  nexus_file.GetDatumInfo (index_data, &rank, dims, &type); 

  int nHeight = dims[0];
  int nWidth = dims[1];

  if( nHeight & (nHeight - 1 ) != 0  || nWidth & (nWidth - 1 ) != 0 ){

    cout << "This version need the projection dimensions to be power of 2" << endl; // power
    exit(1);
  }

  nexus_file.GetDatum (index_data, data); 

}

void get_dim_tif( char* dir, char* file, int* xm, int* ym ){

  char file_path_name[ 256 ];
  strcpy( file_path_name, dir );
  strcat( file_path_name, file );
  // string str_file_name = str_dir;
  // str_file_name.append( str_file );

  itk::ImageFileReader<Uint16ImageType>::Pointer ImageReader;
  ImageReader = itk::ImageFileReader<Uint16ImageType>::New();
  // ImageReader->SetFileName( str_file_name.c_str() );
  ImageReader->SetFileName( file_path_name );
  ImageReader->Update();

  Uint16ImageType::Pointer image = ImageReader->GetOutput();

  Uint16ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize(); 
  *xm = size[ 0 ];
  *ym = size[ 1 ];

}


void read_tif( char* dir, char* file, int xm, int ym, HDF4_DATATYPE* data ){

  char file_path_name[ 256 ];
  strcpy( file_path_name, dir );
  strcat( file_path_name, file );

  // string str_file_name = str_dir;
  // str_file_name.append( str_file );

  itk::ImageFileReader<Uint16ImageType>::Pointer ImageReader;
  ImageReader = itk::ImageFileReader<Uint16ImageType>::New();
  // ImageReader->SetFileName( str_file_name.c_str() );
  ImageReader->SetFileName( file_path_name );
  ImageReader->Update();

  Uint16ImageType::Pointer image = ImageReader->GetOutput();

  Uint16ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize(); 
  if( xm != size[ 0 ] || ym != size[ 1 ] ){
    cout << "The image size does not match!" << endl;
    exit(1);
  }

  Uint16ImageType::IndexType index;

  for( unsigned int ny = 0; ny < ym; ny++ ){
    for( unsigned int nx = 0; nx < xm; nx++ ){

      index[ 0 ] = nx;
      index[ 1 ] = ny;

      data[ ny * xm + nx ] = image->GetPixel( index );
    }
  }

}

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

// Image Concatecation using cross correlation

#include <iostream>   
#include <fstream>
#include <cstdlib>
#include <string>

#include "teem/nrrd.h"
#include "fftw3.h"
#include "nexusbox.h"

#define VERBOSE

using std::cout;
using std::endl;
using std::fstream;
using std::ifstream;
using std::string;
using std::ios;

#define HDF4_DATATYPE unsigned short int  // unit16 for hdf4 

int main(int argc, char**  argv){ 

  if( argc < 3 || argc > 5 ){   
    cout << "Usage: Tomo2DataExchange  /data/tom2/Sam06/  Sam06_exp.hdf [ 0(hdf5)/1(nrrd) outfilename ]" << endl;
    cout << "Note: hdf5 is used as default" << endl;
    exit( 1 );
  }

  char strDataExchangeFile[256];

  int outputType = 0;
  if( argc == 3 ){
    strcpy(strDataExchangeFile, "Tomo2DataExchange.hdf5");
  }
  else if( argc == 4 ){
    outputType = atoi( argv[ 3 ] ); 

    if( outputType == 0 ){
      strcpy(strDataExchangeFile, "Tomo2DataExchange.hdf5");
    }
    if( outputType == 1 ){
      strcpy(strDataExchangeFile, "Tomo2DataExchange.nrrd");
      
    }
  }
  else if( argc == 5 ){
    outputType = atoi( argv[ 3 ] ); 
    strcat(strDataExchangeFile, argv[4] );
  }

  //Step1: read the experimental file using NexusBoxClass
  NexusBoxClass exp_file;
  
  exp_file.SetReadScheme (ENTIRE_CONTENTS); 
  exp_file.ReadAll( argv[1], argv[2] ); 

  // 
  char index[256];
  int rank, dims[2], type; 

  // get the name for the image data set
  char index_data_group[256];

  strcpy (index_data_group, ";experiment;reconstruction;cluster_config;data_group_index"); 
  if (!exp_file.IndexExists(index_data_group)) { 
    cout << "Index " << index_data_group << " does not exist. Use default." << endl; 
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

#ifdef VERBOSE
  cout << "The index for the projection data file is " << index_data << endl; 
#endif

  //Projection File Names 
  char index_proj_name[256];
  strcpy (index_proj_name, ";experiment;acquisition;projections;names"); 
  if (!exp_file.IndexExists(index_proj_name)) {

#ifdef VERBOSE
    cout << "Index " << index_proj_name << " does not exist." << endl;
#endif 

    exit(1);
  } 
  exp_file.GetDatumInfo (index_proj_name, &rank, dims, &type); 

  int num_proj = dims[ 0 ];

  char* proj_file_list = (char *) new char[ dims[0]*dims[1] ]; 
  if (proj_file_list == NULL) {
    cout << "Could not allocat memory for file_list." << endl;
    exit(1);
  } 
  exp_file.GetDatum (index_proj_name, proj_file_list); 

  // get detector size (projection data size)
  char detector_size_x_name[256];
  strcpy (detector_size_x_name, ";experiment;setup;detector;size_x"); 
  if (!exp_file.IndexExists(detector_size_x_name)) {
    cout << "Index " << detector_size_x_name << " does not exist." << endl;
    exit(1);
  } 

  long int num_ray; 
  exp_file.GetDatum (detector_size_x_name, &num_ray); 

#ifdef VERBOSE
  cout << detector_size_x_name << " is " << num_ray << endl;
#endif 

  // 
  char detector_size_y_name[256];
  strcpy (detector_size_y_name, ";experiment;setup;detector;size_y"); 
  if (!exp_file.IndexExists(detector_size_y_name)) {
    cout << "Index " << detector_size_y_name << " does not exist." << endl;
    exit(1);
  } 

  long int num_elevation; 
  exp_file.GetDatum (detector_size_y_name, &num_elevation); 

#ifdef VERBOSE
  cout << detector_size_y_name << " is " << num_elevation << endl;
#endif 

  // retrieve projection data
  
  HDF4_DATATYPE* proj_data_total = new HDF4_DATATYPE[ num_elevation * num_proj * num_ray ];
  HDF4_DATATYPE* proj_data = new HDF4_DATATYPE[ num_elevation * num_ray ];
  if( !proj_data_total || !proj_data ){
    cout << "Error allocating memory for proj_data_total and proj_data!" << endl;
  }

  // 
  exp_file.CreateGroup(";", "entry", "TreeTop" );
  exp_file.CreateGroup(";entry", "exchange", "exchange" );

  int dims_field[2]; 
  dims_field[0] = 1;
  dims_field[1] = 1; 

  exp_file.CreateField (";entry;exchange", "data_x", 1, dims_field, NX_INT32, &num_ray);
  exp_file.CreateField (";entry;exchange", "data_y", 1, dims_field, NX_INT32, &num_proj);
  exp_file.CreateField (";entry;exchange", "data_z", 1, dims_field, NX_INT32, &num_elevation);

  // 
  char strRawDataPath[256];
  strcpy(strRawDataPath, argv[1]);
  strcat(strRawDataPath, "raw/");

  for( int i = 0; i < num_proj; i++ ){

    if( i > 0 && i % 20 == 0 ){
      cout << i << " projection files read! " << endl;
    }

    char proj_file_name[256];     
    strncpy(proj_file_name, &proj_file_list[ i * dims[1] ], dims[1]);
    proj_file_name[dims[1]] = '\0'; 

    // cout << proj_file_name << endl; // test
    NexusBoxClass nexus_proj_file;
  
    nexus_proj_file.SetReadScheme (ENTIRE_CONTENTS); 
    nexus_proj_file.ReadAll( strRawDataPath, proj_file_name ); 

    if (!nexus_proj_file.IndexExists(index_data)) {
      cout << "Error: index " << index_data << " does not exist in " << proj_file_name << endl; 
      exit(1);
    } 
    int rank_proj, dims_proj[2], type_proj; 
    nexus_proj_file.GetDatumInfo (index_data, &rank_proj, dims_proj, &type_proj); 

    if( dims_proj[0] != num_elevation || dims_proj[1] != num_ray ){
      cout << "Error: the dimension of dataset " << index_data << " does not match in " << proj_file_name << endl;
      exit(1);
    }

    nexus_proj_file.GetDatum (index_data, proj_data); 

    for( int index_elevation = 0; index_elevation < num_elevation; index_elevation++ ){
      for( int index_ray = 0; index_ray < num_ray; index_ray++ ){
  	proj_data_total[ (index_elevation * num_proj + i) * num_ray + index_ray ] = proj_data[ index_elevation * num_ray + index_ray ];
      }
    }
  }

  cout << "Finish reading projection files!" << endl;

  int dims_data[3];
  dims_data[0] = num_elevation;
  dims_data[1] = num_proj;
  dims_data[2] = num_ray;
  exp_file.CreateField (";entry;exchange", "data", 3, dims_data, NX_UINT16, proj_data_total);

  cout << "Finish writing projection files!" << endl;

  delete [] proj_data_total; 
  delete [] proj_file_list;

  //White Field File Names 
  char index_white_name[256];
  strcpy (index_white_name, ";experiment;acquisition;white_field;names"); 
  if (!exp_file.IndexExists(index_white_name)) { 

#ifdef VERBOSE
    cout << "Index " << index_white_name << " does not exist." << endl;
#endif

    exit(1);
  } 
  exp_file.GetDatumInfo (index_white_name, &rank, dims, &type); 

  char* white_file_list = (char *) new char[ dims[0]*dims[1] ]; 
  exp_file.GetDatum (index_white_name, white_file_list); 

  if( dims[0] <= 1 ){
    cout << "The current program needs at least two white field images! Exiting..." << endl;
    exit(1);
  }

  // white field data
  HDF4_DATATYPE* white_data_total = new HDF4_DATATYPE[ dims[0] * num_elevation * num_ray ];
  if( !white_data_total ){
    cout << "Error allocating memory for white_data_total!" << endl;
  }
  
  for( int i = 0; i < dims[0]; i++ ){

    char white_file_name[256];     
    strncpy( white_file_name, &white_file_list[ i * dims[1] ], dims[1] );
    white_file_name[dims[1]] = '\0'; 

    NexusBoxClass nexus_white_file;
  
    nexus_white_file.SetReadScheme (ENTIRE_CONTENTS); 
    nexus_white_file.ReadAll( strRawDataPath, white_file_name ); 

    if (!nexus_white_file.IndexExists(index_data)) {
      cout << "Error: index " << index_data << " does not exist in " << white_file_name << endl; 
      exit(1);
    } 
    int rank_white, dims_white[2], type_white; 
    nexus_white_file.GetDatumInfo (index_data, &rank_white, dims_white, &type_white); 

    if( dims_white[0] != num_elevation || dims_white[1] != num_ray ){
      cout << "Error: the dimension of dataset " << index_data << " does not match in " << white_file_name << endl;
      exit(1);
    }

    nexus_white_file.GetDatum (index_data, proj_data); 

    for( int index_elevation = 0; index_elevation < num_elevation; index_elevation++ ){
      for( int index_ray = 0; index_ray < num_ray; index_ray++ ){
  	white_data_total[ (i * num_elevation + index_elevation ) * num_ray + index_ray ] = proj_data[ index_elevation * num_ray + index_ray ];
      }
    }
  }

  int dims_white[3];
  dims_white[0] = dims[0];
  dims_white[1] = num_elevation;
  dims_white[2] = num_ray;
  exp_file.CreateField (";entry;exchange", "white_data", 3, dims_white, NX_UINT16, white_data_total);

  delete [] white_data_total; 
  delete [] white_file_list;

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
    cout << "The current version supports one dark field image only! Double check!" << endl;
  }

  char dark_file_name[256];
  strncpy(dark_file_name, dark_file_list, dims[1]); 
  dark_file_name[dims[1]] = '\0'; 

#ifdef VERBOSE
  cout << "The dark field file name is " << dark_file_name << endl; 
#endif

  NexusBoxClass nexus_dark_file;
  
  nexus_dark_file.SetReadScheme (ENTIRE_CONTENTS); 
  nexus_dark_file.ReadAll( strRawDataPath, dark_file_name ); 

  if (!nexus_dark_file.IndexExists(index_data)) {
    cout << "Error: index " << index_data << " does not exist in " << dark_file_name << endl; 
    exit(1);
  } 
  int rank_dark, dims_dark[2], type_dark; 
  nexus_dark_file.GetDatumInfo (index_data, &rank_dark, dims_dark, &type_dark); 

  if( dims_dark[0] != num_elevation || dims_dark[1] != num_ray ){
    cout << "Error: the dimension of dataset " << index_data << " does not match in " << dark_file_name << endl;
    exit(1);
  }

  nexus_dark_file.GetDatum (index_data, proj_data); 

  int dims_dark2[3];
  dims_dark2[0] = 1;
  dims_dark2[1] = dims_dark[0];
  dims_dark2[2] = dims_dark[1];
  exp_file.CreateField (";entry;exchange", "dark_data", 3, dims_dark2, NX_UINT16, proj_data);

  // 
  exp_file.WriteAll( argv[1], strDataExchangeFile );

  delete [] dark_file_list;
  delete [] proj_data;
}

// This is a simulation program for parallel beam CT reconstruction using SART
// Yongsheng Pan
// 1/28/2011
// All rights reserved


// Definition of the imaging geometry

// Source location: (S * cos(theta), S * sin(theta), 0 )
//                   theta \in [0, PI]

// Detector size (um, vm).
// Object size: (xm, ym, zm)

// These parameters are stored in the parallel_params.txt data file

#include "tinyxml.h"         // Put it in the front to avoid compiling errors 

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>

#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

// Note: these header files should locate AFTER ITK header files
//       and before CUDA header files

#include "nr3.h"
#include "ran.h"
#include "gamma.h"
#include "deviates.h"

// Note: CUDA header files should locate AFTER ITK header files

#include <cutil.h>
#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_math.h>

#include <multithreading.h>

#include <mpi.h>

using std::cout;
using std::endl;
using std::string;

#include "tomo_recon.h"

#include "gridrec.h"               
#include "filteredbackprojection.h"

typedef struct{ 
  // device ID
  int deviceID;
  int deviceShareMemSizePerBlock;

  // Host-side input (projection) data
  float* h_input_data;

  // Host-side output (object) data 
  float* h_output_data;

  // parameters
  int num_width;
  int num_height;
  int num_depth;
  int num_ray;
  int num_elevation;
  int num_proj;
  int stop_type;
  int iter_max;
  float proj_thres;
  float diff_thres_percent;
  float diff_thres_value;
  float lambda;
  float spacing;
  float voxel_size;
  float proj_pixel_size;
  float SOD;
  float inv_rot;
  float xoffset;
  float yoffset;
  float zoffset;
  float start_rot;
  float end_rot;

  // 
  GridRec* gridrec_algorithm;
  float* sinogram;
  float* reconstruction;

}ThreadGPUPlan;

typedef struct{
    int   SOD;
    int   num_proj;
    int   um;
    int   vm;
    int   xm;
    int   ym;
    int   zm;
    int   recon_algorithm;          // 0: CPU Gridrec   1: GPU SART
    int   stop_type;
    int   iter_max;
    float          proj_thres;
    float          diff_thres_percent;
    float          diff_thres_value;
    float          proj_pixel_size;  // detector pixel size
    float          voxel_size;       // voxel size in the object
    float          spacing;
    float          inv_rot; 
    float          lambda;
    float          xoffset;
    float          yoffset;
    float          zoffset;
    float          start_rot;
    float          end_rot;

    int            nSliceStart;
    int            nSliceEnd;
    int            n_proj_image_type;
    string         str_proj_image_directory;
    string         str_proj_image_base_name;

    int            n_flat_field_normalization;
    int            n_proj_offset_y; 

    string         str_white_field_directory_name;
    string         str_white_field_base_name;
    int            n_white_field_index_start;
    int            n_white_field_index_end;
    string         str_dark_field_directory_name;
    string         str_dark_field_base_name;
    int            n_dark_field_index_start;
    int            n_dark_field_index_end;

    string         str_proj_hdf5_dataset_name;
    int            n_proj_hdf5_dataset_normalization;
    int            n_proj_seq_index_start, n_proj_seq_index_end;
    int            n_recon_image_type; 
    string         str_recon_image_directory;
    string         str_recon_image_base_name;
    float          f_recon_threshold_lower_value, f_recon_threshold_upper_value;
    string         str_recon_hdf5_dataset_name;
    int            nReconRadiusCrop;
    int            nGPUDeviceID;
} ParallelParams;

// 
typedef itk::Image<float, 3> Float3DImageType;
typedef itk::Image<unsigned char, 2> UChar2DImageType;

extern "C"
void fbp_wrapper(float* , float* , cudaArray*, cufftComplex *, 
		 float* ,
		 int, int, float, float);

extern "C"
void proj_cal_wrapper( cudaArray*, float* , float* , int , int , int , 
		       int, int , int , float , float , float , float , float,
		       float , float , float , float , float);

extern "C" 
void proj_cal_wrapper_sharemem( float* , float* , float* , int , int , int , 
				int , int , int , float , float , float , 
				float , float , unsigned int );

extern "C"
void proj_gen_wrapper( cudaArray*, float*, int, int, int,
		       int, int, int, float, float, float, float, float);
extern "C"
void backproj_cal_wrapper( cudaArray* , float* , float* , float* , int , int , int ,
			   int , int , int , float , float , float , float , float,
			   float , float , float , float , float);

extern "C"
void backproj_cal_wrapper_sharemem( float* , float* ,  float*, float* , int, int, int,  
				    int, int, int, float , float, float, float, float,
				    unsigned int);

extern "C" 
void tv3d_cal_wrapper( float *, float* , float* , int , int , int, int , float, float , float );

static CUT_THREADPROC SART3DThread(ThreadGPUPlan* );
static CUT_THREADPROC GridRecThread(ThreadGPUPlan* );  
static CUT_THREADPROC FBPThread(ThreadGPUPlan* );  

extern "C"
float reduction_wrapper( float*, float*, int, int, int );

extern "C"
float reduction2_wrapper( float*, float*, float*, int, int, int );

extern "C"
int nextPow2( int );

extern "C"
void Hdf5SerialReadZ(const char* , const char* , int , int ,  HDF5_DATATYPE *);

extern "C"
void Hdf5SerialReadZ_f(const char* , const char* , int , int ,  float *);

extern "C"
void Hdf5SerialReadY(char* , char* , int , int ,  HDF5_DATATYPE *);

extern "C"
void Hdf5SerialReconDataSet(const char* , const char* , int , int , int );

extern "C"
void Hdf5SerialWriteZ( const char* , const char* , int , int , int , int , int , float * );

extern "C"
void Hdf5ParallelReadZ(const char* , const char* , int , int, int,  HDF5_DATATYPE * );

extern "C"
void Hdf5ParallelReadZ_f(const char* , const char* , int , int, int,  float * );

extern "C"
void Hdf5ParallelReadY(char* , char* , int , int ,  int , HDF5_DATATYPE * );

extern "C"
void Hdf5ParallelReconDataSet(const char* , const char* , int , int , int );

extern "C"
void Hdf5ParallelWriteZ( const char* , const char* , int , int , int , int , int , float * );

int GetSharedMemSize(int);

void GridRecSlices(ThreadGPUPlan* plan, GridRec* recon_algorithm); 

void RingCorrectionSinogram (float *data, float ring_coeff, int um, int num_proj,
			     float *mean_vect, float* mean_sino_line_data,
			     float *low_pass_sino_lines_data); 

string num2str( int num );

// Error handling macros for MPI

// #define MPI_CHECK(call) \
//     if((call) != MPI_SUCCESS) { \
//         cerr << "MPI error calling \""#call"\"\n"; \
//         my_abort(-1); }

// Shut down MPI cleanly if something goes wrong
// void my_abort(int err)
// {
//     cout << "Test FAILED\n";
//     MPI_Abort(MPI_COMM_WORLD, err);
// }

//////// TV parameters //////////

float param_fidelity_weight_TV = 0.1; 
float param_epsilon_TV = 1e-5; 
int   param_iter_max_TV = 10; // 1000;
float param_thres_TV = 1.0;

int   param_iter_TV = 1;// 10;
float param_timestep_TV = 0.1;

int main(int argc, char** argv){

  // projection generation by simulation using ray tracing
  // for each angle during source/detector rotation

  if( argc != 2 ){

    cout << "Usage: tomo_recon  params.xml" << endl;
#ifdef STEVE_DATA
    cout << "Note: STEVE_DATA defined for bindary projection data!" << endl;
#endif

#ifdef PHASE_CONTRAST_DATA
    cout << "Note: PHASE_CONTRAST_DATA defined for bindary projection data!" << endl;
#endif

    exit(1); 
  }

  unsigned int timerTotal = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timerTotal ) );
  CUT_SAFE_CALL( cutStartTimer( timerTotal ) );

  unsigned int timerPrep = 0;
  CUT_SAFE_CALL( cutCreateTimer( &timerPrep ) );
  CUT_SAFE_CALL( cutStartTimer( timerPrep ) );

  // // Initialize MPI state
  // MPI_CHECK(MPI_Init(&argc, &argv));
    
  // // Get our MPI node number and node count
  // int commSize, commRank;
  // MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
  // MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

  // read the parameter file
  ParallelParams parallel_param; 
  
  // if( commRank == 0 ){

  // read the parameter file

  TiXmlDocument doc( argv[1] );
  bool loadOkay = doc.LoadFile();

  if ( !loadOkay ){
    cout << "Could not load test file " << argv[1] << " Error= " << doc.ErrorDesc() << " Exiting.  " << endl;
    exit( 1 );
  }

  TiXmlNode* node = 0;
  TiXmlElement* paramsElement = 0;
  TiXmlElement* itemElement = 0;

  node = doc.FirstChild( "Params" );
  paramsElement = node->ToElement();

  // parameters for projection image
  node = paramsElement->FirstChild("PROJ_IMAGE_TYPE");
  if( node == NULL ){
    cout << "Note: PROJ_IMAGE_TYPE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.n_proj_image_type = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_IMAGE_DIRECTORY_NAME");
  if( node == NULL ){
    cout << "Note: PROJ_IMAGE_DIRECTORY_NAME does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.str_proj_image_directory = string( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_IMAGE_BASE_NAME");
  if( node == NULL ){
    cout << "Note: PROJ_IMAGE_BASE_NAME does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.str_proj_image_base_name = string( node->Value() );
  }

  // read parameters for flat field correction
  if( parallel_param.n_proj_image_type == PROJ_IMAGE_TYPE_BIN ||  parallel_param.n_proj_image_type == PROJ_IMAGE_TYPE_BMP ||  parallel_param.n_proj_image_type == PROJ_IMAGE_TYPE_TIF ){

    // 
    node = paramsElement->FirstChild("PROJ_SEQ_INDEX_START");
    if( node == NULL ){
      cout << "Note: PROJ_SEQ_INDEX_START does not exist in XML file. Abort!" << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      parallel_param.n_proj_seq_index_start = atoi( node->Value() );
    }

    // 
    node = paramsElement->FirstChild("PROJ_SEQ_INDEX_END");
    if( node == NULL ){
      cout << "Note: PROJ_SEQ_INDEX_END does not exist in XML file. Abort!" << endl;
      exit(1);
    }
    else{
      node = node->FirstChild();
      parallel_param.n_proj_seq_index_end = atoi( node->Value() );
    }

    //
    node = paramsElement->FirstChild("FLAT_FIELD_NORMALIZATION");
    if( node == NULL ){
      cout << "Note: FLAT_FIELD_NORMALIZATION does not exist in XML file. Using default 0 (no normalization)!" << endl;
      parallel_param.n_flat_field_normalization = 0;
    }
    else{
      node = node->FirstChild();
      parallel_param.n_flat_field_normalization = atoi( node->Value() );
    }

    // 
    node = paramsElement->FirstChild("PROJ_OFFSET_Y");
    if( node == NULL ){
      cout << "Note: PROJ_OFFSET_Y does not exist in XML file. Using default 0!" << endl;
      parallel_param.n_proj_offset_y = 0;
    }
    else{
      node = node->FirstChild();
      parallel_param.n_proj_offset_y = atoi( node->Value() );
    }

    // read parameters used for flat field correction
    if( parallel_param.n_flat_field_normalization == 1 ){

      // white field
      node = paramsElement->FirstChild("WHITE_FIELD_DIRECTORY_NAME");
      if( node == NULL ){
	cout << "Note: WHITE_FIELD_DIRECTORY_NAME does not exist in XML file. Abort!" << endl;
	exit(1);
      }
      else{
	node = node->FirstChild();
	parallel_param.str_white_field_directory_name = string( node->Value() );
      }

      // 
      node = paramsElement->FirstChild("WHITE_FIELD_BASE_NAME");
      if( node == NULL ){
	cout << "Note: WHITE_FIELD_BASE_NAME does not exist in XML file. Abort!" << endl;
	exit(1);
      }
      else{
	node = node->FirstChild();
	parallel_param.str_white_field_base_name = string( node->Value() );
      }
     
      // 
      node = paramsElement->FirstChild("WHITE_FIELD_INDEX_START");
      if( node == NULL ){
	cout << "Note: WHITE_FIELD_INDEX_START does not exist in XML file. Abort!" << endl;
	exit(1);
      }
      else{
	node = node->FirstChild();
	parallel_param.n_white_field_index_start = atoi( node->Value() );
      }

      // 
      node = paramsElement->FirstChild("WHITE_FIELD_INDEX_END");
      if( node == NULL ){
	cout << "Note: WHITE_FIELD_INDEX_END does not exist in XML file. Abort!" << endl;
	exit(1);
      }
      else{
	node = node->FirstChild();
	parallel_param.n_white_field_index_end = atoi( node->Value() );
      }


      // dark field
      node = paramsElement->FirstChild("DARK_FIELD_DIRECTORY_NAME");
      if( node == NULL ){
	cout << "Note: DARK_FIELD_DIRECTORY_NAME does not exist in XML file. Abort!" << endl;
	exit(1);
      }
      else{
	node = node->FirstChild();
	parallel_param.str_dark_field_directory_name = string( node->Value() );
      }

      // 
      node = paramsElement->FirstChild("DARK_FIELD_BASE_NAME");
      if( node == NULL ){
	cout << "Note: DARK_FIELD_BASE_NAME does not exist in XML file. Abort!" << endl;
	exit(1);
      }
      else{
	node = node->FirstChild();
	parallel_param.str_dark_field_base_name = string( node->Value() );
      }
     
      // 
      node = paramsElement->FirstChild("DARK_FIELD_INDEX_START");
      if( node == NULL ){
	cout << "Note: DARK_FIELD_INDEX_START does not exist in XML file. Abort!" << endl;
	exit(1);
      }
      else{
	node = node->FirstChild();
	parallel_param.n_dark_field_index_start = atoi( node->Value() );
      }

      // 
      node = paramsElement->FirstChild("DARK_FIELD_INDEX_END");
      if( node == NULL ){
	cout << "Note: DARK_FIELD_INDEX_END does not exist in XML file. Abort!" << endl;
	exit(1);
      }
      else{
	node = node->FirstChild();
	parallel_param.n_dark_field_index_end = atoi( node->Value() );
      }                  

    } // for flat-field normalization

  }  // for image sequences  

  //
  if( parallel_param.n_proj_image_type == PROJ_IMAGE_TYPE_HDF5 ){

    //
    node = paramsElement->FirstChild("PROJ_HDF5_DATASET_NORMALIZATION");
    if( node == NULL ){
      cout << "Note: PROJ_HDF5_DATASET_NORMALIZATION does not exist in XML file. Using 0 (with data normalization) if applicable!" << endl;
      parallel_param.n_proj_hdf5_dataset_normalization = 0;
    }
    else{
      node = node->FirstChild();
      parallel_param.n_proj_hdf5_dataset_normalization = atoi( node->Value() );
    }

    // 
    node = paramsElement->FirstChild("PROJ_HDF5_DATASET_NAME");
    if( node == NULL ){
      cout << "Note: PROJ_HDF5_DATASET_NAME does not exist in XML file. Using /entry/exchange/data if applicable!" << endl;
      parallel_param.str_proj_hdf5_dataset_name = string( "/entry/exchange/data" );
    }
    else{
      node = node->FirstChild();
      parallel_param.str_proj_hdf5_dataset_name = string( node->Value() );
    }

  }

  // 
  node = paramsElement->FirstChild("RECON_IMAGE_TYPE");
  if( node == NULL ){
    cout << "Note: RECON_IMAGE_TYPE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.n_recon_image_type = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_IMAGE_DIRECTORY_NAME");
  if( node == NULL ){
    cout << "Note: RECON_IMAGE_DIRECTORY_NAME does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.str_recon_image_directory = string( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_IMAGE_BASE_NAME");
  if( node == NULL ){
    cout << "Note: RECON_IMAGE_BASE_NAME does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.str_recon_image_base_name = string( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_THRESHOLD_LOWER_VALUE");
  if( node == NULL ){
    cout << "Note: RECON_THRESHOLD_LOWER_VALUE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.f_recon_threshold_lower_value = atof( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_THRESHOLD_UPPER_VALUE");
  if( node == NULL ){
    cout << "Note: RECON_THRESHOLD_UPPER_VALUE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.f_recon_threshold_upper_value = atof( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_HDF5_DATASET_NAME");
  if( node == NULL ){
    cout << "Note: RECON_HDF5_DATASET_NAME does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.str_recon_hdf5_dataset_name = string( node->Value() );
  }

  // parameters for imaging geometry
  node = paramsElement->FirstChild("SOD");
  if( node == NULL ){
    cout << "Note: SOD does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.SOD = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("NUM_PROJ");
  if( node == NULL ){
    cout << "Note: NUM_PROJ does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.num_proj = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("DETECTOR_SIZE_X");
  if( node == NULL ){
    cout << "Note: DETECTOR_SIZE_X does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.um = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("DETECTOR_SIZE_Y");
  if( node == NULL ){
    cout << "Note: DETECTOR_SIZE_Y does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.vm = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("OBJECT_SIZE_X");
  if( node == NULL ){
    cout << "Note: OBJECT_SIZE_X does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.xm = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("OBJECT_SIZE_Y");
  if( node == NULL ){
    cout << "Note: OBJECT_SIZE_Y does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.ym = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("OBJECT_SIZE_Z");
  if( node == NULL ){
    cout << "Note: OBJECT_SIZE_Z does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.zm = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("RECON_ALG");
  if( node == NULL ){
    cout << "Note: RECON_ALG does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.recon_algorithm = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("STOP_TYPE");
  if( node == NULL ){
    cout << "Note: STOP_TYPE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.stop_type = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("ITER_MAX");
  if( node == NULL ){
    cout << "Note: ITER_MAX does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.iter_max = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_THRES");
  if( node == NULL ){
    cout << "Note: PROJ_THRES does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.proj_thres = atof( node->Value() );
  }

  node = paramsElement->FirstChild("DIFF_THRES_PERCENT");
  if( node == NULL ){
    cout << "Note: DIFF_THRES_PERCENT does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.diff_thres_percent = atof( node->Value() );
  }

  node = paramsElement->FirstChild("DIFF_THRES_VALUE");
  if( node == NULL ){
    cout << "Note: DIFF_THRES_VALUE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.diff_thres_value = atof( node->Value() );
  }

  node = paramsElement->FirstChild("PROJ_PIXEL_SIZE");
  if( node == NULL ){
    cout << "Note: PROJ_PIXEL_SIZE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.proj_pixel_size = atof( node->Value() );
  }

  node = paramsElement->FirstChild("VOXEL_SIZE");
  if( node == NULL ){
    cout << "Note: VOXEL_SIZE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.voxel_size = atof( node->Value() );
  }

  node = paramsElement->FirstChild("SPACING");
  if( node == NULL ){
    cout << "Note: SPACING does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.spacing = atof( node->Value() );
  }

  node = paramsElement->FirstChild("INV_ROT");
  if( node == NULL ){
    cout << "Note: INV_ROT does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.inv_rot = atof( node->Value() );
  }

  node = paramsElement->FirstChild("STEP_SIZE");
  if( node == NULL ){
    cout << "Note: STEP_SIZE does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.lambda = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ROTATION_OFFSET_X");
  if( node == NULL ){
    cout << "Note: ROTATION_OFFSET_X does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.xoffset = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ROTATION_OFFSET_Y");
  if( node == NULL ){
    cout << "Note: ROTATION_OFFSET_Y does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.yoffset = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ROTATION_OFFSET_Z");
  if( node == NULL ){
    cout << "Note: ROTATION_OFFSET_Z does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.zoffset = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ANGLE_START");
  if( node == NULL ){
    cout << "Note: ANGLE_START does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.start_rot = atof( node->Value() );
  }

  node = paramsElement->FirstChild("ANGLE_END");
  if( node == NULL ){
    cout << "Note: ANGLE_END does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.end_rot = atof( node->Value() );
  }

  node = paramsElement->FirstChild("SLICE_START");
  if( node == NULL ){
    cout << "Note: SLICE_START does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.nSliceStart = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("SLICE_END");
  if( node == NULL ){
    cout << "Note: SLICE_END does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.nSliceEnd = atoi( node->Value() );
  }


  node = paramsElement->FirstChild("RECON_RADIUS_CROP");
  if( node == NULL ){
    cout << "Note: RECON_RADIUS_CROP does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.nReconRadiusCrop = atoi( node->Value() );
  }

  node = paramsElement->FirstChild("GPU_DEVICE_ID");
  if( node == NULL ){
    cout << "Note: GPU_DEVICE_ID does not exist in XML file. Abort!" << endl;
    exit(1);
  }
  else{
    node = node->FirstChild();
    parallel_param.nGPUDeviceID = atoi( node->Value() );
  }

  // check the compatibility of parameters for parallel beam
  if( parallel_param.xm != parallel_param.um ){
    cout << "Object width in projection data does not match in parameter file!" << endl;
    exit(1);
  }

  if( parallel_param.zm != parallel_param.vm ){
    cout << "Object height in projection data does not match in parameter file!" << endl;
    exit(1);
  }

#ifdef VERBOSE
  cout << "parameter file read" << endl;
  // cout << "Rank " << commRank << " : parameter file read" << endl;
#endif

  // }

  // // non-efficient way to broadcast parameters
  // MPI_Bcast( &parallel_param.SOD,             1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.num_proj,        1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.um,              1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.vm,              1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.xm,              1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.ym,              1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.zm,              1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.recon_algorithm, 1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.stop_type,       1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.iter_max,        1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.proj_thres,      1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.diff_thres_percent, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.diff_thres_value,   1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.proj_pixel_size, 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.voxel_size,      1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.spacing,         1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.inv_rot,         1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.lambda,          1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.xoffset,         1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.yoffset,         1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.zoffset,         1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.start_rot,       1, MPI_FLOAT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.end_rot,         1, MPI_FLOAT, 0, MPI_COMM_WORLD );

  // MPI_Bcast( &parallel_param.nSliceStart,     1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.nSliceEnd,       1, MPI_INT, 0, MPI_COMM_WORLD );
  // MPI_Bcast( &parallel_param.nGPUDeviceID,    1, MPI_INT, 0, MPI_COMM_WORLD );

  // MPI_Bcast( &parallel_param.n_proj_image_type, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // int size_str;
  // if( commRank == 0 ){
  //   size_str = parallel_param.str_proj_image_directory.size();
  // }
  // MPI_Bcast( &size_str, 1, MPI_INT, 0, MPI_COMM_WORLD );
  // parallel_param.str_proj_image_directory.resize( size_str );
  // MPI_Bcast( (void*) parallel_param.str_proj_image_directory.data(), size_str, MPI_CHAR, 0, MPI_COMM_WORLD);

  // if( commRank == 0 ){
  //   size_str = parallel_param.str_proj_image_base_name.size();
  // }
  // MPI_Bcast( &size_str, 1, MPI_INT, 0, MPI_COMM_WORLD );
  // parallel_param.str_proj_image_base_name.resize( size_str );
  // MPI_Bcast( (void*) parallel_param.str_proj_image_base_name.data(), 
  // 	     size_str, MPI_CHAR, 0, MPI_COMM_WORLD);

  // if( commRank == 0 ){
  //   size_str = parallel_param.str_proj_hdf5_dataset_name.size();
  // }
  // MPI_Bcast( &size_str, 1, MPI_INT, 0, MPI_COMM_WORLD );
  // parallel_param.str_proj_hdf5_dataset_name.resize( size_str );
  // MPI_Bcast( (void*) parallel_param.str_proj_hdf5_dataset_name.data(), size_str, MPI_CHAR, 0, MPI_COMM_WORLD);

  // MPI_Bcast( &parallel_param.n_proj_hdf5_dataset_normalization, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // MPI_Bcast( &parallel_param.n_proj_seq_index_start, 1, MPI_INT, 0, MPI_COMM_WORLD);
  // MPI_Bcast( &parallel_param.n_proj_seq_index_end, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // if( commRank == 0 ){
  //   size_str = parallel_param.str_recon_image_directory.size();
  // }
  // MPI_Bcast( &size_str, 1, MPI_INT, 0, MPI_COMM_WORLD );
  // parallel_param.str_recon_image_directory.resize( size_str );
  // MPI_Bcast( (void*) parallel_param.str_recon_image_directory.data(), size_str, MPI_CHAR, 0, MPI_COMM_WORLD);

  // if( commRank == 0 ){
  //   size_str = parallel_param.str_recon_image_base_name.size();
  // }
  // MPI_Bcast( &size_str, 1, MPI_INT, 0, MPI_COMM_WORLD );
  // parallel_param.str_recon_image_base_name.resize( size_str );
  // MPI_Bcast( (void*)parallel_param.str_recon_image_base_name.data(), size_str, MPI_CHAR, 0, MPI_COMM_WORLD);

  // MPI_Bcast( &parallel_param.n_recon_image_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
  // MPI_Bcast( &parallel_param.f_recon_threshold_lower_value, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Bcast( &parallel_param.f_recon_threshold_upper_value, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Bcast( &parallel_param.nReconRadiusCrop, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // if( commRank == 0 ){
  //   size_str = parallel_param.str_recon_hdf5_dataset_name.size();
  // }
  // MPI_Bcast( &size_str, 1, MPI_INT, 0, MPI_COMM_WORLD );
  // parallel_param.str_recon_hdf5_dataset_name.resize( size_str );
  // MPI_Bcast( (void*) parallel_param.str_recon_hdf5_dataset_name.data(), size_str, MPI_CHAR, 0, MPI_COMM_WORLD);


#ifdef VERBOSE
  // cout << "Rank " << commRank << " : parameters received" << endl;
  cout << "parameters received" << endl;
#endif

  int SOD          = parallel_param.SOD;
  int num_proj     = parallel_param.num_proj;
  int um           = parallel_param.um;
  int vm           = parallel_param.vm;
  int xm           = parallel_param.xm;
  int ym           = parallel_param.ym;
  int zm           = parallel_param.zm;
  int recon_algorithm = parallel_param.recon_algorithm; // 0: GridRec CPU; 1: SART GPU
  int stop_type    = parallel_param.stop_type;
  int iter_max     = parallel_param.iter_max;
  float proj_thres = parallel_param.proj_thres;
  float diff_thres_percent = parallel_param.diff_thres_percent;
  float diff_thres_value = parallel_param.diff_thres_value;
  
  float proj_pixel_size = parallel_param.proj_pixel_size;
  float voxel_size      = parallel_param.voxel_size;
  float spacing         = parallel_param.spacing;
  float inv_rot         = parallel_param.inv_rot;
  float lambda          = parallel_param.lambda;
  float xoffset         = parallel_param.xoffset;
  float yoffset         = parallel_param.yoffset;
  float zoffset         = parallel_param.zoffset;
  float start_rot       = parallel_param.start_rot;
  float end_rot         = parallel_param.end_rot;

  int nSliceStart       = parallel_param.nSliceStart;
  int nSliceEnd         = parallel_param.nSliceEnd;

  int     n_proj_image_type = parallel_param.n_proj_image_type;
  string  str_proj_image_directory = parallel_param.str_proj_image_directory;
  string  str_proj_image_base_name = parallel_param.str_proj_image_base_name;
  int     n_proj_seq_index_start = parallel_param.n_proj_seq_index_start;
  int     n_proj_seq_index_end   = parallel_param.n_proj_seq_index_end;

  int     n_flat_field_normalization = parallel_param.n_flat_field_normalization;
  int     n_proj_offset_y = parallel_param.n_proj_offset_y; 

  string  str_white_field_directory_name = parallel_param.str_white_field_directory_name;
  string  str_white_field_base_name = parallel_param.str_white_field_base_name;
  int     n_white_field_index_start = parallel_param.n_white_field_index_start;
  int     n_white_field_index_end = parallel_param.n_white_field_index_end;
  string  str_dark_field_directory_name = parallel_param.str_dark_field_directory_name;
  string  str_dark_field_base_name = parallel_param.str_dark_field_base_name;
  int     n_dark_field_index_start = parallel_param.n_dark_field_index_start;
  int     n_dark_field_index_end = parallel_param.n_dark_field_index_end;

  string  str_proj_hdf5_dataset_name = parallel_param.str_proj_hdf5_dataset_name;
  int     n_proj_hdf5_dataset_normalization = parallel_param.n_proj_hdf5_dataset_normalization;

  int     n_recon_image_type = parallel_param.n_recon_image_type; 
  string  str_recon_image_directory = parallel_param.str_recon_image_directory;
  string  str_recon_image_base_name = parallel_param.str_recon_image_base_name;
  float   f_recon_threshold_lower_value = parallel_param.f_recon_threshold_lower_value;
  float   f_recon_threshold_upper_value = parallel_param.f_recon_threshold_upper_value;
  string  str_recon_hdf5_dataset_name = parallel_param.str_recon_hdf5_dataset_name;
  int     nReconRadiusCrop = parallel_param.nReconRadiusCrop;
  int     nGPUDeviceID = parallel_param.nGPUDeviceID;

  if( recon_algorithm == 0 && xm != ym ){ 
    cout << "GridRec supports xm = ym only" << endl;
    exit(1); 
  }

  // check GPU and get the needed parameters
  int deviceCount = 0;

#ifdef MULTI_GPU_GRIDREC

  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    cout << "cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched." << endl;
    exit(0);
  }

  if( deviceCount == 0 ){
    cout << "There is no device supporting CUDA." << endl;
    exit(0);
  }

  if( deviceCount > MAX_NUM_GPU ){
    deviceCount = MAX_NUM_GPU;
    cout << "CUDA-capable device count: " << deviceCount << endl;
  }

  // gpu0035 needs deviceCount = 1 for NUM_SLICE_INV = 4 and proj_stu_2048_8_1536*.nrrd. 
  // deviceCount = 2 is ok for NUM_SLICE_INV = 2 at first. Stuck later. 
  // NUM_SLICE_INV = 8 does not work even for deviceCount = 1

#ifdef VERBOSE
  cout << "Rank " << commRank << " :  " << deviceCount << " GPUs used for reconstruction." << endl;
#endif

#else  // current

  if( recon_algorithm == 0 ){       
    deviceCount = 1; 
  }
  else if( recon_algorithm == 1 ){

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
      cout << "cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched." << endl;
      exit(0);
    }

    if( deviceCount == 0 ){
      cout << "There is no device supporting CUDA." << endl;
      exit(0);
    }

    if( deviceCount > MAX_NUM_GPU ){
      deviceCount = MAX_NUM_GPU;
      cout << "CUDA-capable device count: " << deviceCount << endl;
    }

    // gpu0035 needs deviceCount = 1 for NUM_SLICE_INV = 4 and proj_stu_2048_8_1536*.nrrd. 
    // deviceCount = 2 is ok for NUM_SLICE_INV = 2 at first. Stuck later. 
    // NUM_SLICE_INV = 8 does not work even for deviceCount = 1

#ifdef VERBOSE
    // cout << "Rank " << commRank << " :  " << deviceCount << " GPUs used for reconstruction." << endl;
    cout << deviceCount << " GPUs used for reconstruction." << endl;
#endif

  }
#endif

  // OS thread ID
  CUTThread* threadID = new CUTThread[ deviceCount ];
  if( !threadID ){
    cout << "Error allocating memory for threadID" << endl;
    exit(1); 
  }

  // Solver configuration
  ThreadGPUPlan *plan = new ThreadGPUPlan[ deviceCount ];
  if( !plan ){
    cout << "Error allocating memory for plan" << endl;
    exit(1); 
  }

  // read projection data

  bool bProjFormatNrrd = ( n_proj_image_type == PROJ_IMAGE_TYPE_NRRD );
  bProjFormatNrrd = bProjFormatNrrd || ( n_proj_image_type == PROJ_IMAGE_TYPE_BIN );
  bProjFormatNrrd = bProjFormatNrrd || ( n_proj_image_type == PROJ_IMAGE_TYPE_BMP );
  bProjFormatNrrd = bProjFormatNrrd || ( n_proj_image_type == PROJ_IMAGE_TYPE_TIF );

  bool bProjFormatHdf5 = ( n_proj_image_type == PROJ_IMAGE_TYPE_HDF5 );

  if( !bProjFormatNrrd && !bProjFormatHdf5 ){
    cout << "Only NRRD, BIN, BMP, TIFF, HDF5 formats are supported for projection data!" << endl;
    exit(1);
  }

  // prepare for ring artifact removal
  float* mean_vect = new float[ num_proj ];
  float* low_pass_sino_lines_data = new float[ um ]; 
  float* mean_sino_line_data = new float[ um ]; 
  float* data_sino = new float[ um * num_proj ]; 

  Float3DImageType::Pointer ProjPointer; 
  Float3DImageType::IndexType index;

  // if( bProjFormatNrrd && commRank == 0 ){
  if( bProjFormatNrrd ){
    
    if( n_proj_image_type == PROJ_IMAGE_TYPE_NRRD ){

      string strNrrd = str_proj_image_directory;
      strNrrd.append( str_proj_image_base_name );

      itk::ImageFileReader<Float3DImageType>::Pointer RealProjReader;
      RealProjReader = itk::ImageFileReader<Float3DImageType>::New();
      RealProjReader->SetFileName( strNrrd.c_str() );  
      RealProjReader->Update();

      ProjPointer = (RealProjReader->GetOutput());
      Float3DImageType::SizeType size = ProjPointer->GetLargestPossibleRegion().GetSize();

      // check the compatibility of parameters
      if( size[0] != xm || size[0] != um ){
	cout << "Object width in projection data does not match parameter file!" << endl;
	exit(1);
      }

      if( size[1] != zm || size[1] != vm ){
	cout << "Object height in projection data does not match parameter file!" << endl;
	exit(1);
      }

      if( size[2] != num_proj ){
	cout << "Rotations in projection data does not match parameter file!" << endl;
	exit(1);
      }

      um =  size[0];
      vm =  size[1];
      num_proj =  size[2];

      cout << "Projection Data (nrrd) Read from file !" << endl;
    }
    else if( n_proj_image_type == PROJ_IMAGE_TYPE_BIN ){

      Float3DImageType::SizeType size;
      Float3DImageType::RegionType region;

      size[0] = um;
      size[1] = vm;
      size[2] = num_proj;

      region.SetSize( size );

      ProjPointer = Float3DImageType::New(); 
      ProjPointer->SetRegions( region );
      ProjPointer->Allocate(); 

      string strPosix = string(".bin");

      Float3DImageType::PixelType pixelImg;
      PROJ_IMAGE_BIN_TYPE pixel_bin; 

      float* dataWhite = NULL;
      float* dataDark = NULL;

      if( n_flat_field_normalization == 1 ){  // perform flat field normalization 

	dataWhite = new float[ um * vm ];  
	dataDark = new float[ um * vm ];

	// read the white field image sequence
	for( int z = n_white_field_index_start; z <= n_white_field_index_end; z++ ){

	  string strIndex = num2str( z );
	  string strCurrName = str_white_field_directory_name;
	  strCurrName.append( str_white_field_base_name );

	  int pos = strCurrName.length() - strPosix.length() - strIndex.length();
	  strCurrName.replace(pos, strIndex.length(), strIndex); 

	  fstream datafile( strCurrName.c_str(), ios::in | ios::binary );

	  if( !datafile.is_open() ){
	    cout << "    Skip reading white field file " << strCurrName << endl;
	    continue; 
	  }
	  else{
	    cout << "    Reading white field file " << strCurrName << endl;  // test

	    for( int y = 0; y < vm; y++ ){
	      for( int x = 0; x < um; x++ ){  

		index[0] = x;
		index[1] = vm - 1 - y; // Note

		datafile.read( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
		dataWhite[ (vm - 1 - y) * um + x ] += (float) pixel_bin; 
	      }
	    }
	  }

	  cout << "  White field image " << strCurrName << " read " << endl;
	}

	for( int ny = 0; ny < vm; ny++ ){
	  for( int nx = 0; nx < um; nx++ ){  
	
	    dataWhite[ ny * um + nx ] /=  (n_white_field_index_end - n_white_field_index_start + 1 );

	  }
	}
  
	// read the dark field image sequence
	for( int z = n_dark_field_index_start; z <= n_dark_field_index_end; z++ ){

	  string strIndex = num2str( z );
	  string strCurrName = str_dark_field_directory_name;
	  strCurrName.append( str_dark_field_base_name );

	  int pos = strCurrName.length() - strPosix.length() - strIndex.length();
	  strCurrName.replace(pos, strIndex.length(), strIndex); 

	  fstream datafile( strCurrName.c_str(), ios::in | ios::binary );

	  if( !datafile.is_open() ){
	    cout << "    Skip reading dark field file " << strCurrName << endl;
	    continue; 
	  }
	  else{
	    cout << "    Reading dark field file " << strCurrName << endl;  // test

	    for( int y = 0; y < vm; y++ ){
	      for( int x = 0; x < um; x++ ){  

		index[0] = x;
		index[1] = vm - 1 - y; // Note

		datafile.read( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
		dataDark[ (vm - 1 - y) * um + x ] += (float) pixel_bin; 
	      }
	    }
	  }
   
	  cout << "  Dark field image " << strCurrName << " read " << endl;

	}

	for( int ny = 0; ny < vm; ny++ ){
	  for( int nx = 0; nx < um; nx++ ){  
	
	    dataDark[ ny * um + nx ] /=  (n_dark_field_index_end - n_dark_field_index_start + 1 );

	  }
	}

      } // if (n_flat_field_normalization == 1)

      for( int z = n_proj_seq_index_start; z <= n_proj_seq_index_end; z++ ){

	string strIndex = num2str( z );
	string strCurrBinName = str_proj_image_directory;
	strCurrBinName.append( str_proj_image_base_name );

	int pos = strCurrBinName.length() - strPosix.length() - strIndex.length();
	strCurrBinName.replace(pos, strIndex.length(), strIndex); 

	fstream datafile( strCurrBinName.c_str(), ios::in | ios::binary );

	float fWhite, fDark, fValue; 

	if( !datafile.is_open() ){
	  cout << "    Skip reading file " << strCurrBinName << endl;
	  continue; 
	}
	else{
	  cout << "    Reading file " << strCurrBinName << endl;  // test

	  index[2] = z - n_proj_seq_index_start;

	  for( int y = 0; y < vm; y++ ){
	    for( int x = 0; x < um; x++ ){  

	      index[0] = x;
	      index[1] = vm - 1 - y; // Note

	      datafile.read( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
  	      pixelImg = (Float3DImageType::PixelType) pixel_bin; 

#ifdef STEVE_DATA
	      if( pixelImg > 100.0f ){
		pixelImg = 0.0f;
	      }
	      else if( pixelImg < 0.0f ){
		cout << "Pixel value should not be negative at " << index << endl;
		exit(1);
	      }
	      else{
		pixelImg = log( 100 / pixelImg );
	      }
#endif

#ifdef PHASE_CONTRAST_DATA
	      pixelImg *= 1e5; 
#endif 

	      if ( n_flat_field_normalization == 1 ) {

		fWhite = dataWhite[ y * um + x ];
		fDark  = dataDark[ y * um + x ];

		fValue = pixelImg - fDark;

		if( fValue <= 0.0f )
		  ProjPointer->SetPixel( index, (Float3DImageType::PixelType) 0.0f );
		else{
		  fValue = (1.0f * fWhite - fDark) / fValue;
		  if( fValue > 0.0f )
		    ProjPointer->SetPixel( index, (Float3DImageType::PixelType) log(fValue) );
		  else{
		    ProjPointer->SetPixel( index, (Float3DImageType::PixelType) 0.0f );
		  }
		}

	      }
	      else{
		ProjPointer->SetPixel( index, pixelImg );
	      }
	    }
	  }
	}
      }

      cout << "Projection Data (bin) Read from file !" << endl;

      if ( n_flat_field_normalization == 1 ) {
	delete [] dataWhite;  
	delete [] dataDark;
      }

    }
    else if( n_proj_image_type == PROJ_IMAGE_TYPE_BMP ||  n_proj_image_type == PROJ_IMAGE_TYPE_TIF ){

      Float3DImageType::SizeType size;
      Float3DImageType::RegionType region;

      size[0] = um;
      size[1] = vm;
      size[2] = num_proj;

      region.SetSize( size );

      ProjPointer = Float3DImageType::New(); 
      ProjPointer->SetRegions( region );
      ProjPointer->Allocate(); 

      string strPosix = string(".bmp"); // string(".tif");

      itk::ImageFileReader<UChar2DImageType>::Pointer SliceReader;
      SliceReader = itk::ImageFileReader<UChar2DImageType>::New();

      UChar2DImageType::IndexType indexImg;

      float* dataWhite = NULL;
      float* dataDark = NULL;

      if( n_flat_field_normalization == 1 ){  // perform flat field normalization 

	dataWhite = new float[ um * vm ];  
	dataDark = new float[ um * vm ];

	// read the white field image sequence
	for( int z = n_white_field_index_start; z <= n_white_field_index_end; z++ ){

	  string strIndex = num2str( z );
	  string strCurrName = str_white_field_directory_name;
	  strCurrName.append( str_white_field_base_name );

	  int pos = strCurrName.length() - strPosix.length() - strIndex.length();
	  strCurrName.replace(pos, strIndex.length(), strIndex); 

	  // 
	  SliceReader->SetFileName( strCurrName.c_str() );  
	  SliceReader->Update();

	  UChar2DImageType::Pointer curr_image = SliceReader->GetOutput();

	  UChar2DImageType::SizeType sizeImg = curr_image->GetLargestPossibleRegion().GetSize();
	  if( sizeImg[ 0 ] != um || sizeImg[ 1 ] != vm + n_proj_offset_y ){
	    cout << "The image size of " << strCurrName << " is " << sizeImg << ", not equal to ( " <<  um << " , " << vm + n_proj_offset_y << " ) " << endl;
	    exit(1);
	  }

	  // 
	  for( int ny = 0; ny < vm; ny++ ){
	    for( int nx = 0; nx < um; nx++ ){  
	
	      indexImg[0] = nx;
	      indexImg[1] = ny + n_proj_offset_y;
	
	      dataWhite[ ny * um + nx ] += curr_image->GetPixel( indexImg );
	    }
	  }
    
	  cout << "  White field image " << strCurrName << " read " << endl;
	}

	for( int ny = 0; ny < vm; ny++ ){
	  for( int nx = 0; nx < um; nx++ ){  
	
	    dataWhite[ ny * um + nx ] /=  (n_white_field_index_end - n_white_field_index_start + 1 );

	  }
	}
  
	// read the dark field image sequence
	for( int z = n_dark_field_index_start; z <= n_dark_field_index_end; z++ ){

	  string strIndex = num2str( z );
	  string strCurrName = str_dark_field_directory_name;
	  strCurrName.append( str_dark_field_base_name );

	  int pos = strCurrName.length() - strPosix.length() - strIndex.length();
	  strCurrName.replace(pos, strIndex.length(), strIndex); 

	  // 
	  SliceReader->SetFileName( strCurrName.c_str() );  
	  SliceReader->Update();

	  UChar2DImageType::Pointer curr_image = SliceReader->GetOutput();

	  UChar2DImageType::SizeType sizeImg = curr_image->GetLargestPossibleRegion().GetSize();
	  if( sizeImg[ 0 ] != um || sizeImg[ 1 ] != vm + n_proj_offset_y ){
	    cout << "The image size of " << strCurrName << " is " << sizeImg << ", not equal to ( " <<  um << " , " << vm + n_proj_offset_y << " ) " << endl;
	    exit(1);
	  }

	  // 
	  for( int ny = 0; ny < vm; ny++ ){
	    for( int nx = 0; nx < um; nx++ ){  
	
	      indexImg[0] = nx;
	      indexImg[1] = ny + n_proj_offset_y;
	
	      dataDark[ ny * um + nx ] += curr_image->GetPixel( indexImg);
	    }
	  }
    
	  cout << "  Dark field image " << strCurrName << " read " << endl;

	}

	for( int ny = 0; ny < vm; ny++ ){
	  for( int nx = 0; nx < um; nx++ ){  
	
	    dataDark[ ny * um + nx ] /=  (n_dark_field_index_end - n_dark_field_index_start + 1 );

	  }
	}

      } // if (n_flat_field_normalization == 1)

      // read projection data
      for( int z = n_proj_seq_index_start; z <= n_proj_seq_index_end; z++ ){

	string strIndex = num2str( z );
	string strCurrBmpTiffName = str_proj_image_directory;
	strCurrBmpTiffName.append( str_proj_image_base_name );

	int pos = strCurrBmpTiffName.length() - strPosix.length() - strIndex.length();
	strCurrBmpTiffName.replace(pos, strIndex.length(), strIndex); 

	// 
	SliceReader->SetFileName( strCurrBmpTiffName.c_str() );  
	SliceReader->Update();

	UChar2DImageType::Pointer ProjSlicePointer = (SliceReader->GetOutput());
	UChar2DImageType::SizeType size = ProjSlicePointer->GetLargestPossibleRegion().GetSize();

	// check the compatibility of parameters
	if( size[0] != xm || size[0] != um ){
	  cout << "Object width in projection data does not match parameter file!" << endl;
	  exit(1);
	}

	if( size[1] != zm + n_proj_offset_y || size[1] != vm + n_proj_offset_y ){
	  cout << "Object height in projection data does not match parameter file!" << endl;
	  exit(1);
	}

	// 
	index[2] = z - n_proj_seq_index_start;

	float fWhite, fDark, fValue; 

	for( int y = 0; y < vm; y++ ){
	  for( int x = 0; x < um; x++ ){  

	    index[0] = x;
	    index[1] = y;

	    indexImg[0] = x;
	    indexImg[1] = y;

	    if ( n_flat_field_normalization == 1 ) {

	      fWhite = dataWhite[ y * um + x ];
	      fDark  = dataDark[ y * um + x ];

	      fValue = ProjSlicePointer->GetPixel( indexImg) - fDark;

	      if( fValue <= 0.0f )
		ProjPointer->SetPixel( index, (Float3DImageType::PixelType) 0.0f );
	      else{
		fValue = (1.0f * fWhite - fDark) / fValue;
		if( fValue > 0.0f )
		  ProjPointer->SetPixel( index, (Float3DImageType::PixelType) log(fValue) );
		else{
		  ProjPointer->SetPixel( index, (Float3DImageType::PixelType) 0.0f );
		}
	      }
	    }
	    else{ // if n_flat_field_normalization == 0
	      ProjPointer->SetPixel( index, (Float3DImageType::PixelType) ProjSlicePointer->GetPixel( indexImg) );
	    }
	  }
	}

	cout << strCurrBmpTiffName << " read " << endl;
      }

      if( n_proj_image_type == PROJ_IMAGE_TYPE_BMP ) 
	cout << "Projection Data (bmp) Read from file !" << endl;
      if( n_proj_image_type == PROJ_IMAGE_TYPE_TIF )
	cout << "Projection Data (tif) Read from file !" << endl;

      if ( n_flat_field_normalization == 1 ) {
	delete [] dataWhite;  
	delete [] dataDark;
      }
    
    }
  
    // ring artifact removal
    Float3DImageType::IndexType index;
    // for( int ny = 0; ny < vm; ny++ ){
    for( int ny = nSliceStart; ny <= nSliceEnd; ny++ ){

      for( int nz = 0; nz < num_proj; nz++ ){
	for( int nx = 0; nx < um; nx++ ){
	  index[ 0 ] = nx;
	  index[ 1 ] = ny;
	  index[ 2 ] = nz;
	  data_sino[ nz * um + nx ] = ProjPointer->GetPixel( index );
	}
      }

      RingCorrectionSinogram ( data_sino, RING_COEFF, um, num_proj,
			      mean_vect, mean_sino_line_data, low_pass_sino_lines_data); 

      for( int nz = 0; nz < num_proj; nz++ ){
	for( int nx = 0; nx < um; nx++ ){
	  index[ 0 ] = nx;
	  index[ 1 ] = ny;
	  index[ 2 ] = nz;
	  ProjPointer->SetPixel( index, data_sino[ nz * um + nx ] );
	}
      }
      cout << "    Ring removal for projection data slice " << ny << " performed!" << endl;
    }
    cout << endl;
    cout << "Ring removal for projection data (nrrd) performed!" << endl;

  }

  HDF5_DATATYPE * proj_darkfield = new HDF5_DATATYPE[ um * vm ]; 
  HDF5_DATATYPE * proj_whitefield = new HDF5_DATATYPE[ 2 * um * vm ]; 
  HDF5_DATATYPE * proj_slices = new HDF5_DATATYPE[ um * (NUM_SLICE_INV * deviceCount) * num_proj ];
  float * proj_slices_sino = new float[ um * (NUM_SLICE_INV * deviceCount) * num_proj ];
  float * obj_slices = new float[ xm * ym * (NUM_SLICE_INV * deviceCount) ];

  if( !proj_darkfield || !proj_whitefield || !proj_slices || !proj_slices_sino || !obj_slices){
    cout << "Error allocating memory for proj_darkfield,  proj_whitefield, proj_slices and obj_slices!" << endl;
    exit(1);
  }

  string str_proj_hdf5_name;
  if( bProjFormatHdf5 ){

    str_proj_hdf5_name = str_proj_image_directory;
    str_proj_hdf5_name.append( str_proj_image_base_name );

    // if( commSize == 1 ){
    Hdf5SerialReadZ( str_proj_hdf5_name.c_str(), "/entry/exchange/dark_data", 0, 1, proj_darkfield );  
    Hdf5SerialReadZ( str_proj_hdf5_name.c_str(), "/entry/exchange/white_data", 0, 2, proj_whitefield );  

    // }
    // else{ // see if we can access the same dataset from multiple nodes at the same time
      // If not, read it use Hdf5SerialRead, and broadcast it, or read sequentially

    // Hdf5ParallelReadZ( argv[2], "/entry/exchange/dark_data", 0, 1, proj_darkfield );  
    // Hdf5ParallelReadZ( argv[2], "/entry/exchange/white_data", 0, 2, proj_whitefield );  
   
    // }

  }

#ifdef VERBOSE
  // cout << "Rank " << commRank << " : white and dark fields read" << endl;  
  cout << "white and dark fields read" << endl;  
#endif

  // #ifdef OUTPUT_NRRD
  Float3DImageType::Pointer ImageReconPointer = Float3DImageType::New();
  
  Float3DImageType::RegionType regionObj;
  Float3DImageType::SizeType sizeObj;
  sizeObj[ 0 ] = xm;
  sizeObj[ 1 ] = ym;
  sizeObj[ 2 ] = deviceCount * NUM_SLICE_INV;
    
  regionObj.SetSize( sizeObj );  
  ImageReconPointer->SetRegions( regionObj );
  ImageReconPointer->Allocate( );
  // #endif // OUTPUT_NRRD


  if( n_recon_image_type == RECON_IMAGE_TYPE_HDF5 ){  

#ifdef VERBOSE
    // cout << "Rank " << commRank << " : checking dataRecon dataset" << endl;  
    cout << "checking dataRecon dataset" << endl;  
#endif

    string str_recon_hdf5_name = str_recon_image_directory;
    str_recon_hdf5_name.append( str_recon_image_base_name );

    Hdf5SerialReconDataSet( str_recon_hdf5_name.c_str(), str_recon_hdf5_dataset_name.c_str(), xm, ym, nSliceEnd + 1 - nSliceStart );

    // Hdf5ParallelReconDataSet( str_recon_hdf5_name.c_str(), str_recon_hdf5_dataset_name.c_str(), xm, ym, nSliceEnd + 1 - nSliceStart );

#ifdef VERBOSE
    // cout << "Rank " << commRank << " : dataRecon  dataset checked" << endl;  
    cout << "dataRecon  dataset checked" << endl;  
#endif

  }

  CUT_SAFE_CALL(cutStopTimer(timerPrep));
  cout << "Time for Preparation is " <<  cutGetTimerValue(timerPrep) << " (ms) " << endl;
  CUT_SAFE_CALL(cutDeleteTimer(timerPrep));

  // Prepare for CT recon using GridRec
  int num_ray_half_padding = 2 * um;

  float* vect_angle = NULL;

  // #ifdef MULTI_GPU_GRIDREC
  GridRec  ** gridrec_algorithm = NULL;
  float** sinogram = NULL;
  float** reconstruction = NULL;
// #else
//   GridRec  * gridrec_algorithm = NULL;
// #endif

  if( recon_algorithm == 0 ){
    vect_angle = (float*) new float[ num_proj ];
    for( int i = 0; i < num_proj; i++ )
      vect_angle[ i ] = start_rot + i * inv_rot;

// #ifdef MULTI_GPU_GRIDREC
    gridrec_algorithm = new GridRec*[ deviceCount ];
    sinogram = new float*[ deviceCount ];
    reconstruction = new float* [deviceCount ];

    for( int i = 0; i < deviceCount; i++ ){
      gridrec_algorithm[i] = new GridRec ();
      gridrec_algorithm[i]->setSinogramDimensions (num_ray_half_padding, num_proj);
      gridrec_algorithm[i]->setThetaList (vect_angle, num_proj);
      gridrec_algorithm[i]->setGPUDeviceID( i ); 
      gridrec_algorithm[i]->setFilter ( 3 );  // 3 for HANN. See recon_algorithm.h for definition
      gridrec_algorithm[i]->init();

      sinogram[i] = (float*) new float[ 2 * num_ray_half_padding * num_proj ];
      reconstruction[i] = (float*) new float[ 2 * num_ray_half_padding * num_ray_half_padding ];
    }

// #else
//     gridrec_algorithm = new GridRec ();
//     gridrec_algorithm->setSinogramDimensions (num_ray_half_padding, num_proj);
//     gridrec_algorithm->setThetaList (vect_angle, num_proj);
//     gridrec_algorithm->setFilter ( 3 );  // 3 for HANN. See recon_algorithm.h for definition
//     gridrec_algorithm->init();
// #endif
  }

  // CT reconstruction using SART
  int commRank = 0;
  int commSize = 1;

  unsigned int ndeviceCompute; 
  unsigned int z = nSliceStart;

  float timeParallelRead = 0;
  float timeParallelWrite = 0; 
  float timeRecon = 0; 

  while (z <= nSliceEnd){

    // prepare data for threaded GPU
    ndeviceCompute = 0; 
    if( nSliceEnd + 1 - z >= commSize * deviceCount * NUM_SLICE_INV){

      if( commRank == 0 ){
	cout << "Processing slices " << z << " ~ " << z + commSize * NUM_SLICE_INV * deviceCount - 1 << endl;
      }
      
      ndeviceCompute = deviceCount; 

      if( bProjFormatHdf5 ){

	unsigned int timerRead = 0;
	CUT_SAFE_CALL( cutCreateTimer( &timerRead ) );
	CUT_SAFE_CALL( cutStartTimer( timerRead ) );

// #ifdef HDF5_PARALLEL_READ

// 	if ( n_proj_hdf5_dataset_normalization == 0 ){
// 	  Hdf5ParallelReadZ( str_proj_hdf5_name.c_str(), str_proj_hdf5_dataset_name.c_str(), z, NUM_SLICE_INV * deviceCount, commRank, proj_slices );
// 	}
// 	else{
// 	  Hdf5ParallelReadZ_f( str_proj_hdf5_name.c_str(), str_proj_hdf5_dataset_name.c_str(), z, NUM_SLICE_INV * deviceCount, commRank, proj_slices_sino );
// 	}
// 	MPI_Barrier(MPI_COMM_WORLD);

// #endif

// #ifdef HDF5_SERIAL_READ
	if ( n_proj_hdf5_dataset_normalization == 0 ){

	  Hdf5SerialReadZ( str_proj_hdf5_name.c_str(), str_proj_hdf5_dataset_name.c_str(), z + commRank * NUM_SLICE_INV * deviceCount,
			 NUM_SLICE_INV * deviceCount, proj_slices);
	}
	else{     
	  Hdf5SerialReadZ_f( str_proj_hdf5_name.c_str(), str_proj_hdf5_dataset_name.c_str(), z + commRank * NUM_SLICE_INV * deviceCount,
			 NUM_SLICE_INV * deviceCount, proj_slices_sino);
	}

// #endif

	// cudaThreadSynchronize();    
	CUT_SAFE_CALL(cutStopTimer(timerRead));
	timeParallelRead +=  cutGetTimerValue(timerRead);
	CUT_SAFE_CALL(cutDeleteTimer(timerRead));

	// continue; // test

        // }
      }

      for( int i = 0; i < deviceCount; i++ ){
	if( recon_algorithm == 0 ){
	  plan[ i ].deviceID = nGPUDeviceID;   // One GPU each node for GPU GRIDREC (using cufft)
	}
	else{
	  plan[ i ].deviceID = i;
	}

	plan[ i ].deviceShareMemSizePerBlock = GetSharedMemSize( i );  

	// geometry
	plan[ i ].num_width = xm;
	plan[ i ].num_height = ym;
	plan[ i ].num_depth = NUM_SLICE_INV;

	plan[ i ].num_ray = um;
	plan[ i ].num_elevation = NUM_SLICE_INV;
	plan[ i ].num_proj = num_proj;

	// algorithm params
	plan[ i ].stop_type = stop_type;
	plan[ i ].proj_thres = proj_thres;
	plan[ i ].diff_thres_percent = diff_thres_percent;
	plan[ i ].diff_thres_value = diff_thres_value;

	plan[ i ].lambda = lambda;
	plan[ i ].iter_max = iter_max;
	plan[ i ].spacing = spacing;
	plan[ i ].voxel_size = voxel_size;
	plan[ i ].proj_pixel_size = proj_pixel_size;
	plan[ i ].SOD = SOD;
	plan[ i ].inv_rot = inv_rot;
	plan[ i ].xoffset = xoffset;
	plan[ i ].yoffset = yoffset;
	plan[ i ].zoffset = zoffset;
	plan[ i ].start_rot = start_rot;
	plan[ i ].end_rot = end_rot;

	// 
// #ifdef MULTI_GPU_GRIDREC
	if( recon_algorithm == 0 ){
	  plan[ i ].gridrec_algorithm = gridrec_algorithm[i];
	  plan[ i ].sinogram = sinogram[i];
	  plan[ i ].reconstruction = reconstruction[i];
	}
	else{
	  plan[ i ].gridrec_algorithm = NULL;
	  plan[ i ].sinogram = NULL;
	  plan[ i ].reconstruction = NULL;
	}
// #endif

	// memory for input/output data
	plan[ i ].h_input_data = new float[ plan[i].num_proj * plan[i].num_elevation * plan[i].num_ray ];
	plan[ i ].h_output_data = new float[ plan[i].num_width * plan[i].num_height * plan[i].num_depth ];

	if( !plan[ i ].h_input_data ||  !plan[ i ].h_output_data ){
	  cout << "Error allocating memory for plan [ " << i << " ]" << endl;
	  exit( 1 );
	}

	// input(projection) data from file
	if( bProjFormatNrrd){
	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		index[ 0 ] = nx;
		index[ 1 ] = z + i * NUM_SLICE_INV + ny; // z: object index; corresponds to elevation in projection index
		index[ 2 ] = nz;
		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] =  ProjPointer->GetPixel( index );
	      }
	    }
	  }
	}

	if( bProjFormatHdf5 ){

	  if ( n_proj_hdf5_dataset_normalization == 0 ){

	    float proj, dark, white1, white2;
	    for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	      for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
		for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		  // may need work on i and commRank for deviceCount >= 2. check later
		  proj = proj_slices[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx ];   

		  dark = proj_darkfield[ ( z + commRank * NUM_SLICE_INV + ny) * um + nx ];   
		  white1 = proj_whitefield[ (z + commRank * NUM_SLICE_INV + ny) * um + nx ];   
		  white2 = proj_whitefield[ um * vm +  (z + commRank * NUM_SLICE_INV + ny) * um + nx ];   

		  proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx ] =  1.0f - 1.0f * (proj - dark) / (0.5f * white1 + 0.5f * white2 - dark);

		}
	      }
	    }

	  }

	  // ring arfifact removal
	  for( int p = 0; p < NUM_SLICE_INV * deviceCount; p++ ){
	    RingCorrectionSinogram (&proj_slices_sino[ p * num_proj * um ], RING_COEFF, um, num_proj,
				    mean_vect, mean_sino_line_data, low_pass_sino_lines_data); 
	  }

	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] = proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx ];   

	      }
	    }
	  }


	}

      }
  
    }
    else{
      cout << "Processing slices " << z  << " ~ " << nSliceEnd << endl;

      if( bProjFormatHdf5 ){
	// if( commSize == 1 )
	//   Hdf5SerialReadY( argv[2], "/entry/exchange/data", z, zm -z, proj_slices );  
	// else{

	// needs work
	if ( n_proj_hdf5_dataset_normalization == 0 ){
	  Hdf5ParallelReadZ( str_proj_hdf5_name.c_str(), str_proj_hdf5_dataset_name.c_str(), z, nSliceEnd + 1 - z, commRank, proj_slices ); 
	}
	else{
	  Hdf5ParallelReadZ_f( str_proj_hdf5_name.c_str(), str_proj_hdf5_dataset_name.c_str(), z, nSliceEnd + 1 - z, commRank, proj_slices_sino ); 
	}

	// }
      }

      for( int i = 0; i < deviceCount; i++ ){
	plan[ i ].deviceID = i;
	plan[ i ].deviceShareMemSizePerBlock = GetSharedMemSize( i );  

	// geometry
	plan[ i ].num_width = xm;
	plan[ i ].num_height = ym;

	plan[ i ].num_ray = um;
	plan[ i ].num_proj = num_proj;

	bool bProjRead = false;
	ndeviceCompute++; 
	if( nSliceEnd + 1 - z - i * NUM_SLICE_INV > NUM_SLICE_INV ){
	  plan[ i ].num_depth = NUM_SLICE_INV;
	  plan[ i ].num_elevation =  NUM_SLICE_INV;
	}
	else{
	  plan[ i ].num_depth = nSliceEnd + 1 - z - i * NUM_SLICE_INV;
	  plan[ i ].num_elevation = nSliceEnd + 1 - z - i * NUM_SLICE_INV;
	  bProjRead = true;
 	}

	// algorithm params
	plan[ i ].stop_type = stop_type;
	plan[ i ].proj_thres = proj_thres;
	plan[ i ].diff_thres_percent = diff_thres_percent;
	plan[ i ].diff_thres_value = diff_thres_value;

	plan[ i ].lambda = lambda;
	plan[ i ].iter_max = iter_max;
	plan[ i ].spacing = spacing;
	plan[ i ].voxel_size = voxel_size;
	plan[ i ].proj_pixel_size = proj_pixel_size;
	plan[ i ].SOD = SOD;
	plan[ i ].inv_rot = inv_rot;
	plan[ i ].xoffset = xoffset;
	plan[ i ].yoffset = yoffset;
	plan[ i ].zoffset = zoffset;
	plan[ i ].start_rot = start_rot;
	plan[ i ].end_rot = end_rot;

// #ifdef MULTI_GPU_GRIDREC
	if( recon_algorithm == 0 ){
	  plan[ i ].gridrec_algorithm = gridrec_algorithm[i];
	  plan[ i ].sinogram = sinogram[i];
	  plan[ i ].reconstruction = reconstruction[i];
	}
	else{
	  plan[ i ].gridrec_algorithm = NULL;
	  plan[ i ].sinogram = NULL;
	  plan[ i ].reconstruction = NULL;
	}
// #endif

	// memory for input/output data
	plan[ i ].h_input_data = new float[ plan[i].num_proj * plan[i].num_elevation * plan[i].num_ray ];
	plan[ i ].h_output_data = new float[ plan[i].num_width * plan[i].num_height * plan[i].num_depth ];
	if( !plan[ i ].h_input_data ||  !plan[ i ].h_output_data ){
	  cout << "Error allocating memory for plan [ " << i << " ]" << endl;
	  exit( 1 );
	}

	if( bProjFormatNrrd ){
	  // input(projection) data from file
	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		index[ 0 ] = nx;
		index[ 1 ] = z + i * NUM_SLICE_INV + ny;
		index[ 2 ] = nz;
		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] =  ProjPointer->GetPixel( index );
	      }
	    }
	  }
	}

	if( bProjFormatHdf5 ){

	  if ( n_proj_hdf5_dataset_normalization == 0 ){
	    float proj, dark, white1, white2;
	    for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	      for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
		for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		  // may need work on i and commRank for deviceCount >= 2. check later
		  proj = proj_slices[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx ];   

		  dark = proj_darkfield[ (z + commRank * NUM_SLICE_INV + ny) * um + nx ];  
		  white1 = proj_whitefield[ (z + commRank * NUM_SLICE_INV + ny) * um + nx ];   
		  white2 = proj_whitefield[ vm * um + (z + commRank * NUM_SLICE_INV + ny) * um + nx ];   

		  proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx ] =  1.0f - 1.0f * (proj - dark) / (0.5f * white1 + 0.5f * white2 - dark);

		}
	      }
	    }
	  }

	  // ring arfifact removal
	  for( int p = 0; p < NUM_SLICE_INV * deviceCount; p++ ){
	    RingCorrectionSinogram (&proj_slices_sino[ p * num_proj * um ], RING_COEFF, um, num_proj,
				    mean_vect, mean_sino_line_data, low_pass_sino_lines_data); 
	  }

	  for( int nz = 0; nz < plan[ i ].num_proj; nz++ ){
	    for( int ny = 0; ny < plan[ i ].num_elevation; ny++ ){
	      for( int nx = 0; nx < plan[ i ].num_ray; nx++ ){

		plan[ i ].h_input_data[ (nz * plan[i].num_elevation + ny) * plan[i].num_ray + nx ] = proj_slices_sino[ ( (i * NUM_SLICE_INV + ny ) * num_proj + nz ) * um + nx ];   

	      }
	    }
	  }

	}

	if( bProjRead ){
	  break;
	}
      }

    }

    // run threaded calculation
    unsigned int timerRecon = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timerRecon ) );
    CUT_SAFE_CALL( cutStartTimer( timerRecon ) );

#ifdef MULTI_GPU_GRIDREC

    for( int i = 0; i < ndeviceCompute; i++ ){
      if( recon_algorithm == 0 ){        
	// GridRec CPU
	threadID[ i ] = cutStartThread( (CUT_THREADROUTINE) GridRecThread, (void*) (plan + i) );  
	
	// FBP CPU
	// threadID[ i ] = cutStartThread( (CUT_THREADROUTINE) FBPThread, (void*) (plan + i) );      
      }
      else if( recon_algorithm == 1 ){   // SART GPU
	threadID[ i ] = cutStartThread( (CUT_THREADROUTINE) SART3DThread, (void*) (plan + i) );
      }
      cutWaitForThreads( threadID, ndeviceCompute );
    }

#else  // current

    if( recon_algorithm == 0 ){
      for( int i = 0; i < ndeviceCompute; i++ ){
      	// GridRec CPU
	GridRecSlices( (plan + i), gridrec_algorithm[i] );  
	// MPI_Barrier(MPI_COMM_WORLD);
      }
    }      
    else {
      for( int i = 0; i < ndeviceCompute; i++ ){
	if( recon_algorithm == 0 ){        
	  // GridRec CPU
	  // threadID[ i ] = cutStartThread( (CUT_THREADROUTINE) GridRecThread, (void*) (plan + i) );  
	
	  // FBP CPU
	  // threadID[ i ] = cutStartThread( (CUT_THREADROUTINE) FBPThread, (void*) (plan + i) );      
	}
	else if( recon_algorithm == 1 ){   // SART GPU
	  threadID[ i ] = cutStartThread( (CUT_THREADROUTINE) SART3DThread, (void*) (plan + i) );
	}
	cutWaitForThreads( threadID, ndeviceCompute );
      }
    }

#endif 

    // cudaThreadSynchronize(); 

    CUT_SAFE_CALL(cutStopTimer(timerRecon));
    timeRecon +=  cutGetTimerValue(timerRecon);
    CUT_SAFE_CALL(cutDeleteTimer(timerRecon));

    unsigned int timerWrite = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timerWrite ) );
    CUT_SAFE_CALL( cutStartTimer( timerWrite ) );

    // retrieve the GPU calculation results sequentially
    for( int i = 0; i < ndeviceCompute; i++ ){

      for( int nz = 0; nz < plan[ i ].num_depth; nz++ ){
	for( int ny = 0; ny < plan[ i ].num_height; ny++ ){
	  for( int nx = 0; nx < plan[ i ].num_width; nx++ ){

	    index[ 0 ] = nx;
	    index[ 1 ] = ny; 
	    index[ 2 ] = i * NUM_SLICE_INV + nz;
	    ImageReconPointer->SetPixel( index, plan[ i ].h_output_data[ (nz * plan[i].num_height + ny) * plan[i].num_width + nx ]);
	  }
	}
      }

    }

    if( n_recon_image_type == RECON_IMAGE_TYPE_NRRD ){
  
      itk::ImageFileWriter<Float3DImageType>::Pointer ImageReconWriter;
      ImageReconWriter = itk::ImageFileWriter<Float3DImageType>::New();
      ImageReconWriter->SetInput( ImageReconPointer );

      string strPosix = ".nrrd"; 

      string strNrrdOutput = str_recon_image_directory;
      strNrrdOutput.append( str_recon_image_base_name );

      char buf[256];
      sprintf(buf, "_s%d_s%d", z + commRank * deviceCount * NUM_SLICE_INV, 
	      z + ( commRank + 1 ) * deviceCount * NUM_SLICE_INV - 1); 

      string str_end(buf);

      int pos = strNrrdOutput.length() - strPosix.length();
      strNrrdOutput.insert( pos,  str_end );

      ImageReconWriter->SetFileName( strNrrdOutput );
      ImageReconWriter->Update();  

      //   
      if( nReconRadiusCrop > 0 ){

	for( int nz = 0; nz < NUM_SLICE_INV; nz++ ){
	  for( int ny = 0; ny < ym; ny++ ){
	    for( int nx = 0; nx < xm; nx++ ){

	      index[ 0 ] = nx;
	      index[ 1 ] = ny;
	      index[ 2 ] = nz;

	      if( (nx - xm/2)*(nx-xm/2) + (ny - ym/2) * (ny - ym/2) >= nReconRadiusCrop * nReconRadiusCrop ){
		ImageReconPointer->SetPixel( index, 0.0f );
	      }

	    }
	  }
	}

	string strNrrdCropOutput = strNrrdOutput;
	pos = strNrrdCropOutput.length() - strPosix.length();
	strNrrdCropOutput.insert( pos,  "_crop" );

	ImageReconWriter->SetInput( ImageReconPointer );
	ImageReconWriter->SetFileName( strNrrdCropOutput );
	ImageReconWriter->Update();  
      }

    }
    else if( n_recon_image_type == RECON_IMAGE_TYPE_BIN ){

      string strPosix = string(".bin");

      Float3DImageType::PixelType pixelImg;
      PROJ_IMAGE_BIN_TYPE pixel_bin; 

      for( int nz = z + commRank * deviceCount * NUM_SLICE_INV; 
	   nz <= z + ( commRank + 1 ) * deviceCount * NUM_SLICE_INV - 1;  nz++ ){

	string strIndex = num2str( nz );
	string strCurrBinName = str_recon_image_directory;
	strCurrBinName.append( str_recon_image_base_name );

	int pos = strCurrBinName.length() - strPosix.length() - strIndex.length();
	strCurrBinName.replace(pos, strIndex.length(), strIndex); 

	string strCurrBinCropName = strCurrBinName;
	pos = strCurrBinCropName.length() - strPosix.length();
	strCurrBinCropName.insert( pos,  "_crop" );      

	fstream datafile( strCurrBinName.c_str(), ios::out | ios::binary );
	if( !datafile.is_open()  ){
	  cout << "    Error writing file " << strCurrBinName << endl;
	  continue; 
	}
	
	fstream dataCropfile;

	if( nReconRadiusCrop > 0 ){
	  dataCropfile.open( strCurrBinCropName.c_str(), ios::out | ios::binary );

	  if( !dataCropfile.is_open() ){
	    cout << "    Error writing file " << strCurrBinCropName << endl;
	    continue; 
	  }
	}

	index[2] = nz - (z + commRank * deviceCount * NUM_SLICE_INV); 

	cout << "    Writing file " << strCurrBinName << endl;  // test
	for( int y = 0; y < ym; y++ ){
	  for( int x = 0; x < xm; x++ ){  

	    index[0] = x;
	    index[1] = y;

	    pixel_bin = ImageReconPointer->GetPixel( index );
	    datafile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );

	    if( nReconRadiusCrop > 0 ){

	      if( (x - xm/2)*(x - xm/2) + (y - ym/2) * (y - ym/2) <= nReconRadiusCrop * nReconRadiusCrop ){
		dataCropfile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
	      }
	      else{
		pixel_bin = 0.0f; 
		dataCropfile.write( reinterpret_cast< char*> ( &pixel_bin ), sizeof( float ) );
	      }

	    }
	  }
	}

	datafile.close();

	if( nReconRadiusCrop > 0 ){
	  dataCropfile.close();
	}
	
      }
    }
    else if( n_recon_image_type == RECON_IMAGE_TYPE_BMP ||  n_recon_image_type == RECON_IMAGE_TYPE_TIFF){
      string strPosix;

      if( n_recon_image_type == RECON_IMAGE_TYPE_BMP )
	strPosix = string(".bmp");
    
      if( n_recon_image_type == RECON_IMAGE_TYPE_TIFF )
	strPosix = string(".tif");
    
      UChar2DImageType::Pointer ReconSlicePointer = UChar2DImageType::New();

      UChar2DImageType::SizeType size;
      UChar2DImageType::RegionType region;

      size[0] = xm;
      size[1] = ym;

      region.SetSize( size );

      ReconSlicePointer->SetRegions( region );
      ReconSlicePointer->Allocate(); 

      UChar2DImageType::Pointer ReconSliceCropPointer = NULL;
      if( nReconRadiusCrop > 0 ){
	ReconSliceCropPointer = UChar2DImageType::New();
	ReconSliceCropPointer->SetRegions( region );
	ReconSliceCropPointer->Allocate(); 
      }

      Float3DImageType::PixelType pixelImg;

      // 
      float thres_upper = -1e10, thres_lower = 1e10; 
      for( int nz = z + commRank * deviceCount * NUM_SLICE_INV; 
	   nz <=  z + ( commRank + 1 ) * deviceCount * NUM_SLICE_INV - 1; nz++ ){

	index[2] = nz - (z + commRank * deviceCount * NUM_SLICE_INV); 

	for( int y = 0; y < ym; y++ ){
	  for( int x = 0; x < xm; x++ ){  

	    index[0] = x;
	    index[1] = y;

	    pixelImg = ImageReconPointer->GetPixel( index );
	    if( thres_upper < pixelImg ){
	      thres_upper = pixelImg;
	    }
	    if( thres_lower > pixelImg ){
	      thres_lower = pixelImg;
	    }
	  }
	}
      }
    
      // 
      if( thres_upper > f_recon_threshold_upper_value )
	thres_upper = f_recon_threshold_upper_value;

      if( thres_lower < f_recon_threshold_lower_value )
	thres_lower = f_recon_threshold_lower_value;

      UChar2DImageType::PixelType pixelUShort;
      UChar2DImageType::IndexType indexUChar;

      //
      for( int nz = z + commRank * deviceCount * NUM_SLICE_INV; 
	   nz <= z + ( commRank + 1 ) * deviceCount * NUM_SLICE_INV - 1; nz++ ){

	string strIndex = num2str( nz );
	string strCurrBmpTifName = str_recon_image_directory;
	strCurrBmpTifName.append( str_recon_image_base_name) ;

	int pos = strCurrBmpTifName.length() - strPosix.length() - strIndex.length();
	strCurrBmpTifName.replace(pos, strIndex.length(), strIndex); 

	string strCurrBmpTifCropName = strCurrBmpTifName;
	pos = strCurrBmpTifCropName.length() - strPosix.length();
	strCurrBmpTifCropName.insert( pos,  "_crop" );      

	// 
	index[2] = nz - (z + commRank * deviceCount * NUM_SLICE_INV); 

	cout << "    Writing file " << strCurrBmpTifName << endl;  // test

	for( int y = 0; y < ym; y++ ){
	  for( int x = 0; x < xm; x++ ){  

	    index[0] = x;
	    index[1] = y;

	    indexUChar[0] = x;
	    indexUChar[1] = y;
	  
	    pixelImg = ImageReconPointer->GetPixel( index );
	    if( pixelImg < thres_lower )
	      pixelUShort = 0;
	    else if( pixelImg > thres_upper )
	      pixelUShort = 255;
	    else{
	      pixelUShort = (UChar2DImageType::PixelType ) 255 * (pixelImg - thres_lower) / (thres_upper - thres_lower);
	    }

	    ReconSlicePointer->SetPixel( indexUChar, pixelUShort );

	    if( nReconRadiusCrop > 0 ){

	      if( (x - xm/2)*(x - xm/2) + (y - ym/2) * (y - ym/2) <= nReconRadiusCrop * nReconRadiusCrop ){
		ReconSliceCropPointer->SetPixel( indexUChar, pixelUShort );
	      }
	      else{
		ReconSliceCropPointer->SetPixel( indexUChar, 0 );	  
	      }

	    }
	  }
	}

	itk::ImageFileWriter<UChar2DImageType>::Pointer ImageReconSliceWriter;
	ImageReconSliceWriter = itk::ImageFileWriter<UChar2DImageType>::New();
	ImageReconSliceWriter->SetInput( ReconSlicePointer );
	ImageReconSliceWriter->SetFileName( strCurrBmpTifName.c_str() );
	ImageReconSliceWriter->Update();  
    
	if( nReconRadiusCrop > 0 ){
	  itk::ImageFileWriter<UChar2DImageType>::Pointer ImageReconSliceCropWriter;
	  ImageReconSliceCropWriter = itk::ImageFileWriter<UChar2DImageType>::New();
	  ImageReconSliceCropWriter->SetInput( ReconSliceCropPointer );
	  ImageReconSliceCropWriter->SetFileName( strCurrBmpTifCropName.c_str() );
	  ImageReconSliceCropWriter->Update();  
	}

      }
    }

    if( n_recon_image_type == RECON_IMAGE_TYPE_HDF5 ){
      // retrieve the GPU calculation results sequentially
      for( int i = 0; i < ndeviceCompute; i++ ){

	for( int nz = 0; nz < plan[ i ].num_depth; nz++ ){
	  for( int ny = 0; ny < plan[ i ].num_height; ny++ ){
	    for( int nx = 0; nx < plan[ i ].num_width; nx++ ){

	      int index_obj = ((i * plan[ i ].num_depth + nz) * plan[i].num_height + ny) * plan[i].num_width + nx;
	      int index_plan = (nz * plan[i].num_height + ny) * plan[i].num_width + nx;
	      obj_slices[ index_obj ] = plan[i].h_output_data[ index_plan ];

	    }
	  }
	}

      }

      string str_recon_hdf5_name = str_recon_image_directory;
      str_recon_hdf5_name.append( str_recon_image_base_name );

      // Hdf5ParallelWriteZ(str_recon_hdf5_name.c_str(),  str_recon_hdf5_dataset_name.c_str(), z, NUM_SLICE_INV * ndeviceCompute, commRank, xm, ym, obj_slices);

      Hdf5SerialWriteZ(str_recon_hdf5_name.c_str(),  str_recon_hdf5_dataset_name.c_str(), z, NUM_SLICE_INV * ndeviceCompute, commRank, xm, ym, obj_slices);

#ifdef VERBOSE
      // cout << "Rank " << commRank << " : Reconstructed slices " << z + 1 + commRank * NUM_SLICE_INV * ndeviceCompute << " ~ ";
      cout << "Reconstructed slices " << z + 1 + commRank * NUM_SLICE_INV * ndeviceCompute << " ~ ";
      cout << z + commRank * NUM_SLICE_INV * ndeviceCompute + NUM_SLICE_INV * ndeviceCompute;
      cout << " written to hdf5 file" << endl;
#endif // VERBOSE

    }


    // MPI_Barrier(MPI_COMM_WORLD);

    // cudaThreadSynchronize(); 
    CUT_SAFE_CALL(cutStopTimer(timerWrite));
    timeParallelWrite +=  cutGetTimerValue(timerWrite);
    CUT_SAFE_CALL(cutDeleteTimer(timerWrite));

    // free the allocated memory
    for( int i = 0; i < ndeviceCompute; i++ ){
      if( plan[ i ].h_input_data && plan[ i ].num_depth > 0) {
	delete [] plan[ i ].h_input_data;
	delete [] plan[ i ].h_output_data;
	plan[ i ].h_input_data = NULL;
	plan[ i ].h_output_data = NULL;

      }
    }

    // update z;
    if( nSliceEnd + 1 - z >= commSize * deviceCount * NUM_SLICE_INV){
      z += commSize * deviceCount * NUM_SLICE_INV;
    }
    else{
      z = nSliceEnd + 1;
    }

  }

  if( recon_algorithm == 0 ){
    delete [] vect_angle;

// #ifdef MULTI_GPU_GRIDREC
    for( int i = 0; i < deviceCount; i++ ){
      gridrec_algorithm[i]->destroy(); 
      delete [] sinogram[i];
      delete [] reconstruction[i];
    }
    delete [] gridrec_algorithm;
    delete [] sinogram;
    delete [] reconstruction;

// #else
//     gridrec_algorithm->destroy(); 
// #endif

  }

  // free the allocated memory

  // if( bProjFormatHdf5 ){
  delete [] proj_darkfield; 
  delete [] proj_slices;
  delete [] proj_slices_sino;
  delete [] obj_slices;
  delete [] proj_whitefield; 
  // }

  delete [] threadID;
  delete [] plan;

  //
  delete [] mean_vect;
  delete [] low_pass_sino_lines_data;
  delete [] mean_sino_line_data; 
  delete [] data_sino;

  // MPI_CHECK(MPI_Finalize());

  CUT_SAFE_CALL(cutStopTimer(timerTotal));
  cout << "Total Time is " <<  cutGetTimerValue(timerTotal) << " (ms) " << endl;
  CUT_SAFE_CALL(cutDeleteTimer(timerTotal));

  cout << "Parallel Read Time is " <<  timeParallelRead << " (ms) " << endl;
  cout << "Parallel Recon Time is " <<  timeRecon << " (ms) " << endl;
  cout << "Parallel Write Time is " <<  timeParallelWrite << " (ms) " << endl;

  return 0;
}

// threaded multiple CPU/GPU FBP for CT recon
static CUT_THREADPROC FBPThread(ThreadGPUPlan* plan){ 

  unsigned int   num_width       = plan->num_width;
  unsigned int   num_height      = plan->num_height;  
  unsigned int   num_depth       = plan->num_depth;
  unsigned int   num_ray         = plan->num_ray;
  unsigned int   num_elevation   = plan->num_elevation;
  unsigned int   num_proj        = plan->num_proj;
  float xoffset         = plan->xoffset;
  float start_rot       = plan->start_rot;
  float end_rot         = plan->end_rot;
  float inv_rot         = plan->inv_rot;

  // FBP reconstruction
// #ifdef MULTI_GPU_GRIDREC
  float* sinogram = plan->sinogram;
  float* reconstruction = plan->reconstruction;
// #else
//   float* sinogram = (float*) new float[ num_ray * num_proj ];
//   float* reconstruction = (float*) new float[ num_ray * num_ray ];
// #endif

  float* vect_angle = (float*) new float[ num_proj ];

  int i, j, k; 
  float a, b;

#ifdef GPU_FBP

  float* d_sinogram;
  float* d_recon;
  float* d_filter_lut; 
  cudaArray* d_sino_ifft;
  cufftComplex * d_sinogram_fourier;

  // allocate GPU memory
  cudaMalloc( (void**) &d_sinogram, num_proj * num_ray * sizeof( float ) );
  CUT_CHECK_ERROR("Memory creationg failed for d_sinogram");

  cudaMalloc( (void**) &d_recon, num_ray * num_ray * sizeof( float ) );
  CUT_CHECK_ERROR("Memory creationg failed for d_recon");

  cudaMalloc( (void**) &d_filter_lut, num_ray * sizeof( float ) );
  CUT_CHECK_ERROR("Memory creationg failed for d_filter_lut");

  cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaMallocArray( (cudaArray**) &d_sino_ifft, &float1Desc, num_ray, num_proj);

  cudaMalloc( (void**) &d_sinogram_fourier, sizeof(cufftComplex) *  num_proj * (num_ray/2+1) );

  float* filter_lut = new float[ num_ray ];  // use the codes for OptimizeFBP
  for( int p = 0; p < num_ray; p++ ){
    float temp = (float) (p - (float) (num_ray/2)) / ((float) (num_ray/2));
      
    // filter_lut[ p ] = fabs( sin (PI * temp) / PI );                          // Shepp-Logan filter
    // filter_lut[ p ] = fabs( temp ) * 0.5 * (1.0f + cos( 2 * PI * temp ) );   // Hann filter
    // filter_lut[ p ] = fabs( temp ) * ( 0.54 + 0.46 * cos( 2 * PI * temp ) ); // Hamming filter
    // filter_lut[ p ] = fabs( temp );                                             // ramp filter
    filter_lut[ p ] = 1-sqrt( fabs( temp ) );                                // OptimizedFBP filter
  }

  cudaMemcpy( d_filter_lut, filter_lut, sizeof( float) * num_ray, cudaMemcpyHostToDevice );

#endif 

  // FBP reconstructs 1 slice at a time
  for( int iter = 0; iter < num_depth; iter++ ){

    // Trick: the data in vect_angle will be changed by OptimizedFBP->init().  
    //        reset after every iteration. 8/12/2011
    for( i = 0; i < num_proj; i++ )
      vect_angle[ i ] = start_rot + i * inv_rot;

    // prepare sinogram
    for( i = 0; i < num_ray * num_proj; i++ )
      sinogram[ i ] = 0.0f;
  
    for( j = 0; j < num_proj; j++ ){
      for( k = 0; k < num_ray; k++ ){
  
	float kk = k - xoffset; 
	int nkk = (int)floor(kk);

	float fInterpPixel = 0.0f;
	float fInterpWeight = 0.0f;

	if( nkk >= 0 && nkk < num_ray ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter ) * num_ray + nkk ] * (nkk + 1 - kk);
	  fInterpWeight += nkk + 1 - kk;
	}
	    
	if( nkk + 1 >= 0 && nkk + 1 < num_ray ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter ) * num_ray + nkk + 1] * (kk - nkk);
	  fInterpWeight += kk - nkk;

	}

	if( fInterpWeight < 1e-5 ){
	  fInterpPixel = 0.0f;
	}
	else{
	  fInterpPixel /= fInterpWeight; 
	}

#ifdef GPU_FBP

	if( fInterpPixel < -5.0f || fInterpPixel > 5.0f )  // From OptimizeFBP
	  fInterpPixel = 0.0f;

#endif 

	sinogram[ j * num_ray + k ] = fInterpPixel;

      }
    }

    // output sinogram for debug

    // Float3DImageType::IndexType index;
    // Float3DImageType::SizeType sizeSino;
    // Float3DImageType::RegionType regionSino;

    // Float3DImageType::Pointer imgSino = Float3DImageType::New();

    // sizeSino[0] = num_ray;
    // sizeSino[1] = num_proj;
    // sizeSino[2] = 1;

    // regionSino.SetSize( sizeSino );

    // imgSino->SetRegions( regionSino );
    // imgSino->Allocate();

    // for( int z = 0; z < sizeSino[2]; z++ ){
    //   for( int y = 0; y < sizeSino[1]; y++ ){
    // 	for( int x = 0; x < sizeSino[0]; x++ ){  

    // 	  index[0] = x;
    // 	  index[1] = y;
    // 	  index[2] = z;

    // 	  imgSino->SetPixel( index, sinogram[(z * sizeSino[1] + y) * sizeSino[0] + x] );
    // 	}
    //   }
    // }

    // itk::ImageFileWriter<Float3DImageType>::Pointer dataWriter;
    // dataWriter = itk::ImageFileWriter<Float3DImageType>::New();
    // dataWriter->SetInput( imgSino );
    // dataWriter->SetFileName( "sino.nrrd" );
    // dataWriter->Update();

    // perform FBP reconstruction

#ifdef GPU_FBP

    // transfer data from CPU to GPU
    cudaMemcpy( d_sinogram, sinogram, sizeof(float) * num_ray * num_proj, cudaMemcpyHostToDevice );
    CUT_CHECK_ERROR("Memory copy for d_sinogram failed");

    // GPU FBP reconstruction
    unsigned int timerFBP = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timerFBP ) );
    CUT_SAFE_CALL( cutStartTimer( timerFBP ) );

    fbp_wrapper(d_sinogram, d_filter_lut, d_sino_ifft, d_sinogram_fourier,
		d_recon,
		num_ray, num_proj, start_rot, inv_rot); 

    cudaThreadSynchronize(); 

    CUT_SAFE_CALL(cutStopTimer(timerFBP));
    cout << "Time for FBP is " <<  cutGetTimerValue(timerFBP) << " (ms) " << endl;
    CUT_SAFE_CALL(cutDeleteTimer(timerFBP));

    // retrieve reconstruction results
    cudaMemcpy( reconstruction, d_recon, sizeof(float) * num_ray * num_ray, cudaMemcpyDeviceToHost);
    CUT_CHECK_ERROR("Memory copy for d_sinogram failed");

#else
    // ReconAlgorithm  * recon_algorithm = new FBP ();
    OptimizedFBP  * recon_algorithm = new OptimizedFBP (); 

    recon_algorithm->setSinogramDimensions (num_ray, num_proj);
    recon_algorithm->setThetaList (vect_angle, num_proj);
    recon_algorithm->setFilter ( 3 );  // 3 for HANN. See recon_algorithm.h for definition
    recon_algorithm->init();

    for( int i = 0; i < num_ray * num_ray; i++ )
      reconstruction[ i ] = 0.0f;

    for (int loop=0; loop < recon_algorithm->numberOfSinogramsNeeded(); loop++){  

      recon_algorithm->setSinoAndReconBuffers(loop+1, &sinogram[ loop* num_proj * num_ray ],
					      &reconstruction[ loop* num_ray * num_ray] );

    }

    unsigned int timerFBP = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timerFBP ) );
    CUT_SAFE_CALL( cutStartTimer( timerFBP ) );

    recon_algorithm->reconstruct();

    cudaThreadSynchronize(); 

    CUT_SAFE_CALL(cutStopTimer(timerFBP));
    cout << "Time for FBP is " <<  cutGetTimerValue(timerFBP) << " (ms) " << endl;
    CUT_SAFE_CALL(cutDeleteTimer(timerFBP));

#endif // GPU_FBP

    // retrieve reconstruction results
    for( j = 0; j < num_height; j++ ){        // num_height = num_width for GridRec
      for( k = 0; k < num_width; k++ ){
  
	plan->h_output_data[ ( iter * num_height + j ) * num_width + k] = reconstruction[ j * num_ray + k ];

      }
    }

    // output reconstruction
    // Float3DImageType::Pointer imgRecon = Float3DImageType::New();

    // Float3DImageType::SizeType sizeRecon;
    // Float3DImageType::RegionType regionRecon;

    // sizeRecon[0] = num_width;
    // sizeRecon[1] = num_height;
    // sizeRecon[2] = 1;

    // regionRecon.SetSize( sizeRecon );

    // imgRecon->SetRegions( regionRecon );
    // imgRecon->Allocate();

    // for( int z = 0; z < sizeRecon[2]; z++ ){
    //   for( int y = 0; y < sizeRecon[1]; y++ ){
    // 	for( int x = 0; x < sizeRecon[0]; x++ ){  

    // 	  index[0] = x;
    // 	  index[1] = y;
    // 	  index[2] = z;

    // 	  imgRecon->SetPixel( index, reconstruction[ (z * num_ray + y) * num_ray + x ] );

    // 	}
    //   }
    // }

    // dataWriter->SetInput( imgRecon );
    // dataWriter->SetFileName( "recon.nrrd" );
    // dataWriter->Update();

#ifndef GPU_FBP
    recon_algorithm->destroy();  
#endif

  }

#ifdef GPU_FBP

  // free GPU memory
  cudaFree( (void*) d_sinogram );        CUT_CHECK_ERROR( "Memory free failed");
  cudaFree( (void*) d_recon );           CUT_CHECK_ERROR( "Memory free failed");
  cudaFree( (void*) d_filter_lut );     CUT_CHECK_ERROR( "Memory free failed");
  cudaFreeArray( d_sino_ifft );  CUT_CHECK_ERROR( "Memory free failed");
  cudaFree( d_sinogram_fourier );

  delete [] filter_lut; 

#endif

// #ifndef MULTI_GPU_GRIDREC
//   delete[] sinogram;
//   delete[] reconstruction;
// #endif

  delete[] vect_angle;

}

// threaded multiple CPU GridRec for CT recon
static CUT_THREADPROC GridRecThread(ThreadGPUPlan* plan){  // NEW

  unsigned int   deviceID        = plan->deviceID;
  unsigned int   num_width       = plan->num_width;
  unsigned int   num_height      = plan->num_height;  
  unsigned int   num_depth       = plan->num_depth;
  unsigned int   num_ray         = plan->num_ray;
  unsigned int   num_elevation   = plan->num_elevation;
  unsigned int   num_proj        = plan->num_proj;
  float xoffset         = plan->xoffset;
  float start_rot       = plan->start_rot;
  float end_rot         = plan->end_rot;
  float inv_rot         = plan->inv_rot;

  // GridRec reconstruction using half padding
  int num_ray_half_padding = 2 * num_ray;
  int i, j, k; 
  int offset_pad = num_ray / 2; 
  float a, b;

  // Prepare GridRec
// #ifdef MULTI_GPU_GRIDREC
  float* sinogram = plan->sinogram;
  float* reconstruction = plan->reconstruction;

  GridRec  * recon_algorithm = plan->gridrec_algorithm; 
// #else
//   float* sinogram = (float*) new float[ 2 * num_ray_half_padding * num_proj ];
//   float* reconstruction = (float*) new float[ 2 * num_ray_half_padding * num_ray_half_padding ];
//   float* vect_angle = (float*) new float[ num_proj ];

//   for( i = 0; i < num_proj; i++ )
//     vect_angle[ i ] = start_rot + i * inv_rot;

//   GridRec  * recon_algorithm = new GridRec ();

//   recon_algorithm->setSinogramDimensions (num_ray_half_padding, num_proj);
//   recon_algorithm->setThetaList (vect_angle, num_proj);
//   recon_algorithm->setFilter ( 3 );  // 3 for HANN. See recon_algorithm.h for definition
//   recon_algorithm->setGPUDeviceID( deviceID ); 
//   recon_algorithm->init();

// #endif

  cout << "GridRec initialized using GPU device " << deviceID << endl;

  for (int loop=0; loop < recon_algorithm->numberOfSinogramsNeeded(); loop++){   // num_sinograms_needed = 2;
    recon_algorithm->setSinoAndReconBuffers(loop+1, &sinogram[ loop* num_proj* num_ray_half_padding ],
					    &reconstruction[ loop* num_ray_half_padding* num_ray_half_padding] );
  }

  // GridRec reconstructs 2 slices at a time
  for( int iter = 0; iter < num_depth / 2; iter++ ){

    // prepare sinogram
  
    for( j = 0; j < num_proj; j++ ){
      for( k = 0; k < num_ray; k++ ){
  
	float kk = k - xoffset; 
	int nkk = (int)floor(kk);

	float fInterpPixel = 0.0f;
	float fInterpWeight = 0.0f;

	float fInterpPixel2= 0.0f;
	float fInterpWeight2 = 0.0f;

	if( nkk >= 0 && nkk < num_ray ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray + nkk ] * (nkk + 1 - kk);
	  fInterpWeight += nkk + 1 - kk;

	  fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1) * num_ray + nkk ] * (nkk + 1 - kk);
	  fInterpWeight2 += nkk + 1 - kk;
	}
	    
	if( nkk + 1 >= 0 && nkk + 1 < num_ray ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray + nkk + 1] * (kk - nkk);
	  fInterpWeight += kk - nkk;

	  fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1 ) * num_ray + nkk + 1] * (kk - nkk);
	  fInterpWeight2 += kk - nkk;
	}

	if( fInterpWeight < 1e-5 ){
	  fInterpPixel = 0.0f;
	}
	else{
	  fInterpPixel /= fInterpWeight; 
	}

	if( fInterpWeight2 < 1e-5 ){
	  fInterpPixel2 = 0.0f;
	}
	else{
	  fInterpPixel2 /= fInterpWeight2; 
	}

	sinogram[ j * num_ray_half_padding + k + offset_pad ] = fInterpPixel;
	sinogram[ (num_proj + j) * num_ray_half_padding + k + offset_pad ] = fInterpPixel2;

      }
    }

    // pad sinogram using boundary values instead of zero
    for( j = 0; j < num_proj; j++ ){

      for( k = 0; k < offset_pad; k++ ){

	sinogram[ j * num_ray_half_padding + k ] = sinogram[ j * num_ray_half_padding + offset_pad ] ;
	sinogram[ (num_proj + j) * num_ray_half_padding + k ] = sinogram[ (num_proj + j) * num_ray_half_padding + offset_pad ];
    
      }

      for( k = 0; k < offset_pad; k++ ){

	sinogram[ j * num_ray_half_padding + offset_pad + num_ray + k  ] = sinogram[ j * num_ray_half_padding + offset_pad + num_ray - 1 ] ;
	sinogram[ (num_proj + j) * num_ray_half_padding + offset_pad + num_ray + k ] = sinogram[ (num_proj + j) * num_ray_half_padding + offset_pad + num_ray - 1 ];
    
      }

    }

    // output sinogram for debug

    // Float3DImageType::IndexType index;
    // Float3DImageType::SizeType sizeSino;
    // Float3DImageType::RegionType regionSino;

    // Float3DImageType::Pointer imgSino = Float3DImageType::New();

    // sizeSino[0] = num_ray_half_padding;
    // sizeSino[1] = num_proj;
    // sizeSino[2] = 2;

    // regionSino.SetSize( sizeSino );

    // imgSino->SetRegions( regionSino );
    // imgSino->Allocate();

    // for( int z = 0; z < sizeSino[2]; z++ ){
    //   for( int y = 0; y < sizeSino[1]; y++ ){
    // 	for( int x = 0; x < sizeSino[0]; x++ ){  

    // 	  index[0] = x;
    // 	  index[1] = y;
    // 	  index[2] = z;

    // 	  imgSino->SetPixel( index, sinogram[(z * sizeSino[1] + y) * sizeSino[0] + x] );
    // 	}
    //   }
    // }

    // itk::ImageFileWriter<Float3DImageType>::Pointer dataWriter;
    // dataWriter = itk::ImageFileWriter<Float3DImageType>::New();
    // dataWriter->SetInput( imgSino );
    // dataWriter->SetFileName( "sino.nrrd" );
    // dataWriter->Update();

    unsigned int timerGridRec = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timerGridRec ) );
    CUT_SAFE_CALL( cutStartTimer( timerGridRec ) );

    recon_algorithm->reconstruct();

    // cudaThreadSynchronize(); 

    CUT_SAFE_CALL(cutStopTimer(timerGridRec));
    cout << "Time for GridRec is " <<  cutGetTimerValue(timerGridRec) << " (ms) " << endl;
    CUT_SAFE_CALL(cutDeleteTimer(timerGridRec));

    // retrieve reconstruction results
    for( j = 0; j < num_height; j++ ){        // num_height = num_width for GridRec
      for( k = 0; k < num_width; k++ ){
  
	plan->h_output_data[ ( (iter * 2) * num_height + j ) * num_width + k] = reconstruction[ (j + offset_pad) * num_ray_half_padding + k + offset_pad ];

	plan->h_output_data[ ( (iter * 2 + 1) * num_height + j ) * num_width + k] = reconstruction[ (num_ray_half_padding + j + offset_pad) * num_ray_half_padding + k + offset_pad ];

      }
    }

    // output reconstruction
    // Float3DImageType::Pointer imgRecon = Float3DImageType::New();

    // Float3DImageType::SizeType sizeRecon;
    // Float3DImageType::RegionType regionRecon;

    // sizeRecon[0] = num_width;
    // sizeRecon[1] = num_height;
    // sizeRecon[2] = 2;

    // regionRecon.SetSize( sizeRecon );

    // imgRecon->SetRegions( regionRecon );
    // imgRecon->Allocate();

    // for( int z = 0; z < sizeRecon[2]; z++ ){
    //   for( int y = 0; y < sizeRecon[1]; y++ ){
    // 	for( int x = 0; x < sizeRecon[0]; x++ ){  

    // 	  index[0] = x;
    // 	  index[1] = y;
    // 	  index[2] = z;

    // 	  imgRecon->SetPixel( index, reconstruction[ (z * num_ray_half_padding + y + offset_pad) * num_ray_half_padding + x + offset_pad ] );

    // 	}
    //   }
    // }

    // dataWriter->SetInput( imgRecon );
    // dataWriter->SetFileName( "recon.nrrd" );
    // dataWriter->Update();

  }

// #ifndef MULTI_GPU_GRIDREC
//   recon_algorithm->destroy();  
//   delete[] sinogram;
//   delete[] reconstruction;
//   delete[] vect_angle;
// #endif

}

// threaded multiple GPU SART for CT recon (using texture memory)

static CUT_THREADPROC SART3DThread(ThreadGPUPlan* plan){

  unsigned int   deviceID        = plan->deviceID;
  unsigned int   num_width       = plan->num_width;
  unsigned int   num_height      = plan->num_height;  
  unsigned int   num_depth       = plan->num_depth;
  unsigned int   num_ray         = plan->num_ray;
  unsigned int   num_elevation   = plan->num_elevation;
  unsigned int   num_proj        = plan->num_proj;
  unsigned int   stop_type       = plan->stop_type;
  unsigned int   iter_max        = plan->iter_max;
  float proj_thres         = plan->proj_thres;
  float diff_thres_percent = plan->diff_thres_percent;
  float diff_thres_value   = plan->diff_thres_value;
  float lambda          = plan->lambda;
  float spacing         = plan->spacing;
  float voxel_size      = plan->voxel_size;
  float proj_pixel_size = plan->proj_pixel_size;
  float SOD             = plan->SOD;
  float inv_rot         = plan->inv_rot;
  float xoffset         = plan->xoffset;
  float yoffset         = plan->yoffset;
  float zoffset         = plan->zoffset;
  float start_rot       = plan->start_rot;
  float end_rot         = plan->end_rot;

  cudaSetDevice( deviceID ); 

#ifdef OUTPUT_GPU_TIME
  unsigned int timerSART3D = 0;
  if( deviceID == 0 ){
    CUT_SAFE_CALL(cutCreateTimer(&timerSART3D));
    CUT_SAFE_CALL(cutStartTimer(timerSART3D));
  }
#endif 

  int volumeImage = num_depth * num_width * num_height;
  int volumeProj = num_proj * num_elevation * num_ray;

  unsigned int x, y, z, posInVolume;
  float intensity_init = 0.001f;

  // SART initialization
  for( z = 0; z < num_depth; z++ ){
    for( y = 0; y < num_height; y++ ){
      for( x = 0; x < num_width; x++ ){

	posInVolume = ( z * num_height + y ) * num_width + x;
	plan->h_output_data[ posInVolume] = intensity_init; 
      }
    }
  }

  // 3D SART Preparation

  int iter = 0; 

  // Allocate memory in the device for GPU implementation

  double time_gpu = 0.0;
  unsigned int timerDataPrepGPU = 0;

#ifdef OUTPUT_GPU_TIME

  if( deviceID == 0 ){
    CUT_SAFE_CALL(cutCreateTimer(&timerDataPrepGPU));
    CUT_SAFE_CALL(cutStartTimer(timerDataPrepGPU));
  }

#endif

  float* d_image_prev;
  float* d_image_curr;
  float* d_proj_cur; 

  const cudaExtent volume_size_proj = make_cudaExtent(num_ray, num_elevation, num_proj);  
  const cudaExtent volume_size_voxel = make_cudaExtent(num_width,  num_height, num_depth);  

//   cutilSafeCall( cudaMalloc( (void**)&d_image_prev, sizeof(float) * num_ray * num_elevation * num_proj ) );
//   cutilSafeCall( cudaMalloc( (void**)&d_image_curr, sizeof(float) * num_ray * num_elevation * num_proj ) );

  // Note that cudaMallocPitch() takes more time than cudaMalloc(). It is kept because it is recommended. 
  size_t pitchImage = 0;
  cudaMallocPitch((void**) &d_image_prev, &pitchImage, num_width * sizeof(float), num_height * num_depth );                  
  CUT_CHECK_ERROR("Memory creation failed");
  cudaMallocPitch((void**) &d_image_curr, &pitchImage, num_width * sizeof(float), num_height * num_depth );                  
  CUT_CHECK_ERROR("Memory creation failed");

  cudaMemcpy( d_image_prev, plan->h_output_data, sizeof(float) * volumeImage, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy for image_prev failed");
  cudaMemcpy( d_image_curr, d_image_prev, sizeof(float) * volumeImage, cudaMemcpyDeviceToDevice);
  CUT_CHECK_ERROR("Memory copy for image_curr failed");

  // cutilSafeCall( cudaMalloc( (void**)&d_proj_cur, sizeof(float) * num_ray * num_elevation * num_proj ) );

  size_t pitchProj = 0;
  cudaMallocPitch((void**) &d_proj_cur, &pitchProj, num_ray * sizeof(float), num_elevation * num_proj);               CUT_CHECK_ERROR("Memory creation failed");

  cudaArray *d_array_voxel = NULL;
  cudaChannelFormatDesc float1Desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // float
  CUDA_SAFE_CALL( cudaMalloc3DArray( &d_array_voxel, &float1Desc, volume_size_voxel ) );

  float* d_proj;

  cudaMalloc((void**) &d_proj, sizeof(float) * volumeProj);                         
  CUT_CHECK_ERROR("Memory creation failed");

  cudaMemcpy( d_proj, (float*)plan->h_input_data, sizeof(float) * volumeProj, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy for Proj failed");

  cudaArray *d_array_proj = NULL;
  CUDA_SAFE_CALL( cudaMalloc3DArray( &d_array_proj, &float1Desc, volume_size_proj ) );
  // cout << "Memcopy allocation for d_array_proj: " << cudaGetErrorString( cudaGetLastError() ) << endl;

  // allocate GPU memory for reduction
  int maxThreads = THREADS_MAX; 

  float * d_proj_sub = NULL;
  int nVolProj = num_ray * num_elevation * num_proj;

  int nThreads = (nVolProj < maxThreads * 2) ? nextPow2((nVolProj + 1)/ 2) : maxThreads;
  int nBlocks = (nVolProj + (nThreads * 2 - 1) ) / (nThreads * 2);

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_proj_sub, nBlocks* sizeof(float) ) );           
  CUT_CHECK_ERROR("Memory creation failed");
  
  float * d_image_sub = NULL;
  int nVolImg = num_width * num_height * num_depth;

  nThreads = (nVolImg < maxThreads * 2) ? nextPow2((nVolImg + 1)/ 2) : maxThreads;
  nBlocks = (nVolImg + (nThreads * 2 - 1) ) / (nThreads * 2);

  CUDA_SAFE_CALL( cudaMalloc( (void**) &d_image_sub, nBlocks* sizeof(float) ) ); 
  CUT_CHECK_ERROR("Memory creation failed");

  float * d_image_iszero = NULL;

#ifndef WEIGHT_CAL
  float* image_org = new float[ volumeImage ];

  for( x = 0; x < num_width; x++ ){
    for( y = 0; y < num_height; y++ ){
      for( z = 0; z < num_depth; z++ ){
	index[ 0 ] = x;
	index[ 1 ] = y;
	index[ 2 ] = z;

	posInVolume = ( z * num_height + y ) * num_width + x;

	image_org[ posInVolume ] = 1.0f;

      }
    }
  }

  cudaMallocPitch((void**) &d_image_iszero, &pitchImage, num_width * sizeof(float), num_height * num_depth);             
  CUT_CHECK_ERROR("Memory creation failed");

  cudaMemcpy( d_image_iszero, image_org, sizeof(float) * volumeImage, cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("Memory copy for d_image_iszero failed");

#endif  // WEIGHT_CAL

  // Perform TV3D on the result
#ifdef USE_TV

  float fidelity_weight_TV = param_fidelity_weight_TV; 
  float epsilon_TV = param_epsilon_TV; 
  int   iter_max_TV = param_iter_max_TV; 
  float thres_TV = param_thres_TV;

  int   iter_TV = param_iter_TV;
  float timestep_TV = param_timestep_TV;

#ifdef OUTPUT_GPU_TIME

  // create timers
  unsigned int timerTVCalGPU = 0;
 if( deviceID == 0 ){
   CUT_SAFE_CALL(cutCreateTimer(&timerTVCalGPU));
  
   // start timer
   CUT_SAFE_CALL(cutStartTimer(timerTVCalGPU));
 }

#endif  

  tv3d_cal_wrapper( d_image_prev, d_image_iszero, 
		    d_image_curr,  
		    num_width, num_height, num_depth, 
		    iter_TV, fidelity_weight_TV, epsilon_TV, timestep_TV );

#ifdef OUTPUT_GPU_TIME
 if( deviceID == 0 ){
   // stop timer
   cudaThreadSynchronize(); 
   CUT_SAFE_CALL(cutStopTimer(timerTVCalGPU));
   time_gpu += cutGetTimerValue(timerTVCalGPU);
   CUT_SAFE_CALL(cutDeleteTimer(timerTVCalGPU));
 }
#endif

#endif // USE_TV

  // create timer
#ifdef OUTPUT_GPU_TIME

  unsigned int timerTotal = 0;

  if( deviceID == 0 ){
    CUT_SAFE_CALL(cutCreateTimer(&timerTotal));
    CUT_SAFE_CALL(cutStartTimer(timerTotal));
  }

  double time_ProjPrep = 0.0;
  double time_ProjCal = 0.0;
  double time_BackprojPrep = 0.0;
  double time_BackprojCal = 0.0;
  double time_ImgTrans = 0.0;

#endif

  // 3D SART
  bool b_iter_max = true;
  bool b_proj_thres = true;
  bool b_diff_thres = true;

  float proj_l1;
  float diff_percent; 

  while( b_iter_max * b_proj_thres * b_diff_thres ){
    // for( int iter = 0; iter < iter_max; iter++ ){
   
#ifdef ADAPTIVE_STEP
    if( iter > 10 )
      lambda = 1.99f;
    else
      lambda = 1.0f;
#endif // ADAPTIVE_STEP
      
    // step 1: calculate the current projection
    // CUDA implementation of the Projection calculation

#ifdef OUTPUT_GPU_TIME
    // create timers
    unsigned int timerProjPrep = 0;

    if( deviceID == 0 ){
      CUT_SAFE_CALL(cutCreateTimer(&timerProjPrep));
      CUT_SAFE_CALL(cutStartTimer(timerProjPrep));
    }

#endif

    // copy volume data to 3D array
    cudaMemcpy3DParms copyParamsVoxel = {0};
    copyParamsVoxel.srcPtr   = make_cudaPitchedPtr((void*)d_image_prev,
    						   volume_size_voxel.width*sizeof(float), 
    						   volume_size_voxel.width, volume_size_voxel.height);
    copyParamsVoxel.dstArray = d_array_voxel;
    copyParamsVoxel.extent   = volume_size_voxel;
    copyParamsVoxel.kind     = cudaMemcpyDeviceToDevice;
   
    CUDA_SAFE_CALL( cudaMemcpy3D(&copyParamsVoxel) );
    // cout << "Memcopy from d_image_prev to d_array_voxel: " << cudaGetErrorString( cudaGetLastError() ) << endl;
    cudaMemset( d_proj_cur, 0, sizeof(float) * volumeProj );  // Initialization turns out to be important

#ifdef OUTPUT_GPU_TIME

    // stop timer
    if( deviceID == 0 ){

      CUT_SAFE_CALL(cutStopTimer(timerProjPrep));
      double time_tmp =  cutGetTimerValue(timerProjPrep);
      time_ProjPrep += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter + 1<< ": GPU processing time for forward projection preparation: " 
	   << time_tmp << " (ms) " << endl; 

      CUT_SAFE_CALL(cutDeleteTimer(timerProjPrep));
    }

    // create timers
    unsigned int timerProjCal = 0;

    if( deviceID == 0 ){
      CUT_SAFE_CALL(cutCreateTimer(&timerProjCal));
      CUT_SAFE_CALL(cutStartTimer(timerProjCal));
    }

#endif

    proj_cal_wrapper( d_array_voxel, d_proj,            
		      d_proj_cur, 
		      num_depth, num_height, num_width, 
		      num_proj, num_elevation, num_ray, 
		      spacing, voxel_size, proj_pixel_size, SOD, inv_rot, 
		      xoffset, yoffset, zoffset, start_rot, end_rot) ;  

#ifdef OUTPUT_GPU_TIME

    if( deviceID == 0 ){
      // stop timer
      cudaThreadSynchronize(); 
      CUT_SAFE_CALL(cutStopTimer(timerProjCal));
      double time_tmp =  cutGetTimerValue(timerProjCal);
      time_ProjCal += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter + 1<< ": GPU processing time for projection calculation: " 
	   << time_tmp << " (ms) " << endl; 

      CUT_SAFE_CALL(cutDeleteTimer(timerProjCal));
    }

#endif

    // // out d_proj_cur for debug
    // float* proj_cur = new float[ volumeProj ];
    // cudaMemcpy( proj_cur, d_proj_cur, sizeof(float) * volumeProj, cudaMemcpyDeviceToHost );

    // Float3DImageType::Pointer outProj = Float3DImageType::New();
    // Float3DImageType::RegionType regionProj;
    // Float3DImageType::SizeType sizeProj;

    // sizeProj[0] = num_ray;
    // sizeProj[1] = num_elevation;
    // sizeProj[2] = num_proj;
    
    // regionProj.SetSize( sizeProj );
    // outProj->SetRegions( regionProj );
    // outProj->Allocate( );

    // for( x = 0; x < num_ray; x++ ){
    //   for( y = 0; y < num_elevation; y++ ){
    // 	for( z = 0; z < num_proj; z++ ){

    // 	  index[ 0 ] = x;
    // 	  index[ 1 ] = y;
    // 	  index[ 2 ] = z;

    // 	  posInVolume = ( z * num_elevation + y ) * num_ray + x;

    // 	  outProj->SetPixel( index, 1000 * proj_cur[posInVolume] );
    // 	}
    //   }
    // }

    // itk::ImageFileWriter<Float3DImageType>::Pointer ImageWriter;
    // ImageWriter = itk::ImageFileWriter<Float3DImageType>::New();
    // ImageWriter->SetInput( outProj );
    // ImageWriter->SetFileName( "proj.nrrd" );
    // ImageWriter->Update(); 

    // delete [] proj_cur;

    // step 1.5: calculate the l2 difference between d_proj_cur and d_proj 
    if( stop_type == 2 && iter > 0 && iter % STOP_INV == 0 ){

      // note that the reduction results from CPU and GPU are not exactly the same (should be), 
      // but they are close enough for stopping iterations (e.g., 1080.34 CPU vs 1079.33 GPU)
      if( 65535 * THREADS_MAX * 2 >= volumeProj ){  // GPU
	proj_l1 = reduction_wrapper( d_proj_cur, d_proj_sub, 
				     num_ray, num_elevation, num_proj );

	b_proj_thres = (proj_l1 > proj_thres);
      }
      else{

	float* proj_cur = new float[ volumeProj ];
	cudaMemcpy( proj_cur, d_proj_cur, sizeof(float) * volumeProj, cudaMemcpyDeviceToHost ) ;
	CUT_CHECK_ERROR("Memory copy for proj_cur failed");

	float proj_l1_cpu = 0.0;
	for( int i = 0; i < volumeProj; i++ ){
	  proj_l1_cpu += fabs( proj_cur[ i] );

	  // if( isnan( proj_l1_cpu ) )  // for patrick data
	  //   cout << endl;
	}
	delete [] proj_cur;

	b_proj_thres = (proj_l1_cpu > proj_thres);
      }
    }
   
    // step 2:  New Implementation of SART 3D

    // GPU implementation of backprojection
#ifdef OUTPUT_GPU_TIME

    unsigned int timerBackprojPrep;
    if( deviceID == 0 ){
      CUT_SAFE_CALL(cutCreateTimer(&timerBackprojPrep));
      CUT_SAFE_CALL(cutStartTimer(timerBackprojPrep));
    }

#endif

    // Copy data from GPU global memory to GPU array directly. 

    cudaMemcpy3DParms copyParamsProj = {0};
    copyParamsProj.srcPtr   = make_cudaPitchedPtr((void*)d_proj_cur, volume_size_proj.width*sizeof(float), 
     						  volume_size_proj.width, volume_size_proj.height);
    copyParamsProj.dstArray = d_array_proj;
    copyParamsProj.extent   = volume_size_proj;
    copyParamsProj.kind     = cudaMemcpyDeviceToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&copyParamsProj) );

    // cout << "Memcopy from d_proj_cur to d_array_proj: " << cudaGetErrorString( cudaGetLastError() ) << endl;
    // the above codes may be accomplished using cudaMemcpyToArray, or cudaMemcpy3DToArray if any

    cudaMemset( d_image_curr, 0, sizeof(float) * volumeImage );  // Initialization turns out to be important

#ifdef OUTPUT_GPU_TIME

    if( deviceID == 0 ){
      CUT_SAFE_CALL(cutStopTimer(timerBackprojPrep));
      double time_tmp =  cutGetTimerValue(timerBackprojPrep); 
      time_BackprojPrep += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter + 1 << ": GPU processing time for backward projection preparation: "
	   << time_tmp << "  (ms) " << endl;

      CUT_SAFE_CALL(cutDeleteTimer(timerBackprojPrep));
    }

    unsigned int timerBackprojCal;

    if( deviceID == 0 ){
      CUT_SAFE_CALL(cutCreateTimer(&timerBackprojCal));
      CUT_SAFE_CALL(cutStartTimer(timerBackprojCal));
    }

#endif

    backproj_cal_wrapper( d_array_proj, d_image_prev, d_image_iszero, 
			  d_image_curr, 
			  num_depth, num_height, num_width,
			  num_proj, num_elevation, num_ray, 
			  lambda, voxel_size, proj_pixel_size, SOD, inv_rot, 
			  xoffset, yoffset, zoffset, start_rot, end_rot);

#ifdef OUTPUT_GPU_TIME

    if( deviceID == 0 ){
      cudaThreadSynchronize(); 
      CUT_SAFE_CALL(cutStopTimer(timerBackprojCal));
      double time_tmp =  cutGetTimerValue(timerBackprojCal); 
      time_BackprojCal += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter + 1 << ": GPU processing time for backward projection: "
	   << time_tmp << "  (ms) " << endl;

      CUT_SAFE_CALL(cutDeleteTimer(timerBackprojCal));
    }

    unsigned int timerImageTrans;

    if( deviceID == 0 ){
      CUT_SAFE_CALL(cutCreateTimer(&timerImageTrans));
      CUT_SAFE_CALL(cutStartTimer(timerImageTrans));
    }

#endif

    // step 2.5: calculate the difference between two iterations
    if( (stop_type == 3 || stop_type == 4 ) && iter > 0 && iter % STOP_INV == 0){

      if( stop_type == 3 ){
	if( 65535 * THREADS_MAX * 2 >= num_width * num_height * num_depth ){  // GPU
	  // 65535: max dimension of grid size
	  // THREADS_MAX: max number of threads per block
	  // Supports 16 slices of 2048 * 2048. Not 32. 
	  diff_percent = reduction2_wrapper( d_image_curr, d_image_prev, d_image_sub, 
					     num_width, num_height, num_depth) ;
	  diff_percent /= reduction_wrapper( d_image_prev, d_image_sub, 
					     num_width, num_height, num_depth) ;
	  b_diff_thres = (diff_percent > diff_thres_percent);

	  cout << diff_percent << " vs " << diff_thres_percent << endl;
	}
	else{ // CPU just in case

	  float* img_prev = new float[ volumeImage ];
	  float* img_curr = new float[ volumeImage ];
	  cudaMemcpy( img_prev, d_image_prev, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost ) ;
	  CUT_CHECK_ERROR("Memory copy for proj_cur failed");
	  cudaMemcpy( img_curr, d_image_curr, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost ) ;
	  CUT_CHECK_ERROR("Memory copy for proj_cur failed");

	  float img_l1_cpu = 0.0, img_diff_cpu = 0.0;
	  for( int i = 0; i < volumeImage; i++ ){
	    img_diff_cpu += fabs( img_prev[ i] - img_curr[i] );
	    img_l1_cpu += fabs( img_prev[ i] );
	  }
	  delete [] img_prev;
	  delete [] img_curr;

	  diff_percent = img_diff_cpu / img_l1_cpu;
	  b_diff_thres = (diff_percent > diff_thres_percent);

	  cout << diff_percent << " vs " << diff_thres_percent << endl;
	}
      }
      if( stop_type == 4 ){

	if( 65535 * THREADS_MAX * 2 >= num_width * num_height * num_depth ){  // GPU
	  // 65535: max dimension of grid size
	  // THREADS_MAX: max number of threads per block
	  // Supports 16 slices of 2048 * 2048. Not 32. 

	  diff_percent = reduction2_wrapper( d_image_curr, d_image_prev, d_image_sub, 
					     num_width, num_height, num_depth) ;
	  b_diff_thres = (diff_percent > diff_thres_value);
	  cout << diff_percent << " vs " << diff_thres_value << endl;
	}
	else
	{ // CPU just in case
	  float* img_prev = new float[ volumeImage ];
	  float* img_curr = new float[ volumeImage ];
	  cudaMemcpy( img_prev, d_image_prev, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost ) ;
	  CUT_CHECK_ERROR("Memory copy for proj_cur failed");
	  cudaMemcpy( img_curr, d_image_curr, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost ) ;
	  CUT_CHECK_ERROR("Memory copy for proj_cur failed");

	  float img_diff_cpu = 0.0;
	  for( int i = 0; i < volumeImage; i++ ){
	    img_diff_cpu += fabs( img_prev[ i] - img_curr[i] );
	  }
	  delete [] img_prev;
	  delete [] img_curr;

	  diff_percent = img_diff_cpu;
	  b_diff_thres = (diff_percent > diff_thres_value);

	  cout << diff_percent << " vs " << diff_thres_value << endl;
	}
      }
    }

    // prepare for next iteration
    cudaMemcpy( d_image_prev, d_image_curr, sizeof(float) * volumeImage, cudaMemcpyDeviceToDevice );	
    CUT_CHECK_ERROR("Memory copy failed");

#ifdef OUTPUT_GPU_TIME

    if( deviceID == 0 ){
      CUT_SAFE_CALL(cutStopTimer(timerImageTrans));
      double time_tmp =  cutGetTimerValue(timerImageTrans); 
      time_ImgTrans += time_tmp;
      time_gpu += time_tmp;

      cout << endl;
      cout << "    Iter " << iter + 1 << ": GPU processing time for image transfer: "
	   << time_tmp << "  (ms) " << endl;

      CUT_SAFE_CALL(cutDeleteTimer(timerImageTrans));
    }
#endif

    // Perform TV3D on the result
#ifdef USE_TV    

#ifdef OUTPUT_GPU_TIME
    unsigned int timerTVCalGPU;
    if( deviceID == 0 ){
      CUT_SAFE_CALL(cutCreateTimer(&timerTVCalGPU));
      CUT_SAFE_CALL(cutStartTimer(timerTVCalGPU));
    }

#endif

    tv3d_cal_wrapper( d_image_prev, d_image_iszero, 
		      d_image_curr,  
		      num_width, num_height, num_depth, 
		      iter_TV, fidelity_weight_TV, epsilon_TV, timestep_TV);

#ifdef OUTPUT_GPU_TIME
    if( deviceID == 0 ){
      cudaThreadSynchronize(); 
      CUT_SAFE_CALL(cutStopTimer(timerTVCalGPU));
      time_gpu += cutGetTimerValue(timerTVCalGPU); 
    
      cout << endl << "    Iter " << iter << ": GPU Total Variation Regularization time: " 
	   << cutGetTimerValue(timerTVCalGPU) << " (ms) " << endl << endl; 
      CUT_SAFE_CALL(cutDeleteTimer(timerTVCalGPU));
    }
#endif

#endif //USE_TV

    // stop criteria by number of iterations
    if( stop_type == 1 ){
      b_iter_max = (iter < iter_max);
    }
    iter++; 
  
  }

#ifdef OUTPUT_GPU_TIME
  if( deviceID == 0 ){
    // stop timer
    CUT_SAFE_CALL(cutStopTimer(timerTotal));
    cout << "    SART3D GPU total processing time: " << cutGetTimerValue(timerTotal) << " ms " << endl;;
    CUT_SAFE_CALL(cutDeleteTimer(timerTotal));
  }
#endif

  cudaMemcpy( plan->h_output_data, d_image_curr, sizeof(float) * volumeImage, cudaMemcpyDeviceToHost );

#ifdef OUTPUT_GPU_TIME
  if( deviceID == 0 ){
    cout << "    Total time for projection preparation is " << time_ProjPrep << " ms " <<  endl;
    cout << "    Total time for projection calculation is " << time_ProjCal << " ms " << endl;
    cout << "    Total time for back projection preparation is " << time_BackprojPrep << " ms " << endl;
    cout << "    Total time for back projection calculation is " << time_BackprojCal << " ms " << endl;
    cout << "    Total time for image transfer is " << time_ImgTrans << " ms " << endl;

    cout << "    SART 3D GPU Processing time: " << time_gpu << " (ms) " << endl;
  }
#endif  

  cudaFree((void*)d_image_prev);        CUT_CHECK_ERROR("Memory free failed"); 
  cudaFree((void*)d_image_curr);        CUT_CHECK_ERROR("Memory free failed"); 
  cudaFree((void*)d_proj_cur);          CUT_CHECK_ERROR("Memory free failed");

  cudaFreeArray( d_array_voxel);        CUT_CHECK_ERROR("Memory free failed");
  cudaFreeArray( d_array_proj);         CUT_CHECK_ERROR("Memory free failed");

  cudaFree((void*)d_proj);              CUT_CHECK_ERROR("Memory free failed");

  cudaFree((void*)d_image_sub);        CUT_CHECK_ERROR("Memory free failed"); 
  cudaFree((void*)d_proj_sub);        CUT_CHECK_ERROR("Memory free failed"); 

#ifndef WEIGHT_CAL
  cudaFree((void*)d_image_iszero);      CUT_CHECK_ERROR("Memory free failed");
  delete[] image_org;
#endif

#ifdef OUTPUT_GPU_TIME
  if( deviceID == 0 ){
    CUT_SAFE_CALL(cutStopTimer(timerSART3D));
    cout << endl <<  "    SART3D_GPU Total time: " 
	 << cutGetTimerValue(timerSART3D) << " (ms) " << endl; 

    CUT_SAFE_CALL(cutDeleteTimer(timerSART3D));
  }
#endif 

}

int GetSharedMemSize(int dev){

  // get the size of shared memory for each GPU block in bytes
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  return deviceProp.sharedMemPerBlock;
}

void GridRecSlices(ThreadGPUPlan* plan, GridRec* gridrec_algorithm){ 

  unsigned int   num_width       = plan->num_width;
  unsigned int   num_height      = plan->num_height;  
  unsigned int   num_depth       = plan->num_depth;
  unsigned int   num_ray         = plan->num_ray;
  unsigned int   num_elevation   = plan->num_elevation;
  unsigned int   num_proj        = plan->num_proj;
  float xoffset         = plan->xoffset;
  float start_rot       = plan->start_rot;
  float end_rot         = plan->end_rot;
  float inv_rot         = plan->inv_rot;

  // GridRec reconstruction using half padding
  int num_ray_half_padding = 2 * num_ray;
  float* sinogram = (float*) new float[ 2 * num_ray_half_padding * num_proj ];
  float* reconstruction = (float*) new float[ 2 * num_ray_half_padding * num_ray_half_padding ];

  int i, j, k; 
  int offset_pad = num_ray / 2; 
  float a, b;

  // GridRec reconstructs 2 slices at a time
  for( int iter = 0; iter < num_depth / 2; iter++ ){

    // prepare sinogram
    for( i = 0; i < 2 * num_ray_half_padding * num_proj; i++ )
      sinogram[ i ] = 0.0f;
  
    for( j = 0; j < num_proj; j++ ){
      for( k = 0; k < num_ray; k++ ){
  
	float kk = k - xoffset; 
	int nkk = (int)floor(kk);

	float fInterpPixel = 0.0f;
	float fInterpWeight = 0.0f;

	float fInterpPixel2= 0.0f;
	float fInterpWeight2 = 0.0f;

	// pad sinogram using zero
	// if( nkk >= 0 && nkk < num_ray ){
	//   fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray + nkk ] * (nkk + 1 - kk);
	//   fInterpWeight += nkk + 1 - kk;

	//   fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1) * num_ray + nkk ] * (nkk + 1 - kk);
	//   fInterpWeight2 += nkk + 1 - kk;
	// }
	    
	// if( nkk + 1 >= 0 && nkk + 1 < num_ray ){
	//   fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray + nkk + 1] * (kk - nkk);
	//   fInterpWeight += kk - nkk;

	//   fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1 ) * num_ray + nkk + 1] * (kk - nkk);
	//   fInterpWeight2 += kk - nkk;
	// }

	// if( fInterpWeight < 1e-5 || fInterpPixel < 0){
	//   fInterpPixel = 0.0f;
	// }
	// else{
	//   fInterpPixel /= fInterpWeight; 
	// }

	// if( fInterpWeight2 < 1e-5 || fInterpPixel2 < 0 ){
	//   fInterpPixel2 = 0.0f;
	// }
	// else{
	//   fInterpPixel2 /= fInterpWeight2; 
	// }

	// pad sinogram using boundary values instead of zero
	if( nkk >= 0 && nkk < num_ray ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray + nkk ] * (nkk + 1 - kk);
	  fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1) * num_ray + nkk ] * (nkk + 1 - kk);
	}
	else if( nkk < 0 ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray ] * (nkk + 1 - kk);
	  fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1) * num_ray ] * (nkk + 1 - kk);
	}
	else if( nkk >= num_ray ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray + num_ray - 1] * (nkk + 1 - kk);
	  fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1) * num_ray + num_ray - 1 ] * (nkk + 1 - kk);
	}
	 
	// 
	if( nkk + 1 >= 0 && nkk + 1 < num_ray ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray + nkk + 1] * (kk - nkk);
	  fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1 ) * num_ray + nkk + 1] * (kk - nkk);
	}
	else if( nkk + 1 < 0 ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray ] * (kk - nkk);
	  fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1 ) * num_ray ] * (kk - nkk);
	}
	else if( nkk + 1 >= num_ray ){
	  fInterpPixel += plan->h_input_data[ ( j * num_elevation + iter * 2 ) * num_ray + num_ray - 1] * (kk - nkk);
	  fInterpPixel2 += plan->h_input_data[ ( j * num_elevation + iter * 2 + 1 ) * num_ray + num_ray - 1] * (kk - nkk);
	}

	sinogram[ j * num_ray_half_padding + k + offset_pad ] = fInterpPixel;
	sinogram[ (num_proj + j) * num_ray_half_padding + k + offset_pad ] = fInterpPixel2;
      }
    }

    // pad sinogram using boundary values instead of zero
    for( j = 0; j < num_proj; j++ ){

      for( k = 0; k < offset_pad; k++ ){

	// for( k = offset_pad/2; k < offset_pad; k++ ){  // artifacts free
	// for( k = offset_pad * 3/4; k < offset_pad; k++ ){  // artifact left over

	sinogram[ j * num_ray_half_padding + k ] = sinogram[ j * num_ray_half_padding + offset_pad ] ;
	sinogram[ (num_proj + j) * num_ray_half_padding + k ] = sinogram[ (num_proj + j) * num_ray_half_padding + offset_pad ];
    
      }

      for( k = 0; k < offset_pad; k++ ){

	// for( k = 0; k < offset_pad / 2; k++ ){  // artifacts free
	// for( k = 0; k < offset_pad / 4; k++ ){  // artifacts left over

	sinogram[ j * num_ray_half_padding + offset_pad + num_ray + k  ] = sinogram[ j * num_ray_half_padding + offset_pad + num_ray - 1 ] ;
	sinogram[ (num_proj + j) * num_ray_half_padding + offset_pad + num_ray + k ] = sinogram[ (num_proj + j) * num_ray_half_padding + offset_pad + num_ray - 1 ];
    
      }

    }

    // output sinogram for debug

    // Float3DImageType::IndexType index;
    // Float3DImageType::SizeType sizeSino;
    // Float3DImageType::RegionType regionSino;

    // Float3DImageType::Pointer imgSino = Float3DImageType::New();

    // sizeSino[0] = num_ray_half_padding;
    // sizeSino[1] = num_proj;
    // sizeSino[2] = 2;

    // regionSino.SetSize( sizeSino );

    // imgSino->SetRegions( regionSino );
    // imgSino->Allocate();

    // for( int z = 0; z < sizeSino[2]; z++ ){
    //   for( int y = 0; y < sizeSino[1]; y++ ){
    // 	for( int x = 0; x < sizeSino[0]; x++ ){  

    // 	  index[0] = x;
    // 	  index[1] = y;
    // 	  index[2] = z;

    // 	  imgSino->SetPixel( index, sinogram[(z * sizeSino[1] + y) * sizeSino[0] + x] );
    // 	}
    //   }
    // }

    // itk::ImageFileWriter<Float3DImageType>::Pointer dataWriter;
    // dataWriter = itk::ImageFileWriter<Float3DImageType>::New();
    // dataWriter->SetInput( imgSino );
    // dataWriter->SetFileName( "sino.nrrd" );
    // dataWriter->Update();

    // perform GridRec reconstruction
    for (int loop=0; loop < gridrec_algorithm->numberOfSinogramsNeeded(); loop++){   // num_sinograms_needed = 2;

      gridrec_algorithm->setSinoAndReconBuffers(loop+1, &sinogram[ loop* num_proj* num_ray_half_padding ],
					      &reconstruction[ loop* num_ray_half_padding* num_ray_half_padding] );
    }

    gridrec_algorithm->reconstruct();

    // retrieve reconstruction results
    for( j = 0; j < num_height; j++ ){        // num_height = num_width for GridRec
      for( k = 0; k < num_width; k++ ){
  
	plan->h_output_data[ ( (iter * 2) * num_height + j ) * num_width + k] = reconstruction[ (j + offset_pad) * num_ray_half_padding + k + offset_pad ];

	plan->h_output_data[ ( (iter * 2 + 1) * num_height + j ) * num_width + k] = reconstruction[ (num_ray_half_padding + j + offset_pad) * num_ray_half_padding + k + offset_pad ];

      }
    }

    // output reconstruction

    // Float3DImageType::Pointer imgRecon = Float3DImageType::New();
    // Float3DImageType::SizeType sizeRecon;
    // Float3DImageType::RegionType regionRecon;

    // sizeRecon[0] = num_width;
    // sizeRecon[1] = num_height;
    // sizeRecon[2] = 2;

    // regionRecon.SetSize( sizeRecon );

    // imgRecon->SetRegions( regionRecon );
    // imgRecon->Allocate();

    // for( int z = 0; z < sizeRecon[2]; z++ ){
    //   for( int y = 0; y < sizeRecon[1]; y++ ){
    // 	for( int x = 0; x < sizeRecon[0]; x++ ){  

    // 	  index[0] = x;
    // 	  index[1] = y;
    // 	  index[2] = z;

    // 	  imgRecon->SetPixel( index, reconstruction[ (z * num_ray_half_padding + y + offset_pad) * num_ray_half_padding + x + offset_pad ] );

    // 	}
    //   }
    // }

    // dataWriter->SetInput( imgRecon );
    // dataWriter->SetFileName( "recon.nrrd" );
    // dataWriter->Update();

  }

  delete[] sinogram;
  delete[] reconstruction;

}

void RingCorrectionSinogram (float *data, float ring_coeff, int um, int num_proj,
			   float *mean_vect, float* mean_sino_line_data,
			   float *low_pass_sino_lines_data) 
{ 
  // ring artifact removal from Brian Tieman's MPI cluster codes (RingCorrectionSingle)
  int         i, j, m; 
  float       mean_total; 
  float       tmp; 
 
  for (m=0;m<20;m++) {
    
    // normalization of each projection: mean values estimation 
    for (i=0;i<num_proj;i++) 
      mean_vect[i] = 0.0; 

    mean_total = 0.0; 
 
    for (i=0;i<num_proj;i++) {
        
      for (j=0;j<um;j++) {
	mean_vect[i] += data[i*um+j]; 
      }

      mean_vect[i] /= um; 
      mean_total += mean_vect[i]; 
    } 
    mean_total /= num_proj; 
 
    // renormalization of each projection to the global mean 
    for (i=0;i<num_proj;i++) {
      for (j=0;j<um;j++) {
	if (mean_vect[i] != 0.0) {
	  data[i*um+j] = data[i*um+j]*mean_total/mean_vect[i];        // ring filtering: sum of projection and low-pass filter of the result 
 
	}
      }
    }

    for (i=0;i<um;i++) 
      mean_sino_line_data[i] = 0.0; 
 
    for (i=0;i<num_proj;i++) 
      for (j=0;j<um;j++) 
	mean_sino_line_data[j] += data[i*um+j]; 
 
    for (i=0;i<um;i++) 
      mean_sino_line_data[i] /= num_proj; 
 
    for (j=1;j<um-1;j++) {
      low_pass_sino_lines_data[j] = (mean_sino_line_data[j-1]+mean_sino_line_data[j]+mean_sino_line_data[j+1])/3.0; 
    }

    low_pass_sino_lines_data[0] = mean_sino_line_data[0]; 
    low_pass_sino_lines_data[um-1] = mean_sino_line_data[um-1]; 
 
    // ring corrections 
    for (i=0;i<num_proj;i++) {
      for (j=0;j<um;j++) { 
	tmp = mean_sino_line_data[j]-low_pass_sino_lines_data[j]; 
	if ((data[i*um+j] - (tmp * ring_coeff) ) > 0.0) 
	  data[i*um+j] -= (tmp * ring_coeff); 
	else 
	  data[i*um+j] = 0.0; 
      } 
    } 
  }
 
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


#define BLOCK_2DIM_X 16 // 32
#define BLOCK_2DIM_Y 16 // 32
#define BLOCK_2DIM_Z 1

#define BLOCK_3DIM_X 8  
#define BLOCK_3DIM_Y 8
#define BLOCK_3DIM_Z 4 // 8 

#define PI 3.1415926f

#define MAX_NUM_GPU     32
#define NUM_SLICE_INV   2 // 8     // number of slices for each GPU. 8 maximum for 2048*2048. 
                             // 3 for Peter's data  
                             // 96 for stu data (256)

#define ROTATION_CLOCKWISE  // default. otherwise counterclockwise  

// #define WEIGHT_CAL          // define it for PROJ_DATA
                               // undefine it for steve data
                               // undefine it for Peter data

#define STEVE_DATA             // define it for steve data
// #define PETER_DATA             // define it for Peter data

// #define COMPOSITE_SHARE_MEM // remove texture memory usage to avoid data transfer
                               // between GPU global memory and texture memory
                               // use shared memory by compositing instead.

                               // It turns out that texture memory is very helpful here. 
                               // Utilization of shared memory slows down the computation
                               // because of data loading and 3D interpolation. Y. P. 4/12/2011

// #define NOISE_POISSON_PROJ
// #define poisson_mag 1  

// #define USE_TV

// #define OUTPUT_INTERMEDIATE
// #define OUTPUT_FILE_INV 10

// #define OUTPUT_ERRFILE
// #define OUTPUT_CONTRAST

#define OUTPUT_GPU_TIME

#define STOP_INV  1            // number of iterations for stopping criteria

#define THREADS_MAX 1024

#define INTENSITY_INIT 0.001f    // initialization. default: 0.001f for high resolution recon
                                 // 1.0f for Dr. Luhong Wang's data
#define RING_COEFF 1.0

// #define INPUT_TXT             // define it to read parameters from txt files
                                 // default: undefine it to use xml file

enum { PROJ_IMAGE_TYPE_NRRD, PROJ_IMAGE_TYPE_BIN, PROJ_IMAGE_TYPE_BMP, PROJ_IMAGE_TYPE_TIFF, PROJ_IMAGE_TYPE_HDF5 };

enum { RECON_IMAGE_TYPE_NRRD, RECON_IMAGE_TYPE_BIN, RECON_IMAGE_TYPE_BMP, RECON_IMAGE_TYPE_TIFF, RECON_IMAGE_TYPE_HDF5 };

typedef float PROJ_IMAGE_BIN_TYPE;

typedef unsigned short HDF5_DATATYPE;

typedef struct{ 
  // device ID
  int deviceID;
  int deviceShareMemSizePerBlock;

  // Host-side input (projection) data
  float* h_input_data;

  // Host-side output (object) data 
  float* h_output_data;

  // Host-side mask data
  float* h_proj_mask_data;
  int    n_proj_mask_image;

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
}ThreadGPUPlan;


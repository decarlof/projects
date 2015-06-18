
#define BLOCK_2DIM_X 32
#define BLOCK_2DIM_Y 32
#define BLOCK_2DIM_Z 1

#define BLOCK_3DIM_X 8   // 8-4-8 for parallel beam. 8-8-4 for others
#define BLOCK_3DIM_Y 8
#define BLOCK_3DIM_Z 8   

#define PI 3.1415926f

#define MAX_NUM_GPU     32

#define NUM_SLICE_INV   2   // 16    // # of slices for each GPU in SART
                               // NUM_SLICE_INV should be multiples of 2 for GridRec. 
                               // NUM_SLICE_INV should be 1 for FBP. I do not know why yet. 8/11/2011

// #define STEVE_DATA

// #define PHASE_CONTRAST_DATA   // enlarge data by 1e5 for phase contrast data

// #define GPU_FBP

// #define OUTPUT_NRRD      // Recon data output file format. choose one. No default. 
// #define OUTPUT_HDF5         // Use this for magellan version

// #define HDF5_PARALLEL_READ     // Using parallel hdf5. Requires parallel file system (GPFS, PVFS, Lustre)  
// #define HDF5_SERIAL_READ

// test H5Pset_alignment / H5Pget_alignment

#define ROTATION_CLOCKWISE  // default. otherwise counterclockwise  

#define WEIGHT_CAL          // define it for PROJ_DATA

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

#define VERBOSE

#define RING_COEFF 1.0

// #define MULTI_GPU_GRIDREC   // Implementation for Multi-GPU GRIDREC. Unfortunately, GPU GRIDREC
                               // requires cufft, which is not thread-safe yet for CUDA 4.0. Try later. 9/30/2011

enum { PROJ_IMAGE_TYPE_NRRD, PROJ_IMAGE_TYPE_BIN, PROJ_IMAGE_TYPE_BMP, PROJ_IMAGE_TYPE_TIF, PROJ_IMAGE_TYPE_HDF5 };

enum { RECON_IMAGE_TYPE_NRRD, RECON_IMAGE_TYPE_BIN, RECON_IMAGE_TYPE_BMP, RECON_IMAGE_TYPE_TIFF, RECON_IMAGE_TYPE_HDF5 };

typedef float PROJ_IMAGE_BIN_TYPE;

typedef unsigned short HDF5_DATATYPE;



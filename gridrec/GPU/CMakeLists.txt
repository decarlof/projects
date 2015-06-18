cmake_minimum_required(VERSION 2.4)

PROJECT(tomo_recon)

# set the version number
SET (FIND_SHIFT_VERSION_MAJOR 1)
SET (FIND_SHIFT_VERSION_MINOR 0)

# configure a header file to pass some of the CMake settings to 
# the source code
CONFIGURE_FILE( 
   "${PROJECT_SOURCE_DIR}/tomo_recon_config.h.in"
   "${PROJECT_BINARY_DIR}/tomo_recon_config.h"
   )

INCLUDE_DIRECTORIES (${PROJECT_BINARY_DIR})

# choose CPU/GPU GRIDREC. choose GRIDREC/SART in param.xml file
OPTION (USE_GRIDREC_GPU       "Use GPU GRIDREC for recon"  ON)

#set the build type (Debug/Release)
# SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)

#
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)
CUDA_INCLUDE_DIRECTORIES (${PROJECT_BINARY_DIR})

# parallel hdf5
SET(CMAKE_C_COMPILER /clhome/TOMO/PANY/utilities/mpich/bin/mpicc)
SET(CMAKE_C_FLAGS "-DH5_USE_16_API")

# CUDA_INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5_serial/include)  # serial hdf5
CUDA_INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5/include)           # parallel hdf5

CUDA_INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/mpich/include)
CUDA_INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/zlib/include)

# INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5_serial/include)
INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5/include)

INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/mpich/include)
INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/zlib/include)

# LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5_serial/lib)
LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5/lib)

LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/mpich/lib)
LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/zlib/lib)

INCLUDE_DIRECTORIES( /clhome/aps_tools/shared/include/ )
LINK_DIRECTORIES( /clhome/aps_tools/shared/lib )

SET(HDF5_LIB hdf5 hdf5_hl z)  
SET(MPI_LIB mpich mpl)
SET(CUDA_LIB cufft)
SET( FFT_LIBS libfftw3f.a )

# SET(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
# FIND_PACKAGE(CUDA 3.2 REQUIRED)
# IF(CUDA_FOUND)
#   SET(CUDA_NVCC_FLAGS -g;-G;-gencode "arch=compute_10, code=sm_10")
# ENDIF(CUDA_FOUND)

# Find ITK.
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)

SET( ITK_LIBS ITKCommon ITKIO libm.a )

# Add current directory to the nvcc include line.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE}
)

INCLUDE_DIRECTORIES(   
  ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE}
)

# Use gridrec_gpu_phase2.cu
CUDA_ADD_EXECUTABLE(tomo_recon
   tomo_recon.cpp proj_cal.cu backproj_cal.cu  fbp.cu len_weight_cal.cu tv3d_cal.cu reduction.cu gridrec.cu
   Hdf5SerialRead.c Hdf5ParallelRead.c Hdf5ParallelWrite.c fft_fftw.c recon_algorithm.cpp filteredbackprojection.cpp
   tinystr.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp Hdf5SerialWrite.c
   # gridrec.cpp
   # fft_nr.cpp
)

TARGET_LINK_LIBRARIES(
  tomo_recon ${CUDA_TARGET_LINK} ${CUDA_CUT_TARGET_LINK} ${ITK_LIBS} ${HDF5_LIB} ${MPI_LIB} ${CUDA_LIB} ${FFT_LIBS}
)

# Use gridrec_gpu.cu (in development)
CUDA_ADD_EXECUTABLE(tomo_recon_dev 
   tomo_recon.cpp proj_cal.cu backproj_cal.cu  fbp.cu len_weight_cal.cu tv3d_cal.cu reduction.cu gridrec_gpu.cu
   Hdf5SerialRead.c Hdf5ParallelRead.c Hdf5ParallelWrite.c fft_fftw.c recon_algorithm.cpp filteredbackprojection.cpp
   tinystr.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp Hdf5SerialWrite.c
   # gridrec.cpp
   # fft_nr.cpp
)

TARGET_LINK_LIBRARIES(
  tomo_recon_dev ${CUDA_TARGET_LINK} ${CUDA_CUT_TARGET_LINK} ${ITK_LIBS} ${HDF5_LIB} ${MPI_LIB} ${CUDA_LIB} ${FFT_LIBS}
)





  




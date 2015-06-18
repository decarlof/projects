cmake_minimum_required(VERSION 2.4)

PROJECT(EM)

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

# SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)

# Serial hdf5
# SET(CMAKE_C_COMPILER /clhome/TOMO/PANY/utilities/mpich/bin/mpicc)
SET(CMAKE_C_FLAGS "-DH5_USE_16_API")

CUDA_INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5_serial/include)
CUDA_INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/mpich/include)
CUDA_INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/zlib/include)

INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5_serial/include)
INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/mpich/include)
INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/zlib/include)

LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5_serial/lib)
LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/mpich/lib)
LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/zlib/lib)

SET(HDF5_LIB hdf5 hdf5_hl z)  

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

# Use one executable only.
CUDA_ADD_EXECUTABLE(EM 
   ParallelBeamEM.cpp
   proj_cal.cu 
   backproj_cal.cu 
   len_weight_cal.cu 
   tv3d_cal.cu
   reduction.cu
   Hdf5SerialRead.c
   Hdf5SerialWrite.c
   Hdf5ParallelWrite.c
   tinystr.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp
)

TARGET_LINK_LIBRARIES(
  EM
  ${CUDA_TARGET_LINK}  
  ${CUDA_CUT_TARGET_LINK}
  ${ITK_LIBS}
  ${HDF5_LIB}
)





  




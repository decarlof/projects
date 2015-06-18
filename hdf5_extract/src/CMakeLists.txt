cmake_minimum_required(VERSION 2.4)

PROJECT(SART)

SET(CMAKE_BUILD_TYPE Debug)

# SET(CMAKE_BUILD_TYPE Release)

# Serial hdf5
INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5_serial/include)
INCLUDE_DIRECTORIES(/clhome/TOMO/PANY/utilities/zlib/include)

LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/hdf5_serial/lib)
LINK_DIRECTORIES(/clhome/TOMO/PANY/utilities/zlib/lib)

SET(HDF5_LIB hdf5 hdf5_hl z)  

# Find ITK.
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)

SET( ITK_LIBS ITKCommon ITKIO libm.a )

# Use one executable only.
ADD_EXECUTABLE(hdf5_extract
   hdf5_extract.cpp
   Hdf5SerialRead.c
)

TARGET_LINK_LIBRARIES(
  hdf5_extract
  ${ITK_LIBS}
  ${HDF5_LIB}
)





  




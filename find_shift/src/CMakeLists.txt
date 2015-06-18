CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT (find_shift)

# set the version number
SET (FIND_SHIFT_VERSION_MAJOR 1)
SET (FIND_SHIFT_VERSION_MINOR 0)

# configure a header file to pass some of the CMake settings to 
# the source code
CONFIGURE_FILE( 
   "${PROJECT_SOURCE_DIR}/find_shift_config.h.in"
   "${PROJECT_BINARY_DIR}/find_shift_config.h"
   )

INCLUDE_DIRECTORIES (${PROJECT_BINARY_DIR})

# choose FFTW3/BRUCE_FORCE/BRUTE_FORCE_GPU
OPTION (USE_FFTW3           "Use FFTW3 library for implementation" OFF)
OPTION (USE_BRUTE_FORCE_CPU "Use GPU brute force for computation"  OFF)
OPTION (USE_BRUTE_FORCE_GPU "Use GPU brute force for computation"  ON)
OPTION (USE_ENTROPY_METRIC  "Use entropy metric for fine tuning"   OFF)
OPTION (USE_VERBOSE         "Use verbose mode"                     OFF)

IF (USE_FFTW3)
   IF (USE_BRUTE_FORCE_GPU)
      MESSAGE( "USE_BRUTE_FORCE_GPU can not be combined with USE_FFTW3! Please disable one!" )
   ENDIF (USE_BRUTE_FORCE_GPU)
   IF (USE_BRUTE_FORCE_CPU)
      MESSAGE( "USE_BRUTE_FORCE_CPU can not be combined with USE_FFTW3! Please disable one!" )
   ENDIF (USE_BRUTE_FORCE_CPU)
ENDIF (USE_FFTW3)

IF (USE_BRUTE_FORCE_GPU)
   IF (USE_BRUTE_FORCE_CPU)
      MESSAGE( "USE_BRUTE_FORCE_GPU can not be combined with USE_BRUTE_FORCE_CPU! Please disable one!" )
   ENDIF (USE_BRUTE_FORCE_CPU)
ENDIF (USE_BRUTE_FORCE_GPU)

# ITK
# SET( ITK_DIR /clhome/TOMO/PANY/utilities/ITK/ITKBINARY)
# SET( ITK_DIR /clhome/aps_tools/ITK-build )

FIND_PACKAGE(ITK)
IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR "Cannot build without ITK.  Please set ITK_DIR.")
ENDIF(ITK_FOUND)


# set the build type (Debug/Release)
SET (CMAKE_BUILD_TYPE Debug) 
# SET (CMAKE_BUILD_TYPE Release) 

# set special flags
SET (HDF_FLAGS -D__unix__ -DHDF4 -DHDF5)

SET (CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

# define variables for root directories
SET (HDF_ROOT	      /clhome/TOMO/PANY/utilities/hdf4)
SET (HDF5_ROOT	      /clhome/TOMO/PANY/utilities/hdf5_serial)
SET (NAPI_ROOT        /clhome/TOMO/PANY/utilities/nexus/)
SET (XML_ROOT	      /clhome/TOMO/PANY/utilities/mxml)
SET (TEEM_ROOT        /clhome/TOMO/PANY/utilities/teem/teem-1.10.0-src)
SET (FFTW_ROOT        /clhome/TOMO/PANY/utilities/fftw)

SET (NEXUSLIB_ROOT    /clhome/TOMO/PANY/utilities/NexusLibrary)
SET (UTILITY_ROOT     /clhome/TOMO/PANY/utilities/NexusLibrary/workspace/utility)

SET(CUDA_LIB cufft)

# include directories
INCLUDE_DIRECTORIES (
                     ${HDF_ROOT}/include
                     ${HDF5_ROOT}/include
                     ${XML_ROOT}/include
                     ${TEEM_ROOT}/include
                     ${FFTW_ROOT}/include
                     ${NAPI_ROOT}/include
                     ${PROJECT_SOURCE_DIR}/../include
                     )

# library directories
LINK_DIRECTORIES    (${HDF_ROOT}/lib
                     ${HDF5_ROOT}/lib
                     ${NAPI_ROOT}/lib
                     ${XML_ROOT}/lib
                     ${TEEM_ROOT}/lib
                     ${FFTW_ROOT}/lib
                     /usr/lib64
                     )

# define variables for libraries
SET (HDF_LIBS   mfhdf  df  jpeg  z)
SET (HDF5_LIBS  hdf5  hdf5_hl)
SET (OTHER_LIBS	pthread  m)
SET (NAPI_LIBS  NeXus NeXusCPP)
SET (XML_LIBS   mxml)
SET (TEEM_LIBS  teem  bz2  png)
SET (FFTW_LIBS  fftw3)

SET (NEXUS_LIBS	nexuslibrary)
SET (LOG_LIBS   logclass)

INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/workspace/utility)
ADD_SUBDIRECTORY (workspace/utility)

INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/nexus_lib)
ADD_SUBDIRECTORY (nexus_lib)

LINK_DIRECTORIES (${PROJECT_SOURCE_DIR}/../build/workspace/utility)   # -llogclass
LINK_DIRECTORIES (${PROJECT_SOURCE_DIR}/../build/nexus_lib)           # -lnexuslibrary
INCLUDE_DIRECTORIES (${PROJECT_SOURCE_DIR}/main)
ADD_SUBDIRECTORY (main)

# testing
ENABLE_TESTING ()
ADD_TEST (find_shift_sam02 ./main/find_shift ../test/Sam02/ Sam02_exp.hdf)
SET_TESTS_PROPERTIES (find_shift_sam02
         PROPERTIES
         PASS_REGULAR_EXPRESSION "-13.3"
         )

# packaging
INCLUDE (InstallRequiredSystemLibraries)
INCLUDE (CPack)


# Note: Use ${ITK_LIBRARIES} for ITK 4 rather than separate ITK libraries. 

CMAKE_MINIMUM_REQUIRED (VERSION 2.8)
PROJECT ( Tomo2DataExchange )

# set the build type (Debug/Release)
# SET (CMAKE_BUILD_TYPE Debug) 
SET (CMAKE_BUILD_TYPE Release) 

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
# ADD_TEST (find_shift_sam02 ./main/find_shift ../test/Sam02/ Sam02_exp.hdf)
# SET_TESTS_PROPERTIES (find_shift_sam02
#          PROPERTIES
#          PASS_REGULAR_EXPRESSION "-13.25"
#          )

# packaging
# INCLUDE (InstallRequiredSystemLibraries)
# INCLUDE (CPack)


# build executable

IF (USE_BRUTE_FORCE_GPU)

   INCLUDE (${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

   CUDA_INCLUDE_DIRECTORIES (
                             ${HDF_ROOT}/include
                             ${HDF5_ROOT}/include
                             ${XML_ROOT}/include
                             ${TEEM_ROOT}/include
                             ${FFTW_ROOT}/include
                             ${NAPI_ROOT}/include
                             ${PROJECT_SOURCE_DIR}/../include
                             )

   INCLUDE_DIRECTORIES(
         ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE}
         )

   LINK_DIRECTORIES( /opt/cuda/lib64 )

   CUDA_INCLUDE_DIRECTORIES(
         ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE} ${PROJECT_BINARY_DIR}
         )

   CUDA_ADD_EXECUTABLE (find_shift_hdf4 
	                find_shift_hdf4.cpp cross_correlation.cu
                        )

   TARGET_LINK_LIBRARIES (find_shift_hdf4
                   ${HDF_LIBS} ${HDF5_LIBS} ${OTHER_LIBS} ${NEXUS_LIBS} ${ITK_LIBRARIES}
                   ${XML_LIBS} ${LOG_LIBS}  ${TEEM_LIBS}  ${FFTW_LIBS} ${NAPI_LIBS} )
  

   CUDA_ADD_EXECUTABLE (find_shift 
	                find_shift.cpp cross_correlation.cu gridrec.cu fft_nr.cpp recon_algorithm.cpp 
			Hdf5SerialRead.c
                        tinystr.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp
                        )

   TARGET_LINK_LIBRARIES (find_shift
                   ${CUDA_TARGET_LINK} ${CUDA_CUT_TARGET_LINK} ${ITK_LIBRARIES}
                   ${HDF_LIBS} ${HDF5_LIBS} ${OTHER_LIBS} ${NEXUS_LIBS} ${CUDA_LIB}
                   ${XML_LIBS} ${LOG_LIBS}  ${TEEM_LIBS}  ${FFTW_LIBS} ${NAPI_LIBS} )

ELSE (USE_BRUTE_FORCE_GPU)

   ADD_EXECUTABLE (find_shift_hdf4 
	           find_shift_hdf4.cpp
                   )

   TARGET_LINK_LIBRARIES (find_shift_hdf4
                   ${HDF_LIBS} ${HDF5_LIBS} ${OTHER_LIBS} ${NEXUS_LIBS}
                   ${XML_LIBS} ${LOG_LIBS}  ${TEEM_LIBS}  ${FFTW_LIBS} ${NAPI_LIBS} )

ENDIF (USE_BRUTE_FORCE_GPU)


# install executable
INSTALL (TARGETS find_shift_hdf4 DESTINATION ../bin)



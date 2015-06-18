# ankaphase_hdf4

ADD_EXECUTABLE( ankaphase_hdf4  ankaphase_hdf4.cpp tinystr.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp )

TARGET_LINK_LIBRARIES ( ankaphase_hdf4 
                      ${HDF_LIBS}   ${HDF5_LIBS}
                      ${OTHER_LIBS} ${NEXUS_LIBS}
                      ${XML_LIBS}   ${LOG_LIBS} 
		      ${ITK_LIBS}   ${FFTW_LIBS}
		      ${NAPI_LIBS}  )

INSTALL( TARGETS ankaphase_hdf4  DESTINATION bin )

# ankaphase: supports hdf5 in data exchange format

IF (USE_FFT_GPU)

   INCLUDE (${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

   SET(CUDA_FFT_LIB cufft)	
   LINK_DIRECTORIES (/opt/cuda/lib64)   #libcufft

   CUDA_INCLUDE_DIRECTORIES (
                             ${HDF_ROOT}/include
                             ${HDF5_ROOT}/include
                             ${NAPI_ROOT}/include
                             ${XML_ROOT}/include
                             ${TEEM_ROOT}/include
                             ${FFTW_ROOT}/include
                             ${PROJECT_SOURCE_DIR}/../include
                             ${PROJECT_BINARY_DIR}
                             )

   INCLUDE_DIRECTORIES(
         ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE}
         )

   CUDA_INCLUDE_DIRECTORIES(
         ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE} ${CUDA_CUT_INCLUDE}
         )

   CUDA_ADD_EXECUTABLE (ankaphase  ankaphase.cpp  freq_mul.cu Hdf5SerialRead.c 
   		        tinystr.cpp  tinyxml.cpp  tinyxmlerror.cpp  tinyxmlparser.cpp
                        )

   TARGET_LINK_LIBRARIES ( ankaphase
                      ${HDF_LIBS}   ${HDF5_LIBS}
                      ${OTHER_LIBS} ${NEXUS_LIBS}
                      ${XML_LIBS}   ${LOG_LIBS} 
		      ${ITK_LIBS}   ${FFTW_LIBS}
		      ${NAPI_LIBS}  ${CUDA_FFT_LIB})


ELSE (USE_FFT_GPU)

    ADD_EXECUTABLE( ankaphase  ankaphase.cpp  Hdf5SerialRead.c
  		    tinystr.cpp  tinyxml.cpp  tinyxmlerror.cpp  tinyxmlparser.cpp )

    TARGET_LINK_LIBRARIES ( ankaphase
                      ${HDF_LIBS}   ${HDF5_LIBS}
                      ${OTHER_LIBS} ${NEXUS_LIBS}
                      ${XML_LIBS}   ${LOG_LIBS} 
		      ${ITK_LIBS}   ${FFTW_LIBS}
		      ${NAPI_LIBS}  )


ENDIF (USE_FFT_GPU)



INSTALL( TARGETS ankaphase  DESTINATION bin )

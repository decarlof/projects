# build executable

ADD_EXECUTABLE (Tomo2DataExchange_v2 Tomo2DataExchange.cpp )

TARGET_LINK_LIBRARIES (Tomo2DataExchange_v2
                      ${HDF_LIBS}
                      ${HDF5_LIBS}
                      ${OTHER_LIBS}
                      ${NEXUS_LIBS}
                      ${XML_LIBS}
                      ${LOG_LIBS} 
                      ${TEEM_LIBS} 
                      ${FFTW_LIBS}
		      ${NAPI_LIBS}
                      )

# install executable
INSTALL (TARGETS Tomo2DataExchange_v2 DESTINATION ../../bin)



	
NEXUSLIB_SRC = nexusbox \
			nexusapi \
			nexusdata \
			nexusgroup \
			nexusfield \
			nexusattribute \
			nexusexceptionclass

NEXUSLIB_OBJS = ./nexus_lib/nexusbox.o \
			./nexus_lib/nexusapi.o \
			./nexus_lib/nexusdata.o \
			./nexus_lib/nexusgroup.o \
			./nexus_lib/nexusfield.o \
			./nexus_lib/nexusattribute.o \
			./nexus_lib/nexusexceptionclass.o

libnexuslibrary.a:
	@echo 'Creating libnexuslibrary.a...'
	ar -r "./nexus_lib/libnexuslibrary.a" $(NEXUSLIB_OBJS)
	@echo '...done creating.'
	cp ./nexus_lib/libnexuslibrary.a ./lib/
	@echo ' '
				
nexusbox: $(nexusbox)
	$(CCC) $(CCFLAGS) \
	$(NAPI_INC) $(HDF_INC) $(HDF5_INC) $(UTILITY_INC) \
	$(NEXUSLIB_INC) $(HDF_FLAGS) \
	-c ./nexus_lib/nexusbox.cpp \
	-o ./nexus_lib/nexusbox.o 
	@echo ' ' 			

nexusapi: $(nexusapi)
	$(CCC) $(CCFLAGS) \
	$(NAPI_INC) $(HDF_INC) $(HDF5_INC) $(UTILITY_INC) \
	$(NEXUSLIB_INC) $(HDF_FLAGS) \
	-c ./nexus_lib/nexusapi.cpp \
	-o ./nexus_lib/nexusapi.o 
	@echo ' ' 			

nexusdata: $(nexusdata)
	$(CCC) $(CCFLAGS) \
	$(NAPI_INC) $(HDF_INC) $(HDF5_INC) $(UTILITY_INC) \
	$(NEXUSLIB_INC) $(HDF_FLAGS) \
	-c ./nexus_lib/nexusdata.cpp \
	-o ./nexus_lib/nexusdata.o 
	@echo ' ' 			

nexusgroup: $(nexusgroup)
	$(CCC) $(CCFLAGS) \
	$(NAPI_INC) $(HDF_INC) $(HDF5_INC) $(UTILITY_INC) \
	$(NEXUSLIB_INC) $(HDF_FLAGS) \
	-c ./nexus_lib/nexusgroup.cpp \
	-o ./nexus_lib/nexusgroup.o 
	@echo ' ' 			

nexusfield: $(nexusfield)
	$(CCC) $(CCFLAGS) \
	$(NAPI_INC) $(HDF_INC) $(HDF5_INC) $(UTILITY_INC) \
	$(NEXUSLIB_INC) $(HDF_FLAGS) \
	-c ./nexus_lib/nexusfield.cpp \
	-o ./nexus_lib/nexusfield.o 
	@echo ' ' 			

nexusattribute: $(nexusattribute)
	$(CCC) $(CCFLAGS) \
	$(NAPI_INC) $(HDF_INC) $(HDF5_INC) $(UTILITY_INC) \
	$(NEXUSLIB_INC) $(HDF_FLAGS) \
	-c ./nexus_lib/nexusattribute.cpp \
	-o ./nexus_lib/nexusattribute.o 
	@echo ' ' 			

nexusexceptionclass: $(nexusexceptionclass)
	$(CCC) $(CCFLAGS) \
	$(NAPI_INC) $(HDF_INC) $(HDF5_INC) $(UTILITY_INC) \
	$(NEXUSLIB_INC) $(HDF_FLAGS) \
	-c ./nexus_lib/nexusexceptionclass.cpp \
	-o ./nexus_lib/nexusexceptionclass.o 
	@echo ' ' 			

	
nexuslib_clean: $(nexuslib_clean)
	echo "cleaning..."
	/bin/rm -f ./nexus_lib/*.o ./nexus_lib/*~ ./nexus_lib/*.a
	/bin/rm -f ./lib/*.o ./lib/*.a 
	



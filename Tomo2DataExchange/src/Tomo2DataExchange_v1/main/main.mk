	
MAIN_SRC = Tomo2DataExchange

MAIN_OBJS = ./main/Tomo2DataExchange.o

MAIN_INSTALL = main_install
				
Tomo2DataExchange: $(Tomo2DataExchange)
	$(CCC) $(CCFLAGS) \
	$(NAPI_INC) $(HDF_INC) $(UTILITY_INC) $(TEEM_INC) $(FFTW_INC) \
	$(NEXUSLIB_INC) $(HDF_FLAGS) \
	-c ./main/Tomo2DataExchange.cpp \
	-o ./main/Tomo2DataExchange.o 
	@echo ' ' 			
	
main_link: $(main_link)
	$(LD) $(NEXUS_LIBS) $(OTHER_LIBS) \
	$(HDF_LIBS) $(HDF5_LIBS) $(XML_LIBS) $(LOG_LIBS) $(TEEM_LIBS) $(FFTW_LIBS) \
	$(NAPI_OBJS) $(UTILITY_OBJS) \
	./main/Tomo2DataExchange.o -o ./main/Tomo2DataExchange

main_install: $(main_install)
	cp ./main/Tomo2DataExchange ../../bin/Tomo2DataExchange
	
main_clean : $(main_clean)
	/bin/rm -f ./main/Tomo2DataExchange
	



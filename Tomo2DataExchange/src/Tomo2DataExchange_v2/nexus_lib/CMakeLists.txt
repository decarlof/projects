
# build library
ADD_LIBRARY (nexuslibrary
	     nexusbox.cpp 
             nexusapi.cpp
             nexusdata.cpp
             nexusgroup.cpp
             nexusfield.cpp
             nexusattribute.cpp
             nexusexceptionclass.cpp
)

# install library
INSTALL(TARGETS nexuslibrary            DESTINATION  ../lib)
INSTALL (FILES  nexusapi.h              DESTINATION  ../include)
INSTALL (FILES  nexusbox.h              DESTINATION  ../include)
INSTALL (FILES  nexusexceptionclass.h   DESTINATION  ../include)
INSTALL (FILES  nexusgroup.h            DESTINATION  ../include)

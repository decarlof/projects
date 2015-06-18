
# build library
ADD_LIBRARY (logclass 
             linkedlistclass.cpp 
             logfileclass.cpp
             errorlogclass.cpp
)

# install library
INSTALL (TARGETS logclass DESTINATION ../lib)

INSTALL (FILES errorlogclass.h   DESTINATION  ../include)
INSTALL (FILES linkedlistclass.h DESTINATION  ../include)
INSTALL (FILES logfileclass.h    DESTINATION  ../include)

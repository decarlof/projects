
#  For more information, please see: http://software.sci.utah.edu
#
#  The MIT License
#
#  Copyright (c) 2007
#  Scientific Computing and Imaging Institute, University of Utah
#
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

# Make2cmake CMake Script
# Abe Stephens and James Bigler
# (c) 2007 Scientific Computing and Imaging Institute, University of Utah
# Note that the REGEX expressions may need to be tweaked for different dependency generators.

FILE(READ ${input_file} depend_text)

IF (${depend_text} MATCHES ".+")

  # MESSAGE("FOUND DEPENDS")

  # Remember, four backslashes is escaped to one backslash in the string.
  STRING(REGEX REPLACE "\\\\ " " " depend_text ${depend_text})
  
  # This works for the nvcc -M generated dependency files.
  STRING(REGEX REPLACE "^.* : " "" depend_text ${depend_text})
  STRING(REGEX REPLACE "[ \\\\]*\n" ";" depend_text ${depend_text})

  FOREACH(file ${depend_text})

    STRING(REGEX REPLACE "^ +" "" file ${file})

    # IF (EXISTS ${file})
	  #   MESSAGE("DEPEND = ${file}")    
    # ELSE (EXISTS ${file})
	  #   MESSAGE("ERROR = ${file}")
    # ENDIF(EXISTS ${file})
  
    SET(cuda_nvcc_depend "${cuda_nvcc_depend} \"${file}\"\n")
  
  ENDFOREACH(file) 

ELSE(${depend_text} MATCHES ".+") 
  # MESSAGE("FOUND NO DEPENDS")
ENDIF(${depend_text} MATCHES ".+")


FILE(WRITE ${output_file} "# Generated by: make2cmake.cmake\nSET(CUDA_NVCC_DEPEND\n ${cuda_nvcc_depend})\n\n")

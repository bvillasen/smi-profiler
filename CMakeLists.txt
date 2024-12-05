cmake_minimum_required(VERSION 3.5.0)


set(BASE_NAME "smiProfiler")
set(LIB_NAME "lib${BASE_NAME}")

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	CACHE STRING "Location of smiProfiler source code.")

set(BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}
	CACHE STRING "Location of smiProfiler build.")

message( "BUILD_DIR: ${BUILD_DIR}")  

project(${BASE_NAME})


if(NOT DEFINED ROCM_PATH)
  if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH
        "/opt/rocm"
        CACHE PATH "Path to ROCm stack.")
  else()
    set(ROCM_PATH
        $ENV{ROCM_PATH}
        CACHE PATH "Path to ROCm stack.")
  endif()
endif()
ADD_DEFINITIONS( -D__HIP_PLATFORM_AMD__ )

set(CMAKE_BUILD_TYPE Release)

set(SOURCES
  src/global.h
  src/smiProfiler.cpp
  src/rocm-smi_backend.cpp
)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -fPIC -pthread")

include_directories( ${ROCM_PATH}/include )
link_directories(${ROCM_PATH}/lib)

add_library(${BASE_NAME} SHARED ${SOURCES}  )

target_link_libraries(${BASE_NAME} rocm_smi64)

set_target_properties( ${BASE_NAME} PROPERTIES PUBLIC_HEADER src/smiProfiler.h )

install( TARGETS ${BASE_NAME} 
  LIBRARY DESTINATION ${BUILD_DIR}/lib 
  PUBLIC_HEADER DESTINATION ${BUILD_DIR}/include 
)




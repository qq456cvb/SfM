cmake_minimum_required(VERSION 3.14)
project(SfM)
set(CMAKE_CXX_STANDARD 17)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 Version: " ${EIGEN3_VERSION_STRING})

include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB_RECURSE SRC_FILES 
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_executable(main ${SRC_FILES})

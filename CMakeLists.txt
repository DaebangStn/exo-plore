cmake_minimum_required(VERSION 3.24.0)

set(CMAKE_C_COMPILER $ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER $ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++)

project(exoskeleton)

# ======================== Useful Configurations ========================
# This flag is used to enable the verbose logging
# add_compile_definitions(LOG_VERBOSE)

cmake_policy(SET CMP0144 NEW)  # Do not ignore the Environment variables when using the find_package command

add_compile_options(-fPIC)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)
add_compile_options(-mavx2 -mfma)

set(CMAKE_AR gcc-ar)
set(CMAKE_RANLIB gcc-ranlib)

set(OpenGL_GL_PREFERENCE LEGACY)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")
set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/libs/dart ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH ${LIBTORCH} ${CMAKE_PREFIX_PATH})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,-O1 -Wl,--as-needed -Wl,--hash-style=gnu")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold -Wl,-O1 -Wl,--as-needed -Wl,--hash-style=gnu")


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(LOG_VERBOSE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(BUILD_RELEASE)
endif()

add_compile_definitions(MASS_ROOT_DIR="${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -funroll-loops")

set(Python3_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python)
set(Python3_INCLUDE_DIRS $ENV{CONDA_PREFIX}/include/python3.8)
find_package(Python3 EXACT 3.8 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")

include_directories(${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})
find_package(pybind11 REQUIRED)

find_package(Boost REQUIRED)
link_libraries(${Boost_LIBRARIES})
include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
find_package(TinyXML2 REQUIRED)

set(CAFFE2_USE_CUDNN ON)

# CUDA paths are set via CMakePresets.json (CUDAToolkit_ROOT, CUDA_TOOLKIT_ROOT_DIR, CMAKE_CUDA_COMPILER)
# Default values if not set by preset (for backward compatibility)
if(NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "/usr/local/cuda")
endif()
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_FLAGS: ${TORCH_CXX_FLAGS}")

link_libraries(stdc++fs)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src/util)
add_subdirectory(src/core)
if (NOT SERVER_BUILD)
    message(STATUS "Build in the PC")
    add_subdirectory(libs)
    # add_subdirectory(src/postprocess)
    add_subdirectory(src/render)
else ()
    message(STATUS "Build in the SERVER")
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Build and Install the ray environment")
    add_subdirectory(src/ray)
endif ()
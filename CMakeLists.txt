cmake_minimum_required(VERSION 3.6)
project(spexy)

add_compile_options(
        -std=c++14
        -Wall
)

include_directories(src)
include_directories(~/.local/include/python3.6m)

find_package(PyBindCpp REQUIRED PATHS ../pybindcpp/cmake)
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${NUMPY_INCLUDE_DIR})

function(pymodule target name)
    py_module(${target} ${name} ${ARGN})
endfunction(pymodule)

pymodule(circular nat
        spexy/bases/circular/nat.cpp
        src/bases/utils.h
        src/bases/circular.h
        src/bases/circular.cpp
        )

pymodule(cardinals nat
        spexy/bases/cardinals/nat.cpp
        src/bases/utils.h
        src/bases/cardinals.h
        src/bases/cardinals.cpp
        )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 fftw3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIRS})
link_directories(${FFTW3_LIBRARY_DIRS})

pymodule(ops nat_raw
        spexy/ops/nat_raw.cpp
        src/ops/ops.cpp
        src/ops/ops.h
        )
target_link_libraries(ops ${FFTW3_LIBRARIES})

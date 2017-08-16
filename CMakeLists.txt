cmake_minimum_required(VERSION 3.6)
project(dec)

add_compile_options(
        -std=c++14
        -Wall
)

include_directories(src)
include_directories(../pybind11/include)
include_directories(../pybindcpp/include)

find_package(PyBindCpp REQUIRED PATHS ../pybindcpp/cmake)
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${NUMPY_INCLUDE_DIR})

add_custom_target(
	    ${PROJECT_NAME}_pytest ALL
		COMMAND py.test -v --doctest-modules dec/types #dec/ops dec/bases dec/other
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

function(pymodule target name)
    py_module(${target} ${name} ${ARGN})
    add_dependencies(${PROJECT_NAME}_pytest ${target}_symlink)
endfunction(pymodule)

pymodule(circular nat
        dec/bases/circular/nat.cpp
        src/dec/bases/utils.h
        src/dec/bases/circular.h
        src/dec/bases/circular.cpp
        )

pymodule(cardinals nat
        dec/bases/cardinals/nat.cpp
        src/dec/bases/utils.h
        src/dec/bases/cardinals.h
        src/dec/bases/cardinals.cpp
        )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 fftw3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIRS})
link_directories(${FFTW3_LIBRARY_DIRS})
pymodule(ops nat
        dec/ops/nat.cpp
        src/dec/ops/ops.cpp
        src/dec/ops/ops.h
        )
target_link_libraries(ops ${FFTW3_LIBRARIES})


#pymodule(nativepybind11 nativepybind11
#        dec/other/nativepybind11.cpp
#        )

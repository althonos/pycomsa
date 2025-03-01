cmake_minimum_required(VERSION 3.20)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C CXX)

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 11)

set_property(GLOBAL PROPERTY PYTHON_EXTENSIONS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- Cython extension support -------------------------------------------------

set(CYTHON_HEADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
include("src/scripts/cmake/CythonExtension.cmake")

# --- Find LZMA ----------------------------------------------------------------

find_package(LibLZMA REQUIRED)

# --- Compile code -------------------------------------------------------------

add_subdirectory(src)

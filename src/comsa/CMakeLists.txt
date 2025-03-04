set(COMSA_SOURCES
    msa.cpp
    msa.h

    transpose.cpp
    transpose.h

    defs.h

    mtf.cpp
    mtf.h

    queue.cpp
    queue.h

    pbwt.cpp
    pbwt.h

    ss.h

    wfc.cpp
    wfc.h

    rle.cpp
    rle.h

    entropy.cpp
    entropy.h

    rc.h
    sub_rc.h

    lzma_wrapper.cpp
    lzma_wrapper.h

    libs/zlib.h
    libs/zconf.h
)

foreach(_file IN ITEMS ${COMSA_SOURCES})
    if(EXISTS ${PROJECT_SOURCE_DIR}/patches/${_file}.patch)
        add_custom_command(
            OUTPUT
                ${_file}
            COMMENT
                "Patching ${_file}"
            COMMAND
                ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/scripts/apply_patch.py
                    --input ${PROJECT_SOURCE_DIR}/vendor/CoMSA/src/${_file}
                    --patch ${PROJECT_SOURCE_DIR}/patches/${_file}.patch
                    --output ${CMAKE_CURRENT_BINARY_DIR}/${_file}
            DEPENDS
                ${PROJECT_SOURCE_DIR}/vendor/CoMSA/src/${_file}
                ${PROJECT_SOURCE_DIR}/patches/${_file}.patch
        )
    else()
        add_custom_command(
            OUTPUT
                ${_file}
            COMMAND
                cmake -E copy ${PROJECT_SOURCE_DIR}/vendor/CoMSA/src/${_file} ${CMAKE_CURRENT_BINARY_DIR}/${_file}
            DEPENDS
                ${PROJECT_SOURCE_DIR}/vendor/CoMSA/src/${_file}
        )
    endif()
    set(COMSA_PATCHED_SOURCES ${COMSA_PATCHED_SOURCES} ${CMAKE_CURRENT_BINARY_DIR}/${_file})
endforeach()

add_library(comsa STATIC ${COMSA_PATCHED_SOURCES})
target_include_directories(comsa PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
set_property(TARGET comsa PROPERTY C_STANDARD 14)

target_link_libraries(comsa liblzma)
target_include_directories(comsa PUBLIC ${PROJECT_SOURCE_DIR}/vendor/xz/src/liblzma/api)

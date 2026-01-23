# TensorFlow C++ library
# Based on https://github.com/tensorflow/minigo/blob/master/cc/tensorflow/BUILD

# Find all TensorFlow headers
file(GLOB_RECURSE TF_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp"
)

# Find TensorFlow shared libraries
file(GLOB TF_LIBS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.so.2"
    "${CMAKE_CURRENT_SOURCE_DIR}/libtensorflow_cc.so.*"
)

add_library(tensorflow INTERFACE)
target_include_directories(tensorflow INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/external/nsync/public
    ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/eigen3
)

# Link TensorFlow framework library if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libtensorflow_framework.so.2")
    target_link_libraries(tensorflow INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/libtensorflow_framework.so.2
        m
    )
endif()

# Link additional TensorFlow libraries if found
foreach(TF_LIB ${TF_LIBS})
    target_link_libraries(tensorflow INTERFACE ${TF_LIB})
endforeach()

target_link_libraries(tensorflow INTERFACE m)

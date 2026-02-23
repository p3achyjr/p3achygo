# TensorFlow Example proto (standalone)
#
# Uses protobuf 3.21.9 to match TensorFlow 2.16

set(PROTO_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/feature.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/example.proto
)

# Generate into the binary directory
set(PROTO_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Custom command to generate protobuf sources
set(PROTO_SRCS
    ${PROTO_OUTPUT_DIR}/feature.pb.cc
    ${PROTO_OUTPUT_DIR}/example.pb.cc
)
set(PROTO_HDRS
    ${PROTO_OUTPUT_DIR}/feature.pb.h
    ${PROTO_OUTPUT_DIR}/example.pb.h
)

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND protobuf::protoc
    ARGS --cpp_out=${PROTO_OUTPUT_DIR}
         -I${CMAKE_CURRENT_SOURCE_DIR}
         ${PROTO_FILES}
    DEPENDS ${PROTO_FILES} protobuf::protoc
    COMMENT "Generating protobuf sources"
)

add_library(tf_example_proto
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_include_directories(tf_example_proto PUBLIC
    ${PROTO_OUTPUT_DIR}
)

target_link_libraries(tf_example_proto PUBLIC
    protobuf::libprotobuf
)

# Suppress warnings in generated code
target_compile_options(tf_example_proto PRIVATE -w)

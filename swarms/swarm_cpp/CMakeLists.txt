cmake_minimum_required(VERSION 3.20)
project(swarm_controller)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------
# Paths for proto generation
# ------------------------------------------------
set(PROTO_SRC "${PROJECT_SOURCE_DIR}/../../proto")
set(PROTO_OUT "${PROJECT_BINARY_DIR}/proto")
file(MAKE_DIRECTORY ${PROTO_OUT})

# ------------------------------------------------
# Protobuf 3.13
# ------------------------------------------------
set(PROTOBUF_ROOT "$ENV{HOME}/protobuf-3.13.0-install")
set(Protobuf_ROOT ${PROTOBUF_ROOT})
set(Protobuf_DIR "${PROTOBUF_ROOT}/lib/cmake/protobuf")

find_package(Protobuf REQUIRED)

message(STATUS "Protobuf include: ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Protobuf libraries: ${Protobuf_LIBRARIES}")

# ------------------------------------------------
# CPU LibTorch
# ------------------------------------------------
# Folder should contain:
#   lib/
#   include/
#   share/cmake/Torch/

set(Torch_DIR "$ENV{HOME}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

message(STATUS "Found LibTorch: ${Torch_DIR}")
message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")

# ------------------------------------------------
# gRPC
# ------------------------------------------------
find_package(gRPC REQUIRED)

# ------------------------------------------------
# Protoc + gRPC plugin
# ------------------------------------------------
set(PROTOC_EXECUTABLE "$ENV{HOME}/protobuf-3.13.0-install/bin/protoc")
find_program(GRPC_PLUGIN_EXECUTABLE grpc_cpp_plugin)

if(NOT GRPC_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "grpc_cpp_plugin not found")
endif()

# ------------------------------------------------
# Generated proto files
# ------------------------------------------------
set(PROTO_SRCS
    ${PROTO_OUT}/swarm.pb.cc
    ${PROTO_OUT}/swarm.grpc.pb.cc
)

set(PROTO_HDRS
    ${PROTO_OUT}/swarm.pb.h
    ${PROTO_OUT}/swarm.grpc.pb.h
)

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
    COMMAND ${PROTOC_EXECUTABLE}
        -I ${PROTO_SRC}
        --cpp_out ${PROTO_OUT}
        --grpc_out ${PROTO_OUT}
        --plugin=protoc-gen-grpc=${GRPC_PLUGIN_EXECUTABLE}
        ${PROTO_SRC}/swarm.proto
    DEPENDS ${PROTO_SRC}/swarm.proto
    COMMENT "Generating protobuf + gRPC sources"
)

add_custom_target(generate_protos DEPENDS ${PROTO_SRCS} ${PROTO_HDRS})

# ------------------------------------------------
# Executable
# ------------------------------------------------
add_executable(swarm
    src/main.cpp
    src/swarm.cpp
    ${PROTO_SRCS}
)

add_dependencies(swarm generate_protos)

# ------------------------------------------------
# Include directories
# ------------------------------------------------
target_include_directories(swarm
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROTO_OUT}
        ${Protobuf_INCLUDE_DIRS}
)

# ------------------------------------------------
# Link libraries
# ------------------------------------------------
target_link_libraries(swarm
    gRPC::grpc++
    protobuf::libprotobuf
    ${TORCH_LIBRARIES}
)

# ------------------------------------------------
# Torch compile flags
# ------------------------------------------------
target_compile_options(swarm PRIVATE ${TORCH_CXX_FLAGS})

# ------------------------------------------------
# Runtime library path
# ------------------------------------------------
set_property(TARGET swarm PROPERTY
    BUILD_RPATH "$ENV{HOME}/libtorch/lib")

set_property(TARGET swarm PROPERTY
    INSTALL_RPATH "$ENV{HOME}/libtorch/lib")

# ------------------------------------------------
# CPU build indicator
# ------------------------------------------------
add_compile_definitions(CPU_ONLY_BUILD)

message(STATUS "Building with CPU-only LibTorch")
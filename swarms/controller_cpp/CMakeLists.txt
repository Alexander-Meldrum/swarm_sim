cmake_minimum_required(VERSION 3.20)
project(swarm_controller)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# gRPC / protobuf (if used)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

add_executable(controller
    src/main.cpp
    src/controller.cpp
    src/proto/swarm.pb.cc
    src/proto/swarm.grpc.pb.cc
)

target_link_libraries(controller
    gRPC::grpc++
    protobuf::libprotobuf
)

target_include_directories(controller
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/build
)

# cmake_minimum_required(VERSION 3.20)
# project(swarm_controller)

# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# # gRPC / protobuf (if used)

# find_package(Protobuf REQUIRED)
# find_package(gRPC REQUIRED)

# set(PROTO_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
# set(PROTO_FILE ${PROTO_ROOT_DIR}/swarm.proto)

# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
# grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_FILE})

# add_executable(controller
#     src/main.cpp
#     src/swarm_controller.cpp
#     ${PROTO_SRCS}
#     ${GRPC_SRCS}
# )

# target_include_directories(controller
#     PRIVATE
#         ${CMAKE_CURRENT_BINARY_DIR}
#         ${PROJECT_SOURCE_DIR}/include
# )

# target_link_libraries(controller
#     gRPC::grpc++
#     protobuf::libprotobuf
# )



# find_package(Protobuf REQUIRED)
# find_package(gRPC REQUIRED)

# set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../proto)
# set(PROTO_FILE ${PROTO_SRC_DIR}/swarm.proto)

# set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
# file(MAKE_DIRECTORY ${GENERATED_DIR})

# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
# grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_FILE})

# set_source_files_properties(
#     ${PROTO_SRCS} ${PROTO_HDRS}
#     ${GRPC_SRCS} ${GRPC_HDRS}
#     PROPERTIES
#         GENERATED TRUE
# )

# target_include_directories(swarm
#     PUBLIC
#         ${GENERATED_DIR}
#         ${PROJECT_SOURCE_DIR}/include
# )

# target_sources(swarm
#     PRIVATE
#         ${PROTO_SRCS}
#         ${GRPC_SRCS}
# )

# add_executable(swarm
#     src/main.cpp
#     src/proto/swarm.pb.cc
#     src/proto/swarm.grpc.pb.cc
# )

# target_link_libraries(swarm
#     gRPC::grpc++
#     protobuf::libprotobuf
# )

# target_include_directories(swarm
#     PUBLIC
#         ${PROJECT_SOURCE_DIR}/include
# )

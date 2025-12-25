cmake_minimum_required(VERSION 3.20)
project(swarm_controller)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# gRPC / protobuf (if used)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

add_executable(swarm
    src/main.cpp
    src/swarm.cpp
    build/proto/swarm.pb.cc
    build/proto/swarm.grpc.pb.cc
)

target_link_libraries(swarm
    gRPC::grpc++
    protobuf::libprotobuf
)

target_include_directories(swarm
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/build
        ${PROJECT_SOURCE_DIR}/build/proto
)
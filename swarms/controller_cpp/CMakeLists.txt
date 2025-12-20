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
    src/proto/swarm.pb.cc
    src/proto/swarm.grpc.pb.cc
)

target_link_libraries(swarm
    gRPC::grpc++
    protobuf::libprotobuf
)

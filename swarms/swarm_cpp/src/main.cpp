#include <iostream>
#include <memory>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "proto/swarm.pb.h"
#include "proto/swarm.grpc.pb.h"
#include <swarm/swarm.h>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using swarm_proto::swarm_proto_service;
using swarm_proto::StepRequest;
using swarm_proto::StepResponse;
using swarm_proto::ResetRequest;
using swarm_proto::ResetResponse;
using swarm_proto::DroneAction;
using swarm_proto::DroneObservation;


int main() {
    std::cout <<"[swarm] Launching Swarm Controller (swarm_cpp)" << std::endl;

    // Connect to simulator
    auto channel = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
    std::unique_ptr<swarm_proto_service::Stub> stub = swarm_proto_service::NewStub(channel);

    swarm::SwarmController controller(std::move(stub));

    uint32_t num_drones = 100;    // number of drones request
    uint64_t max_steps = 1000;   // max simumlation steps request
    float dt = 0.02;            // Simulator step delta time request
    bool randomize_init_pos = true;
    float arena_size = 10;      // volume of arena will be (2*arena_size)³ [m]
    float min_dist = 1;         // min distance from other drones upon random init positions

    controller.reset(num_drones, max_steps, dt, randomize_init_pos, arena_size, min_dist);

    while (true) {
        auto resp = controller.step();

        if (resp.done()) {
            std::cout <<"[swarm] Episode finished, sim time: " << controller.step_count() * controller.dt() <<" sec\n" << std::endl;
            break;
        }
    }
    return 0;
}
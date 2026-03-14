#include <iostream>
#include <memory>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "swarm.pb.h"
#include "swarm.grpc.pb.h"
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


int main() {

    std::cout <<"[swarm] Launching Swarm Controller (swarm_cpp)" << std::endl;

    torch::Device device(torch::kCPU);
    std::cout << "[swarm] Using device: " << device << std::endl;

    auto channel = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
    std::unique_ptr<swarm_proto_service::Stub> stub = swarm_proto_service::NewStub(channel);

    swarm::SwarmController controller(std::move(stub));

    uint64_t seed = 0;
    uint32_t num_drones_team_0 = 15;
    uint32_t num_drones_team_1 = 20;
    uint64_t max_steps = 1000;

    controller.reset(seed, num_drones_team_0, num_drones_team_1, max_steps);

    controller.loadPolicy("swarms/swarm_cpp/policies/policy_scripted_50.pt");

    while (true) {

        auto resp = controller.step();

        if (resp.done()) {
            std::cout <<"[swarm] Episode finished" << std::endl;
            break;
        }
    }

    return 0;
}
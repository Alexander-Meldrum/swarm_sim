#include <iostream>
#include <memory>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "proto/swarm.pb.h"
#include "proto/swarm.grpc.pb.h"

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
    std::cout << "Launching swarm (controller_cpp)" << std::endl;

    // Connect to Rust simulator
    auto channel = grpc::CreateChannel(
        "localhost:50051",
        grpc::InsecureChannelCredentials()
    );

    std::unique_ptr<swarm_proto_service::Stub> stub =
        swarm_proto_service::NewStub(channel);

    constexpr int N = 64;  // number of drones

    bool done = false;
    int step = 0;

    ResetRequest reset_req;
    ResetResponse reset_resp;
    grpc::ClientContext ctx;
    grpc::Status status = stub->Reset(&ctx, reset_req, &reset_resp);

    if (!status.ok()) {
        std::cerr << "Reset failed: " << status.error_message() << std::endl;
        return 1;
    }

    std::cout << "Simulator reset to step 0 "
        << reset_resp.step() << std::endl;

    while (!done) {
        StepRequest request;
        StepResponse response;
        ClientContext context;
        
        // Build actions (one per drone)
        for (int i = 0; i < N; ++i) {
            DroneAction* a = request.add_actions();
            a->set_ax(0.1f);
            a->set_ay(0.0f);
            a->set_az(0.0f);
        }
        // Populate step counter to be sent to simulator
        request.set_step(step);
        request.set_num_drones(5)
        

        // BLOCKING RPC call (synchronous)
        Status status = stub->Step(&context, request, &response);

        if (!status.ok()) {
            std::cerr << "Step RPC failed: "
                      << status.error_message() << std::endl;
            return 1;
        }

        if (response.done()) {
        // TODO
        std::cout << "World Simulator Done" << std::endl;
        // Stop stepping immediately
        // Log episode return
        // Call reset()
        }
        // Consume observations
        const auto& obs = response.observations();
        for (int i = 0; i < obs.size(); ++i) {
            const DroneObservation& o = obs[i];
            // Example usage
            // std::cout << "Drone " << i << ": "
            //           << o.x() << ", " << o.y() << ", " << o.z() << "\n";
        }

        float reward = response.reward();
        done = response.done();

        std::cout << "Step " << step++
                  << " reward=" << reward
                  << " done=" << done << std::endl;
    }

    std::cout << "Episode finished." << std::endl;
    return 0;
}
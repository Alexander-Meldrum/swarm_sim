#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <span>
#include <grpcpp/grpcpp.h>
#include <torch/script.h>
#include <swarm.pb.h>
#include <swarm.grpc.pb.h>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using swarm_proto::DroneAction;
using swarm_proto::ResetRequest;
using swarm_proto::ResetResponse;
using swarm_proto::StepRequest;
using swarm_proto::StepResponse;
using swarm_proto::swarm_proto_service;

namespace swarm
{

    class SwarmController
    {
    public:
        explicit SwarmController(std::unique_ptr<swarm_proto_service::Stub> stub);

        void loadPolicy(
            const std::string &policy_path
            // uint32_t num_drones,
            // uint8_t num_obs_features,
            // float max_acc = 10.0f
        );

        ResetResponse reset(uint64_t seed, uint32_t num_drones_team_0, uint32_t num_drones_team_1, uint64_t max_steps);
        StepResponse step();

        // std::vector<float> computeActions(
        //     const google::protobuf::RepeatedField<float>* obs_flat);

        std::vector<float> computeActions(std::span<const float> obs_flat);

        std::span<const float> obs_;

    private:
        std::unique_ptr<swarm_proto_service::Stub> stub_;

        torch::jit::script::Module policy_;

        uint32_t num_drones_team_0_{0};
        uint32_t num_drones_team_1_{0};

        uint64_t max_steps_{0};
        uint64_t step_{0};

        float dt_{0.0};

        // const google::protobuf::RepeatedField<float>* obs_ = nullptr;

        ResetResponse reset_response_; // owns observation memory
        StepResponse step_response_;   // owns observation memory
        

        // std::vector<float> obs_;

        size_t num_observation_features_{22}; // TODO
        uint8_t action_dim_{3};               // TODO

        float max_acc_{10.0f}; // TODO
    };

}
#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <grpcpp/grpcpp.h>
#include <swarm.pb.h> // in build folder
#include <swarm.grpc.pb.h> // in build folder

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

namespace swarm {

class SwarmController {
public:
    explicit SwarmController(std::unique_ptr<swarm_proto_service::Stub> stub)
        : stub_(std::move(stub)) {}

    ResetResponse reset(uint32_t num_drones, uint64_t max_steps, float dt);
    StepResponse step();

    const std::vector<float>& observations() const {return obs_;}
    uint64_t step_count() const {return step_;}
    float dt() const {return dt_;}
    

private:
    std::unique_ptr<swarm_proto_service::Stub> stub_;
    uint32_t num_drones_{0};
    uint64_t step_{0};
    float dt_{0.0};
    // std::vector<swarm_proto::DroneObservation> obs_;
    // Flat observation vector
    std::vector<float> obs_;
    uint8_t num_observation_features_ = 3;

    // Called internally by reset() and step()
    void consume_observations_from_proto(const google::protobuf::RepeatedPtrField<swarm_proto::DroneObservation>& observations);

};

} // namespace swarm
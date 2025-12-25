#include <iostream>
#include <memory>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "proto/swarm.pb.h"
#include "proto/swarm.grpc.pb.h"
#include "swarm/swarm.h"

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

void SwarmController::consume_observations_from_proto(const google::protobuf::RepeatedPtrField<swarm_proto::DroneObservation>& observations)
{
    obs_.clear();
    for (const auto& o : observations) {
        obs_.push_back(o.ox());
        obs_.push_back(o.oy());
        obs_.push_back(o.oz());
        obs_.push_back(o.collision_count());
        // obs_.push_back(o.vx());
        // obs_.push_back(o.vy());
        // obs_.push_back(o.vz());
    }
}

ResetResponse SwarmController::reset(uint32_t num_drones, uint64_t max_steps, float dt) {
    ResetRequest request;
    ResetResponse response;
    grpc::ClientContext context;
    
    std::cout <<"[controller] Reset, step: " << step_ << std::endl;
    std::cout <<"[controller] Reset, num_drones: " << num_drones << std::endl;
    // std::cout.flush();

    // Data that we request to be set in simulator
    request.set_num_drones(num_drones);
    request.set_max_steps(max_steps);
    request.set_dt(dt);

    std::cout <<"[controller] request.num_drones() " << request.num_drones() << std::endl;

    // BLOCKING RPC call (synchronous)
    auto status = stub_->Reset(&context, request, &response);
    if (!status.ok()) {
        throw std::runtime_error(status.error_message());
    }
    // Populate data that authoritative simulator responds with
    num_drones_ = response.num_drones();
    step_ = response.step();
    dt_ = response.dt();

    // Observations saved until next step
    obs_.reserve(num_drones_ * num_observation_features_);
    SwarmController::consume_observations_from_proto(response.observations());

    std::cout <<"[controller] Simulator reset to step: " << step_ << std::endl;
    return response;
}

StepResponse SwarmController::step() {
    StepRequest request;
    StepResponse response;
    grpc::ClientContext context;
    
    request.set_step(step_);

    // TODO, Actions depend on observations

    // Build actions (one per drone)
    for (uint32_t i = 0; i < num_drones_; ++i) {
        DroneAction* a = request.add_actions();
        a->set_ax(0.1f);
        a->set_ay(0.0f);
        a->set_az(0.0f);
    }

    // BLOCKING RPC call (synchronous)
    auto status = stub_->Step(&context, request, &response);
    if (!status.ok()) {
        throw std::runtime_error(status.error_message());
    }

    step_ = response.step();

    if (response.done()) {
        // TODO
        std::cout <<"[controller] World Simulator Done" << std::endl;
        // Stop stepping immediately
        // Log episode return
        return response;
    }

    // Consume observations, save for next step
    SwarmController::consume_observations_from_proto(response.observations());

    float reward = response.reward();
    std::cout   << "Step:    " << step_
                << "reward:  " << reward
                << std::endl;


    return response;
}

} // namespace swarm
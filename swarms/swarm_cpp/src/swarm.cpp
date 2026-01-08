#include <iostream>
#include <memory>
#include <vector>
#include <grpcpp/grpcpp.h>
#include "proto/swarm.pb.h"
#include "proto/swarm.grpc.pb.h"
#include "swarm/swarm.h"
#include <random>

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
        obs_.push_back(o.collisions_desired());
        // obs_.push_back(o.collisions_undesired());
        // obs_.push_back(o.vx());
        // obs_.push_back(o.vy());
        // obs_.push_back(o.vz());
    }
}

ResetResponse SwarmController::reset(uint32_t num_drones, uint64_t max_steps, float dt, bool randomize_init_pos, float arena_size, float min_dist) {
    ResetRequest request;
    ResetResponse response;
    grpc::ClientContext context;

    std::cout <<"[swarm] Resetting Swarm Controller & Simulator..." << std::endl;

    // Data that we request to be set in simulator
    request.set_num_drones(num_drones);
    request.set_max_steps(max_steps);
    request.set_dt(dt);
    request.set_randomize_init_pos(randomize_init_pos);
    request.set_arena_size(arena_size);
    request.set_min_dist(min_dist);

    // BLOCKING RPC call (synchronous)
    auto status = stub_->Reset(&context, request, &response);
    if (!status.ok()) {
        throw std::runtime_error(status.error_message());
    }
    std::cout <<"[swarm] Simulator Reset Finished" << std::endl;

    // Populate data that authoritative simulator responds with
    num_drones_ = response.num_drones();
    step_ = response.step();
    dt_ = response.dt();

    std::cout <<"[swarm] Simulator Confirms Reset:" << std::endl;
    std::cout <<"step:       " << step_ << std::endl;
    std::cout <<"num_drones: " << num_drones << std::endl;
    std::cout <<"max_steps:  " << max_steps << std::endl;
    std::cout <<"dt:         " << dt << std::endl;

    // Observations, reserve memory and save first step's obs for next step
    obs_.reserve(num_drones_ * num_observation_features_);
    SwarmController::consume_observations_from_proto(response.observations());

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
        a->set_ax(10*dist(rng));
        a->set_ay(10*dist(rng));
        a->set_az(10*dist(rng));
    }

    // Blocking RPC call to step the simulator (synchronous)
    auto status = stub_->Step(&context, request, &response);
    if (!status.ok()) {
        throw std::runtime_error(status.error_message());
    }

    step_ = response.step();

    if (response.done()) {
        // TODO
        std::cout <<"[swarm] World Simulator Done" << std::endl;
        // Stop stepping immediately
        // Log episode return
        return response;
    }

    // Consume & flatten observations, save for next step
    SwarmController::consume_observations_from_proto(response.observations());

    float reward = response.global_reward();
    // std::cout   << "Step:    " << step_
    //             << "reward:  " << reward
    //             << std::endl;


    return response;
}

} // namespace swarm
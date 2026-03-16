#include "swarm/swarm.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <span>

namespace swarm
{

    SwarmController::SwarmController(std::unique_ptr<swarm_proto_service::Stub> stub)
        : stub_(std::move(stub))
    {
    }

    void SwarmController::loadPolicy(
        const std::string &policy_path)
    {
        policy_ = torch::jit::load(policy_path);
        policy_.eval();
        policy_.to(torch::kCPU);

        std::cout << "[swarm] Policy loaded: " << policy_path << std::endl;
        std::cout << "[swarm] Policy module methods:\n";
        for (const auto &m : policy_.get_methods())
        {
            std::cout << "  " << m.name() << std::endl;
        }
    }

    std::vector<float> SwarmController::computeActions(std::span<const float> obs_flat)
    {
        // ---------------------------------------------------------
        // Step 0 — Debug info
        // ---------------------------------------------------------
        // std::cout << "[swarm] obs_flat.size(): " << obs_flat.size() << std::endl;
        // std::cout << "[swarm] drones: " << num_drones_team_0_ << std::endl;
        // std::cout << "[swarm] obs_features: " << num_observation_features_ << std::endl;

        // ---------------------------------------------------------
        // Step 1 — Validate observation size
        // ---------------------------------------------------------
        const size_t expected =
            static_cast<size_t>(num_drones_team_0_) *
            static_cast<size_t>(num_observation_features_);

        if (obs_flat.size() != expected)
        {
            throw std::runtime_error(
                "computeActions: observation size mismatch");
        }
        if (num_drones_team_0_ == 0 || num_observation_features_ == 0)
        {
            throw std::runtime_error(
                "computeActions: invalid swarm configuration");
        }

        // ---------------------------------------------------------
        // Step 2 — Create observation tensor
        // ---------------------------------------------------------
        // from_blob() creates a tensor that directly references the
        // memory in obs_flat (zero-copy).
        // Shape: [num_drones, num_features]
        torch::Tensor obs_tensor = torch::from_blob(
            const_cast<float *>(obs_flat.data()),
            {static_cast<int64_t>(num_drones_team_0_),
             static_cast<int64_t>(num_observation_features_)},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCPU));

        // Optional: Clone to detach from external memory and ensure safety
        // obs_tensor = obs_tensor.clone().contiguous();

        // ---------------------------------------------------------
        // Step 3 — Run policy inference
        // ---------------------------------------------------------
        // This performs ONE batched forward pass:
        // input  : [N_drones, obs_dim]
        // output : [N_drones, action_dim]
        torch::jit::IValue output_ivalue;

        try
        {
            output_ivalue = policy_.forward({obs_tensor});
        }
        catch (const c10::Error &e)
        {
            std::cerr << "[swarm] TorchScript forward error:\n"
                      << e.what() << std::endl;
            throw;
        }

        // ---------------------------------------------------------
        // Step 4 — Extract tensor from output
        // ---------------------------------------------------------
        torch::Tensor mean;

        if (output_ivalue.isTensor())
        {
            mean = output_ivalue.toTensor();
        }
        else if (output_ivalue.isTuple())
        {
            // policy return (mean, std)
            auto tup = output_ivalue.toTuple();
            mean = tup->elements()[0].toTensor();
        }
        else
        {
            throw std::runtime_error(
                "computeActions: unexpected policy output type");
        }

        // ---------------------------------------------------------
        // Step 5 — Apply action scaling
        // ---------------------------------------------------------
        // RL policy output pipeline:
        // raw_mean -> tanh -> scaled action
        torch::Tensor action = torch::tanh(mean) * max_acc_;

        action = action.contiguous();

        // ---------------------------------------------------------
        // Step 6 — Copy tensor to std::vector
        // ---------------------------------------------------------
        const size_t total_actions =
            static_cast<size_t>(num_drones_team_0_) *
            static_cast<size_t>(action_dim_);

        float *ptr = action.data_ptr<float>();

        return std::vector<float>(ptr, ptr + total_actions);
    }

    ResetResponse SwarmController::reset(
        uint64_t seed,
        uint32_t num_drones_team_0,
        uint32_t num_drones_team_1,
        uint64_t max_steps)
    {
        ResetRequest request;
        grpc::ClientContext context;

        request.set_seed(seed);
        request.set_num_drones_team_0(num_drones_team_0);
        request.set_num_drones_team_1(num_drones_team_1);
        request.set_max_steps(max_steps);

        auto status = stub_->Reset(&context, request, &reset_response_);

        if (!status.ok())
            throw std::runtime_error(status.error_message());

        num_drones_team_0_ = reset_response_.num_drones_team_0();
        num_drones_team_1_ = reset_response_.num_drones_team_1();
        step_ = reset_response_.step();
        dt_ = reset_response_.dt();

        obs_ = {reset_response_.observations().data(),
                (size_t)reset_response_.observations().size()};

        std::cout << "[swarm] obs_.size(): " << obs_.size() << std::endl;

        return reset_response_;
    }

    StepResponse SwarmController::step()
    {
        StepRequest request;
        grpc::ClientContext context;

        // std::cout << "[swarm] Step: " << step_ << std::endl;

        request.set_step(step_);

        // Infer actions and set action request
        auto actions = computeActions(obs_);
        for (uint32_t i = 0; i < num_drones_team_0_; ++i)
        {
            DroneAction *a = request.add_actions();
            a->set_ax(actions[i * 3 + 0]);
            a->set_ay(actions[i * 3 + 1]);
            a->set_az(actions[i * 3 + 2]);
        }
        // Step the simulator
        auto status = stub_->Step(&context, request, &step_response_);
        if (!status.ok())
            throw std::runtime_error(status.error_message());

        step_ = step_response_.step();

        // Store the observations for next step, create span view
        obs_ = {step_response_.observations().data(),
                (size_t)step_response_.observations().size()};

        return step_response_;
    }
}
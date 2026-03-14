#include "swarm/swarm.h"
#include <torch/torch.h>  // for TORCH_VERSION and general libtorch stuff
#include <torch/script.h> // for torch::jit::Module

namespace swarm
{

    SwarmController::SwarmController(std::unique_ptr<swarm_proto_service::Stub> stub)
        : stub_(std::move(stub))
    {
    }

    void SwarmController::loadPolicy(
        const std::string &policy_path)
    {
        // auto policy_ = torch::jit::load(policy_path, torch::kCPU);
        // policy_.eval();

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
        std::cout << "[swarm] obs_flat.size(): " << obs_flat.size() << std::endl;
        std::cout << "[swarm] drones: " << num_drones_team_0_ << std::endl;
        std::cout << "[swarm] obs_features: " << num_observation_features_ << std::endl;

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

        // Clone to detach from external memory and ensure safety, TODO
        obs_tensor = obs_tensor.clone().contiguous();

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
            // some policies return (mean, std)
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
        // Typical RL policy output pipeline:
        // raw_mean -> tanh -> scaled action

        torch::Tensor action = torch::tanh(mean) * max_acc_;

        action = action.contiguous();

        // ---------------------------------------------------------
        // Step 6 — Copy tensor to std::vector
        // ---------------------------------------------------------
        // Layout becomes:
        // [d0_ax d0_ay d0_az  d1_ax d1_ay d1_az ...]

        const size_t total_actions =
            static_cast<size_t>(num_drones_team_0_) *
            static_cast<size_t>(action_dim_);

        float *ptr = action.data_ptr<float>();

        return std::vector<float>(ptr, ptr + total_actions);
    }

    // std::vector<float> SwarmController::computeActions(std::span<const float> obs_flat)
    // {
    //     // -------------------------------------------------------
    //     // Step 0: Basic debug + safety checks
    //     // -------------------------------------------------------

    //     std::cout << "[swarm] obs_flat.size(): " << obs_flat.size() << std::endl;

    //     if (num_drones_team_0_ == 0 || num_observation_features_ == 0)
    //     {
    //         throw std::runtime_error(
    //             "computeActions: num_drones_team_0_ or num_observation_features_ is zero");
    //     }

    //     const size_t expected_obs =
    //         static_cast<size_t>(num_drones_team_0_) *
    //         static_cast<size_t>(num_observation_features_);

    //     if (obs_flat.size() != expected_obs)
    //     {
    //         throw std::runtime_error(
    //             "computeActions: observation size mismatch");
    //     }

    //     // -------------------------------------------------------
    //     // Step 1: Create tensor view of observations (ZERO COPY)
    //     // -------------------------------------------------------
    //     //
    //     // from_blob does NOT allocate memory.
    //     // The tensor directly references the observation buffer.
    //     //
    //     // Input shape must match the traced model:
    //     //   [batch_size, OBS_DIM]
    //     //
    //     // OBS_DIM = num_drones * num_features
    //     //

    //     // torch::Tensor obs_tensor = torch::from_blob(
    //     //     const_cast<float *>(obs_flat.data()),       // pointer to observation memory
    //     //     {1, static_cast<int64_t>(obs_flat.size())}, // tensor shape
    //     //     torch::TensorOptions()
    //     //         .dtype(torch::kFloat32)
    //     //         .device(torch::kCPU));

    //     std::cout << "[swarm] action calc 1 " << std::endl;
    //     // Create zero-copy tensor
    //     torch::Tensor obs_tensor = torch::from_blob(
    //         const_cast<float *>(obs_flat.data()),
    //         {1, static_cast<int64_t>(obs_flat.size())},
    //         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    //     // -------------------------------------------------------
    //     // Step 2: Prepare TorchScript inputs
    //     // -------------------------------------------------------
    //     //
    //     // TorchScript forward() expects a vector/array of IValue
    //     //

    //     // std::array<torch::jit::IValue, 1> inputs{obs_tensor};

    //     // -------------------------------------------------------
    //     // Step 3: Run policy forward pass
    //     // -------------------------------------------------------
    //     std::cout << "[swarm] action calc 3 " << std::endl;
    //     // policy_.to(torch::kCPU);
    //     // -------------------------------------------------------
    //     // DEBUG: Module information
    //     // -------------------------------------------------------

    //     std::cout << "\n========== POLICY DEBUG ==========\n";

    //     std::cout << "[debug] policy module ptr: " << &policy_ << std::endl;

    //     // Print module graph if possible
    //     try {
    //         std::cout << "[debug] policy graph:\n";
    //         std::cout << policy_.dump_to_str(true, false, false) << std::endl;
    //     } catch (...) {
    //         std::cout << "[debug] could not dump policy graph\n";
    //     }

    //     // Check if forward exists
    //     try {
    //         auto method = policy_.find_method("forward");
    //         if (method) {
    //             std::cout << "[debug] forward() method FOUND\n";
    //         } else {
    //             std::cout << "[debug] forward() method NOT FOUND\n";
    //         }
    //     } catch (...) {
    //         std::cout << "[debug] exception while checking forward()\n";
    //     }

    //     // -------------------------------------------------------
    //     // DEBUG: Tensor information
    //     // -------------------------------------------------------

    //     std::cout << "\n========== INPUT TENSOR DEBUG ==========\n";

    //     std::cout << "[debug] obs_flat.size(): " << obs_flat.size() << std::endl;

    //     std::cout << "[debug] tensor sizes: " << obs_tensor.sizes() << std::endl;

    //     std::cout << "[debug] tensor dim: " << obs_tensor.dim() << std::endl;

    //     std::cout << "[debug] tensor dtype: " << obs_tensor.dtype() << std::endl;

    //     std::cout << "[debug] tensor device: " << obs_tensor.device() << std::endl;

    //     std::cout << "[debug] tensor contiguous: " << obs_tensor.is_contiguous() << std::endl;

    //     std::cout << "[debug] tensor defined: " << obs_tensor.defined() << std::endl;

    //     // Print first few observation values
    //     std::cout << "[debug] first observations: ";
    //     for (size_t i = 0; i < std::min<size_t>(5, obs_flat.size()); i++)
    //         std::cout << obs_flat[i] << " ";
    //     std::cout << std::endl;

    //     // -------------------------------------------------------
    //     // DEBUG: Torch version
    //     // -------------------------------------------------------

    //     std::cout << "\n========== TORCH DEBUG ==========\n";

    //     std::cout << "[debug] TORCH_VERSION: " << TORCH_VERSION << std::endl;

    //     std::cout << "[debug] CUDA available: " << torch::cuda::is_available() << std::endl;

    //     // -------------------------------------------------------

    //     std::cout << "========== END DEBUG ==========\n\n";

    //     // auto output_ivalue = policy_.forward(inputs);
    //     auto output_ivalue = policy_.forward({obs_tensor});

    //     // -------------------------------------------------------
    //     // Step 4: Extract policy output tensor
    //     // -------------------------------------------------------
    //     //
    //     // TorchScript policies sometimes return:
    //     //   Tensor
    //     //   Tuple(Tensor, ...)
    //     //

    //     std::cout << "[swarm] action calc 4 " << std::endl;
    //     torch::Tensor mean;

    //     if (output_ivalue.isTensor())
    //     {
    //         mean = output_ivalue.toTensor();
    //     }
    //     else if (output_ivalue.isTuple())
    //     {
    //         auto output = output_ivalue.toTuple();
    //         mean = output->elements()[0].toTensor();
    //     }
    //     else
    //     {
    //         throw std::runtime_error(
    //             "computeActions: unexpected policy output type");
    //     }

    //     // -------------------------------------------------------
    //     // Step 5: Compute final actions
    //     // -------------------------------------------------------
    //     //
    //     // Example RL policy post-processing:
    //     //   tanh squashing + acceleration scaling
    //     //
    //     std::cout << "[swarm] action calc 5 " << std::endl;

    //     torch::Tensor action = torch::tanh(mean) * max_acc_;

    //     // Ensure contiguous layout before reading raw memory
    //     action = action.contiguous();

    //     // -------------------------------------------------------
    //     // Step 6: Copy actions into std::vector<float>
    //     // -------------------------------------------------------
    //     //
    //     // This is the only memory copy in the pipeline.
    //     //
    //     std::cout << "[swarm] action calc 6 " << std::endl;

    //     float *ptr = action.data_ptr<float>();

    //     return std::vector<float>(
    //         ptr,
    //         ptr + num_drones_team_0_ * action_dim_);
    // }

    ResetResponse SwarmController::reset(
        uint64_t seed,
        uint32_t num_drones_team_0,
        uint32_t num_drones_team_1,
        uint64_t max_steps)
    {
        ResetRequest request;
        // ResetResponse response;
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
        // obs_ = &reset_response_.observations();

        obs_ = {reset_response_.observations().data(),
                (size_t)reset_response_.observations().size()};

        std::cout << "[swarm] obs_.size(): " << obs_.size() << std::endl;

        return reset_response_;
    }

    StepResponse SwarmController::step()
    {
        StepRequest request;
        // StepResponse response;
        grpc::ClientContext context;

        std::cout << "[swarm] Step: " << step_ << std::endl;

        request.set_step(step_);

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
        // const auto& obs_ = response.observations();
        // obs_ = &response.observations();

        // Store the observations for next step, create span view
        obs_ = {step_response_.observations().data(),
                (size_t)step_response_.observations().size()};

        return step_response_;
    }
}
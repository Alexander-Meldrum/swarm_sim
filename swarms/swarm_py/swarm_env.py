"""
This file defines a SwarmEnv class.
It communicates with the Rust simulator over gRPC.

Responsibilities:
- Own the gRPC channel and stub
- Send ResetRequest / StepRequest
- Convert protobuf observations into PyTorch tensors
- Hide networking details from RL code
"""

import grpc
import torch
import numpy as np

# Generated from swarm.proto using grpc_tools.protoc
import swarm_pb2
import swarm_pb2_grpc


class SwarmEnv:
    """
    RL-facing environment wrapper.

    Team convention:
    - team 0: controlled by this RL agent (Python)
    - team 1: controlled by rule-based logic inside simulator
    """

    def __init__(self, address="localhost:50051", device="cpu", obs_dim = 0, action_dim = 0):
        # Open a gRPC channel to the simulator
        self.channel = grpc.insecure_channel(address)

        # Create a client stub from the proto service
        self.stub = swarm_pb2_grpc.swarm_proto_serviceStub(self.channel)

        # Torch device (cpu / cuda)
        self.device = device

        # Number of drones per team (set during reset)
        self.num_drones_team_0 = 0
        self.num_drones_team_1 = 0
        self.num_drones        = 0
        self.step_count        = 0

        # Observation structure per drone:
        self.obs_dim = obs_dim

        # Action structure per drone:
        # [ax, ay, az]
        self.action_dim = action_dim

    def reset(self, num_drones_team_0: int, num_drones_team_1: int, max_steps: int, seed: int):
        """
        Start a new episode in the simulator.

        Args:
            num_drones_team_0: number of RL-controlled drones
            team1_drones: number of rule-based drones
            max_steps: maximum simulation steps for episode

        Returns:
            torch.Tensor of observations for team 0
        """

        
        self.step_count = 0

        request = swarm_pb2.ResetRequest(
            seed = seed,
            num_drones_team_0=num_drones_team_0,
            num_drones_team_1=num_drones_team_1,
            max_steps=max_steps,
        )

        # Blocking RPC call (synchronous semantics)
        response = self.stub.Reset(request)

        # Store configuration locally
        self.num_drones_team_0 = response.num_drones_team_0
        self.num_drones_team_1 = response.num_drones_team_1
        self.num_drones = self.num_drones_team_0 + self.num_drones_team_1


        self.step_count = response.step

        # Extract only team-0 observations for RL
        return self._extract_team0_obs(response.observations)

    def step(self, actions: torch.Tensor):
        """
        Perform one simulator step.

        Args:
            actions: Tensor of shape [num_team0, 3]

        Returns:
            obs: next observations (team 0)
            rewards: reward tensor [num_team0,]
            global_reward: global reward tensor (scalar)
            done: episode termination flag
        """

        # Create empty StepRequest
        request = swarm_pb2.StepRequest()

        # Convert tensor actions → protobuf messages
        for i in range(self.num_drones_team_0):
            request.actions.append(
                swarm_pb2.DroneAction(
                    ax=float(actions[i, 0]),
                    ay=float(actions[i, 1]),
                    az=float(actions[i, 2]),
                )
            )

        request.step = self.step_count
        request.team_id = 0

        # Send step request to simulator
        response = self.stub.Step(request)

        # Parse step, observations, reward, and done flag
        self.step_count = response.step
        obs_team_0 = self._extract_team0_obs(response.observations)
        rewards = torch.tensor(response.rewards, device=self.device)
        # global_reward = torch.tensor(response.global_reward, device=self.device)
        done = response.done

        return obs_team_0, rewards,  done # global_reward,

    def _extract_team0_obs(self, observations):
        """
        Convert protobuf observations (Already flat) into a torch tensor.

        Simulator returns observations for *all* drones.
        We filter out team 1 and keep team 0 only.

        Returns:
            Tensor of shape [num_team0, obs_dim]
        """

        from train import MAX_DISTANCE, MAX_VELOCITY

        # Assume obs already flat, reshape only, Convert to torch tensor on desired device
        obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
        obs = obs.view(self.num_drones_team_0, self.obs_dim)
        # Cut away team_1 observations
        obs_team0 = obs[:self.num_drones_team_0]

        # Normalize, assuming K_NEIGHBORS = 2
        # obs_team0[:, 0:3] /= MAX_DISTANCE
        # obs_team0[:, 7:10] /= MAX_DISTANCE
        # obs_team0[:, 14:17] /= MAX_DISTANCE
        # obs_team0[:, 21:24] /= MAX_DISTANCE
        # obs_team0[:, 28:31] /= MAX_DISTANCE
        # obs_team0[:, 3:6] /= MAX_VELOCITY
        # obs_team0[:, 10:13] /= MAX_VELOCITY
        # obs_team0[:, 17:20] /= MAX_VELOCITY
        # obs_team0[:, 24:27] /= MAX_VELOCITY
        # obs_team0[:, 31:34] /= MAX_VELOCITY

        # obs_team0[:, 3] /= 10 # Turn alive flag into 0.1
        obs_team0[:, 4:7] /= MAX_DISTANCE
        obs_team0[:, 10:13] /= MAX_DISTANCE
        # obs_team0[:, 18:21] /= MAX_DISTANCE
        # obs_team0[:, 25:28] /= MAX_DISTANCE

        obs_team0[:, 0:3] /= MAX_VELOCITY
        obs_team0[:, 7:10] /= MAX_VELOCITY
        obs_team0[:, 13:16] /= MAX_VELOCITY
        # obs_team0[:, 21:24] /= MAX_VELOCITY
        # obs_team0[:, 28:31] /= MAX_VELOCITY

        return obs_team0
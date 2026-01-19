"""
This file defines a SwarmEnv class that acts like a Gym-style environment.
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

    def __init__(self, address="localhost:50051", device="cpu"):
        # Open a gRPC channel to the simulator
        self.channel = grpc.insecure_channel(address)

        # Create a client stub from the proto service
        self.stub = swarm_pb2_grpc.swarm_proto_serviceStub(self.channel)

        # Torch device (cpu / cuda)
        self.device = device

        # Number of drones per team (set during reset)
        self.num_team0 = 0
        self.num_team1 = 0

        # Observation structure per drone:
        # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        self.obs_dim = 6

        # Action structure per drone:
        # [ax, ay, az]
        self.action_dim = 3

    def reset(self, num_drones_team_0: int, num_drones_team_1: int, max_steps: int):
        """
        Start a new episode in the simulator.

        Args:
            num_drones_team_0: number of RL-controlled drones
            team1_drones: number of rule-based drones
            max_steps: maximum simulation steps for episode

        Returns:
            torch.Tensor of observations for team 0
        """

        # Store configuration locally
        self.num_drones_team_0 = num_drones_team_0
        self.num_drones_team_1 = num_drones_team_1

        # Build ResetRequest protobuf
        #   uint64 seed = 1;
        #   uint32 num_drones_team_0 = 2;  // Number of learning (RL) drones
        #   uint32 num_drones_team_1 = 3;  // Number of rule-based / opponent drones
        #   uint64 max_steps = 4;
        #   float dt = 5;
        #   bool randomize_init_pos = 6;
        #   float arena_size = 7;
        #   float min_dist=8;

        request = swarm_pb2.ResetRequest(
            seed = 0,
            num_drones_team_0=num_drones_team_0,
            num_drones_team_1=num_drones_team_1,
            max_steps=max_steps,
            dt = 0.02,
            randomize_init_pos=True,
            arena_size=float(10.0),  # TODO check
            min_dist = 2.0,
        )

        # Blocking RPC call (synchronous semantics)
        response = self.stub.Reset(request)

        # Extract only team-0 observations for RL
        return self._extract_team0_obs(response.observations)

    def step(self, actions: torch.Tensor):
        """
        Perform one simulator step.

        Args:
            actions: Tensor of shape [num_team0, 3]

        Returns:
            obs: next observations (team 0)
            reward: scalar reward tensor
            done: episode termination flag
        """

        # Create empty StepRequest
        request = swarm_pb2.StepRequest()

        # Convert tensor actions → protobuf messages
        for i in range(self.num_team0):
            request.actions.append(
                swarm_pb2.DroneAction(
                    ax=float(actions[i, 0]),
                    ay=float(actions[i, 1]),
                    az=float(actions[i, 2]),
                )
            )

        # Send step request to simulator
        response = self.stub.Step(request)

        # Parse observations, reward, and done flag
        obs = self._extract_team0_obs(response.observations)
        reward = torch.tensor(response.reward, device=self.device)
        done = response.done

        return obs, reward, done

    def _extract_team0_obs(self, observations):
        """
        Convert protobuf observations into a flat torch tensor.

        Simulator returns observations for *all* drones.
        We filter out team 1 and keep team 0 only.

        Returns:
            Tensor of shape [num_team0, obs_dim]
        """

        # Preallocate NumPy array for speed
        obs = np.zeros((self.num_team0, self.obs_dim), dtype=np.float32)

        idx = 0
        for o in observations:
            if o.team_id != 0:
                continue

            obs[idx] = [
                o.pos_x, o.pos_y, o.pos_z,
                o.vel_x, o.vel_y, o.vel_z,
            ]
            idx += 1

        # Convert to torch tensor on desired device
        return torch.tensor(obs, device=self.device)
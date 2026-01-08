"""
policy.py

Defines a simple neural network policy mapping
drone observations → continuous actions.

This policy is shared across all drones (parameter sharing),
which is common in swarm RL.
"""

import torch
import torch.nn as nn


class SwarmPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        # Simple MLP policy
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        """
        Args:
            obs: Tensor [num_drones, obs_dim]

        Returns:
            actions: Tensor [num_drones, action_dim]
        """
        return self.net(obs)
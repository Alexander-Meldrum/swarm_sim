"""
Defines a simple neural network policy mapping
drone observations → continuous actions.

This policy is shared across all drones (parameter sharing),
which is common in swarm RL.
"""

import torch
import torch.nn as nn


class SwarmPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        # Initialize the base PyTorch Module.
        # This registers parameters so PyTorch can track gradients
        # and so optimizers (Adam, SGD, etc.) can update them.
        super().__init__()

        # Define the policy network as a simple Multi-Layer Perceptron (MLP).
        # In RL terms, this network represents π_θ(a | s):
        #   a function that maps a state (observation) to action preferences.
        self.net = nn.Sequential(
            # First linear layer:
            # Maps the environment observation (state) of dimension obs_dim
            # into a 128-dimensional hidden representation.
            nn.Linear(obs_dim, 128),

            # Nonlinearity:
            # Allows the policy to represent non-linear decision boundaries.
            nn.ReLU(),

            # Second hidden layer:
            # Further transforms the internal state representation.
            # Deeper layers let the policy capture more complex behaviors.
            nn.Linear(128, 128),

            # Another nonlinearity for expressiveness.
            nn.ReLU(),

            # Output layer:
            # Produces one value per discrete action.
            # These outputs are *logits*, not probabilities.
            # Softmax will be applied later (via torch.distributions.Categorical)
            # to form the action distribution π_θ(a | s).
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
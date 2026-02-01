import torch
import torch.nn as nn

class SwarmPolicy(nn.Module):
    """
    Policy network for continuous-action PPO.

    This network:
    - Takes an observation vector per drone
    - Outputs a Gaussian distribution over actions
    - Is shared across all drones (batch dimension = number of drones)

    PPO will:
    - Sample actions from this distribution
    - Adjust the distribution based on reward feedback
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        # Simple MLP
        # Tanh here keeps hidden activations bounded → stable means
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),                # stabilizes hidden state
            nn.Linear(64, act_dim),   # outputs action mean
        )

        # One log_std per action dimension
        # Shared across all drones (simplest setup)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):

        # Mean of Gaussian policy
        raw_mean = self.net(obs)

        # Smoothly bound the mean to prevent tanh saturation downstream
        # This keeps gradients healthy and prevents runaway μ
        mean = 2.0 * torch.tanh(raw_mean / 2.0)

        # Clamp log_std to avoid:
        #  - std → 0  (no exploration)
        #  - std → ∞  (random thrashing)
        log_std = torch.clamp(self.log_std, -5, 2)
        std = log_std.exp()

        return mean, std

class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        # Simple MLP:
        # Input  : observation vector (position, velocity, etc.)
        # Output : single scalar = expected future reward

        # The value network predicts the total discounted future reward from a state, not just the next reward

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # <-- one value per drone
        )

    def forward(self, obs):
        """
        obs: (num_drones, obs_dim)

        returns:
            value: (num_drones, 1)
        """
        return self.net(obs)
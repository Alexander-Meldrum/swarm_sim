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
            nn.Linear(obs_dim, 128),
            nn.Tanh(),          # OK
            nn.Linear(128, 128),
            nn.Tanh(),          # OK
            nn.Linear(128, act_dim)  # UNBOUNDED
        )

        self.log_std = nn.Parameter(torch.ones(act_dim) * 0.5)
        # self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)  # Init with lower std

        # Reasonable bounds for exploration
        # self.LOG_STD_MIN = -1.0   # std ≈ 0.37
        # self.LOG_STD_MAX = 0.5    # std ≈ 1.65
        # self.LOG_STD_MIN = -1.5   # std ≈ 0.22
        # self.LOG_STD_MAX = 0.0    # std = 1.0

    def forward(self, obs):
        """
        obs: (N, obs_dim)
        returns:
            mean: (N, act_dim)
            std:  (N, act_dim)
        """
        raw_mean = self.net(obs)
        # prevent runaway mean
        raw_mean = torch.clamp(raw_mean, -5.0, 5.0)
        mean = torch.tanh(raw_mean)

        std = self.log_std.exp()
        std = torch.clamp(std, min=0.3, max=1.5)  # prevent collapse & runaway
        std = std.unsqueeze(0).expand_as(mean)
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
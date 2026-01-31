import torch
import torch.nn as nn
# import torch.nn.functional as F


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
        mean = self.net(obs)

        # Clamp log_std to avoid:
        #  - std → 0  (no exploration)
        #  - std → ∞  (random thrashing)
        log_std = torch.clamp(self.log_std, -5, 2)
        std = log_std.exp()

        return mean, std
    

    
    # def __init__(self, obs_dim: int, action_dim: int):
    #     """
    #     obs_dim:    number of input features per drone
    #     action_dim: number of continuous action dimensions (e.g. ax, ay, az)
    #     """
    #     super().__init__()

    #     # -------------------------------
    #     # Feature extractor (the "brain")
    #     # -------------------------------
    #     # This part turns raw observations into a useful internal representation.
    #     # It does NOT decide actions yet — it just builds features.
    #     self.net = nn.Sequential(
    #         nn.Linear(obs_dim, 128),  # combine all observation features
    #         nn.ReLU(),                # non-linearity = expressive power
    #         nn.Linear(128, 128),       # deeper representation
    #         nn.ReLU(),
    #     )

    #     # -----------------------------------
    #     # Mean head (what action to take)
    #     # -----------------------------------
    #     # This layer outputs the MEAN of a Gaussian distribution.
    #     # Each output dimension corresponds to one action dimension.
    #     self.mean_head = nn.Linear(128, action_dim)

    #     # -----------------------------------
    #     # Log standard deviation (how uncertain)
    #     # -----------------------------------
    #     # PPO needs stochasticity.
    #     # Instead of predicting std from the network (which can be unstable),
    #     # we learn ONE log_std per action dimension.
    #     #
    #     # log_std is:
    #     # - a learnable parameter
    #     # - shared across all states
    #     # - expanded at runtime to match the batch size
    #     self.log_std = nn.Parameter(
    #         torch.zeros(action_dim)  # start with std = exp(0) = 1
    #     )

    # def forward(self, obs: torch.Tensor):
    #     """
    #     Forward pass of the policy.

    #     obs shape:
    #         (num_drones, obs_dim)
    #         or
    #         (batch_size, obs_dim)

    #     Returns:
    #         mean: (batch_size, action_dim)
    #         std:  (batch_size, action_dim)
    #     """

    #     # -------------------------------
    #     # Feature extraction
    #     # -------------------------------
    #     # Pass observations through the shared network.
    #     # Each drone is processed independently but with shared weights.
    #     x = self.net(obs)

    #     # -------------------------------
    #     # Action mean
    #     # -------------------------------
    #     # The network predicts the center of the action distribution.
    #     mean = self.mean_head(x)

    #     # -------------------------------
    #     # Action standard deviation
    #     # -------------------------------
    #     # Expand log_std so it matches the shape of mean.
    #     # This allows broadcasting over the batch dimension.
    #     log_std = self.log_std.expand_as(mean)

    #     # Standard deviation must be positive.
    #     # Exponentiation guarantees this.
    #     std = torch.exp(log_std)

    #     # PPO will build a Normal(mean, std) distribution from these.
    #     return mean, std



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
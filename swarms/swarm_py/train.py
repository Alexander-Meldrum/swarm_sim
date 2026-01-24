"""
Minimal RL training loop demonstrating:
- environment reset
- stepping simulator
- policy optimization

This is NOT a full RL algorithm (PPO/SAC),
but a scaffold suitable for extension.
"""

import torch
import torch.optim as optim

from swarm_env import SwarmEnv
from policy import SwarmPolicy


def main():
    print("[swarm_py] starting swarm controller")
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment and policy
    env = SwarmEnv(device=device)
    policy = SwarmPolicy(obs_dim=7, action_dim=3).to(device)

    # Optimizer, Adam smooths updates, adapts per-parameter, handles noisy signals
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Training loop
    for episode in range(10):
        # Reset simulator world
        obs = env.reset(
            num_drones_team_0=8,
            num_drones_team_1=8,
            max_steps=500
        )
        # print(obs)
        # break

        done = False
        # ep_reward = 0.0

        # Episode rollout, PPO
        while not done:            
            # ---- forward policy (OLD) ----
            mean, std = policy(obs)
            dist = torch.distributions.Normal(mean, std)

            action = dist.sample()
            log_prob_old = dist.log_prob(action).sum(dim=-1).detach()

            # ---- environment step ----
            next_obs, rewards, global_reward, done = env.step(action)

            # ---- advantage TODO  advantage = returns - value_prediction ----
            advantage = rewards - rewards.mean()  # (num_drones,)
            advantage_global = global_reward  # scalar

            # ---- forward policy (NEW) ----
            mean_new, std_new = policy(obs)
            dist_new = torch.distributions.Normal(mean_new, std_new)

            log_prob_new = dist_new.log_prob(action).sum(dim=-1)

            # ---- PPO ratio + clipping ----
            ratio = torch.exp(log_prob_new - log_prob_old)
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

            # ---- PPO loss (core line) ----
            loss = -torch.min(
                ratio * advantage,
                clipped_ratio * advantage
            ).mean()

            # print(loss.requires_grad)   # MUST be True
            # print(loss.grad_fn)         # MUST NOT be None

            # Clear gradients to prevent build-up
            optimizer.zero_grad()
            # Update gradients
            loss.backward()
            # Adam updates parameters
            optimizer.step()

            obs = next_obs

        # print(f"[swarm_py] Episode {episode}, reward = {ep_reward:.2f}")


if __name__ == "__main__":
    main()
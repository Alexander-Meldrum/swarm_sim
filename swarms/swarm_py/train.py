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
    policy = SwarmPolicy(obs_dim=6, action_dim=3).to(device)

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Training loop
    for episode in range(100):
        # Reset simulator world
        obs = env.reset(
            num_drones_team_0=8,
            num_drones_team_1=8,
            max_steps=500
        )

        done = False
        ep_reward = 0.0

        # Episode rollout
        while not done:
            # Compute actions for all drones
            actions = policy(obs)

            # Step simulator
            obs, reward, done = env.step(actions)
            ep_reward += reward.item()

            # Dummy loss: maximize reward directly
            # (Replace with PPO/SAC logic later)
            loss = -reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Episode {episode}, reward = {ep_reward:.2f}")


if __name__ == "__main__":
    main()
"""
PPO training loop with rollout returns (stable critic).

This implementation trains a swarm policy using:
- PPO (clipped surrogate objective)
- Rollout-based discounted returns (no GAE)
- Separate policy and value networks
- Masking for dead drones
- Tanh-squashed Gaussian actions
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from swarm_env import SwarmEnv
from policy import SwarmPolicy, ValueNet

# ============================================================
# ENVIRONMENT / OBSERVATION CONFIG
# ============================================================
OBS_DIM = 4 + 9*2          # Observation dimension (self state + neighbors etc.)
ACTION_DIM = 3            # Acceleration in 3D
ALIVE_IDX = 3   # Index of "alive flag" inside observation

# ============================================================
# TRAINING CONFIG
# ============================================================
EPISODE_COUNT = 3000
MAX_STEPS = 500
SEED = 0
NUM_DRONES_TEAM_0 = 15
NUM_DRONES_TEAM_1 = 30

# ============================================================
# OPTIMIZER SETTINGS
# ============================================================
POLICY_LEARNING_RATE = 1e-4
VALUE_LEARNING_RATE = 5e-5

# ============================================================
# PPO HYPERPARAMETERS
# ============================================================
GAMMA = 0.99              # Discount factor
VALUE_COEF = 0.1          # Critic loss weight
ENTROPY_COEF = 0.03       # Exploration strength
CLIP = 0.3                # PPO clipping epsilon

# ============================================================
# ROLLOUT CONFIG
# ============================================================
ROLLOUT_STEPS = 256       # Number of steps before PPO update
PPO_EPOCHS = 4            # How many times to iterate over collected data
MINIBATCH_SIZE = int(0.15 * (ROLLOUT_STEPS * NUM_DRONES_TEAM_0))

# ============================================================
# ACTION / NORMALIZATION CONSTANTS
# ============================================================
EPS = 1e-6
MAX_ACC = 10      # Max acceleration magnitude applied in simulation


def main():

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    env = SwarmEnv(device=device, obs_dim=OBS_DIM)

    # Actor (policy) and critic networks
    policy = SwarmPolicy(obs_dim=OBS_DIM, act_dim=ACTION_DIM).to(device)
    value_net = ValueNet(obs_dim=OBS_DIM).to(device)

    # Separate optimizers (standard PPO practice)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=POLICY_LEARNING_RATE)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=VALUE_LEARNING_RATE)

    entropy_coef = ENTROPY_COEF

    # ============================================================
    # MAIN TRAINING LOOP
    # ============================================================

    for idx, episode in enumerate(range(EPISODE_COUNT)):

        # Reset environment with deterministic seed progression
        obs = env.reset(
            seed=SEED + episode,
            num_drones_team_0=NUM_DRONES_TEAM_0,
            num_drones_team_1=NUM_DRONES_TEAM_1,
            max_steps=MAX_STEPS
        )

        done = False
        step_count = 0

        # Rollout buffer (stores one PPO batch)
        rollout = {
            "obs": [],
            "u": [],               # pre-tanh actions
            "log_prob_old": [],
            "reward": [],
            "value": [],
            "alive": [],
            "next_alive": [],
        }

        # ============================================================
        # INTERACTION LOOP
        # ============================================================
        while not done:

            # Alive mask (1 if drone alive, 0 if dead)
            alive = obs[:, ALIVE_IDX].float()

            # ------------------------------------------------------------
            # POLICY FORWARD PASS
            # ------------------------------------------------------------
            mean, std = policy(obs)        # Gaussian parameters
            dist = Normal(mean, std)

            u = dist.rsample()             # Reparameterized sample
            a = torch.tanh(u)              # Squash to (-1, 1)

            action = a * MAX_ACC           # Scale to physical acceleration
            action = action * alive.unsqueeze(-1)  # Zero action for dead drones

            # ### DEBUG
            # action = obs[..., 10:13]
            # action = action / (torch.norm(action, dim=-1, keepdim=True) + 1e-6)
            # ###

            # Log probability correction for tanh squashing
            log_prob_u = dist.log_prob(u).sum(dim=-1)
            log_det_jacobian = torch.log(1.0 - a.pow(2) + EPS).sum(dim=-1)
            log_prob_old = (log_prob_u - log_det_jacobian).detach()

            # ------------------------------------------------------------
            # VALUE PREDICTION
            # ------------------------------------------------------------
            value = value_net(obs).squeeze(-1)

            # ------------------------------------------------------------
            # ENVIRONMENT STEP
            # ------------------------------------------------------------
            next_obs, reward, done = env.step(action)

            # If all drones dead → force termination
            next_alive = next_obs[:, ALIVE_IDX].float()
            if not next_alive.any():
                done = True


            # DEBUG
            # Check which drones reached terminal state
            # Only check terminal rewards if we have at least one step stored
            # print(next_obs)
            # print(next_alive)
            # if len(rollout["alive"]) > 0:
            #     alive_last_step = rollout["alive"][-1]
            #     terminal_mask = (next_alive == 0) & (alive_last_step == 1)
            #     if terminal_mask.any():
            #         dead_indices = terminal_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            #         print(
            #             f"[Episode {episode} | Step {step_count}] Terminal reward applied for drones {dead_indices} "
            #             f"-> reward: {reward[terminal_mask].detach().cpu().numpy()}"
            #         )

            # ------------------------------------------------------------
            # STORE TRANSITION
            # ------------------------------------------------------------
            rollout["obs"].append(obs.detach())
            rollout["u"].append(u.detach())
            rollout["log_prob_old"].append(log_prob_old)
            rollout["reward"].append(reward.detach())
            rollout["value"].append(value.detach())
            rollout["alive"].append(alive.detach())
            rollout["next_alive"].append(next_alive.detach())

            obs = next_obs
            step_count += 1

            # ============================================================
            # PPO UPDATE TRIGGER
            # ============================================================

            if step_count % ROLLOUT_STEPS == 0 or done:

                # Concatenate rollout tensors
                obs_batch = torch.cat(rollout["obs"])
                u_batch = torch.cat(rollout["u"])
                log_prob_old_batch = torch.cat(rollout["log_prob_old"])
                reward_batch = torch.cat(rollout["reward"])
                value_batch = torch.cat(rollout["value"])
                alive_batch = torch.cat(rollout["alive"])
                next_alive_batch = torch.cat(rollout["next_alive"])

                # Determine rollout length T
                total_samples = reward_batch.shape[0]
                T = total_samples // NUM_DRONES_TEAM_0

                # Reshape to [T, N]
                reward_batch = reward_batch.view(T, NUM_DRONES_TEAM_0)
                # value_batch_2d = value_batch.view(T, NUM_DRONES_TEAM_0)
                # alive_batch_2d = alive_batch.view(T, NUM_DRONES_TEAM_0)
                next_alive_batch_2d = next_alive_batch.view(T, NUM_DRONES_TEAM_0)

                # ============================================================
                # ROLLOUT RETURNS (DISCOUNTED SUM)
                # ============================================================

                # Bootstrap from last state value
                with torch.no_grad():
                    # next_value_last = value_net(obs).squeeze(-1)
                    # End bootstrapping on true terminal
                    if done:
                        next_value_last = torch.zeros(NUM_DRONES_TEAM_0, device=device)
                    else:
                        next_value_last = value_net(obs).squeeze(-1)

                returns = torch.zeros_like(reward_batch)
                R = next_value_last
                    
                # Backward recursion:
                # R_t = r_t + gamma * R_{t+1}
                for t in reversed(range(T)):
                    # R = reward_batch[t] + GAMMA * R * alive_batch_2d[t]
                    R = reward_batch[t] + GAMMA * R * next_alive_batch_2d[t]
                    returns[t] = R

                returns = returns.view(-1)

                # ============================================================
                # ADVANTAGE ESTIMATION
                # ============================================================
                # A = R - V(s)
                advantage = returns - value_batch.detach()
                advantage = advantage.view(T, NUM_DRONES_TEAM_0)

                # # Remove swarm-wide bias per timestep
                # # (prevents global directional collapse)
                # advantage = advantage - advantage.mean(dim=1, keepdim=True)

                # # Normalize globally (improves PPO stability)
                # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # Normalize per drone across time
                for i in range(NUM_DRONES_TEAM_0):
                    a = advantage[:, i]
                    
                    a_std = a.std(unbiased=False)
                    if a_std < 1e-6:
                        advantage[:, i] = 0.0
                    else:
                        advantage[:, i] = (a - a.mean()) / (a_std + 1e-8)

                advantage = advantage.view(-1)

                # ============================================================
                # PPO OPTIMIZATION LOOP
                # ============================================================
                batch_size = obs_batch.size(0)
                indices = torch.randperm(batch_size)

                for _ in range(PPO_EPOCHS):
                    for start in range(0, batch_size, MINIBATCH_SIZE):
                        end = start + MINIBATCH_SIZE
                        mb_idx = indices[start:end]

                        mb_alive = alive_batch[mb_idx]
                        mask_sum = mb_alive.sum().clamp(min=1.0)

                        # ---------------- POLICY UPDATE ----------------

                        mean_new, std_new = policy(obs_batch[mb_idx])
                        dist_new = Normal(mean_new, std_new)

                        log_prob_u_new = dist_new.log_prob(u_batch[mb_idx]).sum(dim=-1)
                        a = torch.tanh(u_batch[mb_idx])
                        log_det_jacobian = torch.log(
                            1.0 - a.pow(2) + EPS
                        ).sum(dim=-1)

                        log_prob_new = log_prob_u_new - log_det_jacobian

                        # PPO importance ratio
                        ratio = torch.exp(log_prob_new - log_prob_old_batch[mb_idx])

                        # Clipped objective
                        policy_loss = torch.min(
                            ratio * advantage[mb_idx],
                            torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage[mb_idx]
                        )

                        # Mask dead drones
                        policy_loss = -(policy_loss * mb_alive).sum() / mask_sum

                        # Entropy bonus (encourages exploration)
                        entropy = dist_new.entropy().sum(dim=-1)
                        entropy_bonus = (entropy * mb_alive).sum() / mask_sum

                        actor_loss = policy_loss - entropy_coef * entropy_bonus

                        policy_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                        policy_optimizer.step()

                        # ---------------- CRITIC UPDATE ----------------

                        value_pred = value_net(obs_batch[mb_idx]).squeeze(-1)

                        value_pred_clipped = value_batch[mb_idx] + \
                            (value_pred - value_batch[mb_idx]).clamp(-0.5, 0.5)    # PPO2-style value clipping

                        value_loss_unclipped = (value_pred - returns[mb_idx].detach()) ** 2
                        value_loss_clipped = (value_pred_clipped - returns[mb_idx].detach()) ** 2

                        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                        value_loss = (value_loss * mb_alive).sum() / mask_sum

                        critic_loss = VALUE_COEF * value_loss

                        value_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
                        value_optimizer.step()

                # ============================================================
                # DEBUG PRINTS (TRAINING HEALTH CHECK)
                # ============================================================
                print("action std across drones:", action.std(dim=0))
                print("obs std across drones:", obs.std(dim=0).mean())
                print("mean |action|:", action.abs().mean().item())
                print("mean action norm:", action.norm(dim=1).mean().item())

                print(
                    "action mean:", action.mean(dim=0).detach().cpu().numpy(),
                    "action std :", action.std(dim=0).detach().cpu().numpy()
                )

                print("per-drone action norm std:", action.norm(dim=1).std().item())

                print(
                    "policy mean mean:", mean.mean().item(),
                    "policy mean std :", mean.std().item(),
                    "policy std mean :", std.mean().item(),
                    "policy std std  :", std.std().item(),
                )

                print(
                    "ratio mean:", ratio.mean().item(),
                    "ratio std :", ratio.std().item(),
                    "ratio min/max:",
                    ratio.min().item(),
                    ratio.max().item()
                )

                print(
                    "adv mean:", advantage.mean().item(),
                    "adv std :", advantage.std().item()
                )

                print(
                    "dist mean:", obs[:, 0:3].norm(dim=1).mean().item(),
                    "reward mean:", reward.mean().item()
                )

                with torch.no_grad():
                    v = value_net(obs).squeeze(-1)

                print(
                    "value mean:", v.mean().item(),
                    "value std :", v.std().item(),
                    "reward mean:", reward.mean().item()
                )

                mean_dir = action.mean(dim=0)
                print(
                    "mean action direction:",
                    (mean_dir / (mean_dir.norm() + 1e-8)).detach().cpu().numpy()
                )

                # Clear rollout buffer for next batch
                for k in rollout:
                    rollout[k].clear()


if __name__ == "__main__":
    main()
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
import math
import torch.nn as nn

from torch.distributions import Normal


from swarm_env import SwarmEnv
from policy import SwarmPolicy, ValueNet

OBS_DIM = 7
ACTION_DIM = 3
ALIVE_IDX = OBS_DIM - 1

EPISODE_COUNT = 500     # Number of full environment resets (training episodes)
MAX_STEPS = 1000        # Maximum number of simulation steps per episode, Episode terminates early if all drones are dead
SEED = 0
NUM_DRONES_TEAM_0 = 10
NUM_DRONES_TEAM_1 = 0

V_CLIP = MAX_STEPS      # Assumes magnitude of reward roughly 1, MAX_STEPS * Reward Magnitude, Prevents critic explosion when rewards accumulate over long episodes
EPS = 1e-6              # numerical stability. Used in log(), division, and tanh Jacobian correction
GAMMA = 0.99            # Discount factor. Controls how much future rewards matter
VALUE_COEF = 0.05       # Weight of value (critic) loss relative to policy loss. Smaller values reduce critic dominance
ENTROPY_COEF = 0.0022   # Entropy bonus coefficient. Encourages exploration by preventing premature policy collapse
CLIP = 0.2              # PPO clipping range (ε). Limits policy updates to prevent large, destabilizing changes
SAT_COEF = 0.005        # Penalty for action saturation (|a| close to 1 after tanh). Discourages banging against action limits constantly

ROLLOUT_STEPS = 250     # Number of environment steps collected before each PPO update. Must be large enough for stable advantage estimates
PPO_EPOCHS = 5          # Number of passes over the same rollout data. Higher = more sample efficiency, but risk of overfitting
MINIBATCH_SIZE = int(0.15 * (ROLLOUT_STEPS * NUM_DRONES_TEAM_0))  # Number of samples per PPO minibatch, This should be ≪ total batch size for good SGD behavior

MAX_ACC = 3.0           # max acceleration magnitude
MAX_DISTANCE = 100      # Rougly the maximum size of the arena, used for normalizing observations for NN
MAX_VELOCITY = 15       # Max velocity of drones, only used for normalizing. Real Max Velocity set in sim_server/configs/sim.yaml


def main():
    print("[swarm_py] starting swarm controller")
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment and policy
    env = SwarmEnv(device=device, obs_dim=OBS_DIM)
    policy = SwarmPolicy(obs_dim=OBS_DIM, act_dim=ACTION_DIM).to(device)
    value_net = ValueNet(obs_dim=OBS_DIM).to(device)

    # Optimizer, Adam smooths updates, adapts per-parameter, handles noisy signals
    optimizer = optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=3e-4)
    

    # Training loop
    for idx, episode in enumerate(range(EPISODE_COUNT)):
        # Reset simulator world
        obs = env.reset(
            seed=SEED+idx,
            num_drones_team_0=NUM_DRONES_TEAM_0,
            num_drones_team_1=NUM_DRONES_TEAM_1,
            max_steps=MAX_STEPS
        )
        # Normalize
        obs[:, 0:3] /= MAX_DISTANCE
        obs[:, 3:6] /= MAX_VELOCITY
        # Extract alive flag from observations
        alive = obs[:, ALIVE_IDX].float()          # (num_drones,)

        done = False
        step_count = 0


        # ============================================================
        # ROLLOUT STORAGE (OLD POLICY DATA)
        # ============================================================
        rollout = {
            "obs": [],
            "u": [],
            "log_prob_old": [],
            "reward": [],
            "value": [],
            "next_value": [],
            "alive": [],
        }

        # Episode rollout, PPO (Proximal Policy Optimization)
        # Policy: “What action should I take?”
        # Value: “How good is this situation in the long run?”
        # Advantage: “Was my action better than expected here?”
        while not done:
            # ============================================================
            # 1. FORWARD OLD POLICY (sampling actions)
            # ============================================================
            # The policy answers: "what actions should I try in this state?"
            # It does NOT know how good the state is.
            mean, std = policy(obs)                    # (num_drones, act_dim), note: already squashed
            dist = Normal(mean, std)                   # Gaussian policy π_old

            # Sample in unconstrained ℝ space
            u = dist.sample()                          # (num_drones, act_dim)

            # Squash to (-1, 1) to enforce action bounds
            a = torch.tanh(u)


            # Scale to environment units (e.g. max acceleration)
            action = a * MAX_ACC                       
            action = action * alive.unsqueeze(-1)      # dead drones send zero action, THIS is sent to simulator

            # ============================================================
            # 2. LOG-PROBABILITY UNDER OLD POLICY
            # ============================================================
            # PPO needs the probability of *the exact action sent to env*
            # under the OLD policy, to measure how much the policy changes.

            # Log-prob in Gaussian (unconstrained) space
            log_prob_u = dist.log_prob(u).sum(dim=-1)  # (num_drones,)

            # Tanh change-of-variables correction
            # This fixes the probability after squashing
            log_det_jacobian = torch.log(
                1.0 - a.pow(2) + EPS
            ).sum(dim=-1)

            # Final OLD log-probability (detached!)
            log_prob_old = (log_prob_u - log_det_jacobian).detach()


            # ============================================================
            # 3. VALUE NETWORK (STATE EVALUATION)
            # ============================================================
            # The value network answers:
            # "How much total future reward should I expect from this state?"
            value = value_net(obs).squeeze(-1)         # (num_drones,)


            # ============================================================
            # 4. ENVIRONMENT STEP
            # ============================================================
            next_obs, reward, done = env.step(action)  # reward: (num_drones,)

            # Normalize observations (VERY important for NN stability)
            next_obs[:, 0:3] /= MAX_DISTANCE
            next_obs[:, 3:6] /= MAX_VELOCITY

            alive = next_obs[:, ALIVE_IDX].float()          # (num_drones,), PyTorch does not allow: float_tensor * bool_tensor

            if not alive.any():
                done = True  # end episode logically
            if done:
                alive = torch.ones_like(alive)


            # ============================================================
            # 5. VALUE OF NEXT STATE
            # ============================================================
            # Used to estimate future reward, The value network does NOT predict the next reward.
            # It predicts the total future reward from now on.
            with torch.no_grad():
                next_value = value_net(next_obs).squeeze(-1)  # (num_drones,)


            # ============================================================
            # STORE ROLLOUT DATA (NO LEARNING HERE)
            # ============================================================
            rollout["obs"].append(obs.detach())
            rollout["u"].append(u.detach())
            rollout["log_prob_old"].append(log_prob_old.detach())
            rollout["reward"].append(reward.detach())
            rollout["value"].append(value.detach())
            rollout["next_value"].append(next_value.detach())
            rollout["alive"].append(alive.detach())

            step_count += 1

            # ============================================================
            # BATCH PPO UPDATE EVERY 500 STEPS
            # ============================================================
            if step_count % ROLLOUT_STEPS == 0 or done:     # Make sure to update policy even if all drones crash
                # Stack rollout tensors
                obs_batch = torch.cat(rollout["obs"])
                u_batch = torch.cat(rollout["u"])
                log_prob_old_batch = torch.cat(rollout["log_prob_old"])
                reward_batch = torch.cat(rollout["reward"])
                value_batch = torch.cat(rollout["value"])
                next_value_batch = torch.cat(rollout["next_value"])

                alive_batch = torch.cat(rollout["alive"]).float() # [T*num_drones]
                # ============================================================
                # ADVANTAGE
                # ============================================================
                # Advantage answers:
                # "Was this action better or worse than expected in THIS state?"

                # One-step TD advantage:
                # actual outcome        - expected outcome
                advantage = reward_batch + GAMMA * next_value_batch * alive_batch - value_batch

                # Normalize advantage to reduce gradient variance
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # Value target
                value_target = reward_batch + GAMMA * next_value_batch * alive_batch
                value_target = torch.clamp(value_target, -V_CLIP, V_CLIP)  # Clamping = stability tool, not theory  TODO, is this best approach?

                # ============================================================
                # PPO OPTIMIZATION (MULTIPLE EPOCHS + MINIBATCHES)
                # ============================================================
                batch_size = obs_batch.size(0)
                indices = torch.randperm(batch_size)

                for _ in range(PPO_EPOCHS):
                    for start in range(0, batch_size, MINIBATCH_SIZE):
                        end = start + MINIBATCH_SIZE
                        mb_idx = indices[start:end]
                        mb_alive = alive_batch[mb_idx]
                        mask_sum = mb_alive.sum().clamp(min=1.0)

                        # ============================================================
                        # FORWARD NEW POLICY (same obs, same sampled action u)
                        # ============================================================
                        mean_new, std_new = policy(obs_batch[mb_idx])
                        dist_new = Normal(mean_new, std_new)

                        entropy = dist_new.entropy().sum(dim=-1)   # (num_drones,)
                        # entropy_bonus = entropy.mean()
                        entropy_bonus = (entropy * mb_alive).sum() / mask_sum

                        log_prob_u_new = dist_new.log_prob(u_batch[mb_idx]).sum(dim=-1)

                        a = torch.tanh(u_batch[mb_idx])
                        log_det_jacobian = torch.log(
                            1.0 - a.pow(2) + EPS
                        ).sum(dim=-1)

                        log_prob_new = log_prob_u_new - log_det_jacobian

                        # ============================================================
                        # PPO CLIPPED POLICY LOSS
                        # ============================================================
                        ratio = torch.exp(log_prob_new - log_prob_old_batch[mb_idx])

                        policy_loss_per_sample = torch.min(
                            ratio * advantage[mb_idx],
                            torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage[mb_idx]
                        )

                        policy_loss = -(policy_loss_per_sample * mb_alive).sum() / mask_sum

                        # ============================================================
                        # VALUE LOSS (CRITIC TRAINING)
                        # ============================================================
                        value_pred = value_net(obs_batch[mb_idx]).squeeze(-1)
                        value_error = (value_pred - value_target[mb_idx].detach()) ** 2
                        value_loss = (value_error * mb_alive).sum() / mask_sum

                        # ============================================================
                        # TOTAL LOSS + UPDATE
                        # ============================================================
                        # Value loss stabilizes learning
                        # Policy loss improves actions
                        saturation = (a.abs() > 0.95).float().mean()  # TODO

                        total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_bonus + SAT_COEF * saturation

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                # RL Debug prints
                print(
                "action mean:", action.mean(dim=0).cpu().numpy(),
                "action std :", action.std(dim=0).cpu().numpy()
                )
                print("per-drone action norm std:", action.norm(dim=1).std().item())
                print(
                    "policy mean mean:", mean.mean().item(),
                    "policy mean std :", mean.std().item(),
                    "policy std mean :", std.mean().item()
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
                    (mean_dir / (mean_dir.norm() + 1e-8)).cpu().numpy()
                )
                # ============================================================
                # CLEAR ROLLOUT BUFFER (START FRESH)
                # ============================================================
                for k in rollout:
                    rollout[k].clear()

            # Move forward in time
            obs = next_obs




if __name__ == "__main__":
    main()
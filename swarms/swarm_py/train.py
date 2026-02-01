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

EPS = 1e-6         # numerical stability
GAMMA = 0.99
VALUE_COEF = 0.05
ENTROPY_COEF = 0.0022
CLIP = 0.2         # PPO clipping range
V_CLIP = 500.0
SAT_COEF = 0.005

EPISODE_COUNT = 500  # Amount of simulation runs
MAX_STEPS = 1000
SEED = 0
NUM_DRONES_TEAM_0 = 10
NUM_DRONES_TEAM_1 = 0

ROLLOUT_STEPS = 250
PPO_EPOCHS = 5
MINIBATCH_SIZE = 64 # TODO should consider num_drones batch size

MAX_ACC = 3.0      # max acceleration magnitude
MAX_DISTANCE = 100
MAX_VELOCITY = 15


def main():
    print("[swarm_py] starting swarm controller")
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment and policy
    env = SwarmEnv(device=device, obs_dim=OBS_DIM)
    policy = SwarmPolicy(obs_dim=OBS_DIM, act_dim=ACTION_DIM).to(device)
    value_net = ValueNet(obs_dim=OBS_DIM).to(device)

    # Optimizer, Adam smooths updates, adapts per-parameter, handles noisy signals
    # optimizer = optim.Adam(policy.parameters(), lr=3e-4)
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
        # obs[:, 6:7] /= MAX_DISTANCE*MAX_DISTANCE  # distance squared
        obs[:, 3:6] /= MAX_VELOCITY
        # print(obs[:, 6:7] )
        # print(obs)
        # break
        # Extract alive flag from observations
        alive = obs[:, ALIVE_IDX].float()          # (num_drones,)
        done_agent = 1.0 - alive                # 1 = terminal, 0 = ongoing
        done_agent.shape == (NUM_DRONES_TEAM_0,)

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
            "done": [],
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
            # next_obs[:, 6:7] /= MAX_DISTANCE*MAX_DISTANCE  # distance squared
            next_obs[:, 3:6] /= MAX_VELOCITY

            alive = obs[:, ALIVE_IDX].float()          # (num_drones,), PyTorch does not allow: float_tensor * bool_tensor
            done_agent = 1.0 - alive                   # 1 = terminal, 0 = ongoing
            done_agent.shape == (NUM_DRONES_TEAM_0,)
            if done:
                done_agent = torch.ones_like(done_agent)


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
            # rollout["done"].append(torch.full_like(reward, float(done)))
            rollout["done"].append(done_agent.detach())

            step_count += 1

            # ============================================================
            # BATCH PPO UPDATE EVERY 500 STEPS
            # ============================================================
            if step_count % ROLLOUT_STEPS == 0:
                # Stack rollout tensors
                obs_batch = torch.cat(rollout["obs"])
                u_batch = torch.cat(rollout["u"])
                log_prob_old_batch = torch.cat(rollout["log_prob_old"])
                reward_batch = torch.cat(rollout["reward"])
                value_batch = torch.cat(rollout["value"])
                next_value_batch = torch.cat(rollout["next_value"])

                # Done mask: 1 = non-terminal, 0 = terminal
                done_batch = torch.cat(rollout["done"]).float() # [T*num_drones]
                not_done_batch = 1.0 - done_batch

                # ============================================================
                # ADVANTAGE
                # ============================================================
                # Advantage answers:
                # "Was this action better or worse than expected in THIS state?"

                # One-step TD advantage:
                # actual outcome        - expected outcome
                advantage = reward_batch + GAMMA * next_value_batch * not_done_batch - value_batch

                # Normalize advantage to reduce gradient variance
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # Value target
                value_target = reward_batch + GAMMA * next_value_batch * not_done_batch
                # value_target = reward_batch + GAMMA * next_value_batch
                value_target = torch.clamp(value_target, -V_CLIP, V_CLIP)  # TODO, is this best approach?

                # ============================================================
                # PPO OPTIMIZATION (MULTIPLE EPOCHS + MINIBATCHES)
                # ============================================================
                batch_size = obs_batch.size(0)
                indices = torch.randperm(batch_size)

                for _ in range(PPO_EPOCHS):
                    for start in range(0, batch_size, MINIBATCH_SIZE):
                        end = start + MINIBATCH_SIZE
                        mb_idx = indices[start:end]
                        mb_mask = not_done_batch[mb_idx]        # [mb]
                        mask_sum = mb_mask.sum().clamp(min=1.0)

                        # ============================================================
                        # FORWARD NEW POLICY (same obs, same sampled action u)
                        # ============================================================
                        mean_new, std_new = policy(obs_batch[mb_idx])
                        dist_new = Normal(mean_new, std_new)

                        entropy = dist_new.entropy().sum(dim=-1)   # (num_drones,)
                        entropy_bonus = entropy.mean()

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

                        policy_loss = -torch.mean(
                            torch.min(
                                ratio * advantage[mb_idx],
                                torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage[mb_idx]
                            )
                        )

                        # ============================================================
                        # VALUE LOSS (CRITIC TRAINING)
                        # ============================================================
                        value_pred = value_net(obs_batch[mb_idx]).squeeze(-1)
                        value_loss = torch.mean(
                            (value_pred - value_target[mb_idx].detach()) ** 2
                        )

                        # ============================================================
                        # TOTAL LOSS + UPDATE
                        # ============================================================
                        # Value loss stabilizes learning
                        # Policy loss improves actions

                        
                        saturation = (a.abs() > 0.95).float().mean()

                        total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_bonus + SAT_COEF * saturation

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

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




            




        # while not done:        
            
        #     # ============================================================
        #     # 1. FORWARD OLD POLICY (sampling)
        #     # ============================================================

        #     mean, std = policy(obs)               # obs: (num_drones, obs_dim)
        #     dist = Normal(mean, std)              # Gaussian policy

        #     # Sample in unconstrained space ℝ
        #     u = dist.sample()                     # shape: (num_drones, act_dim)

        #     # Squash to (-1, 1) so actions are bounded
        #     a = torch.tanh(u)

        #     # Scale to environment units (acceleration)
        #     action = a * MAX_ACC                  # THIS is sent to simulator

        #     # ============================================================
        #     # 2. LOG-PROBABILITY OF THE ACTION (OLD POLICY)
        #     # ============================================================

        #     # Log-probability in unconstrained Gaussian space
        #     # Sum over action dimensions → one log_prob per drone
        #     log_prob_u = dist.log_prob(u).sum(dim=-1)

        #     # Change-of-variables correction for tanh:
        #     # tanh(u) squashes space, changing probability density
        #     log_det_jacobian = torch.log(
        #         1.0 - a.pow(2) + EPS
        #     ).sum(dim=-1)

        #     # Final log-probability of the action actually sent to env
        #     # detach() is CRITICAL: old policy must not get gradients
        #     log_prob_old = (log_prob_u - log_det_jacobian).detach()

        #     # ============================================================
        #     # 3. ENVIRONMENT STEP
        #     # ============================================================
        #     # Debug controller showing we can move towards origin
        #     # action = -obs[:, 0:3] * 10

        #     next_obs, reward, done = env.step(action)  # reward: (num_drones,)

        #     # Normalize
        #     next_obs[:, 0:3] /= MAX_DISTANCE
        #     next_obs[:, 3:6] /= MAX_VELOCITY

        #     # ============================================================
        #     # 4. ADVANTAGE (MINIMAL, BUT VALID)
        #     # ============================================================

        #     # Simplest advantage:
        #     # "Was this drone better or worse than average?"
        #     # TODO consider next_obs
        #     # advantage = reward - reward.mean()

        #     # # Normalize advantage → stable gradients
        #     # advantage = advantage / (advantage.std() + 1e-8)

        #     advantage = reward.detach()
        #     advantage = advantage / (advantage.std() + 1e-8)

        #     # ============================================================
        #     # 5. FORWARD NEW POLICY (same states, same sampled u)
        #     # ============================================================

        #     mean_new, std_new = policy(obs)
        #     dist_new = Normal(mean_new, std_new)

        #     # Log-prob under new policy (same u!)
        #     log_prob_u_new = dist_new.log_prob(u).sum(dim=-1)

        #     # Apply SAME tanh correction
        #     log_prob_new = log_prob_u_new - log_det_jacobian

        #     # ============================================================
        #     # 6. PPO RATIO + CLIPPED OBJECTIVE
        #     # ============================================================

        #     # Likelihood ratio: how much did policy change?
        #     ratio = torch.exp(log_prob_new - log_prob_old)

        #     # PPO clipped objective:
        #     # prevents overly large policy updates
        #     loss = -torch.mean(
        #         torch.min(
        #             ratio * advantage,
        #             torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advantage
        #         )
        #     )

        #     # ============================================================
        #     # 7. BACKPROP + UPDATE
        #     # ============================================================

        #     optimizer.zero_grad()   # clear old gradients
        #     loss.backward()         # compute gradients
        #     optimizer.step()        # update policy parameters

        #     # Move to next state
        #     obs = next_obs


            
            
            

        print("policy std mean:", std.mean().item())





        # print(f"[swarm_py] Episode {episode}, reward = {ep_reward:.2f}")


if __name__ == "__main__":
    main()
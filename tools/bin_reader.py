import struct
import numpy as np

# Binary schema for swarm log
# *******************************************************
# ---- step header ----
# step: u64
# num_drones: u32
# global_reward: f64
STEP_HEADER_FMT = "<Q I d"
STEP_HEADER_SIZE = struct.calcsize(STEP_HEADER_FMT)

# ---- per-drone data ----
# px, py, pz, vx, vy, vz (all f32)
# Note: Add "<" in dependant scripts afterwards to enable multiplying this value by num_drones
DRONE_FMT_BASE = "f"
DRONE_BASE_SIZE = struct.calcsize(DRONE_FMT_BASE)
# *******************************************************

DRONE_STATE_FIELDS_NUM = 6


class SwarmLog:
    def __init__(self, steps, rewards, pos, vel):
        self.steps = steps      # (T,)
        self.rewards = rewards  # (T,)
        self.pos = pos          # (T, N, 3)
        self.vel = vel          # (T, N, 3)


def read_swarm_log(path: str) -> SwarmLog:
    steps = []
    rewards = []
    positions = []
    velocities = []

    with open(path, "rb") as f:
        while True:
            # ---- read step header ----
            header = f.read(STEP_HEADER_SIZE)
            if not header:
                break

            step, num_drones, reward = struct.unpack(
                STEP_HEADER_FMT, header
            )

            # ---- read all drone data in one shot ----
            drone_fmt = "<" + DRONE_FMT_BASE * (DRONE_STATE_FIELDS_NUM * num_drones)
            drone_bytes = f.read(DRONE_STATE_FIELDS_NUM * num_drones * DRONE_BASE_SIZE)

            drone_states = struct.unpack(drone_fmt, drone_bytes)

            arr = np.asarray(drone_states, dtype=np.float32).reshape(
                num_drones, DRONE_STATE_FIELDS_NUM
            )

            steps.append(step)
            rewards.append(reward)
            positions.append(arr[:, :3])
            velocities.append(arr[:, 3:])

    return SwarmLog(
        steps=np.asarray(steps, dtype=np.uint64),
        rewards=np.asarray(rewards, dtype=np.float32),
        pos=np.stack(positions),
        vel=np.stack(velocities),
    )
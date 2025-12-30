# bin_reader.py
import struct
import numpy as np

from schema import (
    STEP_HEADER_FMT,
    STEP_HEADER_SIZE,
    DRONE_FMT_BASE,
    DRONE_SIZE,
)


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
            drone_fmt = "<" + "f" * (6 * num_drones)
            drone_bytes = f.read(6 * num_drones * 4)

            data = struct.unpack(drone_fmt, drone_bytes)

            arr = np.asarray(data, dtype=np.float32).reshape(
                num_drones, 6
            )

            steps.append(step)
            rewards.append(reward)
            positions.append(arr[:, :3])
            velocities.append(arr[:, 3:])

    return SwarmLog(
        steps=np.asarray(steps, dtype=np.uint64),
        rewards=np.asarray(rewards, dtype=np.float64),
        pos=np.stack(positions),
        vel=np.stack(velocities),
    )
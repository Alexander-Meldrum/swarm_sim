import struct
import numpy as np

# Binary schema for swarm state log
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

DRONE_STATE_FIELDS_NUM = 6
# *******************************************************
# Binary schema for swarm event log
# *******************************************************

EVENT_FMT = "<B Q I I I"  # kind:u8, step:u64, drone_a:u32, drone_b:u32, target_id:u32
EVENT_SIZE = struct.calcsize(EVENT_FMT)
NONE_U32 = 0xFFFFFFFF
# *******************************************************

class SwarmStateLog:
    def __init__(self, steps, global_rewards, pos, vel):
        self.steps = steps      # (T,)
        self.global_rewards = global_rewards  # (T,)
        self.pos = pos          # (T, N, 3)
        self.vel = vel          # (T, N, 3)

class SwarmEventLog:
    def __init__(self, steps, kind, drone_a, drone_b, target_id):
        self.steps = steps      # (T,)
        self.kind = kind
        self.drone_a = drone_a
        self.drone_b = drone_b
        self.target_id = target_id


def read_swarm_state_log(path: str) -> SwarmStateLog:
    steps = []
    global_rewards = []
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
            global_rewards.append(reward)
            positions.append(arr[:, :3])
            velocities.append(arr[:, 3:])

    return SwarmStateLog(
        steps=np.asarray(steps, dtype=np.uint64),
        global_rewards=np.asarray(global_rewards, dtype=np.float32),
        pos=np.stack(positions),
        vel=np.stack(velocities),
    )


def read_swarm_event_log(path: str) -> SwarmStateLog:
    steps = []
    global_rewards = []
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
            global_rewards.append(reward)
            positions.append(arr[:, :3])
            velocities.append(arr[:, 3:])

    return SwarmStateLog(
        steps=np.asarray(steps, dtype=np.uint64),
        global_rewards=np.asarray(global_rewards, dtype=np.float32),
        pos=np.stack(positions),
        vel=np.stack(velocities),
    )
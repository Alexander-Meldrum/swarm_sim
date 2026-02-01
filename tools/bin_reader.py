import struct
import numpy as np

# Binary schema for swarm state log
# *******************************************************
# ---- step header ----
# step: u64
# num_drones: u32

STEP_HEADER_FMT = "<Q I"
STEP_HEADER_SIZE = struct.calcsize(STEP_HEADER_FMT)

# ---- per-drone data ----
# px, py, pz, vx, vy, vz, rewards (all f32) # 
# Note: Add "<" in dependant scripts afterwards to enable multiplying this value by num_drones
DRONE_FMT_BASE = "f"
DRONE_BASE_SIZE = struct.calcsize(DRONE_FMT_BASE)

DRONE_STATE_FIELDS_NUM = 7
# *******************************************************
# Binary schema for swarm event log
# *******************************************************

EVENT_FMT = "<Q B I fff I fff I"  # step:u64, kind:u8, drone_a:u32, drone_a_position: Vec3<f32>, drone_b:u32, drone_b_position:Vec3<f32>, target_id:u32
EVENT_SIZE = struct.calcsize(EVENT_FMT)
NONE_U32 = 0xFFFFFFFF
# *******************************************************

class SwarmStateLog:
    def __init__(self, steps, rewards, pos, vel):
        self.steps = steps      # (T,)
        self.pos = pos          # (T, N, 3)
        self.vel = vel          # (T, N, 3)
        self.rewards = rewards  # (T, N)

class SwarmEventLog:
    def __init__(self, steps, kind, drone_a, drone_a_position, drone_b, drone_b_position, target_id):
        self.steps = steps      # (T,)
        self.kind = kind
        self.drone_a = drone_a
        self.drone_a_position = drone_a_position
        self.drone_b = drone_b
        self.drone_b_position = drone_b_position
        self.target_id = target_id


def read_swarm_state_log(path: str) -> SwarmStateLog:
    """
    Reads a binary swarm state log file.

    Binary format:
    ------------------
    Step header:
      step: u64
      num_drones: u32

    Per-drone data (for each of num_drones drones):
      px, py, pz, vx, vy, vz, rewards  (all f32)
    """
    steps = []
    positions = []
    velocities = []
    rewards = []

    with open(path, "rb") as f:
        while True:
            # ---- read step header ----
            header = f.read(STEP_HEADER_SIZE)
            if not header:
                # End of file reached
                break

            if len(header) != STEP_HEADER_SIZE:
                raise ValueError(
                    f"Unexpected step header size: expected {STEP_HEADER_SIZE} bytes, got {len(header)}"
                )

            # Unpack step header: step (u64), num_drones (u32)
            step, num_drones = struct.unpack(STEP_HEADER_FMT, header)
            # print(f"Debug: Reading step {step}, num_drones={num_drones}")

            if num_drones == 0:
                # Skip steps with no drones
                print(f"Warning: Step {step} has zero drones, skipping")
                continue

            # ---- read all drone data for this step ----
            expected_bytes = DRONE_STATE_FIELDS_NUM * num_drones * DRONE_BASE_SIZE
            drone_bytes = f.read(expected_bytes)

            if len(drone_bytes) != expected_bytes:
                raise ValueError(
                    f"Unexpected end of file or incomplete drone data for step {step}: "
                    f"expected {expected_bytes} bytes, got {len(drone_bytes)}"
                )

            # Construct format string: "<ffff...f" repeated for each field of each drone
            drone_fmt = "<" + "f" * (DRONE_STATE_FIELDS_NUM * num_drones)

            # Unpack all drone data into a flat tuple
            drone_states = struct.unpack(drone_fmt, drone_bytes)

            # Reshape into array (num_drones, 7) -> [px, py, pz, vx, vy, vz, rewards]
            arr = np.asarray(drone_states, dtype=np.float32).reshape(
                num_drones, DRONE_STATE_FIELDS_NUM
            )

            # Append per-step data
            steps.append(step)
            positions.append(arr[:, :3])         # px, py, pz
            velocities.append(arr[:, 3:6])       # vx, vy, vz
            rewards.append(arr[:, 6]) # rewards

    # Convert lists into numpy arrays
    return SwarmStateLog(
        steps=np.asarray(steps, dtype=np.uint64),                 # shape (T,)
        pos=np.stack(positions) if positions else np.empty((0,0,3)),    # shape (T, N, 3)
        vel=np.stack(velocities) if velocities else np.empty((0,0,3)),  # shape (T, N, 3)
        rewards=np.stack(rewards) if rewards else np.empty((0,0))  # shape (T, N)
    )

class SwarmEventLog:
    def __init__(self, steps, kind, drone_a, drone_a_position, drone_b, drone_b_position, target_id):
        self.steps = steps
        self.kind = kind
        self.drone_a = drone_a
        self.drone_a_position = drone_a_position
        self.drone_b = drone_b
        self.drone_b_position = drone_b_position
        self.target_id = target_id

def read_swarm_event_log(path: str) -> SwarmEventLog:
    steps = []
    kinds = []
    drone_a_list = []
    drone_a_positions = []
    drone_b_list = []
    drone_b_positions = []
    target_ids = []
    with open(path, "rb") as f:
        while True:
            data = f.read(EVENT_SIZE)
            # print(data)
            if not data:
                break
            if len(data) != EVENT_SIZE:
                raise ValueError(
                    f"Incomplete event read: expected {EVENT_SIZE} bytes, got {len(data)}"
                )

            unpacked = struct.unpack(EVENT_FMT, data)
            # print(unpacked)

            # Map fields
            step = unpacked[0]
            kind = unpacked[1]
            drone_a = unpacked[2]
            drone_a_pos = unpacked[3:6]  # x, y, z
            drone_b = unpacked[6]        # This is 0xFFFFFFFF for target hits
            drone_b_pos = unpacked[7:10]  # x, y, z, This is 0 0 0 for target hits
            target_id = unpacked[10]

            steps.append(step)
            kinds.append(kind)
            drone_a_list.append(drone_a)
            drone_a_positions.append(drone_a_pos)
            drone_b_list.append(drone_b)
            drone_b_positions.append(drone_b_pos)
            target_ids.append(target_id)

    return SwarmEventLog(
        steps=np.asarray(steps, dtype=np.uint64),
        kind=np.asarray(kinds, dtype=np.uint8),
        drone_a=np.asarray(drone_a_list, dtype=np.uint32),
        drone_a_position=np.asarray(drone_a_positions, dtype=np.float32),
        drone_b=np.asarray(drone_b_list, dtype=np.uint32),
        drone_b_position=np.asarray(drone_b_positions, dtype=np.float32),
        target_id=np.asarray(target_ids, dtype=np.uint32),
    )
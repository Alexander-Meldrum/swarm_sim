# Shared binary schema for swarm log

import struct

# ---- step header ----
# step: u64
# num_drones: u32
# global_reward: f64
STEP_HEADER_FMT = "<Q I d"
STEP_HEADER_SIZE = struct.calcsize(STEP_HEADER_FMT)

# ---- per-drone data ----
# px, py, pz, vx, vy, vz (all f32)
DRONE_FMT_BASE = "<ffffff"
DRONE_SIZE = struct.calcsize(DRONE_FMT_BASE)
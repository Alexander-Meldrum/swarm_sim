import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.animation import FuncAnimation

from bin_reader import read_swarm_state_log, read_swarm_event_log

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Relative path to file")
args = parser.parse_args()

# ---- load binary logs ----
state_log = read_swarm_state_log(args.path + "_states.bin")
event_log = read_swarm_event_log(args.path + "_events.bin")

pos = state_log.pos          # (T, N, 3)
steps = state_log.steps
rewards = state_log.rewards
T, N, _ = pos.shape

event_steps = event_log.steps
event_kind = event_log.kind
drone_a_positions = event_log.drone_a_position
drone_b_positions = event_log.drone_b_position

# --------------------------------------------------
# Preprocess events by step
# --------------------------------------------------
target_hits_by_step = {}     # kind == 1
collision_hits_by_step = {}  # kind == 2 (A + B)

for step, kind, pos_a, pos_b in zip(
    event_steps, event_kind, drone_a_positions, drone_b_positions
):
    if kind == 1:  # target hit
        target_hits_by_step.setdefault(step, []).append(pos_a)

    elif kind == 2:  # collision
        collision_hits_by_step.setdefault(step, []).extend([pos_a, pos_b])

# Convert to NumPy arrays for speed
for d in (target_hits_by_step, collision_hits_by_step):
    for step in d:
        d[step] = np.asarray(d[step], dtype=np.float32)


# --------------------------------------------------
# Figure & axes
# --------------------------------------------------
max_abs = np.max(np.abs(pos))
L = max_abs

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-L, L)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=25, azim=45)


# --------------------------------------------------
# Static target
# --------------------------------------------------
target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

target_scat = ax.scatter(
    target_pos[0:1], target_pos[1:2], target_pos[2:3],
    s=120,
    c='orange',
    marker='o',
    label="Target"
)


# --------------------------------------------------
# Drone scatter
# --------------------------------------------------
scat = ax.scatter(
    pos[0, :, 0],
    pos[0, :, 1],
    pos[0, :, 2],
    s=40,
    label="Drones"
)


# --------------------------------------------------
# Trajectories
# --------------------------------------------------
lines = [
    ax.plot([], [], [], lw=1, alpha=0.7)[0]
    for _ in range(N)
]


# --------------------------------------------------
# Event scatters
# --------------------------------------------------
hit_target_scat = ax.scatter(
    [], [], [],
    s=90,
    c='none',
    edgecolors='green',
    marker='o',
    label="Target hit"
)

hit_collision_scat = ax.scatter(
    [], [], [],
    s=90,
    c='red',
    marker='x',
    label="Collision"
)


# --------------------------------------------------
# Accumulators (persist during animation)
# --------------------------------------------------
all_target_hits = []
all_collision_hits = []


# --------------------------------------------------
# Update function
# --------------------------------------------------
def update(frame):
    # Reset when animation loops
    if frame == 0:
        all_target_hits.clear()
        all_collision_hits.clear()
        hit_target_scat._offsets3d = ([], [], [])
        hit_collision_scat._offsets3d = ([], [], [])

    # Update drone positions
    scat._offsets3d = (
        pos[frame, :, 0],
        pos[frame, :, 1],
        pos[frame, :, 2],
    )

    # Update trajectories
    for i, line in enumerate(lines):
        line.set_data(pos[:frame+1, i, 0], pos[:frame+1, i, 1])
        line.set_3d_properties(pos[:frame+1, i, 2])

    # Add events for this step
    step = steps[frame]

    if step in target_hits_by_step:
        all_target_hits.extend(target_hits_by_step[step])

    if step in collision_hits_by_step:
        all_collision_hits.extend(collision_hits_by_step[step])

    # Update target hit markers
    if all_target_hits:
        arr = np.asarray(all_target_hits, dtype=np.float32)
        hit_target_scat._offsets3d = (arr[:,0], arr[:,1], arr[:,2])

    # Update collision markers
    if all_collision_hits:
        arr = np.asarray(all_collision_hits, dtype=np.float32)
        hit_collision_scat._offsets3d = (arr[:,0], arr[:,1], arr[:,2])

    ax.set_title(f"Step {step}")
    return [
        scat,
        hit_target_scat,
        hit_collision_scat,
        target_scat,
        *lines
    ]


# --------------------------------------------------
# Animation
# --------------------------------------------------
ani = FuncAnimation(
    fig,
    update,
    frames=range(0, T, 1),
    interval=20,
    blit=False
)

plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.animation import FuncAnimation

from bin_reader import read_swarm_state_log, read_swarm_event_log

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Relative path to file")
args = parser.parse_args()

# ---- load binary logs ----
metadata, state_log = read_swarm_state_log(args.path + "_states.bin")
_, event_log = read_swarm_event_log(args.path + "_events.bin")

N0 = metadata.num_drones_team_0
N1 = metadata.num_drones_team_1
assert N0 + N1 == state_log.pos.shape[1]

pos = state_log.pos          # (T, N, 3)
pos_team_0 = pos[:, :N0, :]          # (T, N0, 3)
pos_team_1 = pos[:, N0:N0+N1, :]     # (T, N1, 3)
steps = state_log.steps
rewards = state_log.rewards
T, N, _ = pos.shape

STEP_STRIDE = 5   # try 5, 10, 20
frame_steps = np.arange(0, T, STEP_STRIDE)

event_steps = event_log.steps
event_kind = event_log.kind
drone_a_positions = event_log.drone_a_position
drone_b_positions = event_log.drone_b_position

# --------------------------------------------------
# Preprocess events by step
# --------------------------------------------------
target_hits_by_step = {}     # kind == 1
collision_hits_by_step = {}  # step -> list[(drone_id, pos)]

for step, kind, drone_a, pos_a, drone_b, pos_b in zip(
    event_steps,
    event_kind,
    event_log.drone_a,
    drone_a_positions,
    event_log.drone_b,
    drone_b_positions,
):
    
    if kind == 1:  # target hit
        target_hits_by_step.setdefault(step, []).append(pos_a)

    if kind == 2:  # collision
        collision_hits_by_step.setdefault(step, []).append((drone_a, pos_a, drone_b, pos_b)
    )

# Convert to NumPy arrays for speed
# Target hits: positions only → safe to convert
for step in target_hits_by_step:
    target_hits_by_step[step] = np.asarray(
        target_hits_by_step[step], dtype=np.float32
    )

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
target_scat = None
if metadata.stationary_target_exists:
    target_pos = metadata.stationary_target_pos

    target_scat = ax.scatter(
        target_pos[0:1],
        target_pos[1:2],
        target_pos[2:3],
        s=120,
        c="orange",
        marker="o",
        label="Target",
    )

# --------------------------------------------------
# Drone scatter
# --------------------------------------------------
scat_team_0 = ax.scatter(
    pos_team_0[0, :, 0],
    pos_team_0[0, :, 1],
    pos_team_0[0, :, 2],
    s=40,
    c="blue",
    label="Team 0",
)

scat_team_1 = ax.scatter(
    pos_team_1[0, :, 0],
    pos_team_1[0, :, 1],
    pos_team_1[0, :, 2],
    s=40,
    c="red",
    label="Team 1",
)

# --------------------------------------------------
# Trajectories
# --------------------------------------------------
lines_team_0 = [
    ax.plot([], [], [], lw=1, alpha=0.7, c="blue")[0]
    for _ in range(N0)
]

lines_team_1 = [
    ax.plot([], [], [], lw=1, alpha=0.7, c="red")[0]
    for _ in range(N1)
]


# --------------------------------------------------
# Event scatters
# --------------------------------------------------
hit_target_scat = ax.scatter(
    [], [], [],
    s=90,
    # c='none',
    edgecolors='green',
    marker='o',
    label="Target hit"
)

collision_team_0_scat = ax.scatter(
    [], [], [],
    s=120,
    c="green",
    marker="^",
    label="Collision (Team 0)",
)

collision_team_1_scat = ax.scatter(
    [], [], [],
    s=90,
    c="red",
    marker="x",
    label="Collision (Team 1)",
)


# --------------------------------------------------
# Accumulators (persist during animation)
# --------------------------------------------------
all_target_hits = []
collision_hits_team_0 = []
collision_hits_team_1 = []


# --------------------------------------------------
# Update function
# --------------------------------------------------
last_step_idx = 0
def update(frame_idx):
    global last_step_idx

    curr_step_idx = frame_steps[frame_idx]

    # Reset when looping
    if frame_idx == 0:
        last_step_idx = 0
        all_target_hits.clear()
        hit_target_scat._offsets3d = ([], [], [])
        collision_hits_team_0.clear()
        collision_hits_team_1.clear()
        collision_team_0_scat._offsets3d = ([], [], [])
        collision_team_1_scat._offsets3d = ([], [], [])

    # -----------------------------
    # Update drone positions (ONLY latest)
    # -----------------------------
    scat_team_0._offsets3d = (
        pos_team_0[curr_step_idx, :, 0],
        pos_team_0[curr_step_idx, :, 1],
        pos_team_0[curr_step_idx, :, 2],
    )

    scat_team_1._offsets3d = (
        pos_team_1[curr_step_idx, :, 0],
        pos_team_1[curr_step_idx, :, 1],
        pos_team_1[curr_step_idx, :, 2],
    )

    # -----------------------------
    # FAST trajectories: fixed tail
    # -----------------------------
    TRAIL = 200  # keep last 30 steps only
    start = max(0, curr_step_idx - TRAIL)

    for i, line in enumerate(lines_team_0):
        line.set_data(
            pos_team_0[start:curr_step_idx+1, i, 0],
            pos_team_0[start:curr_step_idx+1, i, 1],
        )
        line.set_3d_properties(
            pos_team_0[start:curr_step_idx+1, i, 2]
        )

    for i, line in enumerate(lines_team_1):
        line.set_data(
            pos_team_1[start:curr_step_idx+1, i, 0],
            pos_team_1[start:curr_step_idx+1, i, 1],
        )
        line.set_3d_properties(
            pos_team_1[start:curr_step_idx+1, i, 2]
        )

    # -----------------------------
    # EVENT FIX: consume ALL skipped steps
    # -----------------------------
    for step in steps[last_step_idx : curr_step_idx + 1]:
        if step in target_hits_by_step:
            all_target_hits.extend(target_hits_by_step[step])

        if step in collision_hits_by_step:
            
            for drone_a, pos_a, drone_b, pos_b in collision_hits_by_step[step]:

                a_team0 = drone_a < N0
                b_team0 = drone_b < N0

                # Same-team collision → both red crosses
                if a_team0 == b_team0:
                    collision_hits_team_1.append(pos_a)
                    collision_hits_team_1.append(pos_b)

                # Cross-team collision
                else:
                    if a_team0:
                        collision_hits_team_0.append(pos_a)   # green circle
                        collision_hits_team_1.append(pos_b)   # red cross
                    else:
                        collision_hits_team_1.append(pos_a)
                        collision_hits_team_0.append(pos_b)

    last_step_idx = curr_step_idx + 1

    # -----------------------------
    # Update event scatters
    # -----------------------------
    if all_target_hits:
        arr = np.asarray(all_target_hits, dtype=np.float32)
        hit_target_scat._offsets3d = (arr[:, 0], arr[:, 1], arr[:, 2])

    if collision_hits_team_0:
        arr = np.asarray(collision_hits_team_0, dtype=np.float32)
        collision_team_0_scat._offsets3d = (arr[:,0], arr[:,1], arr[:,2])

    if collision_hits_team_1:
        arr = np.asarray(collision_hits_team_1, dtype=np.float32)
        collision_team_1_scat._offsets3d = (arr[:,0], arr[:,1], arr[:,2])

    ax.set_title(f"Episode: {metadata.episode}, Step: {steps[curr_step_idx]}, " f"Time: {steps[curr_step_idx] * metadata.dt:.3f}s")

    artists = [
        scat_team_0,
        scat_team_1,
        hit_target_scat,
        collision_team_0_scat,
        collision_team_1_scat,
        *lines_team_0,
        *lines_team_1,
    ]

    if target_scat is not None:
        artists.append(target_scat)

    return artists


# --------------------------------------------------
# Animation
# --------------------------------------------------
ani = FuncAnimation(
    fig,
    update,
    frames=len(frame_steps),
    interval=metadata.dt*1000,
    blit=False,
)

# --------------------------------------------------
# Legend
# --------------------------------------------------
# plt.legend()
legend_artists = [
    scat_team_0,
    scat_team_1,
    collision_team_0_scat,
    collision_team_1_scat,
]

legend_labels = [
    "Team 0",
    "Team 1",
    "Interception",
    "Collision",
]

if target_scat is not None:
    legend_artists.insert(2, target_scat)
    legend_artists.insert(2, hit_target_scat)
    legend_labels.insert(2, "Target")
    legend_labels.insert(2, "Target Hit")

ax.legend(legend_artists, legend_labels)

plt.show()
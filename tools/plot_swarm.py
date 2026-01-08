import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bin_reader import read_swarm_state_log


# ---- load binary log ----
state_log = read_swarm_state_log("logs/00001_states.bin")

pos = state_log.pos          # (T, N, 3)
steps = state_log.steps
rewards = state_log.rewards
T, N, _ = pos.shape


# ---- figure & axes ----
# Find largest abs pos value
max_abs = np.max(np.abs(pos))
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# world bounds (adjust)
L = 20

ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-L, L)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.view_init(elev=25, azim=45)


# ---- scatter (drones) ----
scat = ax.scatter(
    pos[0, :, 0],
    pos[0, :, 1],
    pos[0, :, 2],
    s=40
)

# ---- trajectory lines (one per drone) ----
lines = [
    ax.plot([], [], [], lw=1, alpha=0.7)[0]
    for _ in range(N)
]


# ---- update ----
def update(frame):
    # update drone positions
    scat._offsets3d = (
        pos[frame, :, 0],
        pos[frame, :, 1],
        pos[frame, :, 2],
    )

    # update trajectories
    for i, line in enumerate(lines):
        line.set_data(
            pos[:frame+1, i, 0],
            pos[:frame+1, i, 1],
        )
        line.set_3d_properties(
            pos[:frame+1, i, 2]
        )

    ax.set_title(
        f"Step {steps[frame]} | Reward {rewards[frame]:.3f}"
    )

    return [scat, *lines]


ani = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=50,
    blit=False,
)

plt.show()
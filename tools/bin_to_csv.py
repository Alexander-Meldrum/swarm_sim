import csv
import argparse
from bin_reader import read_swarm_state_log , read_swarm_event_log

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Relative path to file")
args = parser.parse_args()

# ---- load binary log ----
state_log = read_swarm_state_log(args.path + "_states.bin")

with open(str(args.path) + "_states.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow([
        "step", "drone",
        "px", "py", "pz",
        "vx", "vy", "vz",
        "rewards",
    ])

    for t in range(len(state_log.steps)):
        for i in range(state_log.pos.shape[1]):  # [T, N, 3]  Picked N, the amount of drones
            writer.writerow([
                state_log.steps[t],
                i,                         # Drone id
                *state_log.pos[t, i],
                *state_log.vel[t, i],
                state_log.rewards[t, i],
            ])

event_log = read_swarm_event_log(args.path + "_events.bin")

with open(str(args.path) + "_events.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow([
        "step", "kind",
        "drone_a", "drone_a_position_x", "drone_a_position_y", "drone_a_position_z",
        "drone_b", "drone_b_position_x", "drone_b_position_y", "drone_b_position_z",
        "target_id",
    ])

    # Iterate over all events
    for i in range(len(event_log.steps)):
        writer.writerow([
            event_log.steps[i],
            event_log.kind[i],
            event_log.drone_a[i],
            *event_log.drone_a_position[i],  # x, y, z
            event_log.drone_b[i],
            *event_log.drone_b_position[i],  # x, y, z
            event_log.target_id[i],
        ])
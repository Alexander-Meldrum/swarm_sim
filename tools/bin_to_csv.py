import csv
from bin_reader import read_swarm_state_log, read_swarm_event_log

state_log = read_swarm_state_log("logs/00001_states.bin")

with open("logs/00001_states.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow([
        "step", "drone",
        "px", "py", "pz",
        "vx", "vy", "vz",
        "global_reward",
    ])

    for t in range(len(state_log.steps)):
        for i in range(state_log.pos.shape[1]):  # [T, N, 3]  Picked N, the amount of drones
            writer.writerow([
                state_log.steps[t],
                i,                         # Drone id
                *state_log.pos[t, i],
                *state_log.vel[t, i],
                state_log.global_rewards[t],
            ])

event_log = read_swarm_event_log("logs/00001_events.bin")

with open("logs/00001_events.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow([
        "step", "kind",
        "drone_a", "drone_b", "target_id",
    ])

    for t in range(len(state_log.steps)):
        for i in range(state_log.pos.shape[1]):  # [T, N, 3]  Picked N, the amount of drones
            writer.writerow([
                state_log.steps[t],
                i,                         # Drone id
                *state_log.pos[t, i],
                *state_log.vel[t, i],
                state_log.rewards[t],
            ])
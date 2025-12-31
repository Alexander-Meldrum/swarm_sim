import csv
from bin_reader import read_swarm_log

log = read_swarm_log("logs/episode_000001.bin")

with open("logs/episode_000001.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow([
        "step", "drone",
        "px", "py", "pz",
        "vx", "vy", "vz",
        "reward",
    ])

    for t in range(len(log.steps)):
        for i in range(log.pos.shape[1]):  # [T, N, 3]  Picked N, the amount of drones
            writer.writerow([
                log.steps[t],
                i,                         # Drone id
                *log.pos[t, i],
                *log.vel[t, i],
                log.rewards[t],
            ])
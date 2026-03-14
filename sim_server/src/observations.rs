use crate::config::SimConfig;
use crate::world::{find_k_nearest_per_team, Vec3, World, K_NEIGHBORS};
use std::sync::Arc;

pub const OBS_DIM: usize = 4 + 6 * 2;

pub struct ObservationBuffer {
    pub obs: Vec<f32>,
}

impl ObservationBuffer {
    pub fn new(num_drones: usize, obs_dim: usize) -> Self {
        Self {
            obs: Vec::with_capacity(num_drones * obs_dim),
        }
    }

    pub fn build(&mut self, world: &World, config: Arc<SimConfig>) {
        self.obs.clear();

        // Only building observations for team 0
        for i in 0..world.num_drones_team_0 {
            let vel = world.velocity[i];

            self.obs.push(vel.x);
            self.obs.push(vel.y);
            self.obs.push(vel.z);
            self.obs.push(world.alive[i] as u8 as f32);

            let (team0_neighbors, team1_neighbors) =
                find_k_nearest_per_team::<K_NEIGHBORS>(&world, i, config.sensing.max_sensing);

            let mut nn_input = [0.0f32; 2 * 9 * K_NEIGHBORS]; // 9 fields per team: dx, dy, dz, dir_x, dir_y, dir_z, dvx, dvy, dvz

            // ----------------------------
            // Team 0 neighbors, dx, dy, dz, dir_x, dir_y, dir_z, dvx, dvy, dvz
            // ----------------------------
            for i in 0..K_NEIGHBORS {
                let base = 9 * i;

                if let Some((p, v)) = team0_neighbors[i] {
                    let norm = p.norm();
                    let rel_dir = if norm > 1e-6 { p / norm } else { Vec3::zero() };
                    nn_input[base + 0] = p.x / config.arena.cell_size;
                    nn_input[base + 1] = p.y / config.arena.cell_size;
                    nn_input[base + 2] = p.z / config.arena.cell_size;
                    nn_input[base + 3] = rel_dir.x;
                    nn_input[base + 4] = rel_dir.y;
                    nn_input[base + 5] = rel_dir.z;
                    nn_input[base + 6] = v.x / config.physics.max_velocity;
                    nn_input[base + 7] = v.y / config.physics.max_velocity;
                    nn_input[base + 8] = v.z / config.physics.max_velocity;
                }
            }

            // ----------------------------
            // Team 1 neighbors, dx, dy, dz, dir_x, dir_y, dir_z, dvx, dvy, dvz
            // ----------------------------
            let offset = 9 * K_NEIGHBORS;

            for i in 0..K_NEIGHBORS {
                let base = offset + 9 * i;

                if let Some((p, v)) = team1_neighbors[i] {
                    let norm = p.norm();
                    let rel_dir = if norm > 1e-6 { p / norm } else { Vec3::zero() };
                    nn_input[base + 0] = p.x / config.arena.cell_size;
                    nn_input[base + 1] = p.y / config.arena.cell_size;
                    nn_input[base + 2] = p.z / config.arena.cell_size;
                    nn_input[base + 3] = rel_dir.x;
                    nn_input[base + 4] = rel_dir.y;
                    nn_input[base + 5] = rel_dir.z;
                    nn_input[base + 6] = v.x / config.physics.max_velocity;
                    nn_input[base + 7] = v.y / config.physics.max_velocity;
                    nn_input[base + 8] = v.z / config.physics.max_velocity;
                }
            }
            self.obs.extend_from_slice(&nn_input);
        }

        // println!("self.obs: {:?}", self.obs)
    }
}

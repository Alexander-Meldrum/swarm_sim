use crate::world::{World, K_NEIGHBORS, find_k_nearest_per_team};

pub const OBS_DIM: usize = 4+6*2;

pub struct ObservationBuffer {
    pub obs: Vec<f32>,
}

impl ObservationBuffer {
    pub fn new(num_drones: usize, obs_dim: usize) -> Self {
        Self {
            obs: Vec::with_capacity(num_drones * obs_dim),
        }
    }

    pub fn build(&mut self, world: &World, max_radius: f32) {
        self.obs.clear();

        // Only building observations for team 0
        for i in 0..world.num_drones_team_0 {
            // let pos = world.position[i];
            let vel = world.velocity[i];

            // self.obs.push(pos.x);
            // self.obs.push(pos.y);
            // self.obs.push(pos.z);
            self.obs.push(vel.x);
            self.obs.push(vel.y);
            self.obs.push(vel.z);
            self.obs.push(world.alive[i] as u8 as f32);

            let (team0_neighbors, team1_neighbors) =
            find_k_nearest_per_team::<K_NEIGHBORS>(&world, i, max_radius);
            
            let mut nn_input = [0.0f32; 12 * K_NEIGHBORS]; // 6 fields per team: dx, dy, dz, dvx, dvy, dvz, present flag

            // ----------------------------
            // Team 0 neighbors, dx, dy, dz, dvx, dvy, dvz
            // ----------------------------
            for i in 0..K_NEIGHBORS {
                let base = 7 * i;

                if let Some((p, v)) = team0_neighbors[i] {
                    nn_input[base + 0] = p.x;  
                    nn_input[base + 1] = p.y;
                    nn_input[base + 2] = p.z;
                    nn_input[base + 3] = v.x;
                    nn_input[base + 4] = v.y;
                    nn_input[base + 5] = v.z;
                    // nn_input[base + 6] = 0.1; // presence flag
                }
            }

            // ----------------------------
            // Team 1 neighbors, dx, dy, dz, dvx, dvy, dvz
            // ----------------------------
            let offset = 6 * K_NEIGHBORS;

            for i in 0..K_NEIGHBORS {
                let base = offset + 6 * i;

                if let Some((p, v)) = team1_neighbors[i] {
                    nn_input[base + 0] = p.x;
                    nn_input[base + 1] = p.y;
                    nn_input[base + 2] = p.z;
                    nn_input[base + 3] = v.x;
                    nn_input[base + 4] = v.y;
                    nn_input[base + 5] = v.z;
                    // nn_input[base + 6] = 0.1; // presence flag
                }
            }
            for value in nn_input {
                self.obs.push(value);
            }
          

        }

        // println!("self.obs: {:?}", self.obs)  
    }
}
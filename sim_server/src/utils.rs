use clap::Parser;
use std::fs::File;
use std::io::Read;
use crate::config::SimConfig;
use crate::world::World;

/// Simulator command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Args {
    /// Path to simulator config YAML
    #[arg(long, default_value = "../configs/sim.yaml")]
    pub config: String,

    /// gRPC bind address
    #[arg(long, default_value = "[::1]:50051")]
    pub bind: String,
}

pub fn load_config(path: &str) -> Result<SimConfig, Box<dyn std::error::Error>> {
    let mut file = File::open(path)
        .map_err(|e| format!("{}: {}", path, e))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| format!("{}: {}", path, e))?;

    let config: SimConfig = serde_yaml::from_str(&contents)
        .map_err(|e| format!("{}: {}", path, e))?;

    Ok(config)
}

pub const OBS_DIM: usize = 6;
pub struct ObservationBuffer {
    pub obs: Vec<f32>,
    // pub obs_dim: usize,
}

impl ObservationBuffer {
    pub fn new(num_drones: usize, obs_dim: usize) -> Self {
        Self {
            obs: Vec::with_capacity(num_drones * obs_dim),
            // obs_dim,
        }
    }

    pub fn build(&mut self, world: &World) {
        self.obs.clear();

        for i in 0..world.num_drones {
            let pos = world.position[i];

            self.obs.push(pos.x);
            self.obs.push(pos.y);
            self.obs.push(pos.z);
            self.obs.push(world.collisions_desired[i] as f32);
            self.obs.push(world.collisions_undesired[i] as f32);
            self.obs.push(world.alive[i] as u8 as f32);
        }
    }
}
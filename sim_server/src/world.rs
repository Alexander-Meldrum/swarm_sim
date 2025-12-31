use std::fs::File;
use std::io::{BufWriter};
use rand::rngs::SmallRng;
use rand::{SeedableRng, Rng};

#[derive(Copy, Clone)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub fn zero() -> Self { Self { x:0.0,y:0.0,z:0.0 } }
}

pub struct World {
    pub num_drones: usize,
    pub position: Vec<Vec3>,
    pub velocity: Vec<Vec3>,
    pub acceleration: Vec<Vec3>,
    pub alive: Vec<bool>,
    pub global_reward: f32,
    pub step: u64,
    pub max_steps: u64,
    pub dt: f32,
    pub episode: u64,
    pub log: Option<BufWriter<File>>,
    pub done: bool,
}

impl World {
    pub fn new(n: usize, max_steps: u64, dt: f32, episode: u64, log:Option<BufWriter<File>>) -> Self {
        Self {
            num_drones: n,
            position: vec![Vec3::zero(); n],
            velocity: vec![Vec3::zero(); n],
            acceleration: vec![Vec3::zero(); n],
            alive: vec![true; n],
            global_reward: 0.0,
            step: 0,
            max_steps: max_steps,
            dt: dt,
            episode: episode,
            log: log,
            done: false,
        }
    }

    pub fn init_drones(&mut self, seed: Option<u64>, randomize_init_pos: bool, arena_size: f32, min_dist: f32) {

        let mut rng = SmallRng::seed_from_u64(seed.expect("No Seed Provided"));
        // examples
        // let x: f32 = rng.gen();                 // [0, 1)
        // let y: f32 = rng.random_range(-1.0..1.0);  // range
        // let i: usize = rng.random_range(0..10);    // integer

        let half = arena_size;
        if randomize_init_pos {
            println!("[simulator] Randomizing init positions with seed: {}", seed.expect("No Seed Provided"));
        }


        for i in 0..self.num_drones {
            let pos = loop {
                let candidate = Vec3 {
                    x: rng.random_range(-half..half),
                    y: rng.random_range(-half..half),
                    z: rng.random_range(-half..half),
                };

                if self.position.iter().all(|p| {
                    (p.x - candidate.x).powi(2)
                  + (p.y - candidate.y).powi(2)
                  + (p.z - candidate.z).powi(2)
                  >= min_dist.powi(2)
                }) {
                    break candidate;
                }
            };

            self.position[i] = pos;
        }
    }
}

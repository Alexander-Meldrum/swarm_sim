use std::fs::File;
use std::io::{BufWriter};

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
}

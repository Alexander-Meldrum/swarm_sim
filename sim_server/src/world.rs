#[derive(Copy, Clone)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub fn zero() -> Self { Self { x:0.0,y:0.0,z:0.0 } }
}

pub struct World {
    pub position: Vec<Vec3>,
    pub velocity: Vec<Vec3>,
    pub acceleration: Vec<Vec3>,
    pub alive: Vec<bool>,
    pub step: u64,
    pub dt: f32,
}

impl World {
    pub fn new(n: usize, dt: f32) -> Self {
        Self {
            position: vec![Vec3::zero(); n],
            velocity: vec![Vec3::zero(); n],
            acceleration: vec![Vec3::zero(); n],
            alive: vec![true; n],
            step: 0,
            dt,
        }
    }
    pub fn reset_state(&mut self) {
        self.step = 0;

        for i in 0..self.position.len() {
            self.position[i] = Vec3::zero();
            self.velocity[i] = Vec3::zero();
            self.acceleration[i] = Vec3::zero();
            self.alive[i] = true;
        }
    }
}

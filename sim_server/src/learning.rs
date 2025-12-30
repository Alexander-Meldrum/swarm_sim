use crate::world::{World};
pub struct Rewards {
    pub individual_rewards: Vec<f32>,
    pub global_reward: f32,
}

impl Rewards {
    pub fn new(num_drones: usize) -> Self {
        Self {
            individual_rewards: Vec::with_capacity(num_drones),
            global_reward: 0.0,
        }
    }
}

pub fn calc_rewards(world: &World) -> Rewards {
        // TODO, calculate proper global_reward
        let mut rewards = Rewards::new(world.num_drones);
        rewards.individual_rewards.fill(0.1);
        rewards.global_reward = world.num_drones as f32;
        rewards
    }


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
        // TODO, setup better rewards, distince between defending/attacking team, calculate proper global_reward
        let mut rewards = Rewards::new(world.num_drones);

        for event in &world.hit_events {
            rewards.individual_rewards[event.drone_id] += 10.0;
        }
        rewards.global_reward = world.num_drones as f32;

        rewards


        

    }


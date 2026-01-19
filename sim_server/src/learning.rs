use crate::world::{World};
pub struct Rewards {
    pub individual_rewards: Vec<f32>,
    pub global_reward: f32,
}

impl Rewards {
    pub fn new(num_drones_team_0: usize) -> Self {
        Self {
            individual_rewards: vec![0.0f32; num_drones_team_0],
            global_reward: 0.0,
        }
    }
}

pub fn calc_rewards(world: &World) -> Rewards {
        // TODO, setup better rewards, diffrentiate between defending/attacking team, calculate proper global_reward
        let mut rewards = Rewards::new(world.num_drones_team_0);


        rewards.individual_rewards.fill(0.0);

        for event in &world.events {
            rewards.individual_rewards[event.drone_a as usize] += 10.0;
        }
        rewards.global_reward = world.num_drones_team_0 as f32;

        rewards

    }


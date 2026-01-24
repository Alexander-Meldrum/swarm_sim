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
        rewards.global_reward = 0.0 as f32;




        for i in 0..world.num_drones_team_0 {
            if !world.alive[i] { continue; }
            // rewards.global_reward += world.position[i].z;
            // rewards.global_reward -= (2.0 * world.position[i].x.abs());
            // rewards.global_reward -= (2.0 * world.position[i].y.abs());

            // rewards.individual_rewards[i] += world.position[i].z;
            rewards.individual_rewards[i] -= world.position[i].z.abs();
            rewards.individual_rewards[i] -= (1.0 * world.position[i].x.abs());
            rewards.individual_rewards[i] -= (1.0 * world.position[i].y.abs());
            rewards.individual_rewards[i] -= (10.0 * world.velocity[i].x.abs());
            rewards.individual_rewards[i] -= (10.0 * world.velocity[i].y.abs());
            rewards.individual_rewards[i] -= (10.0 * world.velocity[i].z.abs());


            println!("x y z: {} {} {}, reward: {}", world.position[i].x, world.position[i].y, world.position[i].z, rewards.individual_rewards[i])
        }

        rewards

    }


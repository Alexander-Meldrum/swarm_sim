use crate::world::{World, Vec3};

pub struct Rewards {
    pub rewards: Vec<f32>,
    pub global_reward: f32,
}

impl Rewards {
    pub fn new(num_drones_team_0: usize) -> Self {
        Self {
            rewards: vec![0.0f32; num_drones_team_0],
            global_reward: 0.0,
        }
    }
}

pub fn calc_rewards(world: &World) -> Rewards {
        // TODO, setup better rewards, diffrentiate between defending/attacking team, calculate proper global_reward
        let mut rewards = Rewards::new(world.num_drones_team_0);

        rewards.rewards.fill(0.0);
        rewards.global_reward = 0.0 as f32;

        for event in &world.events {
            rewards.rewards[event.drone_a as usize] += 1000.0;
            println!("Collision by drone: {}, type: {:?}", event.drone_a, event.kind)
        }


        for i in 0..world.num_drones_team_0 {
            if !world.alive[i] { continue; }

            // Angular alignment: reward velocity pointing toward origin
            let eps: f32 = 1e-6;
            let pos = world.position[i];
            let vel = world.velocity[i];
            let vel_norm_squared = vel.norm_squared();
            let pos_norm = pos.norm_squared().sqrt() + eps;
            let vel_norm = vel_norm_squared.sqrt() + eps;

            // Unit vectors
            let radial_in = Vec3 {
                x: -pos.x / pos_norm,
                y: -pos.y / pos_norm,
                z: -pos.z / pos_norm,
            };

            let vel_dir = Vec3 {
                x: vel.x / vel_norm,
                y: vel.y / vel_norm,
                z: vel.z / vel_norm,
            };

            // Cosine alignment in [-1, 1]
            let alignment = radial_in.dot(&vel_dir);
            // rewards.rewards[i] += 5.0 * alignment;
            // let alignment_reward = 10.0 * alignment;
            let mut alignment_reward = 0.0;
            if vel_norm > 0.1 {
                alignment_reward = 1.5 * alignment * vel_norm;
            }
            
            let time_penalty = -1.0;
            let distance_penalty = -0.0005* world.distance_to_origin2[i];
            let velocity_penalty = -0.001* vel_norm_squared;
            let delta_distance_reward = 0.05* (world.previous_position[i].norm_squared() - world.position[i].norm_squared());

            // let damping_reward  = -0.5 * velocity_norm_squared * world.distance_to_origin2[i];         // penalize high speed when close

            
            rewards.rewards[i] += time_penalty + distance_penalty + velocity_penalty + delta_distance_reward + alignment_reward;


            

            // println!("distance_penalty, velocity_penalty, delta_distance_reward, alignment_reward: {} {} {} {}",distance_penalty, velocity_penalty, delta_distance_reward, alignment_reward)

            // println!("rewards.rewards[i]: {}", rewards.rewards[i])


            // println!("x y z: {} {} {}, reward: {}", world.position[i].x, world.position[i].y, world.position[i].z, rewards.rewards[i])
        }

        rewards

    }


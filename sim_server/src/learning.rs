use crate::world::{World, EventKind};

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
        let mut rewards = Rewards::new(world.num_drones_team_0);

        rewards.rewards.fill(0.0);
        rewards.global_reward = 0.0 as f32;

        for event in &world.events {
            if event.kind == EventKind::DroneCollision {
                // punish two controlled drones colliding
                if event.drone_a < world.num_drones_team_0.try_into().unwrap() && event.drone_b < world.num_drones_team_0.try_into().unwrap()
                {
                    rewards.rewards[event.drone_a as usize] -= 20.0;
                    rewards.rewards[event.drone_b as usize] -= 20.0;
                }
                // Reward if a team 0 drone collides into a team 1 drone. We have to check a/b and b/a collisions.
                if event.drone_a < world.num_drones_team_0.try_into().unwrap() && event.drone_b >= world.num_drones_team_0.try_into().unwrap()
                {
                    rewards.rewards[event.drone_a as usize] += 20.0;
                }
                if event.drone_b < world.num_drones_team_0.try_into().unwrap() && event.drone_a >= world.num_drones_team_0.try_into().unwrap()
                {
                    rewards.rewards[event.drone_b as usize] += 20.0;
                }

            println!("Collision by drones: {}-{}, type: {:?}", event.drone_a, event.drone_b, event.kind)
            }
        }

        for i in 0..world.num_drones_team_0 {
            if !world.alive[i] { continue; }

            // rewards.rewards[i] += calc_reward_for_hitting_stationary_target(world, i);
            rewards.rewards[i] += calc_reward_for_hitting_dynamic_target(world, i);
        }

        rewards

}

fn calc_reward_for_hitting_dynamic_target(world: &World, i: usize) -> f32 {

    let Some((rel_pos, rel_vel)) =   // rel_pos points from drone → target
        world.team1_neighbors[i][0]
    else {
        // No target in range → mild time penalty
        return -5.0;
    };
    let eps: f32 = 1e-6;
    let dist = rel_pos.norm() + eps;
    let vel = world.velocity[i];
    let vel_norm = vel.norm() + eps;

    // ----------------------------
    // 1. Distance reduction (progress reward)
    // ----------------------------
    let dist_prev = (rel_pos + (world.position[i] - world.previous_position[i])).norm();
    let dist_reward = 3.0 * (dist_prev - dist); // positive when getting closer

    // // ----------------------------
    // // 2. Alignment reward (pointing roughly toward target)
    // // ----------------------------
    // let drone_dir = vel / vel_norm;
    // let target_dir = rel_pos / dist;
    // let alignment_reward = 0.25 * drone_dir.dot(&target_dir).clamp(-1.0, 1.0);

    // // ----------------------------
    // // 3. Predictive closing reward (future interception)
    // // ----------------------------
    // let closing_reward = (0.1 * (-rel_pos.dot(&rel_vel) / dist)).clamp(-8.0, 8.0); // positive if approaching, predicts future

    // let closing_reward = 1.0 * closing_reward.clamp(-5.0, 5.0);
    // ----------------------------
    // 4. Optional speed penalty to avoid runaway
    // ----------------------------
    let speed_penalty = -0.1 * vel_norm; // keeps velocity reasonable

    // // ----------------------------
    // // 5. Total reward
    // // ----------------------------
    // let reward = dist_reward + alignment_reward + closing_reward + speed_penalty;

    // println!("Episode {}. dist_reward, alignment_reward, closing_reward, speed_penalty, tot_reward: {} {} {} {} {}", world.episode, dist_reward, alignment_reward, closing_reward, speed_penalty, reward);

    // let progress_reward = 5.0 * (dist_prev - dist);

    let shaping_reward = 1.0 / (1.0 + dist);

    let reward = dist_reward + shaping_reward + speed_penalty;

    // println!("Episode {}. dist_reward, shaping_reward, speed_penalty, tot_reward: {} {} {} {}", world.episode, dist_reward, shaping_reward, speed_penalty, reward);

    return reward;
}

// fn calc_reward_for_hitting_stationary_target(world: &World, i: usize) -> f32 {
//     // Target Hitting Rewards:
//     // Angular alignment: reward velocity pointing toward origin
//     let eps: f32 = 1e-6;
//     let pos = world.position[i];
//     let vel = world.velocity[i];
//     let vel_norm_squared = vel.norm_squared();
//     let pos_norm = pos.norm_squared().sqrt() + eps;
//     let vel_norm = vel_norm_squared.sqrt() + eps;

//     // Unit vectors
//     let radial_in = Vec3 {
//         x: -pos.x / pos_norm,
//         y: -pos.y / pos_norm,
//         z: -pos.z / pos_norm,
//     };

//     let vel_dir = Vec3 {
//         x: vel.x / vel_norm,
//         y: vel.y / vel_norm,
//         z: vel.z / vel_norm,
//     };

//     // Cosine alignment in [-1, 1]
//     let alignment = radial_in.dot(&vel_dir);
//     let velocity_alignment_reward = if alignment >= 0.0 {
//         1.0 * alignment.powf(4.0) * vel_norm  // Reward alignment closer to 0.99
//     } else {
//         5.0 * alignment * vel_norm // strongly penalize alignment away from target
//     };
    
//     let time_penalty = -1.0;
//     let distance_penalty = -0.0005* world.distance_to_origin2[i];
//     let velocity_penalty = -0.0005* vel_norm_squared;
//     let delta_distance_reward = 0.5* (world.previous_position[i].norm_squared().sqrt() - world.position[i].norm_squared().sqrt());
//     let reward = time_penalty + distance_penalty + velocity_penalty + delta_distance_reward + velocity_alignment_reward;

//     return reward;
//     // println!("distance_penalty, velocity_penalty, delta_distance_reward, velocity_alignment_reward: {} {} {} {}",distance_penalty, velocity_penalty, delta_distance_reward, velocity_alignment_reward)
//     // println!("rewards.rewards[i]: {}", rewards.rewards[i])
//     // println!("x y z: {} {} {}, reward: {}", world.position[i].x, world.position[i].y, world.position[i].z, rewards.rewards[i])
// }


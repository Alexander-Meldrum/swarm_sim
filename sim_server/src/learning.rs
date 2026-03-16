use crate::world::{EventKind, Vec3, World};

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

    // Calculate terminal rewards when collisions occurs
    for event in &world.events {
        if event.kind == EventKind::DroneCollision {
            // punish two controlled drones colliding
            if event.drone_a < world.num_drones_team_0.try_into().unwrap()
                && event.drone_b < world.num_drones_team_0.try_into().unwrap()
            {
                rewards.rewards[event.drone_a as usize] -= 30.0;
                rewards.rewards[event.drone_b as usize] -= 30.0;
            }
            // Reward if a team 0 drone collides into a team 1 drone. We have to check a/b and b/a collisions.
            if event.drone_a < world.num_drones_team_0.try_into().unwrap()
                && event.drone_b >= world.num_drones_team_0.try_into().unwrap()
            {
                rewards.rewards[event.drone_a as usize] += 30.0;
            }
            if event.drone_b < world.num_drones_team_0.try_into().unwrap()
                && event.drone_a >= world.num_drones_team_0.try_into().unwrap()
            {
                rewards.rewards[event.drone_b as usize] += 30.0;
            }

            println!(
                "Collision by drones: {} & {}, type: {:?}",
                event.drone_a, event.drone_b, event.kind
            )
        }
    }

    for i in 0..world.num_drones_team_0 {
        if !world.alive[i] {
            continue;
        } // Only terminal rewards for step when collision occurs.

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
        return -0.1;
    };

    let mut no_neighbor = false;
    let (friendly_rel_pos, _friendly_rel_vel) =
        if let Some((pos, vel)) = world.team0_neighbors[i][0] {
            (pos, vel)
        } else {
            no_neighbor = true;
            (Vec3::zero(), Vec3::zero())
        };

    let eps: f32 = 1e-6;
    let dist = rel_pos.norm() + eps;
    let dir_to_target = rel_pos / dist;
    let vel = world.velocity[i];

    // ----------------------------
    // Distance reduction (progress reward)
    // ----------------------------
    let shaping_reward = 0.03 / (0.1 + dist);

    // ----------------------------
    // Predictive closing reward (future interception)
    // ----------------------------
    // let raw_closing = -rel_pos.dot(&rel_vel) / dist; // positive if approaching, predicts future
    //                                                  // Rational scaling to keep under 1.0 & weaken near intercept
    // let closing_reward = 0.3 * raw_closing / (1.0 + raw_closing.abs()) * (dist / (dist + 1.0));
    
    // ----------------------------
    // Closing reward
    // ----------------------------
    let rel_closing_speed = -rel_vel.dot(&dir_to_target);
    // let closing_reward = 0.3 * rel_closing_speed / (1.0 + rel_closing_speed.abs()) * (dist / (dist + 1.0));
    // let closing_reward = 0.3 * rel_closing_speed.max(0.0) / (1.0 + rel_closing_speed.abs());

    let closing_reward = 0.3 * rel_closing_speed.max(0.0) / (1.0 + rel_closing_speed.abs());
    
    // ----------------------------
    // Miss-Distance Prediction, intercept_reward
    // ----------------------------
    // let closing_speed = vel.dot(&dir_to_target);
    // let intercept_reward = if closing_speed > 0.0 {
    //     let t = dist / (closing_speed + 0.01);
    //     let predicted_offset = rel_pos + rel_vel * t;
    //     let miss_distance = predicted_offset.norm();
    //     0.4 / (1.0 + miss_distance)
    // } else {
    //     0.0
    // };

    let closing_speed = -rel_vel.dot(&dir_to_target);
    let intercept_reward = if closing_speed > 0.0 {
        let t = dist / closing_speed.max(0.2);  // 0.2 Forces t to be physically reasonable

        let predicted_offset = rel_pos + rel_vel * t;
        let miss_distance = predicted_offset.norm();

        0.4 / (1.0 + miss_distance)
    } else {
        0.0
    };

    // ----------------------------
    // Penalize lateral velocity
    // ----------------------------
    let forward_speed = vel.dot(&dir_to_target);
    let speed_sq = vel.dot(&vel);
    let lateral_speed_sq = speed_sq - forward_speed * forward_speed;
    let lateral_speed = lateral_speed_sq.max(0.0).sqrt();
    let lateral_penalty = -0.1 * lateral_speed;

    // ----------------------------
    // Keep away from friendlies, friendly_prox_penalty
    // ----------------------------
    let mut friendly_prox_penalty = 0.0;
    if !no_neighbor {
        let d = friendly_rel_pos.norm();
        friendly_prox_penalty = -0.1 / (0.2 + d * d);
    }

    // ----------------------------
    // Total reward
    // ----------------------------
    let reward: f32 = shaping_reward
        + closing_reward
        + intercept_reward
        + lateral_penalty
        + friendly_prox_penalty;
    println!("Episode {}. shaping, closing, intercept, lateral, friendly_prox_penalty, tot: {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} ", world.episode, shaping_reward, closing_reward, intercept_reward, lateral_penalty, friendly_prox_penalty, reward);

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

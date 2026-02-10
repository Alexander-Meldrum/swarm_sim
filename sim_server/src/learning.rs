use crate::world::{World, EventKind, Vec3};

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

    let vel = world.velocity[i];
    let vel_norm2 = vel.norm_squared();
    let vel_norm = vel_norm2.sqrt() + eps;
    let dist_now = rel_pos.norm();    

    // ----------------------------
    // Velocity alignment (tiny)
    // ----------------------------
    // let align_reward =  0.05 * (rel_pos.dot(&vel) / (dist_now + eps));
    // let align_reward =  0.002 * (rel_pos.dot(&vel));
    // println!("rel_pos, vel, dist_now: {:?} {:?} {}", rel_pos, vel, dist_now);

    // ----------------------------
    // Distance reduction
    // ----------------------------
    let prev_rel_pos = rel_pos + (world.position[i] - world.previous_position[i]);
    let dist_prev = prev_rel_pos.norm();
    let dist_reward = 0.001 *( dist_prev - dist_now);

    // ------------------------------------
    // alignment toward target
    // ------------------------------------
    let dist2 = rel_pos.norm_squared();
    let dist = dist2.sqrt() + eps;
    let target_dir = Vec3 {
        x: rel_pos.x / dist,
        y: rel_pos.y / dist,
        z: rel_pos.z / dist,
    };

    let vel_dir = Vec3 {
        x: vel.x / vel_norm,
        y: vel.y / vel_norm,
        z: vel.z / vel_norm,
    };

    let alignment = vel_dir.dot(&target_dir);
    let alignment_reward = 1.0* alignment.clamp(-1.0,1.0);     // Direction only, bounded, safe

    // ------------------------------------
    // closing toward target
    // ------------------------------------
    let closing_speed = -rel_pos.dot(&rel_vel) / (rel_pos.norm() + 1e-6);  // positive → approaching
    let closing_reward = 2.0 * closing_speed.clamp(-10.0, 10.0);

    // println!(
    // "dot(rel_pos, rel_vel) = {}",
    // rel_pos.dot(&rel_vel));
    // println!(
    // "toward = {}, closing = {}",
    // rel_pos.dot(&rel_vel),
    // -rel_pos.dot(&rel_vel) / rel_pos.norm()



// );

    let squared_distance_reward = 0.001 * -(dist_now * dist_now);

    let speed_penalty = -0.05 * vel.norm();

    let simple_distance_reward = -rel_pos.norm() / 10.0;

    println!("Episode {}. simple_distance_reward, dist_reward, squared_distance_reward, alignment_reward, closing_reward, speed_penalty:  {} {} {} {} {} {}", world.episode, simple_distance_reward, dist_reward, squared_distance_reward, alignment_reward, closing_reward, speed_penalty);
    // let reward =  squared_distance_reward + alignment_reward + closing_reward;  // dist_reward +
    // let reward =  squared_distance_reward + alignment_reward + closing_reward;


    let reward = simple_distance_reward + alignment_reward + closing_reward; // + speed_penalty;
    



    return reward;
}

    // For moving targets: Reward closing velocity, not just pointing. Enables “intercepting”. instead of “chasing”.
//     let eps: f32 = 1e-6;

//     let vel = world.velocity[i];
//     let vel_norm2 = vel.norm_squared();
//     let vel_norm = vel_norm2.sqrt() + eps;

//     // ------------------------------------
//     // Find closest enemy (team 1)
//     // ------------------------------------
//     let Some((rel_pos, rel_vel)) =
//         world.team1_neighbors[i][0]
//     else {
//         // No target in range → mild time penalty
//         return -0.5;
//     };

//     let dist2 = rel_pos.norm_squared();
//     let dist = dist2.sqrt() + eps;

//     // ------------------------------------
//     // Closing velocity (key for interception)
//     // ------------------------------------
//     let closing_speed = -rel_pos.dot(&rel_vel) / dist;

//     // Positive if approaching, negative if separating
//     let closing_reward = 0.5 * closing_speed;

//     // ------------------------------------
//     // Velocity alignment toward target
//     // ------------------------------------
//     let target_dir = Vec3 {
//         x: rel_pos.x / dist,
//         y: rel_pos.y / dist,
//         z: rel_pos.z / dist,
//     };

//     let vel_dir = Vec3 {
//         x: vel.x / vel_norm,
//         y: vel.y / vel_norm,
//         z: vel.z / vel_norm,
//     };

//     let alignment = vel_dir.dot(&target_dir);

//     let mut alignment_reward = if alignment >= 0.0 {
//         alignment.powf(4.0) * vel_norm
//     } else {
//         3.0 * alignment * vel_norm
//     };

//     alignment_reward = 0.01* alignment_reward;

//     // ------------------------------------
//     // Distance shaping
//     // ------------------------------------
//     let distance_penalty = -0.002 * dist;
//     let distance_progress =
//         (world.previous_position[i]-rel_pos)
//             .norm_squared()
//             .sqrt()
//         - dist;

//     let progress_reward = 0.001 * distance_progress;

//     // ------------------------------------
//     // Velocity regularization
//     // ------------------------------------
//     let velocity_penalty = -0.0005 * vel_norm2;

//     // ------------------------------------
//     // Time pressure
//     // ------------------------------------
//     let time_penalty = -1.0;
//     println!("alignment_reward, closing_reward, progress_reward, distance_penalty, velocity_penalty: {} {} {} {} {}", alignment_reward, closing_reward, progress_reward,distance_penalty,velocity_penalty);
//     time_penalty
//         + alignment_reward
//         + closing_reward
//         + progress_reward
//         + distance_penalty
//         + velocity_penalty
// }


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


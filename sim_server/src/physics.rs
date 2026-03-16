use crate::config::SimConfig;
use crate::world::{Vec3, World};
use std::sync::Arc;

pub fn step(world: &mut World, actions: &[Vec3], config: Arc<SimConfig>) {
    let max_velocity = config.physics.max_velocity;

    for i in 0..world.num_drones_team_0 {
        if !world.alive[i] {
            continue;
        }
        world.acceleration[i] = actions[i];
        world.velocity[i].x += world.acceleration[i].x * world.dt;
        world.velocity[i].y += world.acceleration[i].y * world.dt;
        world.velocity[i].z += world.acceleration[i].z * world.dt;
        world.velocity[i].x = world.velocity[i].x.clamp(-max_velocity, max_velocity);
        world.velocity[i].y = world.velocity[i].y.clamp(-max_velocity, max_velocity);
        world.velocity[i].z = world.velocity[i].z.clamp(-max_velocity, max_velocity);

        world.previous_position[i] = world.position[i];
        world.position[i].x += world.velocity[i].x * world.dt;
        world.position[i].y += world.velocity[i].y * world.dt;
        world.position[i].z += world.velocity[i].z * world.dt;
        world.distance_to_origin2[i] = world.position[i].norm_squared();
    }

    // Team 1 Controller Step
    if config.team_1_controller.enabled {
        match config.team_1_controller.behavior.as_str() {
            "straight_flying" => {
                // Default controller for team 1
                for i in 0..world.num_drones_team_1 {
                    let idx = i + world.num_drones_team_0;
                    if !world.alive[idx] {
                        continue;
                    }

                    // Use three different “irrational” multipliers for x, y, z velocity, different each episode
                    let x = -(world.seed as f32 * 1.3247).cos().abs(); // Force vector away from +x
                    let y = (world.seed as f32 * 2.4563).cos() * 0.2;
                    let z = (world.seed as f32 * 3.5671).cos() * 0.2;

                    

                    // Normalize to make it a unit vector
                    let mag = (x * x + y * y + z * z).sqrt();

                    world.velocity[idx].x = x / mag;
                    world.velocity[idx].y = y / mag;
                    world.velocity[idx].z = z / mag;

                    world.previous_position[idx] = world.position[idx];
                    world.position[idx].x += world.velocity[idx].x * world.dt;
                    world.position[idx].y += world.velocity[idx].y * world.dt;
                    world.position[idx].z += world.velocity[idx].z * world.dt;
                    world.distance_to_origin2[idx] = world.position[idx].norm_squared();
                }
            }
            _ => {
                panic!("Undefined team 1 controller behavior")
            }
        }
    }

    world.step += 1;
}

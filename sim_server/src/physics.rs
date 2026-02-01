use crate::world::{World, Vec3};

pub fn step(world: &mut World, actions: &[Vec3], max_velocity: f32) {
    for i in 0..world.num_drones_team_0 {
        if !world.alive[i] { continue; }
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
        world.distance_to_origin2[i] = world.position[i].x * world.position[i].x + world.position[i].y * world.position[i].y + world.position[i].z * world.position[i].z;
        // println!("world.distance_to_origin2[i]: {}", world.distance_to_origin2[i])
    }
    world.step += 1;
}

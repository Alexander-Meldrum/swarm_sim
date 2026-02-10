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
        world.distance_to_origin2[i] = world.position[i].norm_squared();
        // println!("actions: {:?}", actions[i])
    }


    // Default controllers for team 1
    for i in 0..world.num_drones_team_1 {
        let idx = i + world.num_drones_team_0;
        if !world.alive[idx] { continue; }
        // world.acceleration[i] = actions[i];
        // world.velocity[i].x += world.acceleration[i].x * world.dt;
        // world.velocity[i].y += world.acceleration[i].y * world.dt;
        // world.velocity[i].z += world.acceleration[i].z * world.dt;
        world.velocity[idx].x = 0.01;
        world.velocity[idx].y = 0.01;
        world.velocity[idx].z = 0.01;

        world.previous_position[idx] = world.position[idx];
        world.position[idx].x += world.velocity[idx].x * world.dt;
        world.position[idx].y += world.velocity[idx].y * world.dt;
        world.position[idx].z += world.velocity[idx].z * world.dt;
        world.distance_to_origin2[idx] = world.position[idx].norm_squared();
        // println!("actions: {:?}", actions[i])
    }




    world.step += 1;
}

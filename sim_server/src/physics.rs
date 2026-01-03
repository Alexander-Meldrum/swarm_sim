use crate::world::{World, Vec3};

pub fn step(world: &mut World, actions: &[Vec3]) {
    for i in 0..world.num_drones {
        if !world.alive[i] { continue; }
        world.acceleration[i] = actions[i];
        world.velocity[i].x += world.acceleration[i].x * world.dt;
        world.velocity[i].y += world.acceleration[i].y * world.dt;
        world.velocity[i].z += world.acceleration[i].z * world.dt;
        world.position[i].x += world.velocity[i].x * world.dt;
        world.position[i].y += world.velocity[i].y * world.dt;
        world.position[i].z += world.velocity[i].z * world.dt;
    }
    world.step += 1;
}

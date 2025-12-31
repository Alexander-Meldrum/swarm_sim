use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use std::io::{Write};
use crate::world::{World, Vec3};
use crate::physics;
use crate::learning::{Rewards, calc_rewards};
use crate::logging::{open_episode_log, log_world};
pub mod swarm_proto {
    tonic::include_proto!("swarm_proto");
}
use swarm_proto::swarm_proto_service_server::{
    SwarmProtoService,
};
use swarm_proto::{DroneObservation, StepRequest, StepResponse, ResetRequest, ResetResponse};

pub struct SimServer {
    // world is a tokio async Mutex, to not block async gRPC tokio thread (reset/step from multiple controllers). Standard Mutex would block.
    pub world: Mutex<World>, 
}

#[tonic::async_trait]
impl SwarmProtoService for SimServer {
    async fn step(
        &self,
        request: Request<StepRequest>,
    ) -> Result<Response<StepResponse>, Status> {
        let mut world = self.world.lock().await;

        // Reject stepping after done
        if world.done {
            return Err(Status::failed_precondition(
                "episode is done, call reset()",
            ));}
        
        // Consume step request
        let req = request.into_inner();

        // Step order check
        if req.step != world.step {
            return Err(Status::failed_precondition(format!(
                "Out-of-order step: client={}, server={}",
                req.step, world.step
            )));
        }       

        let actions: Vec<Vec3> = req.actions.into_iter()
            .map(|a| Vec3{x: a.ax, y: a.ay, z: a.az}).collect();

        physics::step(&mut world, &actions);

        // Calculate Rewards
        let rewards: Rewards = calc_rewards(&world);

        // Log World, braces ensure the log lock is released quickly.
        {
            log_world(&mut world).unwrap();
        }

        println!("[simulator] Step: {}, Time: {}", world.step, (world.step as f32) * world.dt);
        println!("[simulator] Actions (drone 0) From Controller: ax = {}, ay = {}, az = {}", actions[0].x, actions[0].y, actions[0].z);
        // Check if simulation done
        if world.step >= world.max_steps {
            world.done = true;
            // Flush log writing
            world.log.as_mut().expect("log file not initialized before flush").flush()?;
            
            println!("[Simulator] Episode {} Done, reached max_steps!", world.episode);
        }
        // TODO: observations, add to stepresponse

        Ok(Response::new(StepResponse{ step: world.step, observations: vec![], done: world.done, global_reward: rewards.global_reward }))
    }

    async fn reset(
        &self,
        request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        println!("[simulator] Resetting Simulator World...");

        // Use async "await" at async network / sync simulator boundary
        let mut world = self.world.lock().await;
        
        // Extract protobuf message
        let request = request.into_inner();

        // Prepare Log for the Episode
        world.episode += 1;
        world.log = Some(open_episode_log(world.episode));

        // Read requested num_drones, max_steps, dt, seed, randomize_init_pos, arena_size, min_dist
        let num_drones = request.num_drones as usize;
        if num_drones == 0 {
            return Err(Status::invalid_argument("[simulator] num_drones must be > 0"));
        }
        let max_steps= request.max_steps;
        let dt= request.dt;
        let seed = request.seed;
        let randomize_init_pos = request.randomize_init_pos;
        let arena_size = request.arena_size;
        let min_dist = request.min_dist;

        // move out log safely from Option<>
        let log = world.log.take(); 

        // Reset world state, replace the World inside the mutex
        *world = World::new(num_drones, max_steps, dt, world.episode, log);

        world.init_drones(Some(seed), randomize_init_pos, arena_size, min_dist);
        

        // TODO improve obs
        let obs = (0..world.position.len()).map(|_| DroneObservation {
            ox: 0.0, oy: 0.0, oz: 0.0, collision_count: 0
        }).collect();

        // Log World, braces ensure the log lock is released quickly.
        {
            log_world(&mut world).unwrap();
        }

        println!("[simulator] Simulator World Setup Complete");
        println!("[simulator] **********************");
        println!("[simulator] num_drones: {}", world.num_drones);
        println!("[simulator] max_steps:  {}", world.max_steps);
        println!("[simulator] step:       {}", world.step);
        println!("[simulator] dt:         {}", world.dt);
        println!("[simulator] done:       {}", world.done);
        println!("[simulator] **********************");

        Ok(Response::new(ResetResponse {
            step: 0,
            num_drones: world.num_drones as u32,
            max_steps: world.max_steps,
            dt: dt,
            observations: obs,
        }))
    }
}

// use std::sync::Mutex;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use crate::world::{World, Vec3};
use crate::physics;
pub mod swarm_proto {
    tonic::include_proto!("swarm_proto");
}
use swarm_proto::swarm_proto_service_server::{
    SwarmProtoService,
};
use swarm_proto::{DroneObservation, StepRequest, StepResponse, ResetRequest, ResetResponse};


pub struct SimServer {
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

        println!("Controller Step: {}", req.step);
        println!("Simulator Step: {}", world.step);
        println!("Actions (drone 0) From Controller: ax = {}, ay = {}, az = {}", actions[0].x, actions[0].y, actions[0].z);

        
        physics::step(&mut world, &actions);

        // Check termination
        if world.step >= world.max_steps {
            world.done = true;
        }

        Ok(Response::new(StepResponse{ step: world.step, observations: vec![], done: world.done, reward: 0.0 }))
    }

    async fn reset(
        &self,
        request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        println!("Resetting Simulator World...");
        let mut world = self.world.lock().await;

        // Extract protobuf message
        let request = request.into_inner();

        // num_drones
        let num_drones = request.num_drones as usize;
        if num_drones == 0 {
            return Err(Status::invalid_argument("num_drones must be > 0"));
        }

        // max_steps
        let max_steps= request.max_steps;

        // dt
        let dt= request.dt;

        // Reset world state, replace the World inside the mutex
        *world = World::new(num_drones, max_steps, dt);

        let obs = (0..world.position.len()).map(|_| DroneObservation {
            ox: 0.0, oy: 0.0, oz: 0.0, collision_count: 0
        }).collect();

        println!("Simulator World Setup Complete");
        println!("************");
        println!("num_drones: {}", world.num_drones);
        println!("max_steps:  {}", world.max_steps);
        println!("************");

        Ok(Response::new(ResetResponse {
            step: 0,
            num_drones: world.num_drones as u32,
            max_steps: world.max_steps,
            dt: dt,
            observations: obs,
        }))
    }
}

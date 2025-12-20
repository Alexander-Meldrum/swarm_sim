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
        let req = request.into_inner();
        let actions: Vec<Vec3> = req.actions.into_iter()
            .map(|a| Vec3{x: a.ax, y: a.ay, z: a.az}).collect();

        println!("Simulator Step: {}", req.step);
        println!("Actions (drone 0) From Controller: ax = {}, ay = {}, az = {}", actions[0].x, actions[0].y, actions[0].z);
        

        let mut world = self.world.lock().await;

        // STRICT ORDERING CHECK
        if req.step != world.step {
            return Err(Status::failed_precondition(format!(
                "Out-of-order step: client={}, server={}",
                req.step, world.step
            )));
        }
        physics::step(&mut world, &actions);

        world.step += 1;
        
        Ok(Response::new(StepResponse{ step: 0, observations: vec![], done: false, reward: 0.0 }))
    }

    async fn reset(
        &self,
        _request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        println!("Resetting Simulator World...");
        let mut world = self.world.lock().await;

        // Reset world state
        world.reset_state(); // positions, velocities, alive, etc.
        // world.step = 0;

        let obs = (0..world.position.len()).map(|_| DroneObservation {
            ox: 0.0, oy: 0.0, oz: 0.0, collision_count: 0
        }).collect();

        Ok(Response::new(ResetResponse {
            step: 0,
            observations: obs,
        }))
    }
}

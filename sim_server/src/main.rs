mod world;
mod physics;
mod grpc;
mod logging;
mod learning;
pub mod swarm_proto {
    tonic::include_proto!("swarm_proto");
}
use world::World;
use tokio::sync::Mutex;

use grpc::SimServer;
use crate::grpc::swarm_proto::swarm_proto_service_server::{SwarmProtoServiceServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO Remove magic numbers
    // let enable_profiling = true;
    let episode = 0;
    let num_drones_default = 1;
    let max_steps_default = 10000;
    let arena_size_default = 10.0;
    let dt_default = 0.02;
    let world = World::new(num_drones_default, max_steps_default, arena_size_default, dt_default, episode, None, None);

    let server = SimServer {
        world: Mutex::new(world),
    };

    let addr = "[::1]:50051".parse()?;
    println!("[simulator] Sim server running and listening for StepRequests on localhost:50051");

    tonic::transport::Server::builder()
        .add_service(SwarmProtoServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}

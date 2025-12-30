mod world;
mod physics;
mod grpc;
mod logging;
mod learning;
// use std::fs::File;
// use std::io::{BufWriter};
pub mod swarm_proto {
    tonic::include_proto!("swarm_proto");
}
use world::World;
// use learning::Rewards;

use tokio::sync::Mutex;
// use std::sync::Mutex;

use grpc::SimServer;
use crate::grpc::swarm_proto::swarm_proto_service_server::{
    SwarmProtoServiceServer,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TODO Remove magic numbers 
    let episode = 0;
    let world = World::new(64, 10000, 0.02, episode, None);

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

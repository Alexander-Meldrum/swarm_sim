mod world;
mod physics;
mod grpc;
mod config;
mod logging;
mod learning;
mod utils;
pub mod swarm_proto {
    tonic::include_proto!("swarm_proto");
}
// use world::World;
use tokio::sync::Mutex;
use std::sync::Arc;


use utils::Args;
use utils::load_config;
use clap::Parser;

use grpc::SimServer;
use crate::grpc::swarm_proto::swarm_proto_service_server::{SwarmProtoServiceServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments
    let args = Args::parse();
    // Load config ONCE
    let config = load_config(&args.config)?;
    let config = Arc::new(config);
    println!("[simulator] ✅ Loaded config from {}", args.config);


    // TODO remove new world function
    // let enable_profiling = true;
    // let episode = 0;
    // let num_drones_team_0_default = 1;
    // let num_drones_team_1_default = 1;
    // let max_steps_default = 10000;
    // let arena_size_default = 10.0;
    // let dt_default = 0.02;
    // let world = World::new(num_drones_team_0_default, num_drones_team_1_default, max_steps_default, arena_size_default, dt_default, episode, None, None);

    let server = SimServer {
        // config: Arc::new(config), 
        // Clone the Arc, does not copy the config data
        config: config.clone(),
        // world: Mutex::new(world),
        world: Mutex::new(None),
    };

    let addr = "[::1]:50051".parse()?;
    println!("[simulator] Sim server running and listening for StepRequests on localhost:50051");

    tonic::transport::Server::builder()
        .add_service(SwarmProtoServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}

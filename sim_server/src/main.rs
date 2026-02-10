mod world;
mod physics;
mod grpc;
mod config;
mod logging;
mod learning;
mod utils;
mod observations;
pub mod swarm_proto {
    tonic::include_proto!("swarm_proto");
}
use world::World;
use tokio::sync::Mutex;
use std::sync::Arc;
use utils::{Args};
use utils::load_config;
use observations::{ObservationBuffer};
use clap::Parser;

use grpc::{SimServer, Simulator};
use crate::grpc::swarm_proto::swarm_proto_service_server::{SwarmProtoServiceServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments
    let args = Args::parse();
    println!("{:#?}", args);
    // println!("[simulator] Starting simulator with args: {}", args);
    // Load config ONCE
    let config = load_config(&args.config)?;
    let config = Arc::new(config);
    println!("[simulator] ✅ Loaded config from {}", args.config);

    // let simulator = Simulator {
    //     world: Mutex::new(World::dummy()),
    //     obs_buf: ObservationBuffer::new(0,0),
    // };
    // let server = SimServer {
    //     // Clone the Arc, does not copy the config data
    //     config: config.clone(),
    //     // World will be setup afte reset gRPC call

    //     sim: Mutex<simulator,
    //     // tokio::sync::Mutex<Simulator>,
    //     // world: Mutex::new(World::dummy()),

    //     // obs_buf: ObservationBuffer::new(0,0),
    // };
    // let config =  config: config.clone(),

    let server = SimServer {
        config: config.clone(),
        sim: Mutex::new(Simulator {
            world: World::dummy(),
            obs_buf: ObservationBuffer::new(0, 0),
        }),
    };

    let addr = "[::1]:50051".parse()?;
    println!("[simulator] Sim server running and listening for StepRequests on localhost:50051");

    tonic::transport::Server::builder()
        .add_service(SwarmProtoServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}

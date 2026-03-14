mod config;
mod grpc;
mod learning;
mod logging;
mod observations;
mod physics;
mod utils;
mod world;
pub mod swarm_proto {
    tonic::include_proto!("swarm_proto");
}
use clap::Parser;
use observations::ObservationBuffer;
use std::sync::Arc;
use tokio::sync::Mutex;
use utils::load_config;
use utils::Args;
use world::World;

use crate::grpc::swarm_proto::swarm_proto_service_server::SwarmProtoServiceServer;
use grpc::{SimServer, Simulator};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments
    let args = Args::parse();
    println!("{:#?}", args);
    // Load config once
    let config = load_config(&args.config)?;
    let config = Arc::new(config);
    println!("[simulator] Loaded config from {}", args.config);

    let server = SimServer {
        config: config.clone(),
        sim: Mutex::new(Simulator {
            world: World::dummy(),
            obs_buf: ObservationBuffer::new(0, 0),
        }),
    };

    let addr = args.bind.parse()?;
    println!("[simulator] Sim server running and listening for StepRequests on localhost:50051");

    tonic::transport::Server::builder()
        .add_service(SwarmProtoServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}

use clap::Parser;
use std::fs::File;
use std::io::Read;
use crate::config::SimConfig;


/// Simulator command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Args {
    /// Path to simulator config YAML
    #[arg(long, default_value = "../configs/sim.yaml")]
    pub config: String,

    /// gRPC bind address
    #[arg(long, default_value = "[::1]:50051")]
    pub bind: String,
}

pub fn load_config(path: &str) -> Result<SimConfig, Box<dyn std::error::Error>> {
    let mut file = File::open(path)
        .map_err(|e| format!("{}: {}", path, e))?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| format!("{}: {}", path, e))?;

    let config: SimConfig = serde_yaml::from_str(&contents)
        .map_err(|e| format!("{}: {}", path, e))?;

    Ok(config)
}

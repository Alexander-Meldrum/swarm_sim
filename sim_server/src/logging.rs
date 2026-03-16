use crate::config::SimConfig;
use crate::learning::Rewards;
use crate::world::World;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub fn open_new_log(type_of_log: &str, world: &World, config: Arc<SimConfig>) -> BufWriter<File> {
    create_dir_all("logs").unwrap();
    let path = format!("logs/{:05}_{}.bin", world.episode, type_of_log);
    println!("[Simulator] Log file ({}): {}", type_of_log, path);

    let mut file = BufWriter::new(File::create(path).unwrap());

    let metadata = LogMetadata {
        episode: world.episode,
        dt: config.physics.dt,
        num_drones_team_0: world.num_drones_team_0 as u32,
        num_drones_team_1: world.num_drones_team_1 as u32,
        stationary_target_exists: config.target.enabled as u8,
        stationary_target_pos: [
            config.target.position[0],
            config.target.position[1],
            config.target.position[2],
        ],
        stationary_target_radius: config.target.radius,
        schema_version: 1,
    };
    
    // Write metadata at the beginning
    metadata.write_to(&mut file).unwrap();
    file.flush().unwrap();

    file
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LogMetadata {
    pub episode: u64,
    pub dt: f32,
    pub num_drones_team_0: u32,
    pub num_drones_team_1: u32,
    pub stationary_target_exists: u8,
    pub stationary_target_pos: [f32; 3],
    pub stationary_target_radius: f32,
    pub schema_version: u16,
}

impl LogMetadata {
    /// Write the metadata in binary form
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.episode.to_le_bytes())?;
        writer.write_all(&self.dt.to_le_bytes())?;
        writer.write_all(&self.num_drones_team_0.to_le_bytes())?;
        writer.write_all(&self.num_drones_team_1.to_le_bytes())?;
        writer.write_all(&self.stationary_target_exists.to_le_bytes())?;
        writer.write_all(&self.stationary_target_pos[0].to_le_bytes())?;
        writer.write_all(&self.stationary_target_pos[1].to_le_bytes())?;
        writer.write_all(&self.stationary_target_pos[2].to_le_bytes())?;
        writer.write_all(&self.stationary_target_radius.to_le_bytes())?;
        writer.write_all(&self.schema_version.to_le_bytes())?;
        Ok(())
    }
}

// When updating logging, tools/bin_reader.py might need edits.
pub fn log_world(world: &mut World, rewards: &Rewards) -> std::io::Result<()> {
    let log = world.state_log.as_mut().expect("state log not initialized");

    log.write_all(&world.step.to_le_bytes())?;
    log.write_all(&(world.num_drones as u32).to_le_bytes())?; // TODO, remove this logging

    for i in 0..world.num_drones_team_0 as usize {
        let p = &world.position[i];
        let v = &world.velocity[i];

        log.write_all(&p.x.to_le_bytes())?;
        log.write_all(&p.y.to_le_bytes())?;
        log.write_all(&p.z.to_le_bytes())?;
        log.write_all(&v.x.to_le_bytes())?;
        log.write_all(&v.y.to_le_bytes())?;
        log.write_all(&v.z.to_le_bytes())?;
        log.write_all(&rewards.rewards[i].to_le_bytes())?;
    }

    for i in 0..world.num_drones_team_1 as usize {
        let p = &world.position[world.num_drones_team_0 + i];
        let v = &world.velocity[world.num_drones_team_0 + i];

        log.write_all(&p.x.to_le_bytes())?;
        log.write_all(&p.y.to_le_bytes())?;
        log.write_all(&p.z.to_le_bytes())?;
        log.write_all(&v.x.to_le_bytes())?;
        log.write_all(&v.y.to_le_bytes())?;
        log.write_all(&v.z.to_le_bytes())?;
        log.write_all(&0.0f32.to_le_bytes())?; // No rewards to team 1
    }

    log.flush()?; // flush once at the end for efficiency

    Ok(())
}

pub fn log_events(world: &mut World) -> std::io::Result<()> {
    // Check if events occured before proceeding
    if world.events.is_empty() {
        return Ok(());
    }
    let log = world.event_log.as_mut().expect("event log not initialized");

    for e in &world.events {
        // Step
        log.write_all(&e.step.to_le_bytes())?;
        // Event kind
        log.write_all(&[e.kind as u8])?;
        // Drone / target fields
        log.write_all(&e.drone_a.to_le_bytes())?;
        log.write_all(&e.drone_a_position.x.to_le_bytes())?;
        log.write_all(&e.drone_a_position.y.to_le_bytes())?;
        log.write_all(&e.drone_a_position.z.to_le_bytes())?;
        log.write_all(&e.drone_b.to_le_bytes())?;
        log.write_all(&e.drone_b_position.x.to_le_bytes())?;
        log.write_all(&e.drone_b_position.y.to_le_bytes())?;
        log.write_all(&e.drone_b_position.z.to_le_bytes())?;
        log.write_all(&e.target_id.to_le_bytes())?;
    }
    log.flush()?; // flush once at the end for efficiency

    Ok(())
}

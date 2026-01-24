use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use crate::world::{World};
use crate::learning::{Rewards};

pub fn open_new_log(type_of_log: &str, episode: u64) -> BufWriter<File> {
    create_dir_all("logs").unwrap();
    let path = format!("logs/{:05}_{}.bin", episode, type_of_log);
    println!("[Simulator] Log file ({}): {}", type_of_log, path);
    BufWriter::new(File::create(path).unwrap())
}

// When updating logging, tools/bin_reader.py might need edits.
pub fn log_world(
    world: &mut World, rewards: &Rewards
) -> std::io::Result<()> {

    let log = world.state_log.as_mut().expect("state log not initialized");

    log.write_all(&world.step.to_le_bytes())?;
    log.write_all(&world.num_drones_team_0.to_le_bytes())?;
    log.write_all(&rewards.global_reward.to_le_bytes())?;

    for i in 0..world.num_drones_team_0 as usize {
        let p = &world.position[i];
        let v = &world.velocity[i];

        log.write_all(&p.x.to_le_bytes())?;
        log.write_all(&p.y.to_le_bytes())?;
        log.write_all(&p.z.to_le_bytes())?;

        log.write_all(&v.x.to_le_bytes())?;
        log.write_all(&v.y.to_le_bytes())?;
        log.write_all(&v.z.to_le_bytes())?;
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

    // log: &mut BufWriter<File>, events: &Vec<Event>
    for e in &world.events {
        // Step
        log.write_all(&e.step.to_le_bytes())?;
        // Event kind
        log.write_all(&[e.kind as u8])?;
        // Drone / target fields
        log.write_all(&e.drone_a.to_le_bytes())?;
        log.write_all(&e.drone_b.to_le_bytes())?;
        log.write_all(&e.target_id.to_le_bytes())?;
    }
    log.flush()?; // flush once at the end for efficiency

    Ok(())
}

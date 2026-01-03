use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use crate::world::{World};


pub fn open_episode_log(episode: u64) -> BufWriter<File> {
    create_dir_all("logs").unwrap();
    let path = format!("logs/episode_{:06}.bin", episode);
    println!("[Simulator] Episode log file: {}", path);
    BufWriter::new(File::create(path).unwrap())
}

// When updating logging, tools/bin_to_csv.py might need edits.
pub fn log_world(
    world: &mut World,
) -> std::io::Result<()> {

    // let num = world.num_drones;
    let log = world.log.as_mut().expect("log not initialized");

    log.write_all(&world.step.to_le_bytes())?;
    log.write_all(&world.num_drones.to_le_bytes())?;
    log.write_all(&world.rewards.global_reward.to_le_bytes())?;

    for i in 0..world.num_drones as usize {
        let p = &world.position[i];
        let v = &world.velocity[i];

        log.write_all(&p.x.to_le_bytes())?;
        log.write_all(&p.y.to_le_bytes())?;
        log.write_all(&p.z.to_le_bytes())?;

        log.write_all(&v.x.to_le_bytes())?;
        log.write_all(&v.y.to_le_bytes())?;
        log.write_all(&v.z.to_le_bytes())?;
    }

    Ok(())
}


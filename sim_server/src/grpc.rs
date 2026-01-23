use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
use std::sync::Arc;
use std::io::{Write};
use pprof::ProfilerGuard;
use std::fs::{self, File};
use crate::world::{Vec3, World, rebuild_grid, detect_collisions, detect_target_hits};
use crate::physics;
use crate::config::SimConfig;
use crate::learning::{Rewards, calc_rewards};
use crate::logging::{open_new_log, log_world, log_events};
use crate::utils::{ObservationBuffer, OBS_DIM};
pub mod swarm_proto {
    tonic::include_proto!("swarm_proto");
}
use swarm_proto::swarm_proto_service_server::{
    SwarmProtoService,
};
use swarm_proto::{StepRequest, StepResponse, ResetRequest, ResetResponse};

pub struct Simulator {
    pub world: World,
    pub obs_buf: ObservationBuffer,
}

pub struct SimServer {
    /// Immutable configuration (arena, physics, rules)
    pub config: Arc<SimConfig>,
    /// Simulator is a tokio async Mutex, to not block async gRPC tokio thread (reset/step from multiple controllers). Standard Mutex would block.
    pub sim: Mutex<Simulator>,
}

#[tonic::async_trait]
impl SwarmProtoService for SimServer {
    async fn reset(
        &self,
        request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        println!("[simulator] Resetting Simulator World...");

        // Use async "await" at async network / sync simulator boundary
        let mut sim = self.sim.lock().await;
        // destructure with a single borrow
        let Simulator { world, obs_buf } = &mut *sim;
        // let mut world = &mut sim.world;
        // let obs_buf = &mut sim.obs_buf;

        let config = self.config.clone();

        // Extract protobuf message
        let request = request.into_inner();

        // Prepare Log for the Episode
        world.episode += 1;
        let mut state_log = None;
        let mut event_log = None; 
        if config.logging.enabled {
            world.state_log = Some(open_new_log("states", world.episode));
            world.event_log = Some(open_new_log("events", world.episode));
            // move out log safely from Option<>
            state_log = world.state_log.take();
            event_log = world.event_log.take(); 
        } 

        // Read request
        let num_drones_team_0 = request.num_drones_team_0 as usize;
        let num_drones_team_1 = request.num_drones_team_1 as usize;
        if num_drones_team_0 == 0 {
            return Err(Status::invalid_argument("[simulator] num_drones_team_0 must be > 0"));
        }
        let max_steps= request.max_steps;
        let seed = request.seed;

        
        // Reset world state, replace the World inside the mutex
        // *world = World::new(config.clone(), num_drones_team_0, num_drones_team_1, max_steps, world.episode, state_log, event_log);
        *world = World::new(config.clone(), num_drones_team_0, num_drones_team_1, max_steps, world.episode, state_log, event_log);

        // Initialize observation buffer once per episode
        // *obs_buf = ObservationBuffer::new(world.num_drones, OBS_DIM);
        *obs_buf = ObservationBuffer::new(world.num_drones, OBS_DIM);

        println!("[simulator] New World, step: {}", world.step);
        // TODO
        // let enable_profiling = true;
        // Start profiling after creating new world
        if config.logging.profiling_enabled {
            world.profiler = Some(ProfilerGuard::new(config.logging.profiling_frequency).expect("failed to start profiler")); // Unit: Hz
        }
        world.init_drones(Some(seed), config.clone());
        

        // build flattened observations, zero copy move, obs_buf.obs set to empty vec
        obs_buf.build(&world);
        let obs = std::mem::take(&mut obs_buf.obs);

        // Log World, braces ensure the log lock is released quickly.
        if config.logging.enabled {
            log_world(world).unwrap();
            log_events(world).unwrap();
        }

        println!("[simulator] Simulator World Setup Complete");
        println!("[simulator] **********************");
        println!("[simulator] num_drones_team_0: {}", world.num_drones_team_0);
        println!("[simulator] num_drones_team_1: {}", world.num_drones_team_1);
        println!("[simulator] cell_size:  {}", world.cell_size);
        println!("[simulator] max_steps:  {}", world.max_steps);
        println!("[simulator] step:       {}", world.step);
        println!("[simulator] dt:         {}", world.dt);
        println!("[simulator] done:       {}", world.done);
        
        println!("[simulator] **********************");

        Ok(Response::new(ResetResponse {
            step: 0,
            num_drones_team_0: world.num_drones_team_0 as u32,
            num_drones_team_1: world.num_drones_team_1 as u32,
            max_steps: world.max_steps,
            dt: world.dt,
            observations: obs,
        }))
    }
    
    async fn step(
        &self,
        request: Request<StepRequest>,
    ) -> Result<Response<StepResponse>, Status> {
        let mut sim = self.sim.lock().await;
        // destructure with a single borrow
        let Simulator { world, obs_buf } = &mut *sim;
        // let mut world = &mut sim.world;
        // let obs_buf = &mut sim.obs_buf;

        // ----- 1. Handle Inputs ------ 
        // Reject stepping after done
        if world.done {
            return Err(Status::failed_precondition(
                "episode is done, call reset()",
            ));}
        
        // Consume step request
        let req = request.into_inner();

        // Step order check
        if req.step != world.step {
            return Err(Status::failed_precondition(format!(
                "Out-of-order step: req.step={}, world.step={}",
                req.step, world.step
            )));
        }       

        let actions: Vec<Vec3> = req.actions.into_iter()
            .map(|a| Vec3{x: a.ax, y: a.ay, z: a.az}).collect();
        
        // ----- 2. Calculate World & Rewards ------
        // Clear per-step events
        world.events.clear();

        physics::step(world, &actions);

        // Build spatial index
        rebuild_grid(world);

        // Detect collisions & target hits
        detect_collisions(world);
        detect_target_hits(world);

        // Calculate Rewards
        let rewards: Rewards = calc_rewards(&world);

        // ----- 3. Log ------ 
        // Log World, braces ensure the log lock is released quickly.
        if self.config.logging.enabled {
            log_world(world).unwrap();
            log_events(world).unwrap();
        }

        println!("[simulator] Step: {}, Time: {}", world.step, (world.step as f32) * world.dt);
        println!("[simulator] Actions (drone 0) From Controller: ax = {}, ay = {}, az = {}", actions[0].x, actions[0].y, actions[0].z);
        // Check if simulation done
        if world.step >= world.max_steps {
            world.done = true;
            // Flush log writing
            world.state_log.as_mut().expect("log file not initialized before flush").flush()?;
            
            println!("[Simulator] Episode {} Done, reached max_steps!", world.episode);

            if self.config.logging.profiling_enabled {
                // Finish profiling
                if let Some(guard) = world.profiler.take() {
                    let report = guard.report().build().unwrap();
                    let mut dir = std::env::current_dir().unwrap();
                    dir.push("profiles");
                    fs::create_dir_all(&dir).unwrap();
                    let filename = format!("flamegraph-episode-{}.svg", world.episode);
                    let path = dir.join(filename);
                    let file = File::create(&path).unwrap();
                    report.flamegraph(file).unwrap();
                    println!("🔥 Flamegraph written to: {}", path.display());                
                }
            }

        }
        // build flattened observations, zero copy move, obs_buf.obs set to empty vec
        obs_buf.build(&world);
        let obs = std::mem::take(&mut obs_buf.obs);

        // ----- 4. Output ------ 
        Ok(Response::new(StepResponse{ step: world.step, observations: obs, done: world.done, global_reward: rewards.global_reward }))
    }
}

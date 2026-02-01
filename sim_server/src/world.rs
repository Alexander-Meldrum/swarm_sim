use std::fs::File;
use std::io::{BufWriter};
use std::sync::Arc;
use rand::rngs::SmallRng;
use rand::{SeedableRng, Rng};
use pprof::ProfilerGuard;
// use crate::learning::{Rewards};
use crate::config::SimConfig;


#[derive(Copy, Clone, Debug)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub fn zero() -> Self { Self { x:0.0,y:0.0,z:0.0 } }

    #[inline(always)]
    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline(always)]
    pub fn norm_squared(&self) -> f32 {
        self.dot(self)
    }

    // #[inline(always)]
    // pub fn norm(&self) -> f32 {
    //     self.norm_squared().sqrt()
    // }
}

pub struct World {
    pub profiler: Option<ProfilerGuard<'static>>,
    pub step: u64,
    pub max_steps: u64,
    pub dt: f32,
    pub num_drones_team_0: usize,
    pub num_drones_team_1: usize,
    pub num_drones: usize,
    pub episode: u64,
    pub state_log: Option<BufWriter<File>>,
    pub event_log: Option<BufWriter<File>>,

    // Drone Characteristics
    pub team: Vec<u8>,
    pub drone_radius: f32,

    // Drone State
    pub position: Vec<Vec3>,
    pub previous_position: Vec<Vec3>,
    pub velocity: Vec<Vec3>,
    pub acceleration: Vec<Vec3>,
    pub alive: Vec<bool>,
    pub distance_to_origin2: Vec<f32>,

    // Collision accounting
    pub collisions_desired: Vec<u32>,
    pub collisions_undesired: Vec<u32>,

    // Targets
    pub targets: Vec<Target>,

    // Events
    pub events: Vec<Event>,

    // Spatial Grid
    pub arena_min: Vec3,
    pub arena_max: Vec3,
    pub cell_size: f32,
    pub grid_dim: (usize, usize, usize),
    pub grid: Vec<Vec<usize>>, // flat grid: cell → drone indices

    // Reinforcement Learning Rewards
    // pub rewards: Rewards,

    pub done: bool,
}
/// Static target that drones can hit
pub struct Target {
    pub position: Vec3,
    pub radius: f32,
    pub alive: bool,
}

#[repr(u8)]  // for binary logging compatibility.
#[derive(Debug, Clone, Copy)]
pub enum EventKind {
    TargetHit = 1,
    DroneCollision = 2,
    // WallHit = 3,
}
/// Discrete event emitted when a meaningful hit occurs
#[derive(Debug, Clone)]
pub struct Event {
    pub kind: EventKind,
    pub step: u64,
    pub drone_a: u32,
    pub drone_a_position: Vec3,
    pub drone_b: u32,   // for drone-drone, u32::MAX if none
    pub drone_b_position: Vec3,
    pub target_id: u32, // for target hits, u32::MAX if none
}

impl World {
    /// Init a new world
    pub fn new(config: Arc<SimConfig>, num_drones_team_0: usize, num_drones_team_1: usize, max_steps: u64, episode: u64, state_log:Option<BufWriter<File>>, event_log:Option<BufWriter<File>>) -> Self {
        // TODO? implement a reset fn to avoid reallocation each episode.

        let num_drones = num_drones_team_0 + num_drones_team_1;
        // Get values from config
        // lowest & highest corner of the arena
        let arena_min = Vec3 { x: config.arena.min[0], y: config.arena.min[1], z: config.arena.min[2] };
        let arena_max = Vec3 { x: config.arena.max[0], y: config.arena.max[1], z: config.arena.max[2] };
        let drone_radius = config.collisions.radius;
        let dt = config.physics.dt;
        let targets = vec![Target{position : Vec3 {x: config.target.position[0], y: config.target.position[1], z: config.target.position[2]}, radius : config.target.radius, alive : true}];

        // events are per-step; reserve aggressively to avoid realloc
        let events = Vec::with_capacity(num_drones / 4);

        
        // Try to keep cell_size close to "interaction_radius". cell_size affects performance a lot
        // Too small → grid overhead dominates, Too large → neighbor checks dominate
        // let cell_size = 50.0;      // TODO
        // Guestimate good value for cell_size
        let arena_extent = Vec3 {
            x: arena_max.x - arena_min.x,
            y: arena_max.y - arena_min.y,
            z: arena_max.z - arena_min.z,
        };
        let desired_average_drones_per_cell = 2.0;
        let num_cells = num_drones as f32 / desired_average_drones_per_cell;
        let cells_per_axis = num_cells.cbrt().ceil();  // Approximation when non-uniform arena
        let max_extent = arena_extent.x
            .max(arena_extent.y)
            .max(arena_extent.z);
        let cell_size = max_extent / cells_per_axis;

        // Compute grid dimensions, axis-aligned bounding box
        let nx = ((arena_max.x - arena_min.x) / cell_size).ceil() as usize;
        let ny = ((arena_max.y - arena_min.y) / cell_size).ceil() as usize;
        let nz = ((arena_max.z - arena_min.z) / cell_size).ceil() as usize;
        let num_cells = nx * ny * nz;
        
        Self {
            profiler: None,
            step: 0,
            max_steps: max_steps,
            dt: dt,
            num_drones_team_0: num_drones_team_0,
            num_drones_team_1: num_drones_team_1,
            num_drones: num_drones,
            episode: episode,
            state_log: state_log,
            event_log: event_log,
            // Pre-allocate per-drone state (SoA layout)
            position: vec![Vec3::zero(); num_drones],
            previous_position: vec![Vec3::zero(); num_drones],
            velocity: vec![Vec3::zero(); num_drones],
            acceleration: vec![Vec3::zero(); num_drones],
            alive: vec![true; num_drones],
            distance_to_origin2: vec![0.0; num_drones],
            team: vec![0; num_drones],
            drone_radius,
            collisions_desired: vec![0; num_drones],
            collisions_undesired: vec![0; num_drones],
            targets: targets,
            events,
            arena_min,
            arena_max,
            cell_size,
            grid_dim: (nx, ny, nz),
            // Pre-allocate spatial grid. Each cell holds drone indices
            grid: vec![Vec::new(); num_cells],
            // rewards: Rewards::new(num_drones),
            done: false,
        }
    }


    // Dummy used for init of world at simulator startup
    pub fn dummy() -> Self {
        Self {
            profiler: None,
            step: 0,
            max_steps: 0,
            dt: 0.02,
            num_drones_team_0: 0,
            num_drones_team_1: 0,
            num_drones: 0,
            episode: 0,
            state_log: None,
            event_log: None,
            position: Vec::new(),
            previous_position: Vec::new(),
            velocity: Vec::new(),
            acceleration: Vec::new(),
            alive: Vec::new(),
            distance_to_origin2: Vec::new(),
            team: Vec::new(),
            drone_radius: 0.0,
            collisions_desired: Vec::new(),
            collisions_undesired: Vec::new(),
            targets: Vec::new(),
            events: Vec::new(),
            arena_min: Vec3 {x: 0.0, y: 0.0, z: 0.0},
            arena_max: Vec3 {x: 0.0, y: 0.0, z: 0.0},
            cell_size: 0.0,
            grid_dim: (0, 0, 0),
            // Pre-allocate spatial grid. Each cell holds drone indices
            grid: Vec::new(),
            // rewards: Rewards::new(0),
            done: false,
        }
    }

    /// Init drone states
    pub fn init_drones(&mut self, seed: Option<u64>, config: Arc<SimConfig>) {

        let mut rng = SmallRng::seed_from_u64(seed.expect("No Seed Provided"));       
        if config.arena.randomize_init_pos {
            println!("[simulator] Randomizing init positions with seed: {}", seed.expect("No Seed Provided"));
        }

        // TODO: handle teams differently?
        // Random init positions
        for i in 0..self.num_drones {
            let pos = loop {
                let candidate = Vec3 {
                    x: rng.random_range(self.arena_min.x..self.arena_max.x),
                    y: rng.random_range(self.arena_min.y..self.arena_max.y),
                    z: rng.random_range(self.arena_min.z..self.arena_max.z),
                };

                if self.position.iter().all(|p| {
                    (p.x - candidate.x).powi(2)
                  + (p.y - candidate.y).powi(2)
                  + (p.z - candidate.z).powi(2)
                  >= config.arena.min_dist.powi(2)
                }) {
                    break candidate;
                }
            };

            self.position[i] = pos;
        }
    }
}

/// Offsets of the current cell and its 26 neighbors
const NEIGHBORS: [(isize, isize, isize); 27] = [
    (-1,-1,-1), (-1,-1, 0), (-1,-1, 1),
    (-1, 0,-1), (-1, 0, 0), (-1, 0, 1),
    (-1, 1,-1), (-1, 1, 0), (-1, 1, 1),

    ( 0,-1,-1), ( 0,-1, 0), ( 0,-1, 1),
    ( 0, 0,-1), ( 0, 0, 0), ( 0, 0, 1),
    ( 0, 1,-1), ( 0, 1, 0), ( 0, 1, 1),

    ( 1,-1,-1), ( 1,-1, 0), ( 1,-1, 1),
    ( 1, 0,-1), ( 1, 0, 0), ( 1, 0, 1),
    ( 1, 1,-1), ( 1, 1, 0), ( 1, 1, 1),
];

/// Returns the flat grid index for a position, or None if out of bounds, Cannot have "&mut World" as input because of mut borrowing issue"
#[inline]
// fn grid_index(world: &mut World, p: Vec3) -> Option<usize> {
fn grid_index(arena_min: &Vec3, cell_size: &f32, grid_dim: &(usize, usize, usize), p: &Vec3) -> Option<usize> {
    // Convert continuous position into discrete grid coordinates
    // TODO precalculate inverse cell_size
    let ix = ((p.x - arena_min.x) / cell_size) as isize;
    let iy = ((p.y - arena_min.y) / cell_size) as isize;
    let iz = ((p.z - arena_min.z) / cell_size) as isize;

    let (nx, ny, nz) = grid_dim;

    // Reject positions outside the arena
    if ix < 0 || iy < 0 || iz < 0 {
        return None;
    }
    if ix >= *nx as isize || iy >= *ny as isize || iz >= *nz as isize {
        return None;
    }

    // Flatten 3D (ix, iy, iz) → 1D index
    Some(ix as usize + iy as usize * nx + iz as usize * nx * ny)
}

/// Clears and repopulates the spatial grid with alive drones
pub fn rebuild_grid(world: &mut World) {
    // Clear vectors but keep their capacity
    for cell in &mut world.grid {
        cell.clear();
    }

    // Insert each alive drone into its grid cell
    for i in 0..world.num_drones {
        if !world.alive[i] {
            continue;
        }

        // if let Some(idx) = grid_index(world, world.position[i]) {
        if let Some(idx) = grid_index(&world.arena_min, &world.cell_size, &world.grid_dim, &world.position[i]) {

            world.grid[idx].push(i);
        }
    }
}

/// Detects physical collisions between drones using the spatial grid
pub fn detect_collisions(world: &mut World) {
    // TODO check logic
    let r2 = (2.0 * world.drone_radius) * (2.0 * world.drone_radius);
    let (nx, ny, nz) = world.grid_dim;

    // Iterate over every grid cell
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let cell_idx = ix + iy * nx + iz * nx * ny;
                let cell = &world.grid[cell_idx];

                // For every drone in this cell
                for &i in cell {
                    if !world.alive[i] {
                        continue;
                    }

                    let pi = world.position[i];
                    let team_i = world.team[i];

                    // Check neighboring cells for possible collisions
                    for (dx, dy, dz) in NEIGHBORS {
                        let nix = ix as isize + dx;
                        let niy = iy as isize + dy;
                        let niz = iz as isize + dz;

                        // Skip invalid neighbor cells
                        if nix < 0 || niy < 0 || niz < 0 {
                            continue;
                        }
                        if nix >= nx as isize || niy >= ny as isize || niz >= nz as isize {
                            continue;
                        }

                        let n_idx = nix as usize
                            + niy as usize * nx
                            + niz as usize * nx * ny;

                        // Compare against drones in the neighbor cell
                        for &j in &world.grid[n_idx] {
                            // Prevent double checking and dead drones
                            if j <= i || !world.alive[j] {
                                continue;
                            }

                            let pj = world.position[j];

                            // Squared distance (no sqrt)
                            let dx = pi.x - pj.x;
                            let dy = pi.y - pj.y;
                            let dz = pi.z - pj.z;
                            let dist2 = dx*dx + dy*dy + dz*dz;

                            if dist2 < r2 {
                                // Physical collision confirmed
                                world.alive[i] = false;
                                world.alive[j] = false;
                                // Bindary logging uses u32 instead of usize for portability. 
                                world.events.push(Event::drone_collision(world.step, i.try_into().unwrap(), pi, j.try_into().unwrap(), pj));

                                // TODO improve team goal logic, now both teams want to hit eachother
                                // Count all physical collisions
                                world.collisions_undesired[i] += 1;
                                world.collisions_undesired[j] += 1;

                                // Tactical collision: only if teams differ
                                if team_i != world.team[j] {
                                    world.collisions_desired[i] += 1;
                                    world.collisions_desired[j] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Detects meaningful drone–target hits using the spatial grid
pub fn detect_target_hits(world: &mut World, drone_radius: f32) {
    let (nx, ny, nz) = world.grid_dim;

    // let (targets, grid) = (&mut world.targets, &world.grid);

    let (arena_min, cell_size, grid_dim) =  (&world.arena_min, &world.cell_size, &world.grid_dim);

    // let targets = &world.targets;

    for (target_id, target) in world.targets.iter_mut().enumerate() {
    // for (target_id, target) in targets.iter().enumerate() {
        if !target.alive {
            continue;
        }

        // Find which grid cell the target occupies
        let Some(cell_idx) = grid_index(&arena_min, &cell_size, grid_dim, &target.position) else {
            continue;
        };

        // Convert flat index back to 3D grid coordinates
        let ix = cell_idx % nx;
        let iy = (cell_idx / nx) % ny;
        let iz = cell_idx / (nx * ny);

        // Check target cell and neighbors
        for (dx, dy, dz) in NEIGHBORS {
            let nix = ix as isize + dx;
            let niy = iy as isize + dy;
            let niz = iz as isize + dz;

            if nix < 0 || niy < 0 || niz < 0 {
                continue;
            }
            if nix >= nx as isize || niy >= ny as isize || niz >= nz as isize {
                continue;
            }

            let n_idx = nix as usize
                + niy as usize * nx
                + niz as usize * nx * ny;

            // Only drones near the target are checked
            for &drone_id in &world.grid[n_idx] {
                if !world.alive[drone_id] {
                    continue;
                }

                // Only attackers can score hits
                if world.team[drone_id] != 0 {
                    continue;
                }

                let dp = world.position[drone_id];
                let dt = target.position;

                let dx = dp.x - dt.x;
                let dy = dp.y - dt.y;
                let dz = dp.z - dt.z;

                let dist2 = dx*dx + dy*dy + dz*dz;
                let combined_radius = target.radius + drone_radius;

                if dist2 < combined_radius * combined_radius {
                    // Hit confirmation event
                    world.events.push(Event::target_hit(world.step, drone_id.try_into().unwrap(), dp, target_id.try_into().unwrap()));

                    // Apply hit consequences
                    // target.alive = false;
                    world.alive[drone_id] = false;

                    // One hit per target # TODO, flag?
                    // break;
                }
            }

            if !target.alive {
                break;
            }
        }
    }
}

pub const NONE_U32: u32 = u32::MAX;

impl Event {
    pub fn target_hit(step: u64, drone_id: u32, drone_a_position: Vec3, target_id: u32) -> Self {
        Self {
            kind: EventKind::TargetHit,
            step,
            drone_a: drone_id,
            drone_a_position: drone_a_position,
            drone_b: NONE_U32,
            drone_b_position: Vec3 {x: 0.0, y: 0.0, z: 0.0},
            target_id: target_id,
        }
    }

    pub fn drone_collision(step: u64, a: u32, drone_a_position: Vec3, b: u32, drone_b_position: Vec3) -> Self {
        Self {
            kind: EventKind::DroneCollision,
            step,
            drone_a: a,
            drone_a_position: drone_a_position,
            drone_b: b,
            drone_b_position: drone_b_position,
            target_id: NONE_U32,
        }
    }

    // TODO implement wall/ground hit
    // pub fn wall_hit(step: u64, drone_id: u32) -> Self {
    //     Self {
    //         kind: EventKind::WallHit,
    //         step,
    //         drone_a: drone_id,
    //         drone_b: NONE_U32,
    //         target_id: NONE_U32,
    //     }
    // }
}
use std::fs::File;
use std::io::{BufWriter};
use std::sync::Arc;
use rand::rngs::SmallRng;
use rand::{SeedableRng, Rng};
use pprof::ProfilerGuard;
// use crate::learning::{Rewards};
use crate::config::SimConfig;

use std::ops::{Add, Sub};

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

    #[inline(always)]
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }
}
impl Add for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}
impl Sub for Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

pub const K_NEIGHBORS: usize = 1;

pub struct World {
    pub profiler: Option<ProfilerGuard<'static>>,
    pub step: u64,
    pub max_steps: u64,
    pub dt: f32,
    pub num_drones_team_0: usize,
    pub num_drones_team_1: usize,
    pub num_drones: usize,
    // Per-drone nearest neighbors
    pub team0_neighbors: Vec<[Option<(Vec3, Vec3)>; K_NEIGHBORS]>,
    pub team1_neighbors: Vec<[Option<(Vec3, Vec3)>; K_NEIGHBORS]>,
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

    pub done: bool,
}
/// Static target that drones can hit
pub struct Target {
    pub position: Vec3,
    pub radius: f32,
    pub alive: bool,
}

#[repr(u8)]  // for binary logging compatibility.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
        let num_drones = num_drones_team_0 + num_drones_team_1;

        let mut team = Vec::with_capacity(num_drones_team_0 + num_drones_team_1);
        // First team 0 drones
        team.extend(std::iter::repeat(0).take(num_drones_team_0));
        // Then team 1 drones
        team.extend(std::iter::repeat(1).take(num_drones_team_1));
        // Get values from config
        // lowest & highest corner of the arena
        let arena_min = Vec3 { x: config.arena.min[0], y: config.arena.min[1], z: config.arena.min[2] };
        let arena_max = Vec3 { x: config.arena.max[0], y: config.arena.max[1], z: config.arena.max[2] };
        let drone_radius = config.collisions.radius;
        let dt = config.physics.dt;
        let targets = vec![Target{position : Vec3 {x: config.target.position[0], y: config.target.position[1], z: config.target.position[2]}, radius : config.target.radius, alive : true}];

        // events are per-step; reserve aggressively to avoid realloc
        let events = Vec::with_capacity(num_drones / 4);
        let cell_size = config.arena.cell_size;


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
            team0_neighbors: vec![[None; K_NEIGHBORS]; num_drones],
            team1_neighbors: vec![[None; K_NEIGHBORS]; num_drones],
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
            team: team,
            drone_radius,
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
            team0_neighbors: vec![[None; K_NEIGHBORS]; 0],
            team1_neighbors: vec![[None; K_NEIGHBORS]; 0],
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

        if let Some(idx) = grid_index(&world.arena_min, &world.cell_size, &world.grid_dim, &world.position[i]) {
            world.grid[idx].push(i);
        }
    }
}

/// Detects physical collisions between drones using the spatial grid
pub fn detect_collisions(world: &mut World) {
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
    let (arena_min, cell_size, grid_dim) =  (&world.arena_min, &world.cell_size, &world.grid_dim);

    for (target_id, target) in world.targets.iter_mut().enumerate() {
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

pub fn find_k_nearest_per_team<const K: usize>(
    world: &World,
    drone_id: usize,
    max_radius: f32,
) -> (
    [Option<(Vec3, Vec3)>; K],
    [Option<(Vec3, Vec3)>; K],
) {
    // --------------------------------------------------
    // Setup
    // --------------------------------------------------
    let pos = world.position[drone_id];
    let vel = world.velocity[drone_id];

    let max_r2 = max_radius * max_radius;

    let cell_size = world.cell_size;
    let arena_min = world.arena_min;
    let (nx, ny, nz) = world.grid_dim;

    // Compute grid coordinates of this drone
    let ix = ((pos.x - arena_min.x) / cell_size) as isize;
    let iy = ((pos.y - arena_min.y) / cell_size) as isize;
    let iz = ((pos.z - arena_min.z) / cell_size) as isize;

    // Fixed-size k-nearest buffers
    let mut team0: [Option<(usize, f32)>; K] = [None; K];
    let mut team1: [Option<(usize, f32)>; K] = [None; K];

    // --------------------------------------------------
    // Scan neighbor cells
    // --------------------------------------------------
    for (dx, dy, dz) in NEIGHBORS {
        let nix = ix + dx;
        let niy = iy + dy;
        let niz = iz + dz;

        // Skip invalid cells
        if nix < 0 || niy < 0 || niz < 0 {
            continue;
        }
        if nix >= nx as isize || niy >= ny as isize || niz >= nz as isize {
            continue;
        }

        let cell_idx =
            nix as usize
            + niy as usize * nx
            + niz as usize * nx * ny;

        // --------------------------------------------------
        // Check drones in this cell
        // --------------------------------------------------
        for &other in &world.grid[cell_idx] {
            if other == drone_id || !world.alive[other] {
                continue;
            }

            let delta = world.position[other] - pos;
            let dist2 = delta.norm_squared();

            // Max sensing radius check
            if dist2 > max_r2 {
                continue;
            }

            // Select team buffer
            let buf = match world.team[other] {
                0 => &mut team0,
                1 => &mut team1,
                _ => continue,
            };

            // --------------------------------------------------
            // Insert into k-nearest buffer (sorted)
            // --------------------------------------------------
            if let Some((_, worst)) = buf[K - 1] {
                if dist2 >= worst {
                    continue;
                }
            }

            let mut i = 0;
            while i < K {
                match buf[i] {
                    Some((_, d)) if d <= dist2 => i += 1,
                    _ => {
                        for j in (i + 1..K).rev() {
                            buf[j] = buf[j - 1];
                        }
                        buf[i] = Some((other, dist2));
                        break;
                    }
                }
            }
        }
    }

    // --------------------------------------------------
    // Convert to relative position & velocity, points from the agent toward the target
    // --------------------------------------------------
    let mut out0: [Option<(Vec3, Vec3)>; K] = [None; K];
    let mut out1: [Option<(Vec3, Vec3)>; K] = [None; K];

    for i in 0..K {
        if let Some((id, _)) = team0[i] {
            out0[i] = Some((
                world.position[id] - pos,
                world.velocity[id] - vel,
            ));
        }
        if let Some((id, _)) = team1[i] {
            out1[i] = Some((
                world.position[id] - pos,
                world.velocity[id] - vel,
            ));
        }
    }

    (out0, out1)
}

pub fn find_k_nearest_drones(world: &mut World, max_radius: f32, ) {
    
    for i in 0..world.num_drones {
        if !world.alive[i] {
            continue;
        }

        let (t0, t1) =
            find_k_nearest_per_team::<K_NEIGHBORS>(&world, i, max_radius);

        world.team0_neighbors[i] = t0;
        world.team1_neighbors[i] = t1;
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
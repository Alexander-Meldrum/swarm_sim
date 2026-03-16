use serde::Deserialize;

// ===============================
// Top-level simulator config
// ===============================
#[derive(Debug, Deserialize)]
pub struct SimConfig {
    pub arena: ArenaConfig,
    pub target: TargetConfig,
    pub physics: PhysicsConfig,
    pub sensing: SensingConfig,
    pub collisions: CollisionConfig,
    pub team_1_controller: Team1ControllerConfig,
    pub logging: LoggingConfig,
}

// ===============================
// Arena / World Geometry
// ===============================
#[derive(Debug, Deserialize)]
pub struct ArenaConfig {
    /// Minimum corner of the arena (x, y, z) [m]
    pub min: [f32; 3],
    /// Maximum corner of the arena (x, y, z) [m]
    pub max: [f32; 3],
    /// Minimum corner of the start area (x, y, z) [m]
    pub min_start_area: [f32; 3],
    /// Maximum corner of the start area (x, y, z) [m]
    pub max_start_area: [f32; 3],
    /// Minimum distance between drones when randomizing start pos. [m]
    pub min_dist: f32,
    /// Flad to randomize init positions of all drones
    pub randomize_init_pos: bool,
    /// Cell size for the grid used for collision detection, nearest neighbor etc. [m]
    /// Try to keep cell_size close to "interaction_radius". cell_size affects performance a lot
    pub cell_size: f32,
}

// ===============================
// Target
// ===============================
#[derive(Debug, Deserialize)]
pub struct TargetConfig {
    pub enabled: bool,
    pub position: [f32; 3], // [m]
    pub radius: f32,        // [m]
}

// ===============================
// Physics Parameters
// ===============================
#[derive(Debug, Deserialize)]
pub struct PhysicsConfig {
    /// Fixed simulation timestep [seconds]
    pub dt: f32,
    /// Maximum possible velocity magnitude [m/s]
    pub max_velocity: f32,
}

// ===============================
// Sensing Parameters
// ===============================
#[derive(Debug, Deserialize)]
pub struct SensingConfig {
    /// Max sensing distance, has to be smaller than 2*cell_size. [m]
    pub max_sensing: f32,
}

// ===============================
// Collision Handling
// ===============================
#[derive(Debug, Deserialize)]
pub struct CollisionConfig {
    /// Collision radius per drone [m]
    pub radius: f32,
}

// ===============================
// team_1_controller, Built-in Controllers
// ===============================
#[derive(Debug, Deserialize)]
pub struct Team1ControllerConfig {
    /// Enable simulator-owned rule-based drones
    pub enabled: bool,
    /// Behavior name ("straight_flying" etc.)
    pub behavior: String,
    /// init position of team 1 swarm center, positions randomized around this point [m]
    pub init_pos_center: [f32; 3],
}

// ===============================
// Logging Configuration
// ===============================
#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    /// Enable episode logging
    pub enabled: bool,
    /// Enable profiling of simulator
    pub profiling_enabled: bool,
    pub profiling_frequency: i32,
}

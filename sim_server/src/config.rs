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
    // pub controllers: ControllersConfig,
    pub logging: LoggingConfig,
    // pub debug: DebugConfig,
}


// ===============================
// Arena / World Geometry
// ===============================
#[derive(Debug, Deserialize)]
pub struct ArenaConfig {
    /// Minimum corner of the arena (x, y, z)
    pub min: [f32; 3],
    /// Maximum corner of the arena (x, y, z)
    pub max: [f32; 3],
    /// Minimum distance between drones when randomizing start pos.
    pub min_dist: f32,
    pub randomize_init_pos: bool,
    pub cell_size: f32,
}

// ===============================
// Target
// ===============================
#[derive(Debug, Deserialize)]
pub struct TargetConfig {

    pub position: [f32; 3],
    pub radius: f32,

}

// ===============================
// Physics Parameters
// ===============================
#[derive(Debug, Deserialize)]
pub struct PhysicsConfig {
    /// Fixed simulation timestep (seconds)
    pub dt: f32,
    // /// Maximum allowed velocity magnitude
    pub max_velocity: f32,
    // /// Linear drag coefficient
    // pub drag: f32,
}

// ===============================
// Sensing Parameters
// ===============================
#[derive(Debug, Deserialize)]
pub struct SensingConfig {
    /// Max sensing distance, has to be smaller than 2*cell_size
    pub max_sensing: f32,
    // pub nr_of_neighbors: u32  // Hard coded for performance
}

// ===============================
// Collision Handling
// ===============================
#[derive(Debug, Deserialize)]
pub struct CollisionConfig {
    /// Collision radius per drone
    pub radius: f32,
    // /// Disable drone immediately after collision
    // pub disable_on_hit: bool,
}

// ===============================
// Built-in Controllers
// ===============================
// #[derive(Debug, Deserialize)]
// pub struct ControllersConfig {
//     pub rule_based: RuleBasedControllerConfig,
// }

// #[derive(Debug, Deserialize)]
// pub struct RuleBasedControllerConfig {
//     /// Enable simulator-owned rule-based drones
//     pub enabled: bool,
//     /// Team ID controlled by simulator logic
//     pub team_id: u8,
//     /// Behavior name ("patrol", "seek", etc.)
//     pub behavior: String,
// }

// ===============================
// Logging Configuration
// ===============================
#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    /// Enable episode logging
    pub enabled: bool,
    // /// Output directory for logs
    // pub output_dir: String,
    /// Enable profiling of simulator
    pub profiling_enabled: bool,
    pub profiling_frequency: i32,
}

// ===============================
// Debug / Development Options
// ===============================
//
// #[derive(Debug, Deserialize)]
// pub struct DebugConfig {
//     /// Print simulator events to stdout
//     pub verbose: bool,
// }
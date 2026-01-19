use serde::Deserialize;

//
// ===============================
// Top-level simulator config
// ===============================
//
#[derive(Debug, Deserialize)]
pub struct SimConfig {
    pub arena: ArenaConfig,
    pub physics: PhysicsConfig,
    pub collisions: CollisionConfig,
    pub controllers: ControllersConfig,
    pub logging: LoggingConfig,
    pub debug: DebugConfig,
}

//
// ===============================
// Arena / World Geometry
// ===============================
//
#[derive(Debug, Deserialize)]
pub struct ArenaConfig {
    // /// size of Arena, m TODO
    // pub arena_size: f32,
    /// Minimum corner of the arena (x, y, z)
    pub min: [f32; 3],
    /// Maximum corner of the arena (x, y, z)
    pub max: [f32; 3],
    /// Minimum distance between drones when randomizing start pos.
    pub min_dist: f32,

    pub randomize_init_pos: bool,
}

//
// ===============================
// Physics Parameters
// ===============================
//
#[derive(Debug, Deserialize)]
pub struct PhysicsConfig {
    /// Fixed simulation timestep (seconds)
    pub dt: f32,
    /// Maximum allowed velocity magnitude
    pub max_velocity: f32,
    /// Linear drag coefficient
    pub drag: f32,
}

//
// ===============================
// Collision Handling
// ===============================
//
#[derive(Debug, Deserialize)]
pub struct CollisionConfig {
    /// Collision radius per drone
    pub radius: f32,
    /// Disable drone immediately after collision
    pub disable_on_hit: bool,
}

//
// ===============================
// Built-in Controllers
// ===============================
//
#[derive(Debug, Deserialize)]
pub struct ControllersConfig {
    pub rule_based: RuleBasedControllerConfig,
}

#[derive(Debug, Deserialize)]
pub struct RuleBasedControllerConfig {
    /// Enable simulator-owned rule-based drones
    pub enabled: bool,
    /// Team ID controlled by simulator logic
    pub team_id: u8,
    /// Behavior name ("patrol", "seek", etc.)
    pub behavior: String,
}

//
// ===============================
// Logging Configuration
// ===============================
//
#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    /// Enable episode logging
    pub enabled: bool,
    /// Output directory for logs
    pub output_dir: String,
    /// Logging format ("binary", "csv", "json")
    pub format: String,
    /// Write one file per episode
    pub per_episode: bool,
}

//
// ===============================
// Debug / Development Options
// ===============================
//
#[derive(Debug, Deserialize)]
pub struct DebugConfig {
    /// Print simulator events to stdout
    pub verbose: bool,
}
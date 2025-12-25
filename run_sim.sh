#!/usr/bin/env bash
set -euo pipefail

# ---------------- Configuration ----------------
SIM_DIR="sim_server"
SIM_BIN="$SIM_DIR/target/release/sim_server"

CTRL_DIR="swarms/controller_cpp"
CTRL_BUILD="$CTRL_DIR/build"
CTRL_BIN="$CTRL_BUILD/controller"

HOST="::1"
PORT="50051"

# ---------------- Build Rust simulator ----------------
echo "🔨 Building Rust simulator (release)..."
cd "$SIM_DIR"
cargo build --release
cd - >/dev/null

# ---------------- Build C++ controller ----------------
echo "🔨 Building C++ swarm controller..."
mkdir -p "$CTRL_BUILD"
cd "$CTRL_BUILD"
cmake ..
cmake --build . -- -j$(nproc)
cd - >/dev/null

# ---------------- Start simulator ----------------
echo "🚀 Starting simulator..."
"$SIM_BIN" &
SIM_PID=$!

# Ensure simulator is killed on exit
trap "echo '🛑 Stopping simulator'; kill $SIM_PID 2>/dev/null || true" EXIT

# ---------------- Wait for gRPC ----------------
echo "⏳ Waiting for simulator to be ready..."
for i in {1..10}; do
    if nc -z "$HOST" "$PORT"; then
        echo "✅ Simulator is ready"
        break
    fi
    sleep 0.5
done

if ! nc -z "$HOST" "$PORT"; then
    echo "❌ Simulator failed to start"
    exit 1
fi

# ---------------- Run controller ----------------
echo "🎮 Running swarm controller ..."
# "$CTRL_BIN"
stdbuf -oL "$CTRL_BIN"

echo "✅ Controller finished"
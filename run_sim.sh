#!/usr/bin/env bash
set -euo pipefail

# ---------------- Configuration ----------------
SIM_DIR="sim_server"
SIM_BIN="$SIM_DIR/target/release/sim_server"

CTRL_DIR="swarms/swarm_cpp"
CTRL_BUILD="$CTRL_DIR/build"
CTRL_BIN="$CTRL_BUILD/swarm"

HOST="::1"
PORT="50051"

# ---------------- Build C++ swarm controller ----------------
PROTO_SRC="proto"
OUT_DIR="swarms/swarm_cpp/build/proto"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Ensure grpc_cpp_plugin exists
GRPC_PLUGIN="$(command -v grpc_cpp_plugin)"
if [[ -z "$GRPC_PLUGIN" ]]; then
    echo "❌ grpc_cpp_plugin not found in PATH"
    exit 1
fi

# Generate C++ gRPC + protobuf files
protoc \
  --proto_path="$PROTO_SRC" \
  --cpp_out="$OUT_DIR" \
  --grpc_out="$OUT_DIR" \
  --plugin=protoc-gen-grpc="$GRPC_PLUGIN" \
  "$PROTO_SRC/swarm.proto"

echo "✅ Protobuf files generated in $OUT_DIR"

echo "🔨 Building C++ Swarm Controller..."
mkdir -p "$CTRL_BUILD"

cd "$CTRL_BUILD"
cmake ..
cmake --build . -- -j$(nproc)
cd - >/dev/null


# ---------------- Build Rust simulator ----------------
echo "🔨 Building Rust Simulator (release)..."
cd "$SIM_DIR"
cargo build --release
cd - >/dev/null

# # ---------------- Start simulator ----------------
echo "🚀 Starting Simulator..."
"$SIM_BIN" &
SIM_PID=$!

# ---------------- Build & Start Rust simulator in debug/profiling mode ----------------
# RUST_BACKTRACE=full "$SIM_BIN" &
# # perf record -g "$SIM_BIN" &
# echo "🔨 Building & Running Rust Simulator (debug/profiling/release)..."
# cd "$SIM_DIR"
# RUSTFLAGS="-C debuginfo=1" cargo flamegraph --release --bin sim_server --no-perf
# SIM_PID=$!
# cd - >/dev/null

# --------------------------------------------------------------------------------

# Ensure simulator is killed on exit
trap "echo '🛑 Stopping simulator'; kill $SIM_PID 2>/dev/null || true" EXIT

# ---------------- Wait for gRPC ----------------
ready=false
echo "⏳ Waiting for simulator to be ready..."
for i in {1..10}; do
    if nc -z "$HOST" "$PORT"; then
        echo "✅ Simulator is ready"
        ready=true
        break
    fi
    sleep 0.5
done

if [ "$ready" != true ]; then
    echo "❌ Simulator failed to start"
    exit 1
fi

# ---------------- Run Swarm controller ----------------
echo "🎮 Running Swarm Controller ..."
# "$CTRL_BIN"
stdbuf -oL "$CTRL_BIN"

echo "✅ Swarm Controller Finished"
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
# PROTO_SRC="proto"
# OUT_DIR="swarms/swarm_cpp/build/proto"

# # Ensure output directory exists
# mkdir -p "$OUT_DIR"

# # Ensure grpc_cpp_plugin exists
# GRPC_PLUGIN="$(command -v grpc_cpp_plugin)"
# if [[ -z "$GRPC_PLUGIN" ]]; then
#     echo "❌ grpc_cpp_plugin not found in PATH"
#     exit 1
# fi

# # Generate C++ gRPC + protobuf files
# protoc \
#   --proto_path="$PROTO_SRC" \
#   --cpp_out="$OUT_DIR" \
#   --grpc_out="$OUT_DIR" \
#   --plugin=protoc-gen-grpc="$GRPC_PLUGIN" \
#   "$PROTO_SRC/swarm.proto"

# echo "✅ Protobuf files generated in $OUT_DIR"

# echo "🔨 Building C++ Swarm Controller..."
# mkdir -p "$CTRL_BUILD"

# cd "$CTRL_BUILD"
# cmake ..
# cmake --build . -- -j$(nproc)
# cd - >/dev/null

# ---------------- Activate Python Venv ----------------
source .venv/bin/activate
# ---------------- Generate Python gRPC bindings ----------------
echo "🐍 Generating Python gRPC bindings..."

PY_CTRL_DIR="swarms/swarm_py"
PROTO_DIR="proto"

cd "$PY_CTRL_DIR"

python -m grpc_tools.protoc \
  -I "../../$PROTO_DIR" \
  --python_out=. \
  --grpc_python_out=. \
  "../../$PROTO_DIR/swarm.proto"

cd - >/dev/null

# ---------------- Build Rust simulator ----------------
echo "🔨 Building Rust Simulator (release)..."
cd "$SIM_DIR"
cargo build --release
cd - >/dev/null

# ---------------- Start simulator ----------------
echo "🚀 Starting Simulator..."
"$SIM_BIN" &
SIM_PID=$!


# ---------------- Run swarm controller (Python) ----------------
echo "🐍 Running Python swarm controller..."

cd swarms/swarm_py

# Optional but recommended: activate venv
# source .venv/bin/activate

python train.py

cd - >/dev/null

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
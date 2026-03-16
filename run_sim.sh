#!/usr/bin/env bash
set -euo pipefail

# ---------------- Configuration ----------------
# Flip controllers, allowed values ["python", "cpp"]
controller="python"

SIM_DIR="sim_server"
SIM_BIN="$SIM_DIR/target/release/sim_server"

CTRL_DIR="swarms/swarm_cpp"
CTRL_BUILD="$CTRL_DIR/build"
CTRL_BIN="$CTRL_BUILD/swarm"

HOST="::1"
PORT="50051"

# ---------------- Build swarm controller ----------------
if [ "$controller" = "cpp" ]; then
    # --- 1. Build Proto ---
    PROTO_SRC="proto"
    OUT_DIR="swarms/swarm_cpp/build/proto"
    # Ensure output directory exists
    mkdir -p "$OUT_DIR"

    # Ensure grpc_cpp_plugin exists
    GRPC_PLUGIN="$(command -v grpc_cpp_plugin)"
    if [[ -z "$GRPC_PLUGIN" ]]; then
        echo "[shell] grpc_cpp_plugin not found in PATH"
        exit 1
    fi

    # Generate C++ gRPC + protobuf files
    $HOME/protobuf-3.13.0-install/bin/protoc \
    -I"$PROTO_SRC" \
    -I"/home/s0001033/libtorch/include" \
    --cpp_out="$OUT_DIR" \
    --grpc_out="$OUT_DIR" \
    --plugin=protoc-gen-grpc="$GRPC_PLUGIN" \
    "$PROTO_SRC/swarm.proto"
    echo "[shell] Protobuf files generated in $OUT_DIR"

    # --- 2. Build C++ with cmake ---
    echo "[shell] Building C++ Swarm Controller..."
    rm -rf "$CTRL_BUILD"
    mkdir -p "$CTRL_BUILD"

    cd "$CTRL_BUILD"
    cmake ..
    cmake --build . -- -j$(nproc)
    cd - >/dev/null

elif [ "$controller" = "python" ]; then
    # --- 1. Activate Python Venv ---
    source .venv/bin/activate
    # --- 2. Generate Python gRPC bindings ---
    echo "[shell] Generating Python gRPC bindings..."
    PY_CTRL_DIR="swarms/swarm_py"
    PROTO_DIR="proto"
    cd "$PY_CTRL_DIR"

    python -m grpc_tools.protoc \
    -I "../../$PROTO_DIR" \
    --python_out=. \
    --grpc_python_out=. \
    "../../$PROTO_DIR/swarm.proto"
    cd - >/dev/null
else
    echo "[shell] Unknown controller"
fi


# ---------------- Build Rust simulator ----------------
echo "[shell] Building Rust Simulator (release)..."
cd "$SIM_DIR"
cargo build --release
cd - >/dev/null

# ---------------- Start simulator ----------------
echo "[shell] Starting Simulator..."
"$SIM_BIN" --config sim_server/configs/sim.yaml --bind "[${HOST}]:${PORT}" &
SIM_PID=$!

# Ensure simulator is killed on exit
trap "echo 'Stopping simulator'; kill $SIM_PID 2>/dev/null || true" EXIT

# ---------------- Wait for gRPC ----------------
ready=false
echo "[shell] Waiting for simulator to be ready..."
for i in {1..10}; do
    if nc -z "$HOST" "$PORT"; then
        echo "[shell] Simulator is ready"
        ready=true
        break
    fi
    sleep 0.5
done

if [ "$ready" != true ]; then
    echo "[shell] Simulator failed to start"
    exit 1
fi


# ---------------- Run swarm controller ----------------
if [ "$controller" = "cpp" ]; then
    echo "[shell] Running Cpp Swarm Controller ..."
    # "$CTRL_BIN"
    stdbuf -oL "$CTRL_BIN"
    echo "[shell] Swarm Controller Finished"

elif [ "$controller" = "python" ]; then
    echo "[shell] Running Python swarm controller..."
    cd swarms/swarm_py
    python train.py
    cd - >/dev/null
else
    echo "[shell] Could not run unknown controller"
fi

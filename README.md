# swarm_sim
Deterministic swarm simulator for testing & training swarm controllers

## Components
- Simulator Server (Rust)
- Swarm Controller for Reinforcement Learning (RL) (Python)
- Swarm Controller Deployment Example (C++)
- gRPC (protobuf) Communication
- Log Decoder & Visualizer (Python)

## Simulator Server (Rust)
- Deterministic Physics Simulator for Swarm Controllers [Rust]
- ECS (Entity Component System) for cache efficiency supporting large number of agents in a swarm.
- Binary State & Event (Collisions) Logger

## Swarm Controller for Reinforcement Learning (RL) (Python)
TODO

## Swarm Controller Deployment Example (C++)
Roadmap:
A production-style inference client that loads trained policies and controls the swarm through the same gRPC API as the training controller. This demonstrates how learned policies can be deployed without Python dependencies, suitable for embedded systems, robotics, or large-scale evaluation.

## gRPC (protobuf) Communication
- gRPC for bidirectional streaming, lockstep for enabling Reinforcment Learning

## Log Decoder & Visualizer
TODO

## How to run

To run the general build/run script:
./run_sim.sh

For manual build see below.

### Simulation Server [Rust]

protobuf-compiler  (sudo apt install protobuf-compiler)  
cd sim_server  
cargo build --release  
cargo run --release  


### Swarm Controller for Reinforcement Learning (RL) (Python)

Build proto files:

python -m grpc_tools.protoc \
  -I ../../proto \
  --python_out=. \
  --grpc_python_out=. \
  ../../proto/swarm.proto

### Swarm Controller Example [C++]

cd swarms/swarm_cpp  
mkdir build && cd build  
mkdir proto
cmake ..  
make  
./swarm  

#### How to update proto bindings
protoc \
  --proto_path=proto \
  --cpp_out=swarms/swarm_cpp/build/proto \
  --grpc_out=swarms/swarm_cpp/build/proto \
  --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
  proto/swarm.proto

Dependencies:
sudo apt install protobuf-compiler libgrpc++-dev  
sudo apt install protobuf-compiler-grpc  

### Visualization

TODO

python tools/plot_swarm.py logs/00001_states.bin
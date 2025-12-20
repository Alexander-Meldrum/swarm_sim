# swarm_sim
Deterministic swarm simulator for testing & training swarm controllers

## Roadmap

- Deterministic Physics Simulator for Swarm Controllers [Rust]
    - ECS (Entity Component System) for cache efficiency supporting large number of agents in a swarm.
    - Binary State & Event (Collisions) Logger
- Example Drone Swarm [C++]
- gRPC for bidirectional streaming, lockstep for enabling Reinforcment Learning
- Log Visualizer [Python]

## How to run
### Simulation Server [Rust]

protobuf-compiler  (sudo apt install protobuf-compiler)  
cd sim_server
cargo build --release
cargo run --release

### Swarm Controller Example [C++]

cd swarms/controller_cpp  
mkdir build && cd build  
cmake ..  
make  
./swarm

#### How to update proto bindings

(sudo apt install protobuf-compiler libgrpc++-dev)
sudo apt install protobuf-compiler-grpc

protoc \
  --proto_path=proto \
  --cpp_out=swarms/controller_cpp/src/proto \
  --grpc_out=swarms/controller_cpp/src/proto \
  --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
  proto/swarm.proto

### Visualization

TODO
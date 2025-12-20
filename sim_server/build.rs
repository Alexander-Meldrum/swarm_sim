// fn main() {
//     let proto_files = &["swarm.proto"];
//     let proto_include_dirs = &["../protobuf"];

//     tonic_build::configure()
//         .build_server(true)
//         .compile(proto_files, proto_include_dirs)
//         .unwrap();
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(
            &["../proto/swarm.proto"], // ← parent folder
            &["../proto"],             // ← include path
        )?;
    Ok(())
}

use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();
    
    println!("cargo:rerun-if-changed=build.rs");
    
    // Configure build based on target architecture
    if target.contains("aarch64") {
        println!("cargo:rustc-cfg=target_arch_aarch64");
        println!("cargo:rustc-cfg=feature=\"aarch64\"");
    } else if target.contains("x86_64") {
        println!("cargo:rustc-cfg=target_arch_x86_64");
        println!("cargo:rustc-cfg=feature=\"x86_64\"");
    }
    
    // Enable conditional compilation for architecture-specific features
    println!("cargo:rustc-cfg=hardware_monitoring");
    
    println!("Building for target: {}", target);
}
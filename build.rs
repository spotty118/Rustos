use std::env;

fn main() {
    // Simplified build script - just set up rerun triggers
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    
    // For now, skip the complex bootloader creation to get basic compilation working
    // This can be re-enabled once the kernel compiles successfully
}
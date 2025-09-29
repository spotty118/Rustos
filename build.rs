fn main() {
    // Enable link-time optimization for release builds
    #[cfg(not(debug_assertions))]
    println!("cargo:rustc-link-arg=-flto");

    // Add the linker script for the target
    println!("cargo:rustc-link-arg=--gc-sections");
}
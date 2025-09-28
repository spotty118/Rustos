use std::{env, path::PathBuf};

use bootloader::{BiosBoot, BootConfig, UefiBoot};

const KERNEL_BUILD_GUARD: &str = "RUSTOS_KERNEL_BUILD";

fn main() {
    if env::var(KERNEL_BUILD_GUARD).is_ok() {
        // We're in the recursive build used to compile the kernel artifact.
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");

    let kernel_path = build_kernel();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));

    let bios_image = out_dir.join("rustos-bios.img");
    let uefi_image = out_dir.join("rustos-uefi.img");

    let config = BootConfig::default();

    let mut bios_builder = BiosBoot::new(&kernel_path);
    bios_builder.set_boot_config(&config);
    bios_builder
        .create_disk_image(&bios_image)
        .expect("failed to create BIOS disk image");

    let mut uefi_builder = UefiBoot::new(&kernel_path);
    uefi_builder.set_boot_config(&config);
    uefi_builder
        .create_disk_image(&uefi_image)
        .expect("failed to create UEFI disk image");

    println!(
        "cargo:rustc-env=RUSTOS_BIOS_IMAGE={}",
        bios_image.display()
    );
    println!(
        "cargo:rustc-env=RUSTOS_UEFI_IMAGE={}",
        uefi_image.display()
    );
}

fn build_kernel() -> PathBuf {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("missing manifest dir"));
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("target"));
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let target_triple = "x86_64-unknown-none";

    let mut cmd = std::process::Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.current_dir(&manifest_dir)
        .env(KERNEL_BUILD_GUARD, "1")
        .args(["build", "--target", target_triple]);

    match profile.as_str() {
        "release" => {
            cmd.arg("--release");
        }
        "debug" => {}
        other => {
            cmd.args(["--profile", other]);
        }
    }

    let status = cmd.status().expect("failed to compile kernel");

    if !status.success() {
        panic!("kernel build failed");
    }

    let kernel_path = target_dir
        .join(target_triple)
        .join(&profile)
        .join("rustos");

    if !kernel_path.exists() {
        panic!("kernel binary not found at {}", kernel_path.display());
    }

    kernel_path
}
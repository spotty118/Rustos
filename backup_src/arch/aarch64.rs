/// ARM64 (AArch64) architecture-specific implementations

pub fn halt_cpu() {
    unsafe {
        core::arch::asm!("wfi"); // Wait For Interrupt
    }
}

pub fn read_cpu_cycles() -> u64 {
    // Read the Performance Monitor Cycle Count Register
    unsafe {
        let mut cycles: u64;
        core::arch::asm!(
            "mrs {}, pmccntr_el0",
            out(reg) cycles,
        );
        cycles
    }
}

pub fn get_cpu_features() -> &'static str {
    "aarch64 NEON FP-ARMV8"
}

pub fn enable_floating_point() {
    // Enable FPU and NEON on ARM64
    unsafe {
        let mut cpacr: u64;
        core::arch::asm!(
            "mrs {}, cpacr_el1",
            out(reg) cpacr,
        );
        cpacr |= (3 << 20); // Enable FP and SIMD
        core::arch::asm!(
            "msr cpacr_el1, {}",
            in(reg) cpacr,
        );
    }
}

pub fn get_cpu_cycles() -> u64 {
    read_cpu_cycles()
}

pub fn get_timestamp() -> u64 {
    // Return timestamp in microseconds based on cycle counter
    read_cpu_cycles() / 1000 // Simplified conversion
}

pub fn cpu_relax() {
    unsafe {
        core::arch::asm!("yield", options(nomem, nostack));
    }
}

pub fn read_performance_counter() -> u64 {
    read_cpu_cycles()
}

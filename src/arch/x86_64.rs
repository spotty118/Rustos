/// x86_64 architecture-specific implementations

use x86_64::instructions::hlt;

pub fn halt_cpu() {
    hlt();
}

pub fn read_cpu_cycles() -> u64 {
    // Read Time Stamp Counter
    unsafe {
        let mut low: u32;
        let mut high: u32;
        core::arch::asm!(
            "rdtsc",
            out("eax") low,
            out("edx") high,
        );
        ((high as u64) << 32) | (low as u64)
    }
}

pub fn get_cpu_features() -> &'static str {
    "x86_64 SSE AVX"
}

pub fn enable_floating_point() {
    // x86_64 has floating point enabled by default
}

pub fn read_performance_counter() -> u64 {
    read_cpu_cycles()
}
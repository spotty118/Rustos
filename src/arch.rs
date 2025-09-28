//! Architecture-specific functionality for RustOS
//!
//! This module provides CPU feature detection, architecture-specific
//! initialization, and low-level hardware access functions.

use core::sync::atomic::{AtomicBool, Ordering};
use alloc::{string::{String, ToString}, format};
use spin::Mutex;
use lazy_static::lazy_static;

/// CPU feature flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_ssse3: bool,
    pub has_sse4_1: bool,
    pub has_sse4_2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_fma: bool,
    pub has_rdrand: bool,
    pub has_rdseed: bool,
    pub has_tsx: bool,
    pub has_bmi1: bool,
    pub has_bmi2: bool,
    pub has_adx: bool,
    pub has_sha: bool,
    pub has_mpx: bool,
    pub has_sgx: bool,
    pub has_cet: bool,
    pub has_x87: bool,
    pub has_mmx: bool,
    pub has_cmov: bool,
    pub has_clflush: bool,
    pub has_monitor: bool,
    pub has_vmx: bool,
    pub has_smx: bool,
    pub has_speedstep: bool,
    pub has_thermal: bool,
    pub has_tm2: bool,
    pub has_sdbg: bool,
    pub has_xsave: bool,
    pub has_osxsave: bool,
    pub has_invariant_tsc: bool,
    pub has_apic: bool,
    pub has_x2apic: bool,
    pub has_hypervisor: bool,
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self {
            has_sse: false,
            has_sse2: false,
            has_sse3: false,
            has_ssse3: false,
            has_sse4_1: false,
            has_sse4_2: false,
            has_avx: false,
            has_avx2: false,
            has_fma: false,
            has_rdrand: false,
            has_rdseed: false,
            has_tsx: false,
            has_bmi1: false,
            has_bmi2: false,
            has_adx: false,
            has_sha: false,
            has_mpx: false,
            has_sgx: false,
            has_cet: false,
            has_x87: false,
            has_mmx: false,
            has_cmov: false,
            has_clflush: false,
            has_monitor: false,
            has_vmx: false,
            has_smx: false,
            has_speedstep: false,
            has_thermal: false,
            has_tm2: false,
            has_sdbg: false,
            has_xsave: false,
            has_osxsave: false,
            has_invariant_tsc: false,
            has_apic: false,
            has_x2apic: false,
            has_hypervisor: false,
        }
    }
}

/// CPU information structure
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub vendor_id: String,
    pub brand_string: String,
    pub family: u32,
    pub model: u32,
    pub stepping: u32,
    pub features: CpuFeatures,
    pub cache_line_size: u32,
    pub core_count: u32,
    pub thread_count: u32,
    pub base_frequency: u32,  // MHz
    pub max_frequency: u32,   // MHz
}

impl Default for CpuInfo {
    fn default() -> Self {
        Self {
            vendor_id: String::new(),
            brand_string: String::new(),
            family: 0,
            model: 0,
            stepping: 0,
            features: CpuFeatures::default(),
            cache_line_size: 64, // Common default
            core_count: 1,
            thread_count: 1,
            base_frequency: 1000, // 1 GHz default
            max_frequency: 1000,  // 1 GHz default
        }
    }
}

lazy_static! {
    /// Global CPU information
    static ref CPU_INFO: Mutex<CpuInfo> = Mutex::new(CpuInfo::default());
}

/// CPU initialization status
static CPU_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// CPUID instruction wrapper
#[inline]
fn cpuid(leaf: u32) -> (u32, u32, u32, u32) {
    // Stub implementation for compilation
    // In a real implementation, this would execute the CPUID instruction
    (0, 0, 0, 0)
}

/// CPUID instruction with sub-leaf
#[inline]
fn cpuid_count(leaf: u32, sub_leaf: u32) -> (u32, u32, u32, u32) {
    // Stub implementation for compilation
    // In a real implementation, this would execute the CPUID instruction
    (0, 0, 0, 0)
}

/// Initialize CPU detection and feature enumeration
pub fn init() -> Result<(), &'static str> {
    if CPU_INITIALIZED.load(Ordering::SeqCst) {
        return Ok(());
    }

    detect_cpu_features()?;
    CPU_INITIALIZED.store(true, Ordering::SeqCst);

    Ok(())
}

/// Detect CPU features using CPUID
fn detect_cpu_features() -> Result<(), &'static str> {
    let mut cpu_info = CPU_INFO.lock();

    // Check if CPUID is supported
    if !cpuid_supported() {
        return Err("CPUID not supported");
    }

    // Get vendor ID
    let (max_leaf, ebx, ecx, edx) = cpuid(0);
    cpu_info.vendor_id = format!("{}{}{}", 
        u32_to_string(ebx), 
        u32_to_string(edx), 
        u32_to_string(ecx)
    );

    if max_leaf >= 1 {
        // Get processor info and feature bits
        let (eax, ebx, ecx, edx) = cpuid(1);
        
        // Extract family, model, stepping
        cpu_info.stepping = eax & 0xF;
        cpu_info.model = (eax >> 4) & 0xF;
        cpu_info.family = (eax >> 8) & 0xF;
        
        // Extended family and model
        if cpu_info.family == 0xF {
            cpu_info.family += (eax >> 20) & 0xFF;
        }
        if cpu_info.family == 0xF || cpu_info.family == 0x6 {
            cpu_info.model += ((eax >> 16) & 0xF) << 4;
        }

        // Cache line size
        cpu_info.cache_line_size = ((ebx >> 8) & 0xFF) * 8;

        // Core count (approximate)
        cpu_info.core_count = ((ebx >> 16) & 0xFF).max(1);

        // Feature detection from ECX and EDX
        let mut features = CpuFeatures::default();
        
        // EDX features (leaf 1)
        features.has_x87 = (edx & (1 << 0)) != 0; // FPU
        features.has_mmx = (edx & (1 << 23)) != 0; // MMX
        features.has_sse = (edx & (1 << 25)) != 0; // SSE
        features.has_sse2 = (edx & (1 << 26)) != 0; // SSE2
        features.has_thermal = (edx & (1 << 29)) != 0; // Thermal Monitor
        features.has_apic = (edx & (1 << 9)) != 0; // APIC
        features.has_cmov = (edx & (1 << 15)) != 0; // CMOV
        features.has_clflush = (edx & (1 << 19)) != 0; // CLFLUSH

        // ECX features (leaf 1)
        features.has_sse3 = (ecx & (1 << 0)) != 0; // SSE3
        features.has_ssse3 = (ecx & (1 << 9)) != 0; // SSSE3
        features.has_fma = (ecx & (1 << 12)) != 0; // FMA
        features.has_sse4_1 = (ecx & (1 << 19)) != 0; // SSE4.1
        features.has_sse4_2 = (ecx & (1 << 20)) != 0; // SSE4.2
        features.has_x2apic = (ecx & (1 << 21)) != 0; // x2APIC
        features.has_avx = (ecx & (1 << 28)) != 0; // AVX
        features.has_rdrand = (ecx & (1 << 30)) != 0; // RDRAND
        features.has_hypervisor = (ecx & (1 << 31)) != 0; // Hypervisor
        features.has_monitor = (ecx & (1 << 3)) != 0; // MONITOR/MWAIT
        features.has_vmx = (ecx & (1 << 5)) != 0; // VMX
        features.has_smx = (ecx & (1 << 6)) != 0; // SMX
        features.has_speedstep = (ecx & (1 << 7)) != 0; // Enhanced SpeedStep
        features.has_tm2 = (ecx & (1 << 8)) != 0; // Thermal Monitor 2
        features.has_xsave = (ecx & (1 << 26)) != 0; // XSAVE
        features.has_osxsave = (ecx & (1 << 27)) != 0; // OSXSAVE

        cpu_info.features = features;
    }

    // Get extended features if available
    if max_leaf >= 7 {
        let (_, ebx, ecx, edx) = cpuid_count(7, 0);
        
        cpu_info.features.has_bmi1 = (ebx & (1 << 3)) != 0; // BMI1
        cpu_info.features.has_avx2 = (ebx & (1 << 5)) != 0; // AVX2
        cpu_info.features.has_bmi2 = (ebx & (1 << 8)) != 0; // BMI2
        cpu_info.features.has_rdseed = (ebx & (1 << 18)) != 0; // RDSEED
        cpu_info.features.has_adx = (ebx & (1 << 19)) != 0; // ADX
        cpu_info.features.has_sha = (ebx & (1 << 29)) != 0; // SHA
        cpu_info.features.has_mpx = (ebx & (1 << 14)) != 0; // MPX
        cpu_info.features.has_sgx = (ebx & (1 << 2)) != 0; // SGX
        cpu_info.features.has_tsx = (ebx & (1 << 11)) != 0; // TSX
        cpu_info.features.has_cet = (ecx & (1 << 7)) != 0; // CET
    }

    // Get brand string if available
    if cpuid(0x80000000).0 >= 0x80000004 {
        let mut brand = String::new();
        for i in 0x80000002..=0x80000004 {
            let (eax, ebx, ecx, edx) = cpuid(i);
            brand.push_str(&u32_to_string(eax));
            brand.push_str(&u32_to_string(ebx));
            brand.push_str(&u32_to_string(ecx));
            brand.push_str(&u32_to_string(edx));
        }
        cpu_info.brand_string = brand.trim().to_string();
    }

    Ok(())
}

/// Check if CPUID instruction is supported
fn cpuid_supported() -> bool {
    // For compilation purposes, assume CPUID is always supported
    // In a real implementation, this would use assembly to test the EFLAGS ID bit
    true
}

/// Convert u32 to string (little-endian)
fn u32_to_string(value: u32) -> String {
    let bytes = value.to_le_bytes();
    String::from_utf8_lossy(&bytes).to_string()
}

/// Get CPU features
pub fn get_cpu_features() -> CpuFeatures {
    CPU_INFO.lock().features
}

/// Get CPU information
pub fn get_cpu_info() -> CpuInfo {
    CPU_INFO.lock().clone()
}

/// CPU relax hint (pause instruction)
pub fn cpu_relax() {
    unsafe {
        core::arch::asm!("pause", options(nomem, nostack));
    }
}

/// Halt the CPU until next interrupt
pub fn halt() {
    unsafe {
        core::arch::asm!("hlt", options(nomem, nostack));
    }
}

/// Enable interrupts
pub fn enable_interrupts() {
    unsafe {
        core::arch::asm!("sti", options(nomem, nostack));
    }
}

/// Disable interrupts
pub fn disable_interrupts() {
    unsafe {
        core::arch::asm!("cli", options(nomem, nostack));
    }
}

/// Read Time Stamp Counter
pub fn read_tsc() -> u64 {
    unsafe {
        let mut low: u32;
        let mut high: u32;
        core::arch::asm!(
            "rdtsc",
            out("eax") low,
            out("edx") high,
            options(nomem, nostack)
        );
        ((high as u64) << 32) | (low as u64)
    }
}

/// Memory barrier
pub fn memory_barrier() {
    unsafe {
        core::arch::asm!("mfence", options(nomem, nostack));
    }
}

/// Load fence (serialize load operations)
pub fn load_fence() {
    unsafe {
        core::arch::asm!("lfence", options(nomem, nostack));
    }
}

/// Store fence (serialize store operations)
pub fn store_fence() {
    unsafe {
        core::arch::asm!("sfence", options(nomem, nostack));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_features_default() {
        let features = CpuFeatures::default();
        assert!(!features.has_sse); // Should be false by default
    }

    #[test]
    fn test_cpu_info_default() {
        let info = CpuInfo::default();
        assert_eq!(info.core_count, 1);
        assert_eq!(info.cache_line_size, 64);
    }

    #[test]
    fn test_u32_to_string() {
        let s = u32_to_string(0x41424344); // "DCBA" in little-endian
        assert_eq!(s, "DCBA");
    }
}
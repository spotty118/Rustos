//! Kernel Status and System Information Module for RustOS
//!
//! This module provides:
//! - System status reporting and diagnostics
//! - Performance metrics collection and display
//! - Hardware information summary
//! - AI system status monitoring
//! - Subsystem health checks
//! - Boot progress tracking

use alloc::vec::Vec;
use alloc::string::String;
use core::fmt;
use spin::Mutex;
use lazy_static::lazy_static;

/// Overall system status enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemStatus {
    Initializing,
    Running,
    Warning,
    Critical,
    Shutdown,
}

impl fmt::Display for SystemStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SystemStatus::Initializing => write!(f, "Initializing"),
            SystemStatus::Running => write!(f, "Running"),
            SystemStatus::Warning => write!(f, "Warning"),
            SystemStatus::Critical => write!(f, "Critical"),
            SystemStatus::Shutdown => write!(f, "Shutdown"),
        }
    }
}

/// Individual subsystem status
#[derive(Debug, Clone)]
pub struct SubsystemStatus {
    pub name: String,
    pub status: SystemStatus,
    pub description: String,
    pub last_update: u64,
    pub error_count: u32,
}

impl SubsystemStatus {
    pub fn new(name: &str, status: SystemStatus, description: &str) -> Self {
        Self {
            name: name.to_string(),
            status,
            description: description.to_string(),
            last_update: crate::time::uptime_ms(),
            error_count: 0,
        }
    }

    pub fn update_status(&mut self, status: SystemStatus, description: &str) {
        self.status = status;
        self.description = description.to_string();
        self.last_update = crate::time::uptime_ms();

        if matches!(status, SystemStatus::Warning | SystemStatus::Critical) {
            self.error_count += 1;
        }
    }
}

/// System performance metrics
#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub uptime_ms: u64,
    pub cpu_cycles: u64,
    pub memory_usage_kb: u64,
    pub task_count: u32,
    pub interrupt_count: u64,
    pub syscall_count: u64,
    pub file_handles: u32,
    pub ai_inference_count: u32,
}

/// Hardware information structure
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub architecture: String,
    pub cpu_features: String,
    pub gpu_available: bool,
    pub gpu_vendor: String,
    pub memory_total_kb: u64,
    pub storage_devices: u32,
    pub network_interfaces: u32,
    pub peripheral_count: u32,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        Self {
            architecture: "Unknown".to_string(),
            cpu_features: "None".to_string(),
            gpu_available: false,
            gpu_vendor: "None".to_string(),
            memory_total_kb: 0,
            storage_devices: 0,
            network_interfaces: 0,
            peripheral_count: 0,
        }
    }
}

/// Main kernel status manager
pub struct KernelStatus {
    pub system_status: SystemStatus,
    pub subsystems: Vec<SubsystemStatus>,
    pub boot_time: u64,
    pub metrics: SystemMetrics,
    pub hardware: HardwareInfo,
    pub version: String,
}

impl KernelStatus {
    pub fn new() -> Self {
        let boot_time = crate::time::uptime_ms();

        Self {
            system_status: SystemStatus::Initializing,
            subsystems: Vec::new(),
            boot_time,
            metrics: SystemMetrics::default(),
            hardware: HardwareInfo::default(),
            version: "0.1.0".to_string(),
        }
    }

    /// Initialize kernel status tracking
    pub fn init(&mut self) {
        crate::println!("[STATUS] Kernel status tracking initialized");

        // Register core subsystems
        self.register_subsystem("Core", SystemStatus::Running, "Kernel core initialized");
        self.register_subsystem("Memory", SystemStatus::Running, "Memory management active");
        self.register_subsystem("Interrupts", SystemStatus::Running, "Interrupt handling enabled");
        self.register_subsystem("Timer", SystemStatus::Running, "System timer operational");
        self.register_subsystem("VGA", SystemStatus::Running, "VGA buffer initialized");
        self.register_subsystem("Serial", SystemStatus::Running, "Serial communication ready");

        // Initialize hardware detection
        self.detect_hardware();

        self.system_status = SystemStatus::Running;
    }

    /// Register a new subsystem for monitoring
    pub fn register_subsystem(&mut self, name: &str, status: SystemStatus, description: &str) {
        let subsystem = SubsystemStatus::new(name, status, description);
        self.subsystems.push(subsystem);
    }

    /// Update status of an existing subsystem
    pub fn update_subsystem(&mut self, name: &str, status: SystemStatus, description: &str) {
        if let Some(subsystem) = self.subsystems.iter_mut().find(|s| s.name == name) {
            subsystem.update_status(status, description);
        }
    }

    /// Collect current system metrics
    pub fn update_metrics(&mut self) {
        self.metrics.uptime_ms = crate::time::uptime_ms();
        self.metrics.cpu_cycles = crate::arch::get_cpu_cycles();

        // Get task statistics
        let task_stats = crate::task::get_executor_stats();
        self.metrics.task_count = crate::task::EXECUTOR.lock().pending_tasks() as u32;

        // Get timer statistics
        let timer_stats = crate::time::get_timer_stats();
        self.metrics.interrupt_count = timer_stats.total_ticks;

        // Get syscall statistics
        let syscall_stats = crate::syscall::get_syscall_stats();
        self.metrics.syscall_count = syscall_stats.total_calls;

        // Get AI statistics if available
        if let crate::ai::AIStatus::Ready = crate::ai::get_ai_status() {
            // AI system is operational
            self.metrics.ai_inference_count += 1;
        }

        // Estimate memory usage (simplified)
        self.metrics.memory_usage_kb = (crate::allocator::HEAP_SIZE / 1024) as u64;
    }

    /// Detect available hardware
    fn detect_hardware(&mut self) {
        self.hardware.architecture = match cfg!(target_arch = "x86_64") {
            true => "x86_64".to_string(),
            false => "aarch64".to_string(),
        };

        self.hardware.cpu_features = crate::arch::get_cpu_features().to_string();
        self.hardware.gpu_available = crate::gpu::is_gpu_acceleration_available();

        if self.hardware.gpu_available {
            if let Some(gpu) = crate::gpu::GPU_SYSTEM.lock().get_active_gpu() {
                self.hardware.gpu_vendor = match gpu.vendor {
                    crate::gpu::GPUVendor::Intel => "Intel".to_string(),
                    crate::gpu::GPUVendor::Nvidia => "NVIDIA".to_string(),
                    crate::gpu::GPUVendor::AMD => "AMD".to_string(),
                    crate::gpu::GPUVendor::Unknown => "Unknown".to_string(),
                };
            }
        }

        // Estimate system memory (simplified - would need proper detection)
        self.hardware.memory_total_kb = 128 * 1024; // 128 MB estimate
    }

    /// Perform health check on all subsystems
    pub fn health_check(&mut self) -> bool {
        let mut all_healthy = true;

        for subsystem in &mut self.subsystems {
            match subsystem.name.as_str() {
                "Timer" => {
                    let timer_stats = crate::time::get_timer_stats();
                    if timer_stats.total_ticks == 0 {
                        subsystem.update_status(SystemStatus::Critical, "Timer not ticking");
                        all_healthy = false;
                    } else if timer_stats.max_jitter > 10 {
                        subsystem.update_status(SystemStatus::Warning, "High timer jitter detected");
                    }
                }
                "Memory" => {
                    // Check if allocations are working
                    if self.metrics.memory_usage_kb > (self.hardware.memory_total_kb * 9 / 10) {
                        subsystem.update_status(SystemStatus::Warning, "Memory usage high");
                    }
                }
                "AI" => {
                    match crate::ai::get_ai_status() {
                        crate::ai::AIStatus::Error => {
                            subsystem.update_status(SystemStatus::Critical, "AI system error");
                            all_healthy = false;
                        }
                        crate::ai::AIStatus::Ready => {
                            subsystem.update_status(SystemStatus::Running, "AI system operational");
                        }
                        _ => {
                            subsystem.update_status(SystemStatus::Warning, "AI system busy");
                        }
                    }
                }
                _ => {} // Other subsystems assumed healthy if registered
            }
        }

        // Update overall system status
        if !all_healthy {
            self.system_status = SystemStatus::Critical;
        } else if self.subsystems.iter().any(|s| s.status == SystemStatus::Warning) {
            self.system_status = SystemStatus::Warning;
        } else {
            self.system_status = SystemStatus::Running;
        }

        all_healthy
    }

    /// Generate comprehensive status report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("RustOS Kernel Status Report v{}\n", self.version));
        report.push_str(&format!("==========================================\n\n"));

        // System overview
        report.push_str(&format!("System Status: {}\n", self.system_status));
        report.push_str(&format!("Uptime: {} ms (Boot time: {} ms)\n",
                                self.metrics.uptime_ms, self.boot_time));
        report.push_str(&format!("Architecture: {}\n\n", self.hardware.architecture));

        // Hardware information
        report.push_str("Hardware Information:\n");
        report.push_str(&format!("  CPU Features: {}\n", self.hardware.cpu_features));
        report.push_str(&format!("  GPU Available: {} ({})\n",
                                self.hardware.gpu_available, self.hardware.gpu_vendor));
        report.push_str(&format!("  Memory: {} KB\n", self.hardware.memory_total_kb));
        report.push_str(&format!("  Peripherals: {}\n\n", self.hardware.peripheral_count));

        // Performance metrics
        report.push_str("Performance Metrics:\n");
        report.push_str(&format!("  CPU Cycles: {}\n", self.metrics.cpu_cycles));
        report.push_str(&format!("  Active Tasks: {}\n", self.metrics.task_count));
        report.push_str(&format!("  Interrupts: {}\n", self.metrics.interrupt_count));
        report.push_str(&format!("  System Calls: {}\n", self.metrics.syscall_count));
        report.push_str(&format!("  AI Inferences: {}\n", self.metrics.ai_inference_count));
        report.push_str(&format!("  File Handles: {}\n\n", self.metrics.file_handles));

        // Subsystem status
        report.push_str("Subsystem Status:\n");
        for subsystem in &self.subsystems {
            let status_indicator = match subsystem.status {
                SystemStatus::Running => "✓",
                SystemStatus::Warning => "⚠",
                SystemStatus::Critical => "✗",
                SystemStatus::Initializing => "○",
                SystemStatus::Shutdown => "●",
            };

            report.push_str(&format!("  {} {}: {} (Errors: {})\n",
                                   status_indicator, subsystem.name,
                                   subsystem.description, subsystem.error_count));
        }

        report
    }

    /// Print status summary to console
    pub fn print_status_summary(&self) {
        use crate::vga_buffer::{print_colored, print_banner, Color};

        print_banner("RustOS Kernel Status", Color::LightCyan, Color::Black);

        let status_color = match self.system_status {
            SystemStatus::Running => Color::LightGreen,
            SystemStatus::Warning => Color::Yellow,
            SystemStatus::Critical => Color::LightRed,
            SystemStatus::Initializing => Color::LightBlue,
            SystemStatus::Shutdown => Color::DarkGray,
        };

        print_colored(&format!("Overall Status: {}", self.system_status), status_color, Color::Black);
        print_colored(&format!("Uptime: {} seconds", self.metrics.uptime_ms / 1000), Color::White, Color::Black);
        print_colored(&format!("Architecture: {}", self.hardware.architecture), Color::White, Color::Black);

        print_colored("", Color::White, Color::Black); // Empty line
        print_colored("Subsystem Health:", Color::LightCyan, Color::Black);

        for subsystem in &self.subsystems {
            let (symbol, color) = match subsystem.status {
                SystemStatus::Running => ("✓", Color::LightGreen),
                SystemStatus::Warning => ("⚠", Color::Yellow),
                SystemStatus::Critical => ("✗", Color::LightRed),
                SystemStatus::Initializing => ("○", Color::LightBlue),
                SystemStatus::Shutdown => ("●", Color::DarkGray),
            };

            print_colored(&format!("  {} {}: {}", symbol, subsystem.name, subsystem.description),
                         color, Color::Black);
        }

        print_colored("", Color::White, Color::Black); // Empty line
        print_colored(&format!("Performance: {} tasks, {} syscalls, {} interrupts",
                              self.metrics.task_count, self.metrics.syscall_count,
                              self.metrics.interrupt_count), Color::LightBlue, Color::Black);
    }

    /// Export status data for external monitoring
    pub fn export_metrics(&self) -> SystemMetrics {
        self.metrics.clone()
    }

    /// Get current system load average (simplified calculation)
    pub fn get_load_average(&self) -> f32 {
        // Simplified load calculation based on task count and CPU usage
        let base_load = self.metrics.task_count as f32 / 10.0;
        let interrupt_load = (self.metrics.interrupt_count as f32 / 10000.0).min(1.0);
        (base_load + interrupt_load).min(100.0)
    }

    /// Check if system is ready for user applications
    pub fn is_system_ready(&self) -> bool {
        self.system_status == SystemStatus::Running &&
        !self.subsystems.iter().any(|s| s.status == SystemStatus::Critical)
    }
}

/// Global kernel status instance
lazy_static! {
    pub static ref KERNEL_STATUS: Mutex<KernelStatus> = Mutex::new(KernelStatus::new());
}

/// Initialize kernel status tracking
pub fn init() {
    KERNEL_STATUS.lock().init();
    crate::println!("[STATUS] Kernel status system ready");
}

/// Update system metrics
pub fn update_metrics() {
    KERNEL_STATUS.lock().update_metrics();
}

/// Perform system health check
pub fn health_check() -> bool {
    KERNEL_STATUS.lock().health_check()
}

/// Print current status summary
pub fn print_summary() {
    KERNEL_STATUS.lock().print_status_summary();
}

/// Generate full status report
pub fn generate_report() -> String {
    KERNEL_STATUS.lock().generate_report()
}

/// Register a new subsystem for monitoring
pub fn register_subsystem(name: &str, status: SystemStatus, description: &str) {
    KERNEL_STATUS.lock().register_subsystem(name, status, description);
}

/// Update subsystem status
pub fn update_subsystem(name: &str, status: SystemStatus, description: &str) {
    KERNEL_STATUS.lock().update_subsystem(name, status, description);
}

/// Get current system load
pub fn get_load_average() -> f32 {
    KERNEL_STATUS.lock().get_load_average()
}

/// Check if system is ready
pub fn is_system_ready() -> bool {
    KERNEL_STATUS.lock().is_system_ready()
}

/// Export current metrics
pub fn export_metrics() -> SystemMetrics {
    KERNEL_STATUS.lock().export_metrics()
}

/// Periodic status update task
pub async fn status_monitor_task() {
    loop {
        update_metrics();
        health_check();

        // Sleep for 1 second
        crate::time::sleep_ms(1000).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_kernel_status_creation() {
        let status = KernelStatus::new();
        assert_eq!(status.system_status, SystemStatus::Initializing);
        assert_eq!(status.version, "0.1.0");
    }

    #[test_case]
    fn test_subsystem_registration() {
        let mut status = KernelStatus::new();
        status.register_subsystem("Test", SystemStatus::Running, "Test subsystem");
        assert_eq!(status.subsystems.len(), 1);
        assert_eq!(status.subsystems[0].name, "Test");
    }

    #[test_case]
    fn test_system_status_display() {
        assert_eq!(format!("{}", SystemStatus::Running), "Running");
        assert_eq!(format!("{}", SystemStatus::Critical), "Critical");
    }

    #[test_case]
    fn test_load_calculation() {
        let mut status = KernelStatus::new();
        status.metrics.task_count = 50;
        status.metrics.interrupt_count = 10000;
        let load = status.get_load_average();
        assert!(load > 0.0);
        assert!(load <= 100.0);
    }
}

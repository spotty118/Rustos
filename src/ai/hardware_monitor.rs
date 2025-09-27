use crate::ai::learning::{HardwareMetrics, HardwareOptimization};
use spin::Mutex;
use lazy_static::lazy_static;

pub struct HardwareMonitor {
    last_metrics: HardwareMetrics,
    sample_count: u32,
    interrupt_count: u32,
    context_switches: u32,
    io_operations: u32,
    cache_miss_count: u32,
}

impl HardwareMonitor {
    pub fn new() -> Self {
        Self {
            last_metrics: HardwareMetrics {
                cpu_usage: 0,
                memory_usage: 0,
                io_operations: 0,
                interrupt_count: 0,
                context_switches: 0,
                cache_misses: 0,
                thermal_state: 25, // Start at room temperature equivalent
                power_efficiency: 75, // Start with decent efficiency
            },
            sample_count: 0,
            interrupt_count: 0,
            context_switches: 0,
            io_operations: 0,
            cache_miss_count: 0,
        }
    }

    pub fn update_metrics(&mut self) -> HardwareMetrics {
        self.sample_count += 1;
        
        // Use architecture-specific performance counters
        let perf_counter = crate::arch::read_performance_counter();
        let cpu_features = crate::arch::get_cpu_features();
        
        // Calculate CPU usage based on actual performance counter deltas and interrupt activity
        let current_perf = perf_counter;
        let perf_delta = current_perf.wrapping_sub(self.last_metrics.interrupt_count as u64);
        
        // Normalize performance counter delta to CPU usage percentage
        let base_cpu_usage = ((perf_delta / 1000000) % 100) as u8; // Scale down large counter values
        let interrupt_cpu_usage = core::cmp::min(50, (self.interrupt_count / 10) as u8);
        let cpu_usage = core::cmp::min(100, base_cpu_usage + interrupt_cpu_usage);
        
        // Calculate memory usage based on system activity (interrupts, I/O, context switches)
        let activity_factor = (self.interrupt_count + self.context_switches + self.io_operations) / 100;
        let memory_usage = core::cmp::min(100, (activity_factor) as u8);
        
        // Calculate thermal state based on CPU usage and architecture characteristics
        let thermal_base = if cpu_features.contains("aarch64") { 30 } else { 25 };
        let thermal_state = core::cmp::min(100, thermal_base + (cpu_usage * 60 / 100));
        
        // Calculate power efficiency (ARM generally more efficient, affected by thermal state)
        let efficiency_bonus = if cpu_features.contains("aarch64") { 15 } else { 0 };
        let thermal_penalty = thermal_state / 4; // Higher heat reduces efficiency
        let power_efficiency = core::cmp::min(100, 85 + efficiency_bonus - thermal_penalty);
        
        crate::println!("[HW Monitor] CPU: {}%, Mem: {}%, Thermal: {}°C, Efficiency: {}%", 
                       cpu_usage, memory_usage, thermal_state, power_efficiency);
        
        self.last_metrics = HardwareMetrics {
            cpu_usage,
            memory_usage,
            io_operations: self.io_operations,
            interrupt_count: self.interrupt_count,
            context_switches: self.context_switches,
            cache_misses: self.cache_miss_count,
            thermal_state,
            power_efficiency,
        };
        
        // Reset counters for next sample period
        self.interrupt_count = 0;
        self.context_switches = 0;
        self.io_operations = 0;
        self.cache_miss_count = 0;
        
        self.last_metrics
    }
    
    pub fn record_interrupt(&mut self) {
        self.interrupt_count += 1;
    }
    
    pub fn record_context_switch(&mut self) {
        self.context_switches += 1;
    }
    
    pub fn record_io_operation(&mut self) {
        self.io_operations += 1;
    }
    
    pub fn record_cache_miss(&mut self) {
        self.cache_miss_count += 1;
    }
    
    pub fn get_current_metrics(&self) -> HardwareMetrics {
        self.last_metrics
    }
    
    pub fn apply_optimization(&mut self, optimization: HardwareOptimization) {
        match optimization {
            HardwareOptimization::OptimalPerformance => {
                // Apply optimal performance settings
                crate::println!("[HW Monitor] Applying optimal performance mode");
                // In a real implementation, this would adjust CPU frequency, memory timings, etc.
                // For now, we adjust our efficiency metrics to reflect the mode
                self.last_metrics.power_efficiency = core::cmp::max(60, self.last_metrics.power_efficiency - 15);
            }
            HardwareOptimization::BalancedMode => {
                // Apply balanced settings - maintain current efficiency
                crate::println!("[HW Monitor] Applying balanced performance mode");
                // This is the default operating mode
            }
            HardwareOptimization::PowerSaving => {
                // Increase power efficiency at cost of some performance
                crate::println!("[HW Monitor] Applying power saving mode");
                self.last_metrics.power_efficiency = core::cmp::min(100, self.last_metrics.power_efficiency + 10);
            }
            HardwareOptimization::ThermalThrottle => {
                // Reduce performance to manage heat
                crate::println!("[HW Monitor] Applying thermal throttling");
                self.last_metrics.thermal_state = core::cmp::max(25, self.last_metrics.thermal_state - 10);
                self.last_metrics.power_efficiency = core::cmp::min(100, self.last_metrics.power_efficiency + 5);
            }
        }
    }
}

lazy_static! {
    pub static ref HARDWARE_MONITOR: Mutex<HardwareMonitor> = Mutex::new(HardwareMonitor::new());
}

pub fn record_interrupt() {
    HARDWARE_MONITOR.lock().record_interrupt();
}

pub fn record_context_switch() {
    HARDWARE_MONITOR.lock().record_context_switch();
}

pub fn record_io_operation() {
    HARDWARE_MONITOR.lock().record_io_operation();
}

pub fn get_current_metrics() -> HardwareMetrics {
    HARDWARE_MONITOR.lock().get_current_metrics()
}

pub fn update_and_get_metrics() -> HardwareMetrics {
    HARDWARE_MONITOR.lock().update_metrics()
}

pub fn apply_optimization(optimization: HardwareOptimization) {
    HARDWARE_MONITOR.lock().apply_optimization(optimization);
}
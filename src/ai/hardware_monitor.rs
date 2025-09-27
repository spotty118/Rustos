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
        
        // Simulate CPU usage based on interrupt activity and performance counter delta
        let cpu_usage = core::cmp::min(100, (self.interrupt_count / 100) as u8);
        
        // Simulate memory usage (gradually increasing with activity)
        let memory_usage = core::cmp::min(100, (self.sample_count / 10) as u8);
        
        // Calculate thermal state based on CPU usage and architecture
        let thermal_base = if cpu_features.contains("aarch64") { 30 } else { 25 };
        let thermal_state = thermal_base + (cpu_usage * 75 / 100);
        
        // Calculate power efficiency (better on ARM typically)
        let efficiency_bonus = if cpu_features.contains("aarch64") { 10 } else { 0 };
        let power_efficiency = core::cmp::min(100, 100 - (thermal_state * 100 / 125) + efficiency_bonus);
        
        crate::println!("[HW Monitor] Arch: {}, Perf Counter: {}", cpu_features, perf_counter);
        
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
                // Boost performance settings (simulated)
                crate::println!("[HW Monitor] Applying optimal performance mode");
            }
            HardwareOptimization::BalancedMode => {
                // Apply balanced settings
                crate::println!("[HW Monitor] Applying balanced performance mode");
            }
            HardwareOptimization::PowerSaving => {
                // Reduce power consumption
                crate::println!("[HW Monitor] Applying power saving mode");
            }
            HardwareOptimization::ThermalThrottle => {
                // Reduce performance to manage heat
                crate::println!("[HW Monitor] Applying thermal throttling");
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
//! Advanced Performance Monitoring and Real-time Optimization System
//!
//! This module provides comprehensive performance monitoring, profiling, and
//! real-time optimization capabilities for the RustOS kernel. It integrates
//! with the AI system to provide intelligent performance tuning.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of performance metrics to track
const MAX_METRICS: usize = 128;
/// Maximum number of optimization strategies
const MAX_STRATEGIES: usize = 32;
/// Performance sampling interval in milliseconds
const SAMPLING_INTERVAL_MS: u64 = 100;

/// Performance metric categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricCategory {
    CPU,
    Memory,
    IO,
    Network,
    GPU,
    Thermal,
    Power,
    Scheduler,
    Interrupt,
    Cache,
    Storage,
    AI,
}

impl fmt::Display for MetricCategory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MetricCategory::CPU => write!(f, "CPU"),
            MetricCategory::Memory => write!(f, "Memory"),
            MetricCategory::IO => write!(f, "I/O"),
            MetricCategory::Network => write!(f, "Network"),
            MetricCategory::GPU => write!(f, "GPU"),
            MetricCategory::Thermal => write!(f, "Thermal"),
            MetricCategory::Power => write!(f, "Power"),
            MetricCategory::Scheduler => write!(f, "Scheduler"),
            MetricCategory::Interrupt => write!(f, "Interrupt"),
            MetricCategory::Cache => write!(f, "Cache"),
            MetricCategory::Storage => write!(f, "Storage"),
            MetricCategory::AI => write!(f, "AI"),
        }
    }
}

/// Performance optimization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationStrategy {
    /// Increase CPU performance at cost of power
    AggressivePerformance,
    /// Balance performance and power consumption
    Balanced,
    /// Prioritize power savings
    PowerEfficient,
    /// Thermal throttling to prevent overheating
    ThermalProtection,
    /// Optimize for low latency
    LowLatency,
    /// Optimize for high throughput
    HighThroughput,
    /// AI-driven adaptive optimization
    AIAdaptive,
}

impl fmt::Display for OptimizationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OptimizationStrategy::AggressivePerformance => write!(f, "Aggressive Performance"),
            OptimizationStrategy::Balanced => write!(f, "Balanced"),
            OptimizationStrategy::PowerEfficient => write!(f, "Power Efficient"),
            OptimizationStrategy::ThermalProtection => write!(f, "Thermal Protection"),
            OptimizationStrategy::LowLatency => write!(f, "Low Latency"),
            OptimizationStrategy::HighThroughput => write!(f, "High Throughput"),
            OptimizationStrategy::AIAdaptive => write!(f, "AI Adaptive"),
        }
    }
}

/// Performance metric data point
#[derive(Debug, Clone, Copy)]
pub struct MetricSample {
    pub timestamp_ms: u64,
    pub value: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub average: f32,
    pub variance: f32,
    pub trend: MetricTrend,
}

impl MetricSample {
    pub fn new(timestamp: u64, value: f32) -> Self {
        Self {
            timestamp_ms: timestamp,
            value,
            min_value: value,
            max_value: value,
            average: value,
            variance: 0.0,
            trend: MetricTrend::Stable,
        }
    }
}

/// Trend analysis for metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Critical,
}

/// Performance bottleneck detection
#[derive(Debug, Clone, Copy)]
pub struct PerformanceBottleneck {
    pub category: MetricCategory,
    pub severity: BottleneckSeverity,
    pub impact_score: f32,
    pub suggested_strategy: OptimizationStrategy,
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for BottleneckSeverity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BottleneckSeverity::Low => write!(f, "Low"),
            BottleneckSeverity::Medium => write!(f, "Medium"),
            BottleneckSeverity::High => write!(f, "High"),
            BottleneckSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// Real-time performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub cpu_utilization: f32,
    pub memory_usage_percent: f32,
    pub io_throughput_mbps: f32,
    pub network_utilization: f32,
    pub gpu_utilization: f32,
    pub thermal_state: f32,
    pub power_consumption_watts: f32,
    pub scheduler_efficiency: f32,
    pub interrupt_rate: u32,
    pub cache_hit_rate: f32,
    pub ai_inference_rate: f32,
    pub system_responsiveness: f32,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_usage_percent: 0.0,
            io_throughput_mbps: 0.0,
            network_utilization: 0.0,
            gpu_utilization: 0.0,
            thermal_state: 20.0, // 20°C baseline
            power_consumption_watts: 10.0,
            scheduler_efficiency: 95.0,
            interrupt_rate: 0,
            cache_hit_rate: 90.0,
            ai_inference_rate: 0.0,
            system_responsiveness: 100.0,
        }
    }
}

/// Performance monitoring and optimization system
pub struct PerformanceMonitor {
    metrics: FnvIndexMap<MetricCategory, Vec<MetricSample, 64>, MAX_METRICS>,
    current_stats: PerformanceStats,
    active_strategy: OptimizationStrategy,
    bottlenecks: Vec<PerformanceBottleneck, 16>,
    optimization_history: Vec<(u64, OptimizationStrategy), 32>,
    monitoring_enabled: bool,
    ai_optimization_enabled: bool,
    thermal_throttling_active: bool,
    performance_baseline: PerformanceStats,
    last_optimization_timestamp: u64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        let mut monitor = Self {
            metrics: FnvIndexMap::new(),
            current_stats: PerformanceStats::default(),
            active_strategy: OptimizationStrategy::Balanced,
            bottlenecks: Vec::new(),
            optimization_history: Vec::new(),
            monitoring_enabled: false,
            ai_optimization_enabled: true,
            thermal_throttling_active: false,
            performance_baseline: PerformanceStats::default(),
            last_optimization_timestamp: 0,
        };

        // Initialize metric categories
        let categories = [
            MetricCategory::CPU,
            MetricCategory::Memory,
            MetricCategory::IO,
            MetricCategory::Network,
            MetricCategory::GPU,
            MetricCategory::Thermal,
            MetricCategory::Power,
            MetricCategory::Scheduler,
            MetricCategory::Interrupt,
            MetricCategory::Cache,
            MetricCategory::Storage,
            MetricCategory::AI,
        ];

        for category in &categories {
            let _ = monitor.metrics.insert(*category, Vec::new());
        }

        monitor
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        crate::println!("[PERF] Initializing performance monitoring system...");

        // Establish performance baseline
        self.establish_baseline()?;

        // Enable monitoring
        self.monitoring_enabled = true;

        crate::println!("[PERF] Performance monitoring system initialized");
        crate::println!("[PERF] Active strategy: {}", self.active_strategy);
        crate::println!("[PERF] AI optimization: {}", if self.ai_optimization_enabled { "Enabled" } else { "Disabled" });

        Ok(())
    }

    fn establish_baseline(&mut self) -> Result<(), &'static str> {
        crate::println!("[PERF] Establishing performance baseline...");

        // Collect initial metrics
        self.collect_system_metrics()?;
        self.performance_baseline = self.current_stats.clone();

        crate::println!("[PERF] Baseline established - CPU: {:.1}%, Memory: {:.1}%, Thermal: {:.1}°C",
                       self.performance_baseline.cpu_utilization,
                       self.performance_baseline.memory_usage_percent,
                       self.performance_baseline.thermal_state);

        Ok(())
    }

    pub fn collect_system_metrics(&mut self) -> Result<(), &'static str> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        let timestamp = crate::time::get_current_timestamp_ms();

        // Collect CPU metrics
        let cpu_usage = self.measure_cpu_utilization();
        self.record_metric(MetricCategory::CPU, timestamp, cpu_usage)?;
        self.current_stats.cpu_utilization = cpu_usage;

        // Collect memory metrics
        let memory_usage = self.measure_memory_usage();
        self.record_metric(MetricCategory::Memory, timestamp, memory_usage)?;
        self.current_stats.memory_usage_percent = memory_usage;

        // Collect thermal metrics
        let thermal_state = self.measure_thermal_state();
        self.record_metric(MetricCategory::Thermal, timestamp, thermal_state)?;
        self.current_stats.thermal_state = thermal_state;

        // Collect GPU metrics if available
        if crate::gpu::is_gpu_acceleration_available() {
            let gpu_usage = self.measure_gpu_utilization();
            self.record_metric(MetricCategory::GPU, timestamp, gpu_usage)?;
            self.current_stats.gpu_utilization = gpu_usage;
        }

        // Collect I/O metrics
        let io_throughput = self.measure_io_throughput();
        self.record_metric(MetricCategory::IO, timestamp, io_throughput)?;
        self.current_stats.io_throughput_mbps = io_throughput;

        // Collect scheduler metrics
        let scheduler_efficiency = self.measure_scheduler_efficiency();
        self.record_metric(MetricCategory::Scheduler, timestamp, scheduler_efficiency)?;
        self.current_stats.scheduler_efficiency = scheduler_efficiency;

        // Collect cache metrics
        let cache_hit_rate = self.measure_cache_performance();
        self.record_metric(MetricCategory::Cache, timestamp, cache_hit_rate)?;
        self.current_stats.cache_hit_rate = cache_hit_rate;

        // Update system responsiveness score
        self.current_stats.system_responsiveness = self.calculate_responsiveness_score();

        Ok(())
    }

    fn record_metric(&mut self, category: MetricCategory, timestamp: u64, value: f32) -> Result<(), &'static str> {
        if let Some(samples) = self.metrics.get_mut(&category) {
            let sample = MetricSample::new(timestamp, value);

            if samples.is_full() {
                // Remove oldest sample to make room
                samples.remove(0);
            }

            let _ = samples.push(sample);
            Ok(())
        } else {
            Err("Metric category not initialized")
        }
    }

    pub fn analyze_performance(&mut self) -> Result<(), &'static str> {
        self.detect_bottlenecks()?;
        self.analyze_trends()?;

        if self.ai_optimization_enabled {
            self.ai_driven_optimization()?;
        }

        self.apply_thermal_protection()?;

        Ok(())
    }

    fn detect_bottlenecks(&mut self) -> Result<(), &'static str> {
        self.bottlenecks.clear();

        // CPU bottleneck detection
        if self.current_stats.cpu_utilization > 90.0 {
            let bottleneck = PerformanceBottleneck {
                category: MetricCategory::CPU,
                severity: if self.current_stats.cpu_utilization > 95.0 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::High
                },
                impact_score: self.current_stats.cpu_utilization / 100.0,
                suggested_strategy: OptimizationStrategy::HighThroughput,
                description: "High CPU utilization detected",
            };
            let _ = self.bottlenecks.push(bottleneck);
        }

        // Memory bottleneck detection
        if self.current_stats.memory_usage_percent > 85.0 {
            let bottleneck = PerformanceBottleneck {
                category: MetricCategory::Memory,
                severity: if self.current_stats.memory_usage_percent > 95.0 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::High
                },
                impact_score: self.current_stats.memory_usage_percent / 100.0,
                suggested_strategy: OptimizationStrategy::PowerEfficient,
                description: "High memory usage detected",
            };
            let _ = self.bottlenecks.push(bottleneck);
        }

        // Thermal bottleneck detection
        if self.current_stats.thermal_state > 80.0 {
            let bottleneck = PerformanceBottleneck {
                category: MetricCategory::Thermal,
                severity: if self.current_stats.thermal_state > 90.0 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::High
                },
                impact_score: (self.current_stats.thermal_state - 20.0) / 80.0,
                suggested_strategy: OptimizationStrategy::ThermalProtection,
                description: "High thermal state detected",
            };
            let _ = self.bottlenecks.push(bottleneck);
        }

        // Cache performance bottleneck
        if self.current_stats.cache_hit_rate < 80.0 {
            let bottleneck = PerformanceBottleneck {
                category: MetricCategory::Cache,
                severity: BottleneckSeverity::Medium,
                impact_score: (100.0 - self.current_stats.cache_hit_rate) / 100.0,
                suggested_strategy: OptimizationStrategy::LowLatency,
                description: "Low cache hit rate detected",
            };
            let _ = self.bottlenecks.push(bottleneck);
        }

        if !self.bottlenecks.is_empty() {
            crate::println!("[PERF] Detected {} performance bottleneck(s)", self.bottlenecks.len());
            for bottleneck in &self.bottlenecks {
                crate::println!("[PERF] {} bottleneck: {} (severity: {}, impact: {:.2})",
                               bottleneck.category,
                               bottleneck.description,
                               bottleneck.severity,
                               bottleneck.impact_score);
            }
        }

        Ok(())
    }

    fn analyze_trends(&self) -> Result<(), &'static str> {
        // Analyze CPU usage trend
        if let Some(cpu_samples) = self.metrics.get(&MetricCategory::CPU) {
            if cpu_samples.len() >= 5 {
                let recent_avg = cpu_samples.iter()
                    .rev()
                    .take(5)
                    .map(|s| s.value)
                    .sum::<f32>() / 5.0;

                let older_avg = cpu_samples.iter()
                    .rev()
                    .skip(5)
                    .take(5)
                    .map(|s| s.value)
                    .sum::<f32>() / 5.0;

                let trend_change = (recent_avg - older_avg) / older_avg * 100.0;

                if trend_change.abs() > 10.0 {
                    crate::println!("[PERF] CPU usage trend: {:.1}% change", trend_change);
                }
            }
        }

        Ok(())
    }

    fn ai_driven_optimization(&mut self) -> Result<(), &'static str> {
        // Convert current stats to AI input format for hardware optimization
        let hardware_metrics = crate::ai::learning::HardwareMetrics {
            cpu_usage: self.current_stats.cpu_utilization as u32,
            memory_usage: self.current_stats.memory_usage_percent as u32,
            io_operations: (self.current_stats.io_throughput_mbps * 10.0) as u32,
            interrupt_count: self.current_stats.interrupt_rate,
            context_switches: 100, // Placeholder
            cache_misses: ((100.0 - self.current_stats.cache_hit_rate) * 10.0) as u32,
            thermal_state: self.current_stats.thermal_state as u32,
            power_efficiency: 80, // Placeholder
            gpu_usage: self.current_stats.gpu_utilization as u32,
            gpu_memory_usage: 50, // Placeholder
            gpu_temperature: (self.current_stats.thermal_state + 5.0) as u32,
        };

        // Send metrics to AI system for analysis
        crate::ai::process_hardware_metrics(hardware_metrics);

        Ok(())
    }

    fn apply_thermal_protection(&mut self) -> Result<(), &'static str> {
        let should_throttle = self.current_stats.thermal_state > 85.0;

        if should_throttle && !self.thermal_throttling_active {
            crate::println!("[PERF] Activating thermal protection (temp: {:.1}°C)", self.current_stats.thermal_state);
            self.set_optimization_strategy(OptimizationStrategy::ThermalProtection)?;
            self.thermal_throttling_active = true;
        } else if !should_throttle && self.thermal_throttling_active {
            crate::println!("[PERF] Deactivating thermal protection (temp: {:.1}°C)", self.current_stats.thermal_state);
            self.set_optimization_strategy(OptimizationStrategy::Balanced)?;
            self.thermal_throttling_active = false;
        }

        Ok(())
    }

    pub fn set_optimization_strategy(&mut self, strategy: OptimizationStrategy) -> Result<(), &'static str> {
        let timestamp = crate::time::get_current_timestamp_ms();

        if self.active_strategy != strategy {
            crate::println!("[PERF] Switching optimization strategy: {} -> {}",
                           self.active_strategy, strategy);

            self.active_strategy = strategy;
            self.last_optimization_timestamp = timestamp;

            // Record strategy change in history
            if self.optimization_history.is_full() {
                self.optimization_history.remove(0);
            }
            let _ = self.optimization_history.push((timestamp, strategy));

            // Apply strategy-specific optimizations
            self.apply_optimization_strategy(strategy)?;
        }

        Ok(())
    }

    fn apply_optimization_strategy(&self, strategy: OptimizationStrategy) -> Result<(), &'static str> {
        match strategy {
            OptimizationStrategy::AggressivePerformance => {
                crate::println!("[PERF] Applying aggressive performance optimizations");
                // Increase CPU frequency, disable power saving features
            },
            OptimizationStrategy::Balanced => {
                crate::println!("[PERF] Applying balanced performance/power optimizations");
                // Set moderate CPU frequency, enable selective power saving
            },
            OptimizationStrategy::PowerEfficient => {
                crate::println!("[PERF] Applying power-efficient optimizations");
                // Reduce CPU frequency, enable aggressive power saving
            },
            OptimizationStrategy::ThermalProtection => {
                crate::println!("[PERF] Applying thermal protection measures");
                // Reduce CPU frequency, increase fan speeds, throttle non-essential tasks
            },
            OptimizationStrategy::LowLatency => {
                crate::println!("[PERF] Applying low-latency optimizations");
                // Disable CPU idle states, optimize scheduler quantum
            },
            OptimizationStrategy::HighThroughput => {
                crate::println!("[PERF] Applying high-throughput optimizations");
                // Maximize CPU utilization, optimize I/O batch processing
            },
            OptimizationStrategy::AIAdaptive => {
                crate::println!("[PERF] Applying AI-adaptive optimizations");
                // Let AI system determine optimal settings
            },
        }

        Ok(())
    }

    pub fn get_current_stats(&self) -> &PerformanceStats {
        &self.current_stats
    }

    pub fn get_active_strategy(&self) -> OptimizationStrategy {
        self.active_strategy
    }

    pub fn get_bottlenecks(&self) -> &[PerformanceBottleneck] {
        &self.bottlenecks
    }

    pub fn generate_performance_report(&self) -> Result<(), &'static str> {
        crate::println!("=== Performance Monitor Report ===");
        crate::println!("Active Strategy: {}", self.active_strategy);
        crate::println!("Monitoring Enabled: {}", self.monitoring_enabled);
        crate::println!("AI Optimization: {}", if self.ai_optimization_enabled { "Enabled" } else { "Disabled" });
        crate::println!("Thermal Throttling: {}", if self.thermal_throttling_active { "Active" } else { "Inactive" });
        crate::println!();

        crate::println!("Current Performance Metrics:");
        crate::println!("  CPU Utilization: {:.1}%", self.current_stats.cpu_utilization);
        crate::println!("  Memory Usage: {:.1}%", self.current_stats.memory_usage_percent);
        crate::println!("  I/O Throughput: {:.1} MB/s", self.current_stats.io_throughput_mbps);
        crate::println!("  GPU Utilization: {:.1}%", self.current_stats.gpu_utilization);
        crate::println!("  Thermal State: {:.1}°C", self.current_stats.thermal_state);
        crate::println!("  Cache Hit Rate: {:.1}%", self.current_stats.cache_hit_rate);
        crate::println!("  System Responsiveness: {:.1}%", self.current_stats.system_responsiveness);
        crate::println!();

        if !self.bottlenecks.is_empty() {
            crate::println!("Active Bottlenecks:");
            for bottleneck in &self.bottlenecks {
                crate::println!("  {}: {} ({}) - Suggested: {}",
                               bottleneck.category,
                               bottleneck.description,
                               bottleneck.severity,
                               bottleneck.suggested_strategy);
            }
            crate::println!();
        }

        crate::println!("Performance vs Baseline:");
        let cpu_delta = self.current_stats.cpu_utilization - self.performance_baseline.cpu_utilization;
        let mem_delta = self.current_stats.memory_usage_percent - self.performance_baseline.memory_usage_percent;
        let thermal_delta = self.current_stats.thermal_state - self.performance_baseline.thermal_state;

        crate::println!("  CPU Usage: {:.1}% ({:+.1}%)", self.current_stats.cpu_utilization, cpu_delta);
        crate::println!("  Memory Usage: {:.1}% ({:+.1}%)", self.current_stats.memory_usage_percent, mem_delta);
        crate::println!("  Thermal State: {:.1}°C ({:+.1}°C)", self.current_stats.thermal_state, thermal_delta);

        Ok(())
    }

    // Hardware measurement methods (simplified for demonstration)
    fn measure_cpu_utilization(&self) -> f32 {
        // Simulate CPU utilization measurement
        use crate::time::get_current_timestamp_ms;
        let time = get_current_timestamp_ms();
        ((time % 100) as f32) * 0.8 + 10.0 // 10-90% range
    }

    fn measure_memory_usage(&self) -> f32 {
        // Simulate memory usage measurement
        45.0 + ((crate::time::get_current_timestamp_ms() % 50) as f32) * 0.8
    }

    fn measure_thermal_state(&self) -> f32 {
        // Simulate thermal measurement
        35.0 + ((crate::time::get_current_timestamp_ms() % 30) as f32)
    }

    fn measure_gpu_utilization(&self) -> f32 {
        // Simulate GPU utilization measurement
        20.0 + ((crate::time::get_current_timestamp_ms() % 60) as f32) * 0.6
    }

    fn measure_io_throughput(&self) -> f32 {
        // Simulate I/O throughput measurement
        150.0 + ((crate::time::get_current_timestamp_ms() % 100) as f32) * 2.0
    }

    fn measure_scheduler_efficiency(&self) -> f32 {
        // Simulate scheduler efficiency measurement
        92.0 + ((crate::time::get_current_timestamp_ms() % 8) as f32)
    }

    fn measure_cache_performance(&self) -> f32 {
        // Simulate cache hit rate measurement
        85.0 + ((crate::time::get_current_timestamp_ms() % 15) as f32) * 0.8
    }

    fn calculate_responsiveness_score(&self) -> f32 {
        // Calculate overall system responsiveness based on various metrics
        let cpu_factor = (100.0 - self.current_stats.cpu_utilization.min(100.0)) / 100.0;
        let memory_factor = (100.0 - self.current_stats.memory_usage_percent.min(100.0)) / 100.0;
        let cache_factor = self.current_stats.cache_hit_rate / 100.0;
        let scheduler_factor = self.current_stats.scheduler_efficiency / 100.0;

        (cpu_factor + memory_factor + cache_factor + scheduler_factor) * 25.0
    }
}

lazy_static! {
    static ref PERFORMANCE_MONITOR: Mutex<PerformanceMonitor> = Mutex::new(PerformanceMonitor::new());
}

/// Initialize the performance monitoring system
pub fn init_performance_monitor() -> Result<(), &'static str> {
    let mut monitor = PERFORMANCE_MONITOR.lock();
    monitor.initialize()
}

/// Collect system performance metrics
pub fn collect_metrics() -> Result<(), &'static str> {
    let mut monitor = PERFORMANCE_MONITOR.lock();
    monitor.collect_system_metrics()
}

/// Perform performance analysis and optimization
pub fn analyze_and_optimize() -> Result<(), &'static str> {
    let mut monitor = PERFORMANCE_MONITOR.lock();
    monitor.analyze_performance()
}

/// Set optimization strategy
pub fn set_strategy(strategy: OptimizationStrategy) -> Result<(), &'static str> {
    let mut monitor = PERFORMANCE_MONITOR.lock();
    monitor.set_optimization_strategy(strategy)
}

/// Get current performance statistics
pub fn get_performance_stats() -> PerformanceStats {
    let monitor = PERFORMANCE_MONITOR.lock();
    monitor.get_current_stats().clone()
}

/// Generate and display performance report
pub fn generate_report() -> Result<(), &'static str> {
    let monitor = PERFORMANCE_MONITOR.lock();
    monitor.generate_performance_report()
}

/// Performance monitoring task (to be called periodically)
pub fn performance_monitor_task() {
    if collect_metrics().is_ok() {
        let _ = analyze_and_optimize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        assert_eq!(monitor.active_strategy, OptimizationStrategy::Balanced);
        assert!(!monitor.monitoring_enabled);
    }

    #[test_case]
    fn test_metric_recording() {
        let mut monitor = PerformanceMonitor::new();
        let result = monitor.record_metric(MetricCategory::CPU, 1000, 50.0);
        assert!(result.is_ok());
    }

    #[test_case]
    fn test_bottleneck_detection() {
        let mut monitor = PerformanceMonitor::new();
        monitor.current_stats.cpu_utilization = 95.0;
        monitor.current_stats.thermal_state = 85.0;

        let result = monitor.detect_bottlenecks();
        assert!(result.is_ok());
        assert!(!monitor.bottlenecks.is_empty());
    }
}

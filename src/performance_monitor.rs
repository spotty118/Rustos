//! Performance monitoring and adaptive optimization for RustOS.
//!
//! The goal of this module is not to model real hardware performance but to
//! provide a realistic-feeling interface that higher level kernel components
//! can interact with during tests and demonstrations.  The monitor keeps a
//! rolling history of metrics, performs simple statistical analysis, and
//! derives optimization suggestions that the UI code can render.

use alloc::vec::Vec as AllocVec;
use alloc::collections::BTreeMap;
use core::cmp::Ordering;
use core::fmt;
use heapless::{Vec as HeaplessVec};
use lazy_static::lazy_static;
use spin::Mutex;

/// Maximum number of metric categories that can be tracked simultaneously.
const MAX_TRACKED_METRICS: usize = 16;
/// Maximum history length for each tracked metric.
const MAX_METRIC_HISTORY: usize = 64;
/// Maximum number of simultaneously reported bottlenecks.
const MAX_BOTTLENECKS: usize = 8;

type MetricHistory = HeaplessVec<MetricSample, MAX_METRIC_HISTORY>;

lazy_static! {
    static ref PERFORMANCE_MONITOR: Mutex<PerformanceMonitor> =
        Mutex::new(PerformanceMonitor::new());
}

/// Obtain a reference to the global performance monitor.
pub fn monitor() -> &'static Mutex<PerformanceMonitor> {
    &PERFORMANCE_MONITOR
}

/// Categories of metrics tracked by the performance monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Classification for the trend of a metric sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Critical,
}

/// A sampled value for a metric together with aggregated statistics.
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
    pub fn new(timestamp_ms: u64, value: f32) -> Self {
        Self {
            timestamp_ms,
            value,
            min_value: value,
            max_value: value,
            average: value,
            variance: 0.0,
            trend: MetricTrend::Stable,
        }
    }
}

/// Severity levels assigned to detected performance bottlenecks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for BottleneckSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BottleneckSeverity::Low => write!(f, "Low"),
            BottleneckSeverity::Medium => write!(f, "Medium"),
            BottleneckSeverity::High => write!(f, "High"),
            BottleneckSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// Optimization strategies recommended by the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStrategy {
    AggressivePerformance,
    Balanced,
    PowerEfficient,
    ThermalProtection,
    LowLatency,
    HighThroughput,
    AIAdaptive,
}

impl fmt::Display for OptimizationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Description of a detected performance bottleneck.
#[derive(Debug, Clone, Copy)]
pub struct PerformanceBottleneck {
    pub category: MetricCategory,
    pub severity: BottleneckSeverity,
    pub impact_score: f32,
    pub suggested_strategy: OptimizationStrategy,
    pub description: &'static str,
}

impl PerformanceBottleneck {
    pub const fn new(
        category: MetricCategory,
        severity: BottleneckSeverity,
        impact_score: f32,
        suggested_strategy: OptimizationStrategy,
        description: &'static str,
    ) -> Self {
        Self {
            category,
            severity,
            impact_score,
            suggested_strategy,
            description,
        }
    }
}

/// Aggregate performance statistics used by higher level systems.
#[derive(Debug, Clone, Copy)]
pub struct PerformanceStats {
    pub cpu_utilization: f32,
    pub memory_usage_percent: f32,
    pub io_throughput_mbps: f32,
    pub network_utilization: f32,
    pub gpu_utilization: f32,
    pub thermal_state: f32,
    pub power_consumption_watts: f32,
    pub ai_inference_rate: f32,
    pub scheduler_efficiency: f32,
    pub interrupt_rate: f32,
    pub cache_hit_rate: f32,
    pub storage_latency_ms: f32,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            memory_usage_percent: 0.0,
            io_throughput_mbps: 0.0,
            network_utilization: 0.0,
            gpu_utilization: 0.0,
            thermal_state: 28.0,
            power_consumption_watts: 60.0,
            ai_inference_rate: 0.0,
            scheduler_efficiency: 98.0,
            interrupt_rate: 120.0,
            cache_hit_rate: 95.0,
            storage_latency_ms: 4.2,
        }
    }
}

/// Internal state of the performance monitor.
pub struct PerformanceMonitor {
    metrics: BTreeMap<MetricCategory, MetricHistory>,
    current_stats: PerformanceStats,
    active_strategy: OptimizationStrategy,
    bottlenecks: HeaplessVec<PerformanceBottleneck, MAX_BOTTLENECKS>,
    monitoring_enabled: bool,
    ai_optimization_enabled: bool,
    last_timestamp: u64,
}

impl PerformanceMonitor {
    /// Create a new monitor with default configuration.
    pub fn new() -> Self {
        Self {
            metrics: BTreeMap::new(),
            current_stats: PerformanceStats::default(),
            active_strategy: OptimizationStrategy::Balanced,
            bottlenecks: HeaplessVec::new(),
            monitoring_enabled: true,
            ai_optimization_enabled: true,
            last_timestamp: 0,
        }
    }

    /// Reset the monitor to a clean state.
    pub fn reset(&mut self) {
        self.metrics.clear();
        self.bottlenecks.clear();
        self.current_stats = PerformanceStats::default();
        self.active_strategy = OptimizationStrategy::Balanced;
        self.monitoring_enabled = true;
        self.ai_optimization_enabled = true;
        self.last_timestamp = 0;
    }

    /// Enable or disable metric collection.
    pub fn set_monitoring_enabled(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
    }

    /// Toggle AI guided optimization features.
    pub fn set_ai_optimization(&mut self, enabled: bool) {
        self.ai_optimization_enabled = enabled;
        if !enabled {
            self.active_strategy = OptimizationStrategy::Balanced;
        }
    }

    /// Record a new sample for the given metric category.
    pub fn record_sample(&mut self, category: MetricCategory, timestamp_ms: u64, value: f32) {
        if !self.monitoring_enabled {
            return;
        }

        let mut sample = MetricSample::new(timestamp_ms, value);

        if let Some(history) = self.metrics.get_mut(&category) {
            if history.is_full() {
                history.remove(0);
            }
            Self::populate_sample_statistics(history, &mut sample);
            let _ = history.push(sample);
        } else {
            let mut history: MetricHistory = HeaplessVec::new();
            let _ = history.push(sample);
            let _ = self.metrics.insert(category, history);
        }

        self.update_stats_with_sample(category, value);
        self.last_timestamp = timestamp_ms;
        self.recalculate_bottlenecks();
        self.update_active_strategy();
    }

    /// Replace the aggregated performance statistics with a new snapshot.
    pub fn update_overview(&mut self, stats: PerformanceStats) {
        self.current_stats = stats;
        self.recalculate_bottlenecks();
        self.update_active_strategy();
    }

    /// Retrieve the current aggregated statistics.
    pub fn current_stats(&self) -> PerformanceStats {
        self.current_stats
    }

    /// Return a copy of all detected bottlenecks.
    pub fn bottlenecks(&self) -> AllocVec<PerformanceBottleneck> {
        self.bottlenecks.iter().copied().collect()
    }

    /// Retrieve the currently active optimization strategy.
    pub fn active_strategy(&self) -> OptimizationStrategy {
        self.active_strategy
    }

    /// Compute an overall health score in the range [0.0, 1.0].
    pub fn overall_health_score(&self) -> f32 {
        let stats = &self.current_stats;

        let cpu_penalty = clamp01(stats.cpu_utilization / 100.0) * 0.25;
        let mem_penalty = clamp01(stats.memory_usage_percent / 100.0) * 0.2;
        let gpu_penalty = clamp01(stats.gpu_utilization / 100.0) * 0.15;
        let thermal_penalty = clamp01((stats.thermal_state - 30.0) / 70.0) * 0.2;
        let power_penalty = clamp01(stats.power_consumption_watts / 320.0) * 0.1;
        let bottleneck_penalty = (self.bottlenecks.len() as f32) * 0.03;

        let mut score = 1.0
            - cpu_penalty
            - mem_penalty
            - gpu_penalty
            - thermal_penalty
            - power_penalty
            - bottleneck_penalty;

        if score < 0.0 {
            score = 0.0;
        } else if score > 1.0 {
            score = 1.0;
        }

        score
    }

    /// Return the stored samples for a given metric category.
    pub fn metric_history(&self, category: MetricCategory) -> AllocVec<MetricSample> {
        self.metrics
            .get(&category)
            .map(|history| history.iter().copied().collect())
            .unwrap_or_else(AllocVec::new)
    }

    fn populate_sample_statistics(history: &MetricHistory, sample: &mut MetricSample) {
        let mut min_value = sample.value;
        let mut max_value = sample.value;
        let mut sum = sample.value;
        let mut count = 1.0f32;

        for entry in history.iter() {
            min_value = min_value.min(entry.value);
            max_value = max_value.max(entry.value);
            sum += entry.value;
            count += 1.0;
        }

        let average = sum / count;
        sample.min_value = min_value;
        sample.max_value = max_value;
        sample.average = average;

        let mut variance = 0.0;
        for entry in history.iter() {
            let diff = entry.value - average;
            variance += diff * diff;
        }
        let diff_new = sample.value - average;
        variance += diff_new * diff_new;
        variance /= count;
        sample.variance = variance;

        if let Some(previous) = history.last() {
            let delta = sample.value - previous.value;
            if sample.value >= (average * 1.2).max(95.0) {
                sample.trend = MetricTrend::Critical;
            } else if variance > 50.0 {
                sample.trend = MetricTrend::Volatile;
            } else if delta > 2.0 {
                sample.trend = MetricTrend::Increasing;
            } else if delta < -2.0 {
                sample.trend = MetricTrend::Decreasing;
            } else {
                sample.trend = MetricTrend::Stable;
            }
        }
    }

    fn update_stats_with_sample(&mut self, category: MetricCategory, value: f32) {
        match category {
            MetricCategory::CPU => self.current_stats.cpu_utilization = value,
            MetricCategory::Memory => self.current_stats.memory_usage_percent = value,
            MetricCategory::IO => self.current_stats.io_throughput_mbps = value,
            MetricCategory::Network => self.current_stats.network_utilization = value,
            MetricCategory::GPU => self.current_stats.gpu_utilization = value,
            MetricCategory::Thermal => self.current_stats.thermal_state = value,
            MetricCategory::Power => self.current_stats.power_consumption_watts = value,
            MetricCategory::Scheduler => self.current_stats.scheduler_efficiency = value,
            MetricCategory::Interrupt => self.current_stats.interrupt_rate = value,
            MetricCategory::Cache => self.current_stats.cache_hit_rate = value,
            MetricCategory::Storage => self.current_stats.storage_latency_ms = value,
            MetricCategory::AI => self.current_stats.ai_inference_rate = value,
        }
    }

    fn recalculate_bottlenecks(&mut self) {
        self.bottlenecks.clear();
        let stats = self.current_stats;

        self.maybe_push_bottleneck(
            MetricCategory::CPU,
            stats.cpu_utilization,
            85.0,
            95.0,
            OptimizationStrategy::HighThroughput,
            "CPU utilization is elevated",
        );

        self.maybe_push_bottleneck(
            MetricCategory::Memory,
            stats.memory_usage_percent,
            80.0,
            92.0,
            OptimizationStrategy::LowLatency,
            "Memory pressure detected",
        );

        self.maybe_push_bottleneck(
            MetricCategory::GPU,
            stats.gpu_utilization,
            85.0,
            95.0,
            OptimizationStrategy::AggressivePerformance,
            "GPU workloads near saturation",
        );

        self.maybe_push_bottleneck(
            MetricCategory::Thermal,
            stats.thermal_state,
            78.0,
            88.0,
            OptimizationStrategy::ThermalProtection,
            "Thermal limits approaching",
        );

        self.maybe_push_bottleneck(
            MetricCategory::Power,
            stats.power_consumption_watts,
            220.0,
            300.0,
            OptimizationStrategy::PowerEfficient,
            "High power consumption detected",
        );

        if stats.cache_hit_rate < 85.0 {
            let severity = if stats.cache_hit_rate < 70.0 {
                BottleneckSeverity::High
            } else {
                BottleneckSeverity::Medium
            };
            let _ = self.bottlenecks.push(PerformanceBottleneck::new(
                MetricCategory::Cache,
                severity,
                1.0 - stats.cache_hit_rate / 100.0,
                OptimizationStrategy::AIAdaptive,
                "Cache hit rate dropping",
            ));
        }
    }

    fn maybe_push_bottleneck(
        &mut self,
        category: MetricCategory,
        value: f32,
        warning_threshold: f32,
        critical_threshold: f32,
        suggested_strategy: OptimizationStrategy,
        description: &'static str,
    ) {
        let severity = if value >= critical_threshold {
            BottleneckSeverity::Critical
        } else if value >= warning_threshold + 7.0 {
            BottleneckSeverity::High
        } else if value >= warning_threshold {
            BottleneckSeverity::Medium
        } else {
            return;
        };

        let impact = (value / critical_threshold).min(1.0);
        let _ = self.bottlenecks.push(PerformanceBottleneck::new(
            category,
            severity,
            impact,
            suggested_strategy,
            description,
        ));
    }

    fn update_active_strategy(&mut self) {
        if !self.ai_optimization_enabled {
            return;
        }

        if let Some(bottleneck) = self
            .bottlenecks
            .iter()
            .max_by(|a, b| a.severity.cmp(&b.severity))
        {
            self.active_strategy = match bottleneck.category {
                MetricCategory::Thermal => OptimizationStrategy::ThermalProtection,
                MetricCategory::Power => OptimizationStrategy::PowerEfficient,
                MetricCategory::Memory => OptimizationStrategy::LowLatency,
                MetricCategory::GPU | MetricCategory::Network | MetricCategory::CPU => {
                    OptimizationStrategy::HighThroughput
                }
                MetricCategory::AI => OptimizationStrategy::AIAdaptive,
                _ => OptimizationStrategy::Balanced,
            };
        } else if self.current_stats.cpu_utilization > 60.0
            && self.current_stats.gpu_utilization > 60.0
        {
            self.active_strategy = OptimizationStrategy::AggressivePerformance;
        } else {
            self.active_strategy = OptimizationStrategy::Balanced;
        }
    }
}

fn clamp01(value: f32) -> f32 {
    if value < 0.0 {
        0.0
    } else if value > 1.0 {
        1.0
    } else {
        value
    }
}


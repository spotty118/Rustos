//! Advanced Real-time Observability System
//!
//! This module implements comprehensive system observability, tracing, and
//! real-time metrics collection for the RustOS kernel. It provides detailed
//! insights into system behavior, performance bottlenecks, and operational
//! health with distributed tracing capabilities.

use core::fmt;
use heapless::{Vec, FnvIndexMap, String};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of trace spans to track
const MAX_TRACE_SPANS: usize = 256;
/// Maximum number of metrics
const MAX_METRICS: usize = 128;
/// Maximum number of log entries
const MAX_LOG_ENTRIES: usize = 512;
/// Trace sampling rate (0.0 to 1.0)
const TRACE_SAMPLING_RATE: f32 = 0.1;
/// Metrics aggregation window in milliseconds
const METRICS_WINDOW_MS: u64 = 5000;

/// Observability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObservabilityLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
}

impl fmt::Display for ObservabilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ObservabilityLevel::Trace => write!(f, "TRACE"),
            ObservabilityLevel::Debug => write!(f, "DEBUG"),
            ObservabilityLevel::Info => write!(f, "INFO"),
            ObservabilityLevel::Warn => write!(f, "WARN"),
            ObservabilityLevel::Error => write!(f, "ERROR"),
            ObservabilityLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// System components for observability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SystemComponent {
    KernelCore,
    MemoryManager,
    ProcessScheduler,
    FileSystem,
    NetworkStack,
    GPUSystem,
    AISystem,
    SecurityMonitor,
    PerformanceMonitor,
    RecoverySystem,
    PredictiveHealth,
    Observability,
}

impl fmt::Display for SystemComponent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SystemComponent::KernelCore => write!(f, "Kernel"),
            SystemComponent::MemoryManager => write!(f, "Memory"),
            SystemComponent::ProcessScheduler => write!(f, "Scheduler"),
            SystemComponent::FileSystem => write!(f, "FileSystem"),
            SystemComponent::NetworkStack => write!(f, "Network"),
            SystemComponent::GPUSystem => write!(f, "GPU"),
            SystemComponent::AISystem => write!(f, "AI"),
            SystemComponent::SecurityMonitor => write!(f, "Security"),
            SystemComponent::PerformanceMonitor => write!(f, "Performance"),
            SystemComponent::RecoverySystem => write!(f, "Recovery"),
            SystemComponent::PredictiveHealth => write!(f, "Health"),
            SystemComponent::Observability => write!(f, "Observability"),
        }
    }
}

/// Metric types for different measurement patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricType {
    Counter,     // Monotonically increasing
    Gauge,       // Current value that can go up/down
    Histogram,   // Distribution of values
    Timer,       // Duration measurements
    Rate,        // Events per time unit
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MetricType::Counter => write!(f, "Counter"),
            MetricType::Gauge => write!(f, "Gauge"),
            MetricType::Histogram => write!(f, "Histogram"),
            MetricType::Timer => write!(f, "Timer"),
            MetricType::Rate => write!(f, "Rate"),
        }
    }
}

/// Trace span for distributed tracing
#[derive(Debug, Clone)]
pub struct TraceSpan {
    pub span_id: u64,
    pub parent_id: Option<u64>,
    pub trace_id: u64,
    pub operation_name: String<32>,
    pub component: SystemComponent,
    pub start_time_ms: u64,
    pub end_time_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    pub status: SpanStatus,
    pub tags: Vec<(String<16>, String<32>), 8>,
    pub events: Vec<SpanEvent, 16>,
}

/// Span status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpanStatus {
    Active,
    Success,
    Error,
    Cancelled,
    Timeout,
}

impl fmt::Display for SpanStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SpanStatus::Active => write!(f, "Active"),
            SpanStatus::Success => write!(f, "Success"),
            SpanStatus::Error => write!(f, "Error"),
            SpanStatus::Cancelled => write!(f, "Cancelled"),
            SpanStatus::Timeout => write!(f, "Timeout"),
        }
    }
}

/// Events within a trace span
#[derive(Debug, Clone)]
pub struct SpanEvent {
    pub timestamp_ms: u64,
    pub event_name: String<24>,
    pub level: ObservabilityLevel,
    pub message: String<64>,
    pub attributes: Vec<(String<16>, String<24>), 4>,
}

/// Metric data point
#[derive(Debug, Clone)]
pub struct MetricData {
    pub metric_id: u32,
    pub name: String<32>,
    pub component: SystemComponent,
    pub metric_type: MetricType,
    pub value: f64,
    pub timestamp_ms: u64,
    pub tags: Vec<(String<16>, String<24>), 4>,
    pub samples: Vec<f64, 32>, // For histograms and aggregations
}

/// Log entry for structured logging
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub log_id: u64,
    pub timestamp_ms: u64,
    pub level: ObservabilityLevel,
    pub component: SystemComponent,
    pub message: String<128>,
    pub context: Vec<(String<16>, String<32>), 8>,
    pub span_id: Option<u64>,
    pub trace_id: Option<u64>,
}

/// System performance snapshot
#[derive(Debug, Clone, Copy)]
pub struct PerformanceSnapshot {
    pub timestamp_ms: u64,
    pub cpu_utilization: f32,
    pub memory_usage_mb: u32,
    pub active_processes: u32,
    pub network_throughput_mbps: f32,
    pub disk_io_ops_per_sec: u32,
    pub gpu_utilization: f32,
    pub system_load_average: f32,
    pub context_switches_per_sec: u32,
    pub interrupts_per_sec: u32,
}

/// Observability statistics
#[derive(Debug, Clone, Copy)]
pub struct ObservabilityStats {
    pub total_spans: u32,
    pub active_spans: u32,
    pub total_metrics: u32,
    pub total_logs: u32,
    pub traces_per_second: f32,
    pub metrics_per_second: f32,
    pub logs_per_second: f32,
    pub average_span_duration_ms: u64,
    pub error_rate: f32,
    pub sampling_overhead_percent: f32,
}

/// Main observability system
pub struct ObservabilitySystem {
    trace_spans: Vec<TraceSpan, MAX_TRACE_SPANS>,
    metrics: FnvIndexMap<String<32>, MetricData, MAX_METRICS>,
    log_entries: Vec<LogEntry, MAX_LOG_ENTRIES>,
    performance_snapshots: Vec<PerformanceSnapshot, 64>,
    stats: ObservabilityStats,
    next_span_id: u64,
    next_trace_id: u64,
    next_log_id: u64,
    next_metric_id: u32,
    sampling_enabled: bool,
    collection_enabled: bool,
    last_snapshot_ms: u64,
    last_stats_update_ms: u64,
}

impl ObservabilitySystem {
    pub fn new() -> Self {
        Self {
            trace_spans: Vec::new(),
            metrics: FnvIndexMap::new(),
            log_entries: Vec::new(),
            performance_snapshots: Vec::new(),
            stats: ObservabilityStats {
                total_spans: 0,
                active_spans: 0,
                total_metrics: 0,
                total_logs: 0,
                traces_per_second: 0.0,
                metrics_per_second: 0.0,
                logs_per_second: 0.0,
                average_span_duration_ms: 0,
                error_rate: 0.0,
                sampling_overhead_percent: 0.0,
            },
            next_span_id: 1,
            next_trace_id: 1,
            next_log_id: 1,
            next_metric_id: 1,
            sampling_enabled: true,
            collection_enabled: true,
            last_snapshot_ms: 0,
            last_stats_update_ms: 0,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        crate::println!("[OBSERVABILITY] Initializing observability system...");

        // Initialize baseline metrics
        self.initialize_baseline_metrics()?;

        // Start initial trace
        let _ = self.start_span(
            "system_initialization",
            SystemComponent::KernelCore,
            None,
        );

        crate::println!("[OBSERVABILITY] Observability system initialized successfully");
        Ok(())
    }

    pub fn start_span(&mut self, operation: &str, component: SystemComponent,
                     parent_span_id: Option<u64>) -> Result<u64, &'static str> {
        if !self.collection_enabled {
            return Err("Collection disabled");
        }

        // Apply sampling
        if self.sampling_enabled {
            // Simple sampling based on span ID
            if (self.next_span_id as f32 * TRACE_SAMPLING_RATE) % 1.0 > TRACE_SAMPLING_RATE {
                return Ok(self.next_span_id); // Return ID but don't actually trace
            }
        }

        if self.trace_spans.len() >= MAX_TRACE_SPANS {
            // Remove oldest completed spans
            self.cleanup_completed_spans();
            if self.trace_spans.len() >= MAX_TRACE_SPANS {
                return Err("Trace buffer full");
            }
        }

        let span_id = self.next_span_id;
        self.next_span_id += 1;

        let trace_id = if let Some(parent_id) = parent_span_id {
            // Find parent span's trace ID
            self.trace_spans.iter()
                .find(|span| span.span_id == parent_id)
                .map(|span| span.trace_id)
                .unwrap_or_else(|| {
                    let id = self.next_trace_id;
                    self.next_trace_id += 1;
                    id
                })
        } else {
            let id = self.next_trace_id;
            self.next_trace_id += 1;
            id
        };

        let mut operation_name = String::new();
        let _ = operation_name.push_str(operation);

        let span = TraceSpan {
            span_id,
            parent_id: parent_span_id,
            trace_id,
            operation_name,
            component,
            start_time_ms: self.get_current_time_ms(),
            end_time_ms: None,
            duration_ms: None,
            status: SpanStatus::Active,
            tags: Vec::new(),
            events: Vec::new(),
        };

        let _ = self.trace_spans.push(span);
        self.stats.total_spans += 1;
        self.stats.active_spans += 1;

        Ok(span_id)
    }

    pub fn finish_span(&mut self, span_id: u64, status: SpanStatus) -> Result<(), &'static str> {
        for span in &mut self.trace_spans {
            if span.span_id == span_id && span.end_time_ms.is_none() {
                let end_time = self.get_current_time_ms();
                span.end_time_ms = Some(end_time);
                span.duration_ms = Some(end_time - span.start_time_ms);
                span.status = status;

                self.stats.active_spans = self.stats.active_spans.saturating_sub(1);

                // Update average duration
                if let Some(duration) = span.duration_ms {
                    self.stats.average_span_duration_ms =
                        (self.stats.average_span_duration_ms + duration) / 2;
                }

                return Ok(());
            }
        }

        Err("Span not found or already finished")
    }

    pub fn add_span_event(&mut self, span_id: u64, event_name: &str, level: ObservabilityLevel,
                         message: &str) -> Result<(), &'static str> {
        for span in &mut self.trace_spans {
            if span.span_id == span_id && span.end_time_ms.is_none() {
                if span.events.len() >= 16 {
                    return Err("Span events buffer full");
                }

                let mut name = String::new();
                let _ = name.push_str(event_name);
                let mut msg = String::new();
                let _ = msg.push_str(message);

                let event = SpanEvent {
                    timestamp_ms: self.get_current_time_ms(),
                    event_name: name,
                    level,
                    message: msg,
                    attributes: Vec::new(),
                };

                let _ = span.events.push(event);
                return Ok(());
            }
        }

        Err("Active span not found")
    }

    pub fn record_metric(&mut self, name: &str, component: SystemComponent,
                        metric_type: MetricType, value: f64) -> Result<(), &'static str> {
        if !self.collection_enabled {
            return Ok(());
        }

        let mut metric_name = String::new();
        let _ = metric_name.push_str(name);

        if let Some(existing_metric) = self.metrics.get_mut(&metric_name) {
            // Update existing metric
            existing_metric.value = match metric_type {
                MetricType::Counter => existing_metric.value + value,
                MetricType::Gauge => value,
                MetricType::Timer | MetricType::Histogram => {
                    // Add to samples for aggregation
                    if existing_metric.samples.len() < 32 {
                        let _ = existing_metric.samples.push(value);
                    }
                    value
                }
                MetricType::Rate => value,
            };
            existing_metric.timestamp_ms = self.get_current_time_ms();
        } else {
            // Create new metric
            if self.metrics.len() >= MAX_METRICS {
                return Err("Metrics buffer full");
            }

            let mut samples = Vec::new();
            if metric_type == MetricType::Histogram || metric_type == MetricType::Timer {
                let _ = samples.push(value);
            }

            let metric = MetricData {
                metric_id: self.next_metric_id,
                name: metric_name.clone(),
                component,
                metric_type,
                value,
                timestamp_ms: self.get_current_time_ms(),
                tags: Vec::new(),
                samples,
            };

            self.next_metric_id += 1;
            let _ = self.metrics.insert(metric_name, metric);
            self.stats.total_metrics += 1;
        }

        Ok(())
    }

    pub fn log(&mut self, level: ObservabilityLevel, component: SystemComponent,
              message: &str, span_id: Option<u64>) -> Result<(), &'static str> {
        if !self.collection_enabled {
            return Ok(());
        }

        if self.log_entries.len() >= MAX_LOG_ENTRIES {
            // Remove oldest log entries
            for _ in 0..16 {
                if !self.log_entries.is_empty() {
                    self.log_entries.remove(0);
                }
            }
        }

        let mut log_message = String::new();
        let _ = log_message.push_str(message);

        let trace_id = if let Some(sid) = span_id {
            self.trace_spans.iter()
                .find(|span| span.span_id == sid)
                .map(|span| span.trace_id)
        } else {
            None
        };

        let log_entry = LogEntry {
            log_id: self.next_log_id,
            timestamp_ms: self.get_current_time_ms(),
            level,
            component,
            message: log_message,
            context: Vec::new(),
            span_id,
            trace_id,
        };

        self.next_log_id += 1;
        let _ = self.log_entries.push(log_entry);
        self.stats.total_logs += 1;

        // Print to console for immediate visibility
        crate::println!("[{}] [{}] {}", level, component, message);

        Ok(())
    }

    pub fn take_performance_snapshot(&mut self) {
        let current_time = self.get_current_time_ms();

        if current_time - self.last_snapshot_ms < 1000 {
            return; // Don't snapshot too frequently
        }

        let perf_stats = crate::performance_monitor::get_performance_stats();

        let snapshot = PerformanceSnapshot {
            timestamp_ms: current_time,
            cpu_utilization: perf_stats.cpu_utilization,
            memory_usage_mb: (perf_stats.memory_usage_percent / 100.0 * 1024.0) as u32,
            active_processes: 10, // Simulated
            network_throughput_mbps: 50.0, // Simulated
            disk_io_ops_per_sec: 200, // Simulated
            gpu_utilization: if crate::gpu::is_gpu_acceleration_available() { 30.0 } else { 0.0 },
            system_load_average: perf_stats.cpu_utilization / 100.0,
            context_switches_per_sec: 1000, // Simulated
            interrupts_per_sec: 500, // Simulated
        };

        if self.performance_snapshots.len() >= 64 {
            self.performance_snapshots.remove(0);
        }

        let _ = self.performance_snapshots.push(snapshot);
        self.last_snapshot_ms = current_time;

        // Record as metrics
        let _ = self.record_metric("system.cpu_utilization", SystemComponent::KernelCore,
                                   MetricType::Gauge, snapshot.cpu_utilization as f64);
        let _ = self.record_metric("system.memory_usage_mb", SystemComponent::MemoryManager,
                                   MetricType::Gauge, snapshot.memory_usage_mb as f64);
        let _ = self.record_metric("system.gpu_utilization", SystemComponent::GPUSystem,
                                   MetricType::Gauge, snapshot.gpu_utilization as f64);
    }

    pub fn update_statistics(&mut self) {
        let current_time = self.get_current_time_ms();

        if current_time - self.last_stats_update_ms < 5000 {
            return; // Update every 5 seconds
        }

        let time_window = current_time - self.last_stats_update_ms;
        if time_window > 0 {
            // Calculate rates
            let recent_spans = self.trace_spans.iter()
                .filter(|span| current_time - span.start_time_ms < time_window)
                .count() as f32;

            let recent_logs = self.log_entries.iter()
                .filter(|log| current_time - log.timestamp_ms < time_window)
                .count() as f32;

            self.stats.traces_per_second = recent_spans / (time_window as f32 / 1000.0);
            self.stats.logs_per_second = recent_logs / (time_window as f32 / 1000.0);

            // Calculate error rate
            let error_spans = self.trace_spans.iter()
                .filter(|span| span.status == SpanStatus::Error)
                .count() as f32;

            self.stats.error_rate = if self.stats.total_spans > 0 {
                error_spans / (self.stats.total_spans as f32)
            } else {
                0.0
            };

            // Estimate sampling overhead (simplified)
            self.stats.sampling_overhead_percent = if self.sampling_enabled {
                TRACE_SAMPLING_RATE * 0.1 // Rough estimate: 0.1% overhead per 10% sampling
            } else {
                0.5 // Fixed overhead when not sampling
            };
        }

        self.last_stats_update_ms = current_time;
    }

    pub fn get_observability_stats(&self) -> ObservabilityStats {
        self.stats
    }

    pub fn get_recent_spans(&self, count: usize) -> Vec<&TraceSpan, 32> {
        let mut recent_spans = Vec::new();
        let mut added = 0;

        // Get most recent spans (reverse order)
        for span in self.trace_spans.iter().rev() {
            if added >= count || added >= 32 {
                break;
            }
            let _ = recent_spans.push(span);
            added += 1;
        }

        recent_spans
    }

    pub fn get_recent_logs(&self, count: usize, level: Option<ObservabilityLevel>) -> Vec<&LogEntry, 32> {
        let mut recent_logs = Vec::new();
        let mut added = 0;

        for log in self.log_entries.iter().rev() {
            if added >= count || added >= 32 {
                break;
            }

            if let Some(filter_level) = level {
                if log.level != filter_level {
                    continue;
                }
            }

            let _ = recent_logs.push(log);
            added += 1;
        }

        recent_logs
    }

    pub fn get_metric_value(&self, name: &str) -> Option<f64> {
        let mut metric_name = String::new();
        let _ = metric_name.push_str(name);
        self.metrics.get(&metric_name).map(|metric| metric.value)
    }

    pub fn enable_collection(&mut self, enabled: bool) {
        self.collection_enabled = enabled;
        crate::println!("[OBSERVABILITY] Collection {}", if enabled { "enabled" } else { "disabled" });
    }

    pub fn enable_sampling(&mut self, enabled: bool) {
        self.sampling_enabled = enabled;
        crate::println!("[OBSERVABILITY] Sampling {}", if enabled { "enabled" } else { "disabled" });
    }

    pub fn get_system_health_summary(&self) -> String<256> {
        let mut summary = String::new();

        let current_health = crate::predictive_health::get_overall_system_health();
        let security_metrics = crate::ai_security::get_security_metrics();
        let perf_stats = crate::performance_monitor::get_performance_stats();

        let _ = summary.push_str("Health: ");
        let health_percent = (current_health * 100.0) as u32;
        // Convert u32 to string manually since we can't use format! in no_std
        let _ = summary.push_str(if health_percent >= 90 { "Excellent" }
                                else if health_percent >= 70 { "Good" }
                                else if health_percent >= 50 { "Fair" }
                                else { "Poor" });

        let _ = summary.push_str(", Security: ");
        let security_percent = (security_metrics.system_security_score * 100.0) as u32;
        let _ = summary.push_str(if security_percent >= 90 { "Secure" }
                                else if security_percent >= 70 { "Protected" }
                                else { "At Risk" });

        let _ = summary.push_str(", Performance: ");
        let perf_percent = perf_stats.system_responsiveness as u32;
        let _ = summary.push_str(if perf_percent >= 90 { "Optimal" }
                                else if perf_percent >= 70 { "Good" }
                                else { "Degraded" });

        summary
    }

    fn initialize_baseline_metrics(&mut self) -> Result<(), &'static str> {
        // Initialize core system metrics
        let _ = self.record_metric("system.uptime_ms", SystemComponent::KernelCore,
                                   MetricType::Counter, 0.0);
        let _ = self.record_metric("system.cpu_utilization", SystemComponent::KernelCore,
                                   MetricType::Gauge, 0.0);
        let _ = self.record_metric("memory.usage_percent", SystemComponent::MemoryManager,
                                   MetricType::Gauge, 0.0);
        let _ = self.record_metric("security.threats_detected", SystemComponent::SecurityMonitor,
                                   MetricType::Counter, 0.0);
        let _ = self.record_metric("ai.inference_operations", SystemComponent::AISystem,
                                   MetricType::Counter, 0.0);
        let _ = self.record_metric("recovery.operations_executed", SystemComponent::RecoverySystem,
                                   MetricType::Counter, 0.0);

        Ok(())
    }

    fn cleanup_completed_spans(&mut self) {
        // Remove completed spans that are older than 30 seconds
        let current_time = self.get_current_time_ms();
        let cutoff_time = current_time.saturating_sub(30000);

        let mut i = 0;
        while i < self.trace_spans.len() {
            let span = &self.trace_spans[i];
            if span.end_time_ms.is_some() && span.start_time_ms < cutoff_time {
                self.trace_spans.remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn get_current_time_ms(&self) -> u64 {
        // In a real implementation, this would get actual system time
        // For demo purposes, we'll use a simple counter
        static mut COUNTER: u64 = 0;
        unsafe {
            COUNTER += 1;
            COUNTER * 10 // Simulate millisecond increments
        }
    }
}

lazy_static! {
    static ref OBSERVABILITY: Mutex<ObservabilitySystem> = Mutex::new(ObservabilitySystem::new());
}

pub fn init_observability_system() {
    let mut obs = OBSERVABILITY.lock();
    match obs.initialize() {
        Ok(_) => crate::println!("[OBSERVABILITY] System ready"),
        Err(e) => crate::println!("[OBSERVABILITY] Failed to initialize: {}", e),
    }
}

pub fn start_span(operation: &str, component: SystemComponent, parent_span_id: Option<u64>) -> Result<u64, &'static str> {
    OBSERVABILITY.lock().start_span(operation, component, parent_span_id)
}

pub fn finish_span(span_id: u64, status: SpanStatus) -> Result<(), &'static str> {
    OBSERVABILITY.lock().finish_span(span_id, status)
}

pub fn add_span_event(span_id: u64, event_name: &str, level: ObservabilityLevel, message: &str) -> Result<(), &'static str> {
    OBSERVABILITY.lock().add_span_event(span_id, event_name, level, message)
}

pub fn record_metric(name: &str, component: SystemComponent, metric_type: MetricType, value: f64) -> Result<(), &'static str> {
    OBSERVABILITY.lock().record_metric(name, component, metric_type, value)
}

pub fn log_event(level: ObservabilityLevel, component: SystemComponent, message: &str, span_id: Option<u64>) -> Result<(), &'static str> {
    OBSERVABILITY.lock().log(level, component, message, span_id)
}

pub fn take_performance_snapshot() {
    OBSERVABILITY.lock().take_performance_snapshot();
}

pub fn update_observability_stats() {
    OBSERVABILITY.lock().update_statistics();
}

pub fn get_observability_stats() -> ObservabilityStats {
    OBSERVABILITY.lock().get_observability_stats()
}

pub fn get_system_health_summary() -> String<256> {
    OBSERVABILITY.lock().get_system_health_summary()
}

pub fn enable_observability_collection(enabled: bool) {
    OBSERVABILITY.lock().enable_collection(enabled);
}

pub fn enable_trace_sampling(enabled: bool) {
    OBSERVABILITY.lock().enable_sampling(enabled);
}

pub fn get_recent_error_logs() -> Vec<String<128>, 16> {
    let mut error_messages = Vec::new();
    let obs = OBSERVABILITY.lock();
    let recent_logs = obs.get_recent_logs(16, Some(ObservabilityLevel::Error));

    for log in recent_logs {
        let mut msg = String::new();
        let _ = msg.push_str(&log.message);
        let _ = error_messages.push(msg);
    }

    error_messages
}

pub fn periodic_observability_task() {
    take_performance_snapshot();
    update_observability_stats();

    // Log system status periodically
    let health_summary = get_system_health_summary();
    let _ = log_event(
        ObservabilityLevel::Info,
        SystemComponent::Observability,
        "System status update",
        None
    );
}

// Convenience macros for common observability operations
#[macro_export]
macro_rules! trace_span {
    ($operation:expr, $component:expr) => {
        crate::observability::start_span($operation, $component, None)
    };
    ($operation:expr, $component:expr, $parent:expr) => {
        crate::observability::start_span($operation, $component, Some($parent))
    };
}

#[macro_export]
macro_rules! trace_event {
    ($span_id:expr, $event:expr, $level:expr, $message:expr) => {
        crate::observability::add_span_event($span_id, $event, $level, $message)
    };
}

#[macro_export]
macro_rules! record_gauge {
    ($name:expr, $component:expr, $value:expr) => {
        crate::observability::record_metric($name, $component, crate::observability::MetricType::Gauge, $value as f64)
    };
}

#[macro_export]
macro_rules! record_timer {
    ($name:expr, $component:expr, $value:expr) => {
        crate::observability::record_metric($name, $component, crate::observability::MetricType::Timer, $value as f64)
    };
}

#[test_case]
fn test_observability_system_initialization() {
    let mut obs = ObservabilitySystem::new();
    assert!(obs.initialize().is_ok());
    assert!(obs.collection_enabled);
    assert!(obs.sampling_enabled);
}

#[test_case]
fn test_span_lifecycle() {
    let mut obs = ObservabilitySystem::new();
    let _ = obs.initialize();

    let span_id = obs.start_span("test_operation", SystemComponent::KernelCore, None).unwrap();
    assert!(span_id > 0);
    assert_eq!(obs.stats.active_spans, 1);

    let _ = obs.add_span_event(span_id, "test_event", ObservabilityLevel::Info, "Test event message");
    let _ = obs.finish_span(span_id, SpanStatus::Success);

    assert_eq!(obs.stats.active_spans, 0);
    assert_eq!(obs.stats.total_spans, 1);
}

#[test_case]
fn test_metric_recording() {
    let mut obs = ObservabilitySystem::new();
    let _ = obs.initialize();

    let _ = obs.record_metric("test.counter", SystemComponent::KernelCore, MetricType::Counter, 1.0);
    let _ = obs.record_metric("test.counter", SystemComponent::KernelCore, MetricType::Counter, 2.0);

    let value = obs.get_metric_value("test.counter");
    assert_eq!(value, Some(3.0)); // Counter should accumulate

    let _ = obs.record_metric("test.gauge", SystemComponent::KernelCore, MetricType::Gauge, 42.0);
    let gauge_value = obs.get_metric_value("test.gauge");
    assert_eq!(gauge_value, Some(42.0));
}

#[test_case]
fn test_logging_system() {
    let mut obs = ObservabilitySystem::new();
    let _ = obs.initialize();

    let span_id = obs.start_span("test_span", SystemComponent::KernelCore, None).unwrap();
    let _ = obs.log(ObservabilityLevel::Info, SystemComponent::KernelCore, "Test log message", Some(span_id));

    assert_eq!(obs.stats.total_logs, 1);

    let recent_logs = obs.get_recent_logs(10, Some(ObservabilityLevel::Info));
    assert_eq!(recent_logs.len(), 1);
    assert_eq!(recent_logs[0].span_id, Some(span_id));
}

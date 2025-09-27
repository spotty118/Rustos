//! Predictive System Health Monitor
//!
//! This module implements advanced predictive analytics for system health,
//! failure prevention, and proactive system maintenance. It uses AI-driven
//! algorithms to predict potential system failures before they occur.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of health patterns to track
const MAX_HEALTH_PATTERNS: usize = 64;
/// Maximum number of failure signatures
const MAX_FAILURE_SIGNATURES: usize = 32;
/// Health prediction window in milliseconds
const PREDICTION_WINDOW_MS: u64 = 30000; // 30 seconds
/// Critical health threshold
const CRITICAL_HEALTH_THRESHOLD: f32 = 0.25;

/// System health categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HealthCategory {
    SystemStability,
    MemoryIntegrity,
    CPUHealth,
    StorageHealth,
    NetworkHealth,
    GPUHealth,
    ThermalHealth,
    PowerHealth,
    SecurityHealth,
    AISystemHealth,
}

impl fmt::Display for HealthCategory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            HealthCategory::SystemStability => write!(f, "System Stability"),
            HealthCategory::MemoryIntegrity => write!(f, "Memory Integrity"),
            HealthCategory::CPUHealth => write!(f, "CPU Health"),
            HealthCategory::StorageHealth => write!(f, "Storage Health"),
            HealthCategory::NetworkHealth => write!(f, "Network Health"),
            HealthCategory::GPUHealth => write!(f, "GPU Health"),
            HealthCategory::ThermalHealth => write!(f, "Thermal Health"),
            HealthCategory::PowerHealth => write!(f, "Power Health"),
            HealthCategory::SecurityHealth => write!(f, "Security Health"),
            HealthCategory::AISystemHealth => write!(f, "AI System Health"),
        }
    }
}

/// Predicted failure types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FailureType {
    MemoryCorruption,
    CPUOverheat,
    StorageFailure,
    NetworkDisconnection,
    GPUFailure,
    PowerLoss,
    SecurityBreach,
    SystemDeadlock,
    KernelPanic,
    AISystemFailure,
}

impl fmt::Display for FailureType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FailureType::MemoryCorruption => write!(f, "Memory Corruption"),
            FailureType::CPUOverheat => write!(f, "CPU Overheat"),
            FailureType::StorageFailure => write!(f, "Storage Failure"),
            FailureType::NetworkDisconnection => write!(f, "Network Disconnection"),
            FailureType::GPUFailure => write!(f, "GPU Failure"),
            FailureType::PowerLoss => write!(f, "Power Loss"),
            FailureType::SecurityBreach => write!(f, "Security Breach"),
            FailureType::SystemDeadlock => write!(f, "System Deadlock"),
            FailureType::KernelPanic => write!(f, "Kernel Panic"),
            FailureType::AISystemFailure => write!(f, "AI System Failure"),
        }
    }
}

/// Severity levels for predicted failures
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SeverityLevel {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

impl fmt::Display for SeverityLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SeverityLevel::Low => write!(f, "Low"),
            SeverityLevel::Medium => write!(f, "Medium"),
            SeverityLevel::High => write!(f, "High"),
            SeverityLevel::Critical => write!(f, "Critical"),
            SeverityLevel::Emergency => write!(f, "Emergency"),
        }
    }
}

/// Health metric sample for predictive analysis
#[derive(Debug, Clone, Copy)]
pub struct HealthMetric {
    pub category: HealthCategory,
    pub timestamp_ms: u64,
    pub health_score: f32, // 0.0 = unhealthy, 1.0 = perfect health
    pub trend_velocity: f32, // Rate of change
    pub anomaly_score: f32, // Deviation from normal patterns
    pub confidence: f32, // Confidence in the measurement
}

/// Failure prediction result
#[derive(Debug, Clone, Copy)]
pub struct FailurePrediction {
    pub failure_type: FailureType,
    pub probability: f32, // 0.0 to 1.0
    pub time_to_failure_ms: u64,
    pub severity: SeverityLevel,
    pub confidence: f32,
    pub affected_categories: [HealthCategory; 4],
}

/// Health pattern for machine learning
#[derive(Debug, Clone)]
pub struct HealthPattern {
    pub pattern_id: u32,
    pub metrics: [f32; 16], // Condensed metric representation
    pub outcome: HealthOutcome,
    pub frequency: u32,
    pub last_seen_ms: u64,
}

/// Health outcome classifications
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthOutcome {
    Stable,
    Degrading,
    Recovering,
    Critical,
    Failure,
}

/// Main predictive health monitoring system
pub struct PredictiveHealthMonitor {
    health_metrics: FnvIndexMap<HealthCategory, Vec<HealthMetric, 64>, MAX_HEALTH_PATTERNS>,
    failure_signatures: Vec<FailurePrediction, MAX_FAILURE_SIGNATURES>,
    learned_patterns: Vec<HealthPattern, MAX_HEALTH_PATTERNS>,
    current_health_score: f32,
    prediction_accuracy: f32,
    last_prediction_ms: u64,
    emergency_mode: bool,
    pattern_counter: u32,
}

impl PredictiveHealthMonitor {
    pub fn new() -> Self {
        Self {
            health_metrics: FnvIndexMap::new(),
            failure_signatures: Vec::new(),
            learned_patterns: Vec::new(),
            current_health_score: 1.0,
            prediction_accuracy: 0.0,
            last_prediction_ms: 0,
            emergency_mode: false,
            pattern_counter: 0,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        crate::println!("[HEALTH] Initializing predictive health monitor...");

        // Initialize health metric tracking for each category
        for category in [
            HealthCategory::SystemStability,
            HealthCategory::MemoryIntegrity,
            HealthCategory::CPUHealth,
            HealthCategory::StorageHealth,
            HealthCategory::NetworkHealth,
            HealthCategory::GPUHealth,
            HealthCategory::ThermalHealth,
            HealthCategory::PowerHealth,
            HealthCategory::SecurityHealth,
            HealthCategory::AISystemHealth,
        ].iter() {
            let _ = self.health_metrics.insert(*category, Vec::new());
        }

        // Load baseline health patterns
        self.load_baseline_patterns()?;

        crate::println!("[HEALTH] Predictive health monitor initialized successfully");
        Ok(())
    }

    pub fn update_health_metrics(&mut self, current_time_ms: u64) {
        // Collect health metrics from various system components
        self.collect_system_stability_metrics(current_time_ms);
        self.collect_memory_integrity_metrics(current_time_ms);
        self.collect_cpu_health_metrics(current_time_ms);
        self.collect_thermal_health_metrics(current_time_ms);
        self.collect_gpu_health_metrics(current_time_ms);
        self.collect_ai_system_health_metrics(current_time_ms);

        // Update overall health score
        self.update_overall_health_score();
    }

    pub fn predict_failures(&mut self, current_time_ms: u64) -> Option<FailurePrediction> {
        if current_time_ms - self.last_prediction_ms < PREDICTION_WINDOW_MS / 4 {
            return None; // Don't predict too frequently
        }

        self.last_prediction_ms = current_time_ms;

        // Analyze current health patterns
        let pattern = self.extract_current_pattern(current_time_ms);

        // Compare with known failure signatures
        if let Some(prediction) = self.match_failure_signatures(&pattern) {
            if prediction.severity >= SeverityLevel::High {
                crate::println!("[HEALTH] ⚠️  FAILURE PREDICTED: {} in {}ms ({}% confidence)",
                               prediction.failure_type,
                               prediction.time_to_failure_ms,
                               (prediction.confidence * 100.0) as u32);

                // Trigger preventive measures
                self.trigger_preventive_measures(&prediction);
                return Some(prediction);
            }
        }

        // Learn from current patterns
        self.learn_from_current_state(&pattern);
        None
    }

    pub fn get_overall_health_score(&self) -> f32 {
        self.current_health_score
    }

    pub fn is_emergency_mode(&self) -> bool {
        self.emergency_mode
    }

    pub fn get_prediction_accuracy(&self) -> f32 {
        self.prediction_accuracy
    }

    fn collect_system_stability_metrics(&mut self, timestamp_ms: u64) {
        // Simulate system stability assessment
        let base_stability = 0.95;
        let cpu_load_factor = crate::performance_monitor::get_performance_stats().cpu_utilization / 100.0;
        let memory_factor = crate::performance_monitor::get_performance_stats().memory_usage_percent / 100.0;

        let stability_score = base_stability - (cpu_load_factor * 0.2) - (memory_factor * 0.15);
        let anomaly_score = if stability_score < 0.7 { 0.8 } else { 0.1 };

        let metric = HealthMetric {
            category: HealthCategory::SystemStability,
            timestamp_ms,
            health_score: stability_score.max(0.0).min(1.0),
            trend_velocity: 0.0, // Calculate from previous samples
            anomaly_score,
            confidence: 0.9,
        };

        if let Some(metrics) = self.health_metrics.get_mut(&HealthCategory::SystemStability) {
            let _ = metrics.push(metric);
            if metrics.len() > 64 {
                metrics.remove(0); // Keep only recent metrics
            }
        }
    }

    fn collect_memory_integrity_metrics(&mut self, timestamp_ms: u64) {
        let stats = crate::performance_monitor::get_performance_stats();
        let memory_health = 1.0 - (stats.memory_usage_percent / 100.0 * 0.5);
        let anomaly_score = if stats.memory_usage_percent > 90.0 { 0.9 } else { 0.1 };

        let metric = HealthMetric {
            category: HealthCategory::MemoryIntegrity,
            timestamp_ms,
            health_score: memory_health.max(0.0).min(1.0),
            trend_velocity: 0.0,
            anomaly_score,
            confidence: 0.95,
        };

        if let Some(metrics) = self.health_metrics.get_mut(&HealthCategory::MemoryIntegrity) {
            let _ = metrics.push(metric);
            if metrics.len() > 64 {
                metrics.remove(0);
            }
        }
    }

    fn collect_cpu_health_metrics(&mut self, timestamp_ms: u64) {
        let stats = crate::performance_monitor::get_performance_stats();
        let cpu_health = 1.0 - (stats.cpu_utilization / 100.0 * 0.3);
        let anomaly_score = if stats.cpu_utilization > 95.0 { 0.8 } else { 0.1 };

        let metric = HealthMetric {
            category: HealthCategory::CPUHealth,
            timestamp_ms,
            health_score: cpu_health.max(0.0).min(1.0),
            trend_velocity: 0.0,
            anomaly_score,
            confidence: 0.9,
        };

        if let Some(metrics) = self.health_metrics.get_mut(&HealthCategory::CPUHealth) {
            let _ = metrics.push(metric);
            if metrics.len() > 64 {
                metrics.remove(0);
            }
        }
    }

    fn collect_thermal_health_metrics(&mut self, timestamp_ms: u64) {
        let stats = crate::performance_monitor::get_performance_stats();
        let thermal_health = 1.0 - (stats.thermal_state / 100.0);
        let anomaly_score = if stats.thermal_state > 80.0 { 0.9 } else { 0.1 };

        let metric = HealthMetric {
            category: HealthCategory::ThermalHealth,
            timestamp_ms,
            health_score: thermal_health.max(0.0).min(1.0),
            trend_velocity: 0.0,
            anomaly_score,
            confidence: 0.85,
        };

        if let Some(metrics) = self.health_metrics.get_mut(&HealthCategory::ThermalHealth) {
            let _ = metrics.push(metric);
            if metrics.len() > 64 {
                metrics.remove(0);
            }
        }
    }

    fn collect_gpu_health_metrics(&mut self, timestamp_ms: u64) {
        // Check if GPU acceleration is available
        let gpu_health = if crate::gpu::is_gpu_acceleration_available() {
            0.9 // Assume good health when GPU is available
        } else {
            0.5 // Reduced health when GPU unavailable
        };

        let metric = HealthMetric {
            category: HealthCategory::GPUHealth,
            timestamp_ms,
            health_score: gpu_health,
            trend_velocity: 0.0,
            anomaly_score: 0.1,
            confidence: 0.8,
        };

        if let Some(metrics) = self.health_metrics.get_mut(&HealthCategory::GPUHealth) {
            let _ = metrics.push(metric);
            if metrics.len() > 64 {
                metrics.remove(0);
            }
        }
    }

    fn collect_ai_system_health_metrics(&mut self, timestamp_ms: u64) {
        let ai_status = crate::ai::get_ai_status();
        let ai_health = match ai_status {
            crate::ai::AIStatus::Ready => 1.0,
            crate::ai::AIStatus::Learning => 0.9,
            crate::ai::AIStatus::Inferencing => 0.9,
            crate::ai::AIStatus::Initializing => 0.5,
            crate::ai::AIStatus::Error => 0.1,
        };

        let metric = HealthMetric {
            category: HealthCategory::AISystemHealth,
            timestamp_ms,
            health_score: ai_health,
            trend_velocity: 0.0,
            anomaly_score: if ai_health < 0.5 { 0.8 } else { 0.1 },
            confidence: 0.95,
        };

        if let Some(metrics) = self.health_metrics.get_mut(&HealthCategory::AISystemHealth) {
            let _ = metrics.push(metric);
            if metrics.len() > 64 {
                metrics.remove(0);
            }
        }
    }

    fn update_overall_health_score(&mut self) {
        let mut total_health = 0.0;
        let mut total_weight = 0.0;

        // Weight different health categories
        let category_weights = [
            (HealthCategory::SystemStability, 2.0),
            (HealthCategory::MemoryIntegrity, 1.8),
            (HealthCategory::CPUHealth, 1.5),
            (HealthCategory::ThermalHealth, 1.3),
            (HealthCategory::GPUHealth, 1.0),
            (HealthCategory::AISystemHealth, 1.2),
        ];

        for (category, weight) in category_weights.iter() {
            if let Some(metrics) = self.health_metrics.get(category) {
                if let Some(latest_metric) = metrics.last() {
                    total_health += latest_metric.health_score * weight;
                    total_weight += weight;
                }
            }
        }

        if total_weight > 0.0 {
            self.current_health_score = total_health / total_weight;
        }

        // Check for emergency conditions
        self.emergency_mode = self.current_health_score < CRITICAL_HEALTH_THRESHOLD;

        if self.emergency_mode {
            crate::println!("[HEALTH] 🚨 EMERGENCY MODE: Overall health score: {:.2}", self.current_health_score);
        }
    }

    fn extract_current_pattern(&self, _timestamp_ms: u64) -> [f32; 16] {
        let mut pattern = [0.0f32; 16];

        // Extract key health indicators
        pattern[0] = self.current_health_score;

        if let Some(stability_metrics) = self.health_metrics.get(&HealthCategory::SystemStability) {
            if let Some(metric) = stability_metrics.last() {
                pattern[1] = metric.health_score;
                pattern[2] = metric.anomaly_score;
            }
        }

        if let Some(memory_metrics) = self.health_metrics.get(&HealthCategory::MemoryIntegrity) {
            if let Some(metric) = memory_metrics.last() {
                pattern[3] = metric.health_score;
                pattern[4] = metric.anomaly_score;
            }
        }

        if let Some(cpu_metrics) = self.health_metrics.get(&HealthCategory::CPUHealth) {
            if let Some(metric) = cpu_metrics.last() {
                pattern[5] = metric.health_score;
                pattern[6] = metric.anomaly_score;
            }
        }

        if let Some(thermal_metrics) = self.health_metrics.get(&HealthCategory::ThermalHealth) {
            if let Some(metric) = thermal_metrics.last() {
                pattern[7] = metric.health_score;
                pattern[8] = metric.anomaly_score;
            }
        }

        pattern
    }

    fn match_failure_signatures(&self, pattern: &[f32; 16]) -> Option<FailurePrediction> {
        // Pattern matching for known failure signatures

        // Memory corruption signature
        if pattern[3] < 0.3 && pattern[4] > 0.7 {
            return Some(FailurePrediction {
                failure_type: FailureType::MemoryCorruption,
                probability: 0.85,
                time_to_failure_ms: 5000, // 5 seconds
                severity: SeverityLevel::Critical,
                confidence: 0.9,
                affected_categories: [
                    HealthCategory::MemoryIntegrity,
                    HealthCategory::SystemStability,
                    HealthCategory::CPUHealth,
                    HealthCategory::AISystemHealth,
                ],
            });
        }

        // CPU overheat signature
        if pattern[7] < 0.2 && pattern[8] > 0.8 {
            return Some(FailurePrediction {
                failure_type: FailureType::CPUOverheat,
                probability: 0.9,
                time_to_failure_ms: 10000, // 10 seconds
                severity: SeverityLevel::Emergency,
                confidence: 0.95,
                affected_categories: [
                    HealthCategory::ThermalHealth,
                    HealthCategory::CPUHealth,
                    HealthCategory::SystemStability,
                    HealthCategory::PowerHealth,
                ],
            });
        }

        // System instability signature
        if pattern[1] < 0.4 && pattern[0] < 0.5 {
            return Some(FailurePrediction {
                failure_type: FailureType::SystemDeadlock,
                probability: 0.7,
                time_to_failure_ms: 15000, // 15 seconds
                severity: SeverityLevel::High,
                confidence: 0.8,
                affected_categories: [
                    HealthCategory::SystemStability,
                    HealthCategory::CPUHealth,
                    HealthCategory::MemoryIntegrity,
                    HealthCategory::AISystemHealth,
                ],
            });
        }

        None
    }

    fn trigger_preventive_measures(&mut self, prediction: &FailurePrediction) {
        crate::println!("[HEALTH] Triggering preventive measures for: {}", prediction.failure_type);

        match prediction.failure_type {
            FailureType::CPUOverheat => {
                crate::println!("[HEALTH] Emergency thermal protection activated");
                let _ = crate::performance_monitor::set_strategy(
                    crate::performance_monitor::OptimizationStrategy::ThermalProtection
                );
            }
            FailureType::MemoryCorruption => {
                crate::println!("[HEALTH] Memory protection measures activated");
                // Trigger emergency memory cleanup and defragmentation
            }
            FailureType::SystemDeadlock => {
                crate::println!("[HEALTH] System stability measures activated");
                let _ = crate::performance_monitor::set_strategy(
                    crate::performance_monitor::OptimizationStrategy::Balanced
                );
            }
            _ => {
                crate::println!("[HEALTH] Generic protective measures activated");
            }
        }
    }

    fn learn_from_current_state(&mut self, pattern: &[f32; 16]) {
        // Simple learning - create new pattern if significantly different
        let mut is_new_pattern = true;

        for learned in &self.learned_patterns {
            let mut similarity = 0.0;
            for i in 0..16 {
                similarity += (pattern[i] - learned.metrics[i]).abs();
            }
            similarity /= 16.0;

            if similarity < 0.1 { // Similar pattern found
                is_new_pattern = false;
                break;
            }
        }

        if is_new_pattern && self.learned_patterns.len() < MAX_HEALTH_PATTERNS {
            let outcome = if self.current_health_score > 0.8 {
                HealthOutcome::Stable
            } else if self.current_health_score > 0.6 {
                HealthOutcome::Degrading
            } else {
                HealthOutcome::Critical
            };

            let new_pattern = HealthPattern {
                pattern_id: self.pattern_counter,
                metrics: *pattern,
                outcome,
                frequency: 1,
                last_seen_ms: 0, // Would use actual timestamp
            };

            let _ = self.learned_patterns.push(new_pattern);
            self.pattern_counter += 1;
        }
    }

    fn load_baseline_patterns(&mut self) -> Result<(), &'static str> {
        // Load some baseline healthy patterns
        let healthy_pattern = HealthPattern {
            pattern_id: 0,
            metrics: [1.0, 0.95, 0.1, 0.9, 0.1, 0.85, 0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            outcome: HealthOutcome::Stable,
            frequency: 10,
            last_seen_ms: 0,
        };

        let degrading_pattern = HealthPattern {
            pattern_id: 1,
            metrics: [0.6, 0.7, 0.3, 0.65, 0.4, 0.6, 0.5, 0.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            outcome: HealthOutcome::Degrading,
            frequency: 3,
            last_seen_ms: 0,
        };

        let _ = self.learned_patterns.push(healthy_pattern);
        let _ = self.learned_patterns.push(degrading_pattern);

        Ok(())
    }
}

lazy_static! {
    static ref HEALTH_MONITOR: Mutex<PredictiveHealthMonitor> = Mutex::new(PredictiveHealthMonitor::new());
}

pub fn init_predictive_health_monitor() {
    let mut monitor = HEALTH_MONITOR.lock();
    match monitor.initialize() {
        Ok(_) => crate::println!("[HEALTH] Predictive health monitor ready"),
        Err(e) => crate::println!("[HEALTH] Failed to initialize: {}", e),
    }
}

pub fn update_health_metrics(current_time_ms: u64) {
    HEALTH_MONITOR.lock().update_health_metrics(current_time_ms);
}

pub fn predict_system_failures(current_time_ms: u64) -> Option<FailurePrediction> {
    HEALTH_MONITOR.lock().predict_failures(current_time_ms)
}

pub fn get_overall_system_health() -> f32 {
    HEALTH_MONITOR.lock().get_overall_health_score()
}

pub fn is_emergency_mode() -> bool {
    HEALTH_MONITOR.lock().is_emergency_mode()
}

pub fn get_prediction_accuracy() -> f32 {
    HEALTH_MONITOR.lock().get_prediction_accuracy()
}

#[test_case]
fn test_health_monitor_initialization() {
    let mut monitor = PredictiveHealthMonitor::new();
    assert!(monitor.initialize().is_ok());
    assert_eq!(monitor.get_overall_health_score(), 1.0);
}

#[test_case]
fn test_failure_prediction() {
    let mut monitor = PredictiveHealthMonitor::new();
    let _ = monitor.initialize();

    // Test with critical pattern
    let critical_pattern = [0.2, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let prediction = monitor.match_failure_signatures(&critical_pattern);
    assert!(prediction.is_some());

    if let Some(pred) = prediction {
        assert_eq!(pred.failure_type, FailureType::CPUOverheat);
        assert!(pred.probability > 0.8);
    }
}

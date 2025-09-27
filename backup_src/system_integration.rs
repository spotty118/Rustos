//! Comprehensive System Integration and Reporting Module
//!
//! This module provides centralized coordination, monitoring, and reporting
//! for all advanced kernel systems including AI, GPU, network, storage,
//! and performance monitoring. It enables intelligent cross-system optimization
//! and provides unified system health reporting.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of system alerts to track
const MAX_SYSTEM_ALERTS: usize = 64;
/// Maximum number of optimization events in history
const MAX_OPTIMIZATION_HISTORY: usize = 128;
/// System integration analysis interval
const INTEGRATION_ANALYSIS_INTERVAL_MS: u64 = 5000;

/// System component types for integration management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SystemComponent {
    PerformanceMonitor,
    NetworkOptimizer,
    StorageAnalytics,
    GPUCompute,
    MultiGPU,
    AISystem,
    ThermalManagement,
    PowerManagement,
    MemoryManager,
    TaskScheduler,
}

impl fmt::Display for SystemComponent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SystemComponent::PerformanceMonitor => write!(f, "Performance Monitor"),
            SystemComponent::NetworkOptimizer => write!(f, "Network Optimizer"),
            SystemComponent::StorageAnalytics => write!(f, "Storage Analytics"),
            SystemComponent::GPUCompute => write!(f, "GPU Compute"),
            SystemComponent::MultiGPU => write!(f, "Multi-GPU Manager"),
            SystemComponent::AISystem => write!(f, "AI System"),
            SystemComponent::ThermalManagement => write!(f, "Thermal Management"),
            SystemComponent::PowerManagement => write!(f, "Power Management"),
            SystemComponent::MemoryManager => write!(f, "Memory Manager"),
            SystemComponent::TaskScheduler => write!(f, "Task Scheduler"),
        }
    }
}

/// System alert severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARN"),
            AlertSeverity::Critical => write!(f, "CRIT"),
            AlertSeverity::Emergency => write!(f, "EMRG"),
        }
    }
}

/// Cross-system optimization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntegrationStrategy {
    /// Optimize for maximum system performance
    MaxPerformance,
    /// Balance performance and power consumption
    Balanced,
    /// Minimize power consumption
    PowerEfficient,
    /// Prioritize system stability and reliability
    Stable,
    /// Emergency mode with minimal resource usage
    Emergency,
    /// AI-driven adaptive integration
    AIAdaptive,
    /// User-defined custom strategy
    Custom,
}

impl fmt::Display for IntegrationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntegrationStrategy::MaxPerformance => write!(f, "Max Performance"),
            IntegrationStrategy::Balanced => write!(f, "Balanced"),
            IntegrationStrategy::PowerEfficient => write!(f, "Power Efficient"),
            IntegrationStrategy::Stable => write!(f, "Stable"),
            IntegrationStrategy::Emergency => write!(f, "Emergency"),
            IntegrationStrategy::AIAdaptive => write!(f, "AI Adaptive"),
            IntegrationStrategy::Custom => write!(f, "Custom"),
        }
    }
}

/// System alert for cross-component issues
#[derive(Debug, Clone)]
pub struct SystemAlert {
    pub alert_id: u64,
    pub component: SystemComponent,
    pub severity: AlertSeverity,
    pub message: &'static str,
    pub timestamp: u64,
    pub acknowledged: bool,
    pub auto_resolved: bool,
    pub impact_score: f32,
}

impl SystemAlert {
    pub fn new(id: u64, component: SystemComponent, severity: AlertSeverity, message: &'static str) -> Self {
        Self {
            alert_id: id,
            component,
            severity,
            message,
            timestamp: crate::time::get_current_timestamp_ms(),
            acknowledged: false,
            auto_resolved: false,
            impact_score: match severity {
                AlertSeverity::Info => 0.1,
                AlertSeverity::Warning => 0.3,
                AlertSeverity::Critical => 0.7,
                AlertSeverity::Emergency => 1.0,
            },
        }
    }
}

/// Cross-system optimization event
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    pub event_id: u64,
    pub timestamp: u64,
    pub strategy_from: IntegrationStrategy,
    pub strategy_to: IntegrationStrategy,
    pub trigger_reason: &'static str,
    pub affected_components: Vec<SystemComponent, 8>,
    pub performance_impact: f32,
    pub success: bool,
}

/// Comprehensive system health metrics
#[derive(Debug, Clone)]
pub struct SystemHealth {
    // Overall health indicators
    pub overall_health_score: f32,      // 0.0 to 1.0
    pub system_stability_score: f32,    // 0.0 to 1.0
    pub performance_efficiency: f32,    // 0.0 to 1.0
    pub resource_utilization: f32,      // 0.0 to 1.0

    // Component health scores
    pub performance_health: f32,
    pub network_health: f32,
    pub storage_health: f32,
    pub gpu_health: f32,
    pub ai_health: f32,
    pub thermal_health: f32,
    pub power_health: f32,
    pub memory_health: f32,

    // System-wide metrics
    pub total_alerts: u32,
    pub critical_alerts: u32,
    pub emergency_alerts: u32,
    pub uptime_seconds: u64,
    pub optimization_events: u64,
    pub ai_optimizations_success_rate: f32,

    // Resource coordination metrics
    pub cross_system_efficiency: f32,
    pub bottleneck_detection_accuracy: f32,
    pub adaptive_response_time_ms: f32,
    pub integration_overhead_percent: f32,
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self {
            overall_health_score: 1.0,
            system_stability_score: 1.0,
            performance_efficiency: 0.85,
            resource_utilization: 0.30,

            performance_health: 1.0,
            network_health: 1.0,
            storage_health: 1.0,
            gpu_health: 1.0,
            ai_health: 1.0,
            thermal_health: 1.0,
            power_health: 1.0,
            memory_health: 1.0,

            total_alerts: 0,
            critical_alerts: 0,
            emergency_alerts: 0,
            uptime_seconds: 0,
            optimization_events: 0,
            ai_optimizations_success_rate: 0.85,

            cross_system_efficiency: 0.90,
            bottleneck_detection_accuracy: 0.88,
            adaptive_response_time_ms: 150.0,
            integration_overhead_percent: 3.5,
        }
    }
}

/// Main system integration manager
pub struct SystemIntegrationManager {
    // System state
    integration_strategy: IntegrationStrategy,
    system_health: SystemHealth,
    system_alerts: Vec<SystemAlert, MAX_SYSTEM_ALERTS>,
    optimization_history: Vec<OptimizationEvent, MAX_OPTIMIZATION_HISTORY>,

    // Component status tracking
    component_status: FnvIndexMap<SystemComponent, bool, 16>,
    component_health: FnvIndexMap<SystemComponent, f32, 16>,
    component_last_update: FnvIndexMap<SystemComponent, u64, 16>,

    // Configuration and state
    initialized: bool,
    auto_optimization_enabled: bool,
    emergency_mode_active: bool,
    last_analysis_timestamp: u64,
    system_start_time: u64,

    // Coordination thresholds
    performance_threshold: f32,
    thermal_threshold: f32,
    memory_threshold: f32,
    network_threshold: f32,
    storage_threshold: f32,

    // Counters
    next_alert_id: u64,
    next_event_id: u64,
    successful_optimizations: u64,
    total_optimizations: u64,
}

impl SystemIntegrationManager {
    pub fn new() -> Self {
        Self {
            integration_strategy: IntegrationStrategy::AIAdaptive,
            system_health: SystemHealth::default(),
            system_alerts: Vec::new(),
            optimization_history: Vec::new(),

            component_status: FnvIndexMap::new(),
            component_health: FnvIndexMap::new(),
            component_last_update: FnvIndexMap::new(),

            initialized: false,
            auto_optimization_enabled: true,
            emergency_mode_active: false,
            last_analysis_timestamp: 0,
            system_start_time: 0,

            performance_threshold: 0.7,
            thermal_threshold: 80.0,
            memory_threshold: 85.0,
            network_threshold: 0.8,
            storage_threshold: 0.8,

            next_alert_id: 1,
            next_event_id: 1,
            successful_optimizations: 0,
            total_optimizations: 0,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        if self.initialized {
            return Ok(());
        }

        crate::println!("[INTEGRATION] Initializing system integration manager...");

        self.system_start_time = crate::time::get_current_timestamp_ms();

        // Initialize component tracking
        let components = [
            SystemComponent::PerformanceMonitor,
            SystemComponent::NetworkOptimizer,
            SystemComponent::StorageAnalytics,
            SystemComponent::GPUCompute,
            SystemComponent::MultiGPU,
            SystemComponent::AISystem,
            SystemComponent::ThermalManagement,
            SystemComponent::PowerManagement,
            SystemComponent::MemoryManager,
            SystemComponent::TaskScheduler,
        ];

        for component in &components {
            self.component_status.insert(*component, false)
                .map_err(|_| "Failed to initialize component status")?;
            self.component_health.insert(*component, 1.0)
                .map_err(|_| "Failed to initialize component health")?;
            self.component_last_update.insert(*component, self.system_start_time)
                .map_err(|_| "Failed to initialize component timestamps")?;
        }

        // Set initial component status based on actual system state
        self.update_component_status(SystemComponent::PerformanceMonitor, true);
        self.update_component_status(SystemComponent::AISystem, true);

        // Check for GPU availability
        if crate::gpu::is_gpu_acceleration_available() {
            self.update_component_status(SystemComponent::GPUCompute, true);
        }

        self.initialized = true;

        crate::println!("[INTEGRATION] System integration manager initialized");
        crate::println!("[INTEGRATION] Active strategy: {}", self.integration_strategy);
        crate::println!("[INTEGRATION] Auto-optimization: {}",
                       if self.auto_optimization_enabled { "Enabled" } else { "Disabled" });

        // Generate initial system report
        self.generate_integration_report()?;

        Ok(())
    }

    pub fn update_component_status(&mut self, component: SystemComponent, active: bool) {
        let current_time = crate::time::get_current_timestamp_ms();

        if let Some(status) = self.component_status.get_mut(&component) {
            *status = active;
        }

        if let Some(last_update) = self.component_last_update.get_mut(&component) {
            *last_update = current_time;
        }

        if active {
            crate::println!("[INTEGRATION] Component activated: {}", component);
        } else {
            crate::println!("[INTEGRATION] Component deactivated: {}", component);
        }
    }

    pub fn update_component_health(&mut self, component: SystemComponent, health_score: f32) {
        let clamped_health = health_score.max(0.0).min(1.0);

        if let Some(health) = self.component_health.get_mut(&component) {
            *health = clamped_health;
        }

        // Generate alert if health is critically low
        if clamped_health < 0.3 {
            self.create_alert(component, AlertSeverity::Critical,
                             "Component health critically low");
        } else if clamped_health < 0.6 {
            self.create_alert(component, AlertSeverity::Warning,
                             "Component health degraded");
        }
    }

    pub fn create_alert(&mut self, component: SystemComponent, severity: AlertSeverity, message: &'static str) -> Result<u64, &'static str> {
        if self.system_alerts.is_full() {
            // Remove oldest alert to make room
            self.system_alerts.remove(0);
        }

        let alert_id = self.next_alert_id;
        self.next_alert_id += 1;

        let alert = SystemAlert::new(alert_id, component, severity, message);

        crate::println!("[INTEGRATION] ALERT [{}] {}: {} - {}",
                       alert.severity, alert.component, message, alert_id);

        self.system_alerts.push(alert)
            .map_err(|_| "Failed to create system alert")?;

        // Update alert counters
        self.system_health.total_alerts += 1;
        match severity {
            AlertSeverity::Critical => self.system_health.critical_alerts += 1,
            AlertSeverity::Emergency => self.system_health.emergency_alerts += 1,
            _ => {}
        }

        // Trigger emergency mode if needed
        if severity == AlertSeverity::Emergency {
            self.activate_emergency_mode()?;
        }

        Ok(alert_id)
    }

    fn activate_emergency_mode(&mut self) -> Result<(), &'static str> {
        if !self.emergency_mode_active {
            crate::println!("[INTEGRATION] ACTIVATING EMERGENCY MODE");

            self.emergency_mode_active = true;
            self.integration_strategy = IntegrationStrategy::Emergency;

            // Apply emergency optimizations across all systems
            let _ = crate::performance_monitor::set_strategy(
                crate::performance_monitor::OptimizationStrategy::ThermalProtection
            );

            let _ = crate::network_optimizer::set_network_strategy(
                crate::network_optimizer::NetworkOptimizationStrategy::PowerEfficient
            );

            let _ = crate::storage_analytics::set_storage_strategy(
                crate::storage_analytics::StorageOptimizationStrategy::PowerEfficient
            );

            if crate::gpu::is_gpu_acceleration_available() {
                crate::gpu::multi_gpu::set_multi_gpu_strategy(
                    crate::gpu::multi_gpu::LoadBalancingStrategy::ThermalAware
                );
            }

            self.record_optimization_event(
                self.integration_strategy,
                IntegrationStrategy::Emergency,
                "Emergency mode activation",
                true
            )?;
        }

        Ok(())
    }

    pub fn analyze_system_integration(&mut self) -> Result<(), &'static str> {
        if !self.initialized {
            return Ok(());
        }

        let current_time = crate::time::get_current_timestamp_ms();

        // Only analyze if enough time has passed
        if current_time - self.last_analysis_timestamp < INTEGRATION_ANALYSIS_INTERVAL_MS {
            return Ok(());
        }

        self.last_analysis_timestamp = current_time;

        // Update system health metrics
        self.update_system_health_metrics();

        // Perform cross-system analysis
        self.analyze_cross_system_performance()?;

        // Apply automatic optimizations if enabled
        if self.auto_optimization_enabled && !self.emergency_mode_active {
            self.apply_automatic_optimizations()?;
        }

        // Check for emergency conditions
        self.check_emergency_conditions()?;

        // Clean up old alerts and events
        self.cleanup_old_data();

        Ok(())
    }

    fn update_system_health_metrics(&mut self) {
        // Update uptime
        let current_time = crate::time::get_current_timestamp_ms();
        self.system_health.uptime_seconds = (current_time - self.system_start_time) / 1000;

        // Collect metrics from all active systems
        let perf_stats = crate::performance_monitor::get_performance_stats();
        let network_stats = crate::network_optimizer::get_network_stats();
        let storage_stats = crate::storage_analytics::get_storage_stats();

        // Update component health scores
        self.system_health.performance_health = perf_stats.system_responsiveness / 100.0;
        self.system_health.network_health = network_stats.network_health_score;
        self.system_health.storage_health = storage_stats.overall_health_score;
        self.system_health.thermal_health = if perf_stats.thermal_state < 80.0 {
            (80.0 - perf_stats.thermal_state) / 80.0 + 0.5
        } else {
            (100.0 - perf_stats.thermal_state) / 20.0
        }.max(0.0).min(1.0);

        self.system_health.memory_health = (100.0 - perf_stats.memory_usage_percent) / 100.0;

        // Update GPU health if available
        if crate::gpu::is_gpu_acceleration_available() {
            let multi_gpu_stats = crate::gpu::multi_gpu::get_multi_gpu_stats();
            if multi_gpu_stats.total_gpus > 0 {
                self.system_health.gpu_health = multi_gpu_stats.load_balance_score;
            }
        }

        // Calculate overall health score
        let component_count = 8.0;
        let total_health = self.system_health.performance_health +
                          self.system_health.network_health +
                          self.system_health.storage_health +
                          self.system_health.gpu_health +
                          self.system_health.ai_health +
                          self.system_health.thermal_health +
                          self.system_health.power_health +
                          self.system_health.memory_health;

        self.system_health.overall_health_score = total_health / component_count;

        // Calculate system stability score based on alert frequency
        let stability_factor = if self.system_health.total_alerts > 0 {
            1.0 - (self.system_health.critical_alerts as f32 / self.system_health.total_alerts as f32) * 0.5
        } else {
            1.0
        };

        self.system_health.system_stability_score = stability_factor;

        // Update resource utilization
        self.system_health.resource_utilization = (
            perf_stats.cpu_utilization / 100.0 * 0.4 +
            perf_stats.memory_usage_percent / 100.0 * 0.4 +
            perf_stats.gpu_utilization / 100.0 * 0.2
        );

        // Update AI optimization success rate
        if self.total_optimizations > 0 {
            self.system_health.ai_optimizations_success_rate =
                self.successful_optimizations as f32 / self.total_optimizations as f32;
        }
    }

    fn analyze_cross_system_performance(&mut self) -> Result<(), &'static str> {
        // Analyze bottlenecks across systems
        let perf_stats = crate::performance_monitor::get_performance_stats();
        let network_stats = crate::network_optimizer::get_network_stats();
        let storage_stats = crate::storage_analytics::get_storage_stats();

        // Detect cross-system bottlenecks
        if perf_stats.cpu_utilization > 90.0 && storage_stats.average_read_latency_ms > 10.0 {
            self.create_alert(SystemComponent::StorageAnalytics, AlertSeverity::Warning,
                             "High CPU usage correlating with storage latency")?;
        }

        if network_stats.network_utilization > 80.0 && perf_stats.memory_usage_percent > 85.0 {
            self.create_alert(SystemComponent::NetworkOptimizer, AlertSeverity::Warning,
                             "High network utilization with memory pressure")?;
        }

        if perf_stats.thermal_state > 85.0 && perf_stats.gpu_utilization > 80.0 {
            self.create_alert(SystemComponent::ThermalManagement, AlertSeverity::Critical,
                             "High thermal state with heavy GPU usage")?;
        }

        // Calculate cross-system efficiency
        let avg_system_health = (self.system_health.performance_health +
                               self.system_health.network_health +
                               self.system_health.storage_health) / 3.0;

        self.system_health.cross_system_efficiency = avg_system_health *
            (1.0 - self.system_health.integration_overhead_percent / 100.0);

        Ok(())
    }

    fn apply_automatic_optimizations(&mut self) -> Result<(), &'static str> {
        let current_strategy = self.integration_strategy;
        let mut new_strategy = current_strategy;

        // AI-driven strategy selection
        match self.integration_strategy {
            IntegrationStrategy::AIAdaptive => {
                if self.system_health.overall_health_score < 0.7 {
                    if self.system_health.thermal_health < 0.6 {
                        new_strategy = IntegrationStrategy::PowerEfficient;
                    } else if self.system_health.performance_health < 0.6 {
                        new_strategy = IntegrationStrategy::MaxPerformance;
                    } else {
                        new_strategy = IntegrationStrategy::Stable;
                    }
                } else if self.system_health.overall_health_score > 0.9 {
                    new_strategy = IntegrationStrategy::Balanced;
                }
            },
            _ => {
                // Check if we should return to AI adaptive mode
                if self.system_health.overall_health_score > 0.8 {
                    new_strategy = IntegrationStrategy::AIAdaptive;
                }
            }
        }

        if new_strategy != current_strategy {
            self.set_integration_strategy(new_strategy)?;
        }

        Ok(())
    }

    fn check_emergency_conditions(&mut self) -> Result<(), &'static str> {
        let perf_stats = crate::performance_monitor::get_performance_stats();

        // Check for emergency conditions
        if perf_stats.thermal_state > 95.0 {
            self.create_alert(SystemComponent::ThermalManagement, AlertSeverity::Emergency,
                             "Critical thermal state - emergency shutdown required")?;
        }

        if perf_stats.memory_usage_percent > 98.0 {
            self.create_alert(SystemComponent::MemoryManager, AlertSeverity::Emergency,
                             "Critical memory exhaustion")?;
        }

        if self.system_health.overall_health_score < 0.3 {
            self.create_alert(SystemComponent::PerformanceMonitor, AlertSeverity::Emergency,
                             "System health critically compromised")?;
        }

        Ok(())
    }

    fn cleanup_old_data(&mut self) {
        let current_time = crate::time::get_current_timestamp_ms();

        // Remove alerts older than 1 hour (3600000 ms)
        let mut indices_to_remove = Vec::<usize, 32>::new();
        for (index, alert) in self.system_alerts.iter().enumerate() {
            if current_time - alert.timestamp > 3600000 {
                if indices_to_remove.push(index).is_err() {
                    break;
                }
            }
        }

        // Remove in reverse order to maintain indices
        for &index in indices_to_remove.iter().rev() {
            if index < self.system_alerts.len() {
                self.system_alerts.remove(index);
            }
        }
    }

    pub fn set_integration_strategy(&mut self, strategy: IntegrationStrategy) -> Result<(), &'static str> {
        let old_strategy = self.integration_strategy;
        self.integration_strategy = strategy;

        crate::println!("[INTEGRATION] Strategy change: {} -> {}", old_strategy, strategy);

        // Apply strategy-specific optimizations
        self.apply_integration_strategy(strategy)?;

        // Record optimization event
        self.record_optimization_event(old_strategy, strategy, "Manual strategy change", true)?;

        Ok(())
    }

    fn apply_integration_strategy(&mut self, strategy: IntegrationStrategy) -> Result<(), &'static str> {
        match strategy {
            IntegrationStrategy::MaxPerformance => {
                let _ = crate::performance_monitor::set_strategy(
                    crate::performance_monitor::OptimizationStrategy::AggressivePerformance
                );
                let _ = crate::network_optimizer::set_network_strategy(
                    crate::network_optimizer::NetworkOptimizationStrategy::HighThroughput
                );
                let _ = crate::storage_analytics::set_storage_strategy(
                    crate::storage_analytics::StorageOptimizationStrategy::HighThroughput
                );
            },
            IntegrationStrategy::PowerEfficient => {
                let _ = crate::performance_monitor::set_strategy(
                    crate::performance_monitor::OptimizationStrategy::PowerEfficient
                );
                let _ = crate::network_optimizer::set_network_strategy(
                    crate::network_optimizer::NetworkOptimizationStrategy::PowerEfficient
                );
                let _ = crate::storage_analytics::set_storage_strategy(
                    crate::storage_analytics::StorageOptimizationStrategy::PowerEfficient
                );
            },
            IntegrationStrategy::Balanced => {
                let _ = crate::performance_monitor::set_strategy(
                    crate::performance_monitor::OptimizationStrategy::Balanced
                );
                let _ = crate::network_optimizer::set_network_strategy(
                    crate::network_optimizer::NetworkOptimizationStrategy::Balanced
                );
                let _ = crate::storage_analytics::set_storage_strategy(
                    crate::storage_analytics::StorageOptimizationStrategy::Balanced
                );
            },
            IntegrationStrategy::Emergency => {
                // Already handled in activate_emergency_mode
            },
            _ => {
                // Default to AI adaptive for other strategies
                let _ = crate::performance_monitor::set_strategy(
                    crate::performance_monitor::OptimizationStrategy::AIAdaptive
                );
            }
        }

        Ok(())
    }

    fn record_optimization_event(&mut self, from: IntegrationStrategy, to: IntegrationStrategy,
                                reason: &'static str, success: bool) -> Result<(), &'static str> {
        if self.optimization_history.is_full() {
            self.optimization_history.remove(0);
        }

        let event_id = self.next_event_id;
        self.next_event_id += 1;

        let mut affected_components = Vec::new();
        let _ = affected_components.push(SystemComponent::PerformanceMonitor);
        let _ = affected_components.push(SystemComponent::NetworkOptimizer);
        let _ = affected_components.push(SystemComponent::StorageAnalytics);

        let event = OptimizationEvent {
            event_id,
            timestamp: crate::time::get_current_timestamp_ms(),
            strategy_from: from,
            strategy_to: to,
            trigger_reason: reason,
            affected_components,
            performance_impact: 0.05, // Estimated 5% impact
            success,
        };

        self.optimization_history.push(event)
            .map_err(|_| "Failed to record optimization event")?;

        self.total_optimizations += 1;
        if success {
            self.successful_optimizations += 1;
        }

        self.system_health.optimization_events = self.total_optimizations;

        Ok(())
    }

    pub fn get_system_health(&self) -> &SystemHealth {
        &self.system_health
    }

    pub fn get_integration_strategy(&self) -> IntegrationStrategy {
        self.integration_strategy
    }

    pub fn generate_integration_report(&self) -> Result<(), &'static str> {
        crate::println!("=== System Integration Report ===");
        crate::println!("Integration Strategy: {}", self.integration_strategy);
        crate::println!("Auto-optimization: {}", if self.auto_optimization_enabled { "Enabled" } else { "Disabled" });
        crate::println!("Emergency Mode: {}", if self.emergency_mode_active { "Active" } else { "Inactive" });
        crate::println!("System Uptime: {} seconds", self.system_health.uptime_seconds);
        crate::println!();

        crate::println!("Overall System Health:");
        crate::println!("  Overall Score: {:.1}%", self.system_health.overall_health_score * 100.0);
        crate::println!("  Stability Score: {:.1}%", self.system_health.system_stability_score * 100.0);
        crate::println!("  Performance Efficiency: {:.1}%", self.system_health.performance_efficiency * 100.0);
        crate::println!("  Resource Utilization: {:.1}%", self.system_health.resource_utilization * 100.0);
        crate::println!();

        crate::println!("Component Health:");
        crate::println!("  Performance Monitor: {:.1}%", self.system_health.performance_health * 100.0);
        crate::println!("  Network Optimizer: {:.1}%", self.system_health.network_health * 100.0);
        crate::println!("  Storage Analytics: {:.1}%", self.system_health.storage_health * 100.0);
        crate::println!("  GPU System: {:.1}%", self.system_health.gpu_health * 100.0);
        crate::println!("  AI System: {:.1}%", self.system_health.ai_health * 100.0);
        crate::println!("  Thermal Management: {:.1}%", self.system_health.thermal_health * 100.0);
        crate::println!("  Memory Manager: {:.1}%", self.system_health.memory_health * 100.0);
        crate::println!();

        crate::println!("System Alerts:");
        crate::println!("  Total Alerts: {}", self.system_health.total_alerts);
        crate::println!("  Critical Alerts: {}", self.system_health.critical_alerts);
        crate::println!("  Emergency Alerts: {}", self.system_health.emergency_alerts);
        crate::println!();

        crate::println!("Integration Metrics:");
        crate::println!("  Cross-system Efficiency: {:.1}%", self.system_health.cross_system_efficiency * 100.0);
        crate::println!("  Bottleneck Detection Accuracy: {:.1}%", self.system_health.bottleneck_detection_accuracy * 100.0);
        crate::println!("  Adaptive Response Time: {:.1}ms", self.system_health.adaptive_response_time_ms);
        crate::println!("  Integration Overhead: {:.1}%", self.system_health.integration_overhead_percent);
        crate::println!();

        crate::println!("Optimization History:");
        crate::println!("  Total Optimizations: {}", self.system_health.optimization_events);
        crate::println!("  Success Rate: {:.1}%", self.system_health.ai_optimizations_success_rate * 100.0);
        crate::println!("  Recent Events: {}", self.optimization_history.len());

        if !self.system_alerts.is_empty() {
            crate::println!();
            crate::println!("Recent Alerts:");
            for alert in self.system_alerts.iter().rev().take(5) {
                crate::println!("  [{}] {}: {} ({}ms ago)",
                               alert.severity, alert.component, alert.message,
                               crate::time::get_current_timestamp_ms() - alert.timestamp);
            }
        }

        Ok(())
    }
}

lazy_static! {
    static ref SYSTEM_INTEGRATION: Mutex<SystemIntegrationManager> = Mutex::new(SystemIntegrationManager::new());
}

/// Initialize system integration manager
pub fn init_system_integration() -> Result<(), &'static str> {
    let mut integration = SYSTEM_INTEGRATION.lock();
    integration.initialize()
}

/// Update component status
pub fn update_component_status(component: SystemComponent, active: bool) {
    let mut integration = SYSTEM_INTEGRATION.lock();
    integration.update_component_status(component, active);
}

/// Update component health score
pub fn update_component_health(component: SystemComponent, health_score: f32) {
    let mut integration = SYSTEM_INTEGRATION.lock();
    integration.update_component_health(component, health_score);
}

/// Create system alert
pub fn create_system_alert(component: SystemComponent, severity: AlertSeverity, message: &'static str) -> Result<u64, &'static str> {
    let mut integration = SYSTEM_INTEGRATION.lock();
    integration.create_alert(component, severity, message)
}

/// Analyze system integration and apply optimizations
pub fn analyze_system_integration() -> Result<(), &'static str> {
    let mut integration = SYSTEM_INTEGRATION.lock();
    integration.analyze_system_integration()
}

/// Set integration strategy
pub fn set_integration_strategy(strategy: IntegrationStrategy) -> Result<(), &'static str> {
    let mut integration = SYSTEM_INTEGRATION.lock();
    integration.set_integration_strategy(strategy)
}

/// Get current system health
pub fn get_system_health() -> SystemHealth {
    let integration = SYSTEM_INTEGRATION.lock();
    integration.get_system_health().clone()
}

/// Get current integration strategy
pub fn get_integration_strategy() -> IntegrationStrategy {
    let integration = SYSTEM_INTEGRATION.lock();
    integration.get_integration_strategy()
}

/// Generate and display integration report
pub fn generate_integration_report() -> Result<(), &'static str> {
    let integration = SYSTEM_INTEGRATION.lock();
    integration.generate_integration_report()
}

/// System integration task (to be called periodically)
pub fn system_integration_task() {
    let _ = analyze_system_integration();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_system_integration_creation() {
        let manager = SystemIntegrationManager::new();
        assert!(!manager.initialized);
        assert_eq!(manager.integration_strategy, IntegrationStrategy::AIAdaptive);
    }

    #[test_case]
    fn test_alert_creation() {
        let alert = SystemAlert::new(1, SystemComponent::PerformanceMonitor, AlertSeverity::Warning, "Test alert");
        assert_eq!(alert.alert_id, 1);
        assert_eq!(alert.component, SystemComponent::PerformanceMonitor);
        assert_eq!(alert.severity, AlertSeverity::Warning);
    }

    #[test_case]
    fn test_system_health_default() {
        let health = SystemHealth::default();
        assert_eq!(health.overall_health_score, 1.0);
        assert_eq!(health.total_alerts, 0);
    }
}

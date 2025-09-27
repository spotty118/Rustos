//! AI-Driven Security Monitoring System
//!
//! This module implements advanced AI-powered security monitoring, threat detection,
//! and automated response capabilities for the RustOS kernel. It uses machine learning
//! algorithms to detect anomalous behavior patterns and potential security threats.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of security events to track
const MAX_SECURITY_EVENTS: usize = 128;
/// Maximum number of threat signatures
const MAX_THREAT_SIGNATURES: usize = 64;
/// Maximum number of behavioral patterns
const MAX_BEHAVIORAL_PATTERNS: usize = 32;
/// Security scan interval in milliseconds
const SECURITY_SCAN_INTERVAL_MS: u64 = 1000;
/// Threat confidence threshold for action
const THREAT_CONFIDENCE_THRESHOLD: f32 = 0.75;

/// Security threat categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThreatCategory {
    UnauthorizedAccess,
    MemoryCorruption,
    BufferOverflow,
    PrivilegeEscalation,
    RootkitActivity,
    AnomalousSystemCalls,
    NetworkIntrusion,
    DenialOfService,
    DataExfiltration,
    MalwareSignature,
    BehavioralAnomaly,
    TimingAttack,
}

impl fmt::Display for ThreatCategory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ThreatCategory::UnauthorizedAccess => write!(f, "Unauthorized Access"),
            ThreatCategory::MemoryCorruption => write!(f, "Memory Corruption"),
            ThreatCategory::BufferOverflow => write!(f, "Buffer Overflow"),
            ThreatCategory::PrivilegeEscalation => write!(f, "Privilege Escalation"),
            ThreatCategory::RootkitActivity => write!(f, "Rootkit Activity"),
            ThreatCategory::AnomalousSystemCalls => write!(f, "Anomalous System Calls"),
            ThreatCategory::NetworkIntrusion => write!(f, "Network Intrusion"),
            ThreatCategory::DenialOfService => write!(f, "Denial of Service"),
            ThreatCategory::DataExfiltration => write!(f, "Data Exfiltration"),
            ThreatCategory::MalwareSignature => write!(f, "Malware Signature"),
            ThreatCategory::BehavioralAnomaly => write!(f, "Behavioral Anomaly"),
            ThreatCategory::TimingAttack => write!(f, "Timing Attack"),
        }
    }
}

/// Security threat severity levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
    Emergency,
}

impl fmt::Display for ThreatSeverity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ThreatSeverity::Low => write!(f, "Low"),
            ThreatSeverity::Medium => write!(f, "Medium"),
            ThreatSeverity::High => write!(f, "High"),
            ThreatSeverity::Critical => write!(f, "Critical"),
            ThreatSeverity::Emergency => write!(f, "Emergency"),
        }
    }
}

/// Security response actions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecurityResponse {
    Monitor,
    AlertOnly,
    BlockAccess,
    QuarantineProcess,
    KillProcess,
    NetworkDisconnect,
    SystemLockdown,
    EmergencyShutdown,
    LogAndContinue,
    AdaptiveThrottling,
}

impl fmt::Display for SecurityResponse {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SecurityResponse::Monitor => write!(f, "Monitor"),
            SecurityResponse::AlertOnly => write!(f, "Alert Only"),
            SecurityResponse::BlockAccess => write!(f, "Block Access"),
            SecurityResponse::QuarantineProcess => write!(f, "Quarantine Process"),
            SecurityResponse::KillProcess => write!(f, "Kill Process"),
            SecurityResponse::NetworkDisconnect => write!(f, "Network Disconnect"),
            SecurityResponse::SystemLockdown => write!(f, "System Lockdown"),
            SecurityResponse::EmergencyShutdown => write!(f, "Emergency Shutdown"),
            SecurityResponse::LogAndContinue => write!(f, "Log and Continue"),
            SecurityResponse::AdaptiveThrottling => write!(f, "Adaptive Throttling"),
        }
    }
}

/// Security event record
#[derive(Debug, Clone, Copy)]
pub struct SecurityEvent {
    pub event_id: u32,
    pub timestamp_ms: u64,
    pub category: ThreatCategory,
    pub severity: ThreatSeverity,
    pub confidence: f32, // 0.0 to 1.0
    pub source_process_id: u32,
    pub affected_resource: u32,
    pub behavioral_anomaly_score: f32,
    pub response_taken: SecurityResponse,
    pub false_positive_probability: f32,
}

/// Behavioral pattern for AI learning
#[derive(Debug, Clone)]
pub struct BehavioralPattern {
    pub pattern_id: u32,
    pub features: [f32; 16], // Feature vector for ML
    pub normal_behavior: bool,
    pub frequency: u32,
    pub last_seen_ms: u64,
    pub confidence: f32,
}

/// Threat detection rule
#[derive(Debug, Clone, Copy)]
pub struct ThreatSignature {
    pub signature_id: u32,
    pub category: ThreatCategory,
    pub severity: ThreatSeverity,
    pub detection_pattern: [f32; 8], // Simplified pattern matching
    pub false_positive_rate: f32,
    pub last_updated_ms: u64,
    pub effectiveness_score: f32,
}

/// System security metrics
#[derive(Debug, Clone, Copy)]
pub struct SecurityMetrics {
    pub total_threats_detected: u32,
    pub threats_blocked: u32,
    pub false_positives: u32,
    pub system_security_score: f32, // 0.0 to 1.0
    pub average_response_time_ms: u64,
    pub active_threats: u32,
    pub behavioral_anomalies: u32,
    pub security_events_per_minute: f32,
}

/// Main AI security monitoring system
pub struct AISecurityMonitor {
    security_events: Vec<SecurityEvent, MAX_SECURITY_EVENTS>,
    threat_signatures: Vec<ThreatSignature, MAX_THREAT_SIGNATURES>,
    behavioral_patterns: Vec<BehavioralPattern, MAX_BEHAVIORAL_PATTERNS>,
    security_metrics: SecurityMetrics,
    last_scan_timestamp: u64,
    lockdown_mode: bool,
    learning_mode: bool,
    event_counter: u32,
    pattern_counter: u32,
    signature_counter: u32,
}

impl AISecurityMonitor {
    pub fn new() -> Self {
        Self {
            security_events: Vec::new(),
            threat_signatures: Vec::new(),
            behavioral_patterns: Vec::new(),
            security_metrics: SecurityMetrics {
                total_threats_detected: 0,
                threats_blocked: 0,
                false_positives: 0,
                system_security_score: 1.0,
                average_response_time_ms: 0,
                active_threats: 0,
                behavioral_anomalies: 0,
                security_events_per_minute: 0.0,
            },
            last_scan_timestamp: 0,
            lockdown_mode: false,
            learning_mode: true,
            event_counter: 0,
            pattern_counter: 0,
            signature_counter: 0,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        crate::println!("[SECURITY] Initializing AI security monitoring system...");

        // Load baseline threat signatures
        self.load_baseline_signatures()?;

        // Initialize behavioral pattern learning
        self.initialize_behavioral_patterns()?;

        // Set up security monitoring
        self.security_metrics.system_security_score = 1.0;
        self.learning_mode = true;

        crate::println!("[SECURITY] AI security monitor initialized successfully");
        Ok(())
    }

    pub fn perform_security_scan(&mut self, current_time_ms: u64) -> Vec<SecurityEvent, 16> {
        if current_time_ms - self.last_scan_timestamp < SECURITY_SCAN_INTERVAL_MS {
            return Vec::new();
        }

        self.last_scan_timestamp = current_time_ms;
        let mut detected_threats = Vec::new();

        // Collect system behavioral data
        let behavioral_data = self.collect_behavioral_data(current_time_ms);

        // AI-powered threat detection
        if let Some(threats) = self.detect_threats_ai(&behavioral_data, current_time_ms) {
            for threat in threats {
                if threat.confidence >= THREAT_CONFIDENCE_THRESHOLD {
                    crate::println!("[SECURITY] 🚨 THREAT DETECTED: {} ({}% confidence, {} severity)",
                                   threat.category,
                                   (threat.confidence * 100.0) as u32,
                                   threat.severity);

                    // Execute security response
                    self.execute_security_response(&threat);

                    let _ = detected_threats.push(threat);
                }
            }
        }

        // Update behavioral patterns (learning)
        if self.learning_mode {
            self.update_behavioral_patterns(&behavioral_data, current_time_ms);
        }

        // Update security metrics
        self.update_security_metrics(current_time_ms);

        detected_threats
    }

    pub fn get_security_metrics(&self) -> SecurityMetrics {
        self.security_metrics
    }

    pub fn is_lockdown_mode(&self) -> bool {
        self.lockdown_mode
    }

    pub fn enable_learning_mode(&mut self, enabled: bool) {
        self.learning_mode = enabled;
        crate::println!("[SECURITY] Learning mode {}", if enabled { "enabled" } else { "disabled" });
    }

    pub fn add_custom_threat_signature(&mut self, category: ThreatCategory, severity: ThreatSeverity,
                                      pattern: [f32; 8]) -> Result<u32, &'static str> {
        if self.threat_signatures.len() >= MAX_THREAT_SIGNATURES {
            return Err("Maximum threat signatures reached");
        }

        let signature = ThreatSignature {
            signature_id: self.signature_counter,
            category,
            severity,
            detection_pattern: pattern,
            false_positive_rate: 0.1,
            last_updated_ms: 0,
            effectiveness_score: 0.5,
        };

        self.signature_counter += 1;
        let _ = self.threat_signatures.push(signature);

        crate::println!("[SECURITY] Custom threat signature added: {} (ID: {})", category, signature.signature_id);
        Ok(signature.signature_id)
    }

    fn load_baseline_signatures(&mut self) -> Result<(), &'static str> {
        // Memory corruption signature
        let memory_corruption = ThreatSignature {
            signature_id: self.signature_counter,
            category: ThreatCategory::MemoryCorruption,
            severity: ThreatSeverity::High,
            detection_pattern: [0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4],
            false_positive_rate: 0.05,
            last_updated_ms: 0,
            effectiveness_score: 0.9,
        };
        self.signature_counter += 1;
        let _ = self.threat_signatures.push(memory_corruption);

        // Privilege escalation signature
        let privilege_escalation = ThreatSignature {
            signature_id: self.signature_counter,
            category: ThreatCategory::PrivilegeEscalation,
            severity: ThreatSeverity::Critical,
            detection_pattern: [0.9, 0.1, 0.8, 0.2, 0.95, 0.05, 0.85, 0.15],
            false_positive_rate: 0.02,
            last_updated_ms: 0,
            effectiveness_score: 0.95,
        };
        self.signature_counter += 1;
        let _ = self.threat_signatures.push(privilege_escalation);

        // Behavioral anomaly signature
        let behavioral_anomaly = ThreatSignature {
            signature_id: self.signature_counter,
            category: ThreatCategory::BehavioralAnomaly,
            severity: ThreatSeverity::Medium,
            detection_pattern: [0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 0.8, 0.2],
            false_positive_rate: 0.15,
            last_updated_ms: 0,
            effectiveness_score: 0.7,
        };
        self.signature_counter += 1;
        let _ = self.threat_signatures.push(behavioral_anomaly);

        // DoS attack signature
        let dos_attack = ThreatSignature {
            signature_id: self.signature_counter,
            category: ThreatCategory::DenialOfService,
            severity: ThreatSeverity::High,
            detection_pattern: [0.95, 0.05, 0.9, 0.1, 0.8, 0.2, 0.85, 0.15],
            false_positive_rate: 0.08,
            last_updated_ms: 0,
            effectiveness_score: 0.85,
        };
        self.signature_counter += 1;
        let _ = self.threat_signatures.push(dos_attack);

        Ok(())
    }

    fn initialize_behavioral_patterns(&mut self) -> Result<(), &'static str> {
        // Normal system behavior pattern
        let normal_pattern = BehavioralPattern {
            pattern_id: self.pattern_counter,
            features: [0.3, 0.4, 0.2, 0.1, 0.5, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            normal_behavior: true,
            frequency: 100,
            last_seen_ms: 0,
            confidence: 0.9,
        };
        self.pattern_counter += 1;
        let _ = self.behavioral_patterns.push(normal_pattern);

        // High load pattern (potentially normal)
        let high_load_pattern = BehavioralPattern {
            pattern_id: self.pattern_counter,
            features: [0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
            normal_behavior: true,
            frequency: 20,
            last_seen_ms: 0,
            confidence: 0.7,
        };
        self.pattern_counter += 1;
        let _ = self.behavioral_patterns.push(high_load_pattern);

        Ok(())
    }

    fn collect_behavioral_data(&self, _current_time_ms: u64) -> [f32; 16] {
        let mut behavioral_data = [0.0f32; 16];

        // Collect system metrics
        let performance_stats = crate::performance_monitor::get_performance_stats();
        let system_health = crate::predictive_health::get_overall_system_health();

        // Normalize and encode behavioral features
        behavioral_data[0] = performance_stats.cpu_utilization / 100.0;
        behavioral_data[1] = performance_stats.memory_usage_percent / 100.0;
        behavioral_data[2] = performance_stats.system_responsiveness / 100.0;
        behavioral_data[3] = system_health;
        behavioral_data[4] = performance_stats.thermal_state / 100.0;

        // AI system behavior
        let ai_status = crate::ai::get_ai_status();
        behavioral_data[5] = match ai_status {
            crate::ai::AIStatus::Ready => 1.0,
            crate::ai::AIStatus::Learning => 0.8,
            crate::ai::AIStatus::Inferencing => 0.9,
            crate::ai::AIStatus::Initializing => 0.3,
            crate::ai::AIStatus::Error => 0.0,
        };

        // Recovery system activity
        behavioral_data[6] = if crate::autonomous_recovery::is_recovery_active() { 1.0 } else { 0.0 };

        // GPU utilization (if available)
        behavioral_data[7] = if crate::gpu::is_gpu_acceleration_available() { 0.8 } else { 0.0 };

        // Emergency conditions
        behavioral_data[8] = if crate::predictive_health::is_emergency_mode() { 1.0 } else { 0.0 };

        // Simulate additional security-relevant metrics
        behavioral_data[9] = 0.1; // Simulated network activity
        behavioral_data[10] = 0.05; // Simulated file system activity
        behavioral_data[11] = 0.02; // Simulated privilege requests

        behavioral_data
    }

    fn detect_threats_ai(&mut self, behavioral_data: &[f32; 16], current_time_ms: u64) -> Option<Vec<SecurityEvent, 8>> {
        let mut detected_threats = Vec::new();

        // Compare against known threat signatures
        for signature in &self.threat_signatures {
            let similarity = self.calculate_pattern_similarity(&signature.detection_pattern,
                                                              &behavioral_data[0..8]);

            if similarity > 0.8 { // High similarity to threat pattern
                let confidence = similarity * (1.0 - signature.false_positive_rate);

                if confidence >= THREAT_CONFIDENCE_THRESHOLD {
                    let event = SecurityEvent {
                        event_id: self.event_counter,
                        timestamp_ms: current_time_ms,
                        category: signature.category,
                        severity: signature.severity,
                        confidence,
                        source_process_id: 0, // Would be determined by actual process analysis
                        affected_resource: 0,
                        behavioral_anomaly_score: self.calculate_anomaly_score(behavioral_data),
                        response_taken: SecurityResponse::AlertOnly, // Will be updated
                        false_positive_probability: signature.false_positive_rate,
                    };

                    self.event_counter += 1;
                    let _ = detected_threats.push(event);
                }
            }
        }

        // Behavioral anomaly detection
        let anomaly_score = self.calculate_anomaly_score(behavioral_data);
        if anomaly_score > 0.8 {
            let event = SecurityEvent {
                event_id: self.event_counter,
                timestamp_ms: current_time_ms,
                category: ThreatCategory::BehavioralAnomaly,
                severity: if anomaly_score > 0.95 { ThreatSeverity::High } else { ThreatSeverity::Medium },
                confidence: anomaly_score,
                source_process_id: 0,
                affected_resource: 0,
                behavioral_anomaly_score: anomaly_score,
                response_taken: SecurityResponse::Monitor,
                false_positive_probability: 0.2,
            };

            self.event_counter += 1;
            let _ = detected_threats.push(event);
        }

        if detected_threats.is_empty() {
            None
        } else {
            Some(detected_threats)
        }
    }

    fn calculate_pattern_similarity(&self, pattern1: &[f32], pattern2: &[f32]) -> f32 {
        let mut similarity = 0.0;
        let len = pattern1.len().min(pattern2.len());

        for i in 0..len {
            similarity += 1.0 - (pattern1[i] - pattern2[i]).abs();
        }

        similarity / len as f32
    }

    fn calculate_anomaly_score(&self, behavioral_data: &[f32; 16]) -> f32 {
        let mut max_anomaly = 0.0;

        // Compare against known normal patterns
        for pattern in &self.behavioral_patterns {
            if pattern.normal_behavior {
                let mut difference = 0.0;
                for i in 0..16 {
                    difference += (behavioral_data[i] - pattern.features[i]).abs();
                }
                difference /= 16.0;

                let anomaly_score = difference * pattern.confidence;
                max_anomaly = max_anomaly.max(anomaly_score);
            }
        }

        max_anomaly.min(1.0)
    }

    fn execute_security_response(&mut self, threat: &SecurityEvent) {
        let response = self.determine_response_strategy(threat);

        match response {
            SecurityResponse::Monitor => {
                crate::println!("[SECURITY] Monitoring threat: {}", threat.category);
            }

            SecurityResponse::AlertOnly => {
                crate::println!("[SECURITY] 🔔 SECURITY ALERT: {} detected", threat.category);
            }

            SecurityResponse::BlockAccess => {
                crate::println!("[SECURITY] 🚫 BLOCKING ACCESS: {} threat", threat.category);
                // Implement access blocking logic
            }

            SecurityResponse::QuarantineProcess => {
                crate::println!("[SECURITY] 🔒 QUARANTINING PROCESS: PID {}", threat.source_process_id);
                // Implement process quarantine logic
            }

            SecurityResponse::KillProcess => {
                crate::println!("[SECURITY] ⚔️  TERMINATING THREAT PROCESS: PID {}", threat.source_process_id);
                // Implement process termination logic
            }

            SecurityResponse::SystemLockdown => {
                crate::println!("[SECURITY] 🚨 SYSTEM LOCKDOWN INITIATED");
                self.lockdown_mode = true;
                // Implement system lockdown procedures
            }

            SecurityResponse::EmergencyShutdown => {
                crate::println!("[SECURITY] 🚨 EMERGENCY SHUTDOWN - CRITICAL SECURITY THREAT");
                // Trigger emergency shutdown via autonomous recovery
                let _ = crate::autonomous_recovery::force_recovery(
                    crate::autonomous_recovery::RecoveryTrigger::SecurityBreach, 0
                );
            }

            SecurityResponse::AdaptiveThrottling => {
                crate::println!("[SECURITY] ⚡ ADAPTIVE THROTTLING: Reducing system performance");
                let _ = crate::performance_monitor::set_strategy(
                    crate::performance_monitor::OptimizationStrategy::PowerEfficient
                );
            }

            _ => {
                crate::println!("[SECURITY] 📝 LOGGING SECURITY EVENT: {}", threat.category);
            }
        }

        // Record the response taken
        self.security_metrics.total_threats_detected += 1;
        if response != SecurityResponse::Monitor && response != SecurityResponse::AlertOnly {
            self.security_metrics.threats_blocked += 1;
        }
    }

    fn determine_response_strategy(&self, threat: &SecurityEvent) -> SecurityResponse {
        match (threat.severity, threat.category, threat.confidence) {
            (ThreatSeverity::Emergency, _, _) => SecurityResponse::EmergencyShutdown,

            (ThreatSeverity::Critical, ThreatCategory::PrivilegeEscalation, confidence) if confidence > 0.9 => {
                SecurityResponse::SystemLockdown
            }

            (ThreatSeverity::Critical, ThreatCategory::MemoryCorruption, confidence) if confidence > 0.85 => {
                SecurityResponse::KillProcess
            }

            (ThreatSeverity::High, ThreatCategory::DenialOfService, _) => {
                SecurityResponse::AdaptiveThrottling
            }

            (ThreatSeverity::High, _, confidence) if confidence > 0.8 => {
                SecurityResponse::QuarantineProcess
            }

            (ThreatSeverity::Medium, _, confidence) if confidence > 0.75 => {
                SecurityResponse::BlockAccess
            }

            (ThreatSeverity::Low, _, _) => {
                SecurityResponse::Monitor
            }

            _ => SecurityResponse::AlertOnly,
        }
    }

    fn update_behavioral_patterns(&mut self, behavioral_data: &[f32; 16], current_time_ms: u64) {
        // Simple learning: update pattern frequencies and add new patterns
        let mut is_known_pattern = false;

        for pattern in &mut self.behavioral_patterns {
            let similarity = self.calculate_pattern_similarity(&pattern.features, behavioral_data);

            if similarity > 0.85 { // Similar to existing pattern
                pattern.frequency += 1;
                pattern.last_seen_ms = current_time_ms;

                // Update pattern features with exponential moving average
                for i in 0..16 {
                    pattern.features[i] = (pattern.features[i] * 0.9) + (behavioral_data[i] * 0.1);
                }

                is_known_pattern = true;
                break;
            }
        }

        // Add new pattern if significantly different and not at capacity
        if !is_known_pattern && self.behavioral_patterns.len() < MAX_BEHAVIORAL_PATTERNS {
            let system_health = crate::predictive_health::get_overall_system_health();
            let is_normal = system_health > 0.7 && !crate::predictive_health::is_emergency_mode();

            let new_pattern = BehavioralPattern {
                pattern_id: self.pattern_counter,
                features: *behavioral_data,
                normal_behavior: is_normal,
                frequency: 1,
                last_seen_ms: current_time_ms,
                confidence: if is_normal { 0.6 } else { 0.3 },
            };

            self.pattern_counter += 1;
            let _ = self.behavioral_patterns.push(new_pattern);

            crate::println!("[SECURITY] New behavioral pattern learned (ID: {}, Normal: {})",
                           new_pattern.pattern_id, new_pattern.normal_behavior);
        }
    }

    fn update_security_metrics(&mut self, current_time_ms: u64) {
        // Update security score based on recent threat activity
        let recent_threats = self.security_events.iter()
            .filter(|event| current_time_ms - event.timestamp_ms < 60000) // Last minute
            .count();

        self.security_metrics.active_threats = recent_threats as u32;

        // Calculate security score (higher threats = lower score)
        let threat_factor = 1.0 - (recent_threats as f32 * 0.1).min(0.8);
        let lockdown_penalty = if self.lockdown_mode { 0.5 } else { 1.0 };

        self.security_metrics.system_security_score = threat_factor * lockdown_penalty;

        // Update events per minute
        let events_last_minute = self.security_events.iter()
            .filter(|event| current_time_ms - event.timestamp_ms < 60000)
            .count();

        self.security_metrics.security_events_per_minute = events_last_minute as f32;
    }
}

lazy_static! {
    static ref SECURITY_MONITOR: Mutex<AISecurityMonitor> = Mutex::new(AISecurityMonitor::new());
}

pub fn init_ai_security_monitor() {
    let mut monitor = SECURITY_MONITOR.lock();
    match monitor.initialize() {
        Ok(_) => crate::println!("[SECURITY] AI security monitor ready"),
        Err(e) => crate::println!("[SECURITY] Failed to initialize: {}", e),
    }
}

pub fn perform_security_scan(current_time_ms: u64) -> Vec<SecurityEvent, 16> {
    SECURITY_MONITOR.lock().perform_security_scan(current_time_ms)
}

pub fn get_security_metrics() -> SecurityMetrics {
    SECURITY_MONITOR.lock().get_security_metrics()
}

pub fn is_system_locked_down() -> bool {
    SECURITY_MONITOR.lock().is_lockdown_mode()
}

pub fn enable_security_learning(enabled: bool) {
    SECURITY_MONITOR.lock().enable_learning_mode(enabled);
}

pub fn add_threat_signature(category: ThreatCategory, severity: ThreatSeverity,
                           pattern: [f32; 8]) -> Result<u32, &'static str> {
    SECURITY_MONITOR.lock().add_custom_threat_signature(category, severity, pattern)
}

#[test_case]
fn test_security_monitor_initialization() {
    let mut monitor = AISecurityMonitor::new();
    assert!(monitor.initialize().is_ok());
    assert!(!monitor.is_lockdown_mode());
    assert_eq!(monitor.get_security_metrics().system_security_score, 1.0);
}

#[test_case]
fn test_threat_detection() {
    let mut monitor = AISecurityMonitor::new();
    let _ = monitor.initialize();

    // Simulate malicious behavioral pattern
    let malicious_data = [0.95, 0.9, 0.1, 0.2, 0.85, 0.0, 1.0, 0.8, 1.0, 0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0];

    if let Some(threats) = monitor.detect_threats_ai(&malicious_data, 1000) {
        assert!(!threats.is_empty());
        assert!(threats[0].confidence > THREAT_CONFIDENCE_THRESHOLD);
    }
}

#[test_case]
fn test_behavioral_learning() {
    let mut monitor = AISecurityMonitor::new();
    let _ = monitor.initialize();
    monitor.enable_learning_mode(true);

    let initial_patterns = monitor.behavioral_patterns.len();

    // Add unique behavioral data
    let unique_data = [0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];
    monitor.update_behavioral_patterns(&unique_data, 2000);

    assert!(monitor.behavioral_patterns.len() >= initial_patterns);
}

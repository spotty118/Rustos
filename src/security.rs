//! Security Framework for RustOS
//!
//! This module provides basic security features including permission management,
//! access control, and security monitoring for the kernel.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use alloc::{vec::Vec, string::{String, ToString}, collections::BTreeMap, format, vec};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use crate::println;

/// Security levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecurityLevel {
    /// No security restrictions
    None = 0,
    /// Basic security checks
    Basic = 1,
    /// Enhanced security
    Enhanced = 2,
    /// High security mode
    High = 3,
    /// Maximum security
    Maximum = 4,
}

/// Permission types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// Read permission
    Read,
    /// Write permission
    Write,
    /// Execute permission
    Execute,
    /// Delete permission
    Delete,
    /// Admin permission
    Admin,
    /// Network access
    Network,
    /// File system access
    FileSystem,
    /// Device access
    Device,
    /// Memory access
    Memory,
    /// Interrupt handling
    Interrupt,
}

/// Security context for processes
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub user_id: u32,
    pub group_id: u32,
    pub permissions: Vec<Permission>,
    pub security_level: SecurityLevel,
    pub access_mask: u64,
    pub capabilities: u64,
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self {
            user_id: 0,
            group_id: 0,
            permissions: Vec::new(),
            security_level: SecurityLevel::Basic,
            access_mask: 0,
            capabilities: 0,
        }
    }
}

/// Security event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityEvent {
    /// Unauthorized access attempt
    UnauthorizedAccess,
    /// Permission denied
    PermissionDenied,
    /// Privilege escalation attempt
    PrivilegeEscalation,
    /// Security policy violation
    PolicyViolation,
    /// Suspicious activity detected
    SuspiciousActivity,
    /// System integrity check failed
    IntegrityFailure,
    /// Authentication failure
    AuthenticationFailure,
    /// Security level changed
    SecurityLevelChanged,
}

/// Security alert
#[derive(Debug, Clone)]
pub struct SecurityAlert {
    pub event_type: SecurityEvent,
    pub timestamp: u64,
    pub source: String,
    pub description: String,
    pub severity: u8, // 0-10 scale
    pub process_id: Option<u32>,
    pub user_id: Option<u32>,
}

lazy_static! {
    /// Global security state
    static ref SECURITY_STATE: Mutex<SecurityState> = Mutex::new(SecurityState::new());
    
    /// Security contexts by process
    static ref SECURITY_CONTEXTS: RwLock<BTreeMap<u32, SecurityContext>> = RwLock::new(BTreeMap::new());
    
    /// Security alerts
    static ref SECURITY_ALERTS: Mutex<Vec<SecurityAlert>> = Mutex::new(Vec::new());
}

/// Internal security state
struct SecurityState {
    initialized: bool,
    global_security_level: SecurityLevel,
    monitoring_enabled: bool,
    access_control_enabled: bool,
    audit_enabled: bool,
    threat_detection_enabled: bool,
}

impl SecurityState {
    fn new() -> Self {
        Self {
            initialized: false,
            global_security_level: SecurityLevel::Basic,
            monitoring_enabled: true,
            access_control_enabled: true,
            audit_enabled: true,
            threat_detection_enabled: true,
        }
    }
}

/// Security statistics
static SECURITY_EVENTS: AtomicU64 = AtomicU64::new(0);
static ACCESS_DENIED_COUNT: AtomicU64 = AtomicU64::new(0);
static PRIVILEGE_VIOLATIONS: AtomicU64 = AtomicU64::new(0);

/// Initialize security framework
pub fn init() -> Result<(), &'static str> {
    let mut state = SECURITY_STATE.lock();
    
    if state.initialized {
        return Ok(());
    }

    // Set up default security context for kernel
    let kernel_context = SecurityContext {
        user_id: 0, // root
        group_id: 0,
        permissions: vec![
            Permission::Read,
            Permission::Write,
            Permission::Execute,
            Permission::Delete,
            Permission::Admin,
            Permission::Network,
            Permission::FileSystem,
            Permission::Device,
            Permission::Memory,
            Permission::Interrupt,
        ],
        security_level: SecurityLevel::Maximum,
        access_mask: 0xFFFFFFFFFFFFFFFF,
        capabilities: 0xFFFFFFFFFFFFFFFF,
    };
    
    SECURITY_CONTEXTS.write().insert(0, kernel_context);
    
    state.initialized = true;
    println!("Security: Framework initialized with {} security level", 
             match state.global_security_level {
                 SecurityLevel::None => "None",
                 SecurityLevel::Basic => "Basic",
                 SecurityLevel::Enhanced => "Enhanced", 
                 SecurityLevel::High => "High",
                 SecurityLevel::Maximum => "Maximum",
             });
    
    Ok(())
}

/// Check if a process has permission for an operation
pub fn check_permission(process_id: u32, permission: Permission) -> bool {
    let contexts = SECURITY_CONTEXTS.read();
    
    if let Some(context) = contexts.get(&process_id) {
        if context.permissions.contains(&permission) {
            true
        } else {
            // Log access denied
            ACCESS_DENIED_COUNT.fetch_add(1, Ordering::SeqCst);
            log_security_event(SecurityEvent::PermissionDenied, 
                             &format!("Process {} denied {} permission", process_id, format!("{:?}", permission)));
            false
        }
    } else {
        // No security context - deny by default
        ACCESS_DENIED_COUNT.fetch_add(1, Ordering::SeqCst);
        log_security_event(SecurityEvent::UnauthorizedAccess,
                         &format!("No security context for process {}", process_id));
        false
    }
}

/// Set security context for a process
pub fn set_security_context(process_id: u32, context: SecurityContext) -> Result<(), &'static str> {
    let state = SECURITY_STATE.lock();
    
    if !state.access_control_enabled {
        return Ok(());
    }
    
    // Validate security context
    if context.security_level > state.global_security_level && process_id != 0 {
        PRIVILEGE_VIOLATIONS.fetch_add(1, Ordering::SeqCst);
        log_security_event(SecurityEvent::PrivilegeEscalation,
                         &format!("Process {} attempted to exceed global security level", process_id));
        return Err("Security level exceeds global maximum");
    }
    
    SECURITY_CONTEXTS.write().insert(process_id, context);
    Ok(())
}

/// Get security context for a process
pub fn get_security_context(process_id: u32) -> Option<SecurityContext> {
    SECURITY_CONTEXTS.read().get(&process_id).cloned()
}

/// Set global security level
pub fn set_security_level(level: SecurityLevel) -> Result<(), &'static str> {
    let mut state = SECURITY_STATE.lock();
    
    if level < state.global_security_level {
        // Lowering security level requires admin permission
        if !check_permission(0, Permission::Admin) {
            return Err("Admin permission required to lower security level");
        }
    }
    
    state.global_security_level = level;
    log_security_event(SecurityEvent::SecurityLevelChanged,
                     &format!("Global security level changed to {:?}", level));
    
    Ok(())
}

/// Get current global security level
pub fn get_security_level() -> SecurityLevel {
    SECURITY_STATE.lock().global_security_level
}

/// Get current security status
pub fn get_security_status() -> SecurityLevel {
    get_security_level()
}

/// Log a security event
pub fn log_security_event(event_type: SecurityEvent, description: &str) {
    SECURITY_EVENTS.fetch_add(1, Ordering::SeqCst);
    
    let alert = SecurityAlert {
        event_type,
        timestamp: crate::time::uptime_ms(),
        source: "kernel".to_string(),
        description: description.to_string(),
        severity: match event_type {
            SecurityEvent::UnauthorizedAccess => 8,
            SecurityEvent::PrivilegeEscalation => 9,
            SecurityEvent::IntegrityFailure => 10,
            SecurityEvent::AuthenticationFailure => 7,
            SecurityEvent::SuspiciousActivity => 6,
            SecurityEvent::PolicyViolation => 5,
            SecurityEvent::PermissionDenied => 4,
            SecurityEvent::SecurityLevelChanged => 3,
        },
        process_id: None,
        user_id: None,
    };
    
    let mut alerts = SECURITY_ALERTS.lock();
    alerts.push(alert);
    
    // Keep only the last 1000 alerts
    if alerts.len() > 1000 {
        alerts.remove(0);
    }
}

/// Get recent security alerts
pub fn get_security_alerts() -> Vec<SecurityAlert> {
    SECURITY_ALERTS.lock().clone()
}

/// Clear security alerts
pub fn clear_security_alerts() {
    SECURITY_ALERTS.lock().clear();
}

/// Enable/disable security monitoring
pub fn set_monitoring_enabled(enabled: bool) {
    SECURITY_STATE.lock().monitoring_enabled = enabled;
}

/// Check if monitoring is enabled
pub fn is_monitoring_enabled() -> bool {
    SECURITY_STATE.lock().monitoring_enabled
}

/// Enable/disable access control
pub fn set_access_control_enabled(enabled: bool) {
    SECURITY_STATE.lock().access_control_enabled = enabled;
}

/// Check if access control is enabled
pub fn is_access_control_enabled() -> bool {
    SECURITY_STATE.lock().access_control_enabled
}

/// Security statistics
#[derive(Debug, Clone)]
pub struct SecurityStats {
    pub total_events: u64,
    pub access_denied_count: u64,
    pub privilege_violations: u64,
    pub active_alerts: usize,
    pub security_level: SecurityLevel,
    pub monitoring_enabled: bool,
    pub access_control_enabled: bool,
}

/// Get security statistics
pub fn get_security_stats() -> SecurityStats {
    let state = SECURITY_STATE.lock();
    let alerts = SECURITY_ALERTS.lock();
    
    SecurityStats {
        total_events: SECURITY_EVENTS.load(Ordering::SeqCst),
        access_denied_count: ACCESS_DENIED_COUNT.load(Ordering::SeqCst),
        privilege_violations: PRIVILEGE_VIOLATIONS.load(Ordering::SeqCst),
        active_alerts: alerts.len(),
        security_level: state.global_security_level,
        monitoring_enabled: state.monitoring_enabled,
        access_control_enabled: state.access_control_enabled,
    }
}

/// Validate system integrity
pub fn validate_integrity() -> Result<(), &'static str> {
    // Basic integrity checks
    let state = SECURITY_STATE.lock();
    
    if !state.initialized {
        return Err("Security framework not initialized");
    }
    
    // Check if security contexts are valid
    let contexts = SECURITY_CONTEXTS.read();
    if contexts.is_empty() {
        log_security_event(SecurityEvent::IntegrityFailure,
                         "No security contexts found");
        return Err("Security context integrity failure");
    }
    
    Ok(())
}

/// Threat detection and response
pub fn detect_threats() -> Vec<SecurityAlert> {
    let mut threats = Vec::new();
    let state = SECURITY_STATE.lock();
    
    if !state.threat_detection_enabled {
        return threats;
    }
    
    // Simple threat detection based on statistics
    let stats = get_security_stats();
    
    if stats.access_denied_count > 100 {
        let alert = SecurityAlert {
            event_type: SecurityEvent::SuspiciousActivity,
            timestamp: crate::time::uptime_ms(),
            source: "threat_detector".to_string(),
            description: "High number of access denials detected".to_string(),
            severity: 7,
            process_id: None,
            user_id: None,
        };
        threats.push(alert);
    }
    
    if stats.privilege_violations > 10 {
        let alert = SecurityAlert {
            event_type: SecurityEvent::SuspiciousActivity,
            timestamp: crate::time::uptime_ms(),
            source: "threat_detector".to_string(),
            description: "Multiple privilege escalation attempts detected".to_string(),
            severity: 9,
            process_id: None,
            user_id: None,
        };
        threats.push(alert);
    }
    
    threats
}

/// Security audit function
pub fn audit_security() -> Result<String, &'static str> {
    validate_integrity()?;
    
    let stats = get_security_stats();
    let threats = detect_threats();
    
    let audit_report = format!(
        "Security Audit Report:\n\
         - Security Level: {:?}\n\
         - Total Events: {}\n\
         - Access Denied: {}\n\
         - Privilege Violations: {}\n\
         - Active Alerts: {}\n\
         - Threats Detected: {}\n\
         - Monitoring: {}\n\
         - Access Control: {}",
        stats.security_level,
        stats.total_events,
        stats.access_denied_count,
        stats.privilege_violations,
        stats.active_alerts,
        threats.len(),
        stats.monitoring_enabled,
        stats.access_control_enabled
    );
    
    Ok(audit_report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_context_default() {
        let context = SecurityContext::default();
        assert_eq!(context.user_id, 0);
        assert_eq!(context.security_level, SecurityLevel::Basic);
    }

    #[test]
    fn test_security_level_ordering() {
        assert!(SecurityLevel::Maximum > SecurityLevel::High);
        assert!(SecurityLevel::High > SecurityLevel::Enhanced);
        assert!(SecurityLevel::Enhanced > SecurityLevel::Basic);
        assert!(SecurityLevel::Basic > SecurityLevel::None);
    }

    #[test]
    fn test_security_stats() {
        let stats = get_security_stats();
        assert!(stats.total_events >= 0);
    }
}
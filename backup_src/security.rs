//! Advanced Security Framework for RustOS
//!
//! This module provides:
//! - Capability-based access control system
//! - Process sandboxing and isolation
//! - Security policies and enforcement
//! - Cryptographic primitives and key management
//! - Access control lists (ACLs) and permissions
//! - Security auditing and logging
//! - Secure inter-process communication
//! - Memory protection and data integrity
//! - Authentication and authorization
//! - Security context switching

use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use core::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use spin::Mutex;
use lazy_static::lazy_static;

/// Security capability identifier
pub type CapabilityId = u64;

/// Security context identifier
pub type SecurityContextId = u32;

/// User and group identifiers
pub type UserId = u32;
pub type GroupId = u32;

/// Security levels for classification
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SecurityLevel {
    Public = 0,
    Internal = 1,
    Confidential = 2,
    Secret = 3,
    TopSecret = 4,
}

/// Capability types in the system
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum CapabilityType {
    // File system capabilities
    FileRead,
    FileWrite,
    FileExecute,
    FileCreate,
    FileDelete,
    DirectoryCreate,
    DirectoryDelete,

    // Process capabilities
    ProcessCreate,
    ProcessKill,
    ProcessDebug,
    ProcessSetuid,
    ProcessSetgid,
    ProcessChroot,

    // Network capabilities
    NetworkBind,
    NetworkConnect,
    NetworkListen,
    NetworkRaw,
    NetworkAdmin,

    // System capabilities
    SystemShutdown,
    SystemReboot,
    SystemMount,
    SystemUmount,
    SystemTime,
    SystemHostname,

    // Hardware capabilities
    HardwareAccess,
    HardwareConfig,
    HardwareDMA,

    // IPC capabilities
    IPCCreate,
    IPCConnect,
    IPCBroadcast,

    // Memory capabilities
    MemoryExecute,
    MemoryMap,
    MemoryLock,

    // Administrative capabilities
    UserManagement,
    SecurityAdmin,
    AuditLog,
}

/// Security capability structure
#[derive(Debug, Clone)]
pub struct Capability {
    pub id: CapabilityId,
    pub capability_type: CapabilityType,
    pub target_resource: Option<String>,
    pub permissions: PermissionFlags,
    pub owner: UserId,
    pub group: GroupId,
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub transferable: bool,
    pub inheritable: bool,
}

/// Permission flags for fine-grained access control
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PermissionFlags {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
    pub delete: bool,
    pub admin: bool,
}

impl PermissionFlags {
    pub const fn none() -> Self {
        Self {
            read: false,
            write: false,
            execute: false,
            delete: false,
            admin: false,
        }
    }

    pub const fn read_only() -> Self {
        Self {
            read: true,
            write: false,
            execute: false,
            delete: false,
            admin: false,
        }
    }

    pub const fn read_write() -> Self {
        Self {
            read: true,
            write: true,
            execute: false,
            delete: false,
            admin: false,
        }
    }

    pub const fn full() -> Self {
        Self {
            read: true,
            write: true,
            execute: true,
            delete: true,
            admin: true,
        }
    }
}

/// Security context for processes and users
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub id: SecurityContextId,
    pub user_id: UserId,
    pub group_id: GroupId,
    pub supplementary_groups: Vec<GroupId>,
    pub capabilities: BTreeSet<CapabilityId>,
    pub security_level: SecurityLevel,
    pub sandbox_enabled: bool,
    pub audit_enabled: bool,
    pub created_at: u64,
    pub last_access: u64,
}

impl SecurityContext {
    pub fn new(id: SecurityContextId, user_id: UserId, group_id: GroupId) -> Self {
        let now = crate::time::uptime_ms();
        Self {
            id,
            user_id,
            group_id,
            supplementary_groups: Vec::new(),
            capabilities: BTreeSet::new(),
            security_level: SecurityLevel::Public,
            sandbox_enabled: false,
            audit_enabled: true,
            created_at: now,
            last_access: now,
        }
    }

    pub fn has_capability(&self, capability_id: CapabilityId) -> bool {
        self.capabilities.contains(&capability_id)
    }

    pub fn add_capability(&mut self, capability_id: CapabilityId) {
        self.capabilities.insert(capability_id);
        self.last_access = crate::time::uptime_ms();
    }

    pub fn remove_capability(&mut self, capability_id: CapabilityId) {
        self.capabilities.remove(&capability_id);
        self.last_access = crate::time::uptime_ms();
    }

    pub fn is_member_of_group(&self, group_id: GroupId) -> bool {
        self.group_id == group_id || self.supplementary_groups.contains(&group_id)
    }
}

/// Access Control List entry
#[derive(Debug, Clone)]
pub struct AclEntry {
    pub entry_type: AclEntryType,
    pub principal_id: u32, // User ID or Group ID
    pub permissions: PermissionFlags,
    pub inherited: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AclEntryType {
    User,
    Group,
    Other,
}

/// Access Control List
#[derive(Debug, Clone)]
pub struct AccessControlList {
    pub entries: Vec<AclEntry>,
    pub default_permissions: PermissionFlags,
    pub owner: UserId,
    pub group: GroupId,
}

impl AccessControlList {
    pub fn new(owner: UserId, group: GroupId) -> Self {
        Self {
            entries: Vec::new(),
            default_permissions: PermissionFlags::none(),
            owner,
            group,
        }
    }

    pub fn check_access(&self, context: &SecurityContext, requested: PermissionFlags) -> bool {
        // Check owner permissions
        if context.user_id == self.owner {
            return self.has_owner_permission(requested);
        }

        // Check group permissions
        if context.is_member_of_group(self.group) {
            if let Some(group_perms) = self.get_group_permissions(self.group) {
                if self.permissions_match(group_perms, requested) {
                    return true;
                }
            }
        }

        // Check explicit ACL entries
        for entry in &self.entries {
            match entry.entry_type {
                AclEntryType::User if entry.principal_id == context.user_id => {
                    if self.permissions_match(entry.permissions, requested) {
                        return true;
                    }
                }
                AclEntryType::Group if context.is_member_of_group(entry.principal_id) => {
                    if self.permissions_match(entry.permissions, requested) {
                        return true;
                    }
                }
                AclEntryType::Other => {
                    if self.permissions_match(entry.permissions, requested) {
                        return true;
                    }
                }
                _ => {}
            }
        }

        // Check default permissions
        self.permissions_match(self.default_permissions, requested)
    }

    fn has_owner_permission(&self, requested: PermissionFlags) -> bool {
        // Owner has full permissions by default
        true
    }

    fn get_group_permissions(&self, _group_id: GroupId) -> Option<PermissionFlags> {
        // Default group permissions
        Some(PermissionFlags::read_only())
    }

    fn permissions_match(&self, available: PermissionFlags, requested: PermissionFlags) -> bool {
        (!requested.read || available.read) &&
        (!requested.write || available.write) &&
        (!requested.execute || available.execute) &&
        (!requested.delete || available.delete) &&
        (!requested.admin || available.admin)
    }
}

/// Security audit log entry
#[derive(Debug, Clone)]
pub struct AuditLogEntry {
    pub timestamp: u64,
    pub event_type: SecurityEventType,
    pub user_id: UserId,
    pub process_id: u32,
    pub resource: String,
    pub action: String,
    pub result: SecurityResult,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecurityEventType {
    Authentication,
    Authorization,
    CapabilityGrant,
    CapabilityRevoke,
    AccessDenied,
    PrivilegeEscalation,
    PolicyViolation,
    SystemCall,
    FileAccess,
    NetworkAccess,
    ProcessCreation,
    ConfigChange,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecurityResult {
    Success,
    Failure,
    Blocked,
    Warning,
}

/// Sandbox configuration
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub allow_network: bool,
    pub allow_filesystem: bool,
    pub allowed_paths: Vec<String>,
    pub allowed_syscalls: Vec<u32>,
    pub memory_limit: usize,
    pub cpu_limit_percent: u8,
    pub max_files: u32,
    pub max_processes: u32,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            allow_network: false,
            allow_filesystem: false,
            allowed_paths: vec!["/tmp".to_string()],
            allowed_syscalls: vec![0, 1, 2], // Basic syscalls
            memory_limit: 64 * 1024 * 1024, // 64MB
            cpu_limit_percent: 10,
            max_files: 10,
            max_processes: 1,
        }
    }
}

/// Cryptographic context for secure operations
#[derive(Debug)]
pub struct CryptoContext {
    pub algorithm: CryptoAlgorithm,
    pub key_size: usize,
    pub key_data: Vec<u8>,
    pub initialization_vector: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CryptoAlgorithm {
    AES128,
    AES256,
    ChaCha20,
    RSA2048,
    RSA4096,
    ECDSA256,
    ECDSA384,
}

/// Main security manager
pub struct SecurityManager {
    capabilities: BTreeMap<CapabilityId, Capability>,
    security_contexts: BTreeMap<SecurityContextId, SecurityContext>,
    access_control_lists: BTreeMap<String, AccessControlList>,
    audit_log: Vec<AuditLogEntry>,
    next_capability_id: AtomicU64,
    next_context_id: AtomicU32,
    security_policy: SecurityPolicy,
    crypto_contexts: BTreeMap<u32, CryptoContext>,
    sandbox_configs: BTreeMap<u32, SandboxConfig>,
    failed_auth_attempts: BTreeMap<UserId, u32>,
}

/// System security policy
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    pub enforce_capabilities: bool,
    pub enforce_sandboxing: bool,
    pub audit_all_access: bool,
    pub max_auth_failures: u32,
    pub session_timeout_minutes: u32,
    pub require_strong_passwords: bool,
    pub allow_privilege_escalation: bool,
    pub default_security_level: SecurityLevel,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            enforce_capabilities: true,
            enforce_sandboxing: true,
            audit_all_access: true,
            max_auth_failures: 3,
            session_timeout_minutes: 30,
            require_strong_passwords: true,
            allow_privilege_escalation: false,
            default_security_level: SecurityLevel::Internal,
        }
    }
}

impl SecurityManager {
    pub fn new() -> Self {
        Self {
            capabilities: BTreeMap::new(),
            security_contexts: BTreeMap::new(),
            access_control_lists: BTreeMap::new(),
            audit_log: Vec::new(),
            next_capability_id: AtomicU64::new(1000),
            next_context_id: AtomicU32::new(1),
            security_policy: SecurityPolicy::default(),
            crypto_contexts: BTreeMap::new(),
            sandbox_configs: BTreeMap::new(),
            failed_auth_attempts: BTreeMap::new(),
        }
    }

    /// Initialize security system with default policies
    pub fn init(&mut self) -> Result<(), &'static str> {
        crate::println!("[SECURITY] Initializing security framework...");

        // Create root security context
        let root_context = SecurityContext::new(0, 0, 0);
        self.security_contexts.insert(0, root_context);

        // Create basic capabilities
        self.create_basic_capabilities()?;

        // Set up default ACLs
        self.setup_default_acls()?;

        crate::println!("[SECURITY] Security framework initialized");
        self.log_security_event(
            SecurityEventType::ConfigChange,
            0, 0,
            "system".to_string(),
            "security_init".to_string(),
            SecurityResult::Success,
            "Security framework initialized".to_string()
        );

        Ok(())
    }

    /// Create basic system capabilities
    fn create_basic_capabilities(&mut self) -> Result<(), &'static str> {
        let basic_caps = [
            CapabilityType::FileRead,
            CapabilityType::FileWrite,
            CapabilityType::ProcessCreate,
            CapabilityType::NetworkConnect,
            CapabilityType::IPCCreate,
        ];

        for cap_type in &basic_caps {
            self.create_capability(*cap_type, None, PermissionFlags::full(), 0, 0)?;
        }

        Ok(())
    }

    /// Set up default Access Control Lists
    fn setup_default_acls(&mut self) -> Result<(), &'static str> {
        // Create ACL for root filesystem
        let mut root_acl = AccessControlList::new(0, 0);
        root_acl.default_permissions = PermissionFlags::read_only();
        self.access_control_lists.insert("/".to_string(), root_acl);

        // Create ACL for tmp directory
        let mut tmp_acl = AccessControlList::new(0, 0);
        tmp_acl.default_permissions = PermissionFlags::read_write();
        self.access_control_lists.insert("/tmp".to_string(), tmp_acl);

        Ok(())
    }

    /// Create a new capability
    pub fn create_capability(
        &mut self,
        capability_type: CapabilityType,
        target_resource: Option<String>,
        permissions: PermissionFlags,
        owner: UserId,
        group: GroupId,
    ) -> Result<CapabilityId, &'static str> {
        let id = self.next_capability_id.fetch_add(1, Ordering::AcqRel);

        let capability = Capability {
            id,
            capability_type,
            target_resource,
            permissions,
            owner,
            group,
            created_at: crate::time::uptime_ms(),
            expires_at: None,
            transferable: false,
            inheritable: true,
        };

        self.capabilities.insert(id, capability);

        self.log_security_event(
            SecurityEventType::CapabilityGrant,
            owner, 0,
            target_resource.unwrap_or("system".to_string()),
            format!("create_capability_{:?}", capability_type),
            SecurityResult::Success,
            format!("Capability {} created", id)
        );

        Ok(id)
    }

    /// Create a new security context
    pub fn create_security_context(
        &mut self,
        user_id: UserId,
        group_id: GroupId,
    ) -> Result<SecurityContextId, &'static str> {
        let id = self.next_context_id.fetch_add(1, Ordering::AcqRel);
        let context = SecurityContext::new(id, user_id, group_id);

        self.security_contexts.insert(id, context);

        self.log_security_event(
            SecurityEventType::Authentication,
            user_id, 0,
            "system".to_string(),
            "create_context".to_string(),
            SecurityResult::Success,
            format!("Security context {} created", id)
        );

        Ok(id)
    }

    /// Grant capability to security context
    pub fn grant_capability(
        &mut self,
        context_id: SecurityContextId,
        capability_id: CapabilityId,
        granter_context_id: SecurityContextId,
    ) -> Result<(), &'static str> {
        // Verify granter has permission to grant this capability
        if !self.can_grant_capability(granter_context_id, capability_id) {
            self.log_security_event(
                SecurityEventType::AccessDenied,
                granter_context_id, 0,
                "capability".to_string(),
                "grant_capability".to_string(),
                SecurityResult::Failure,
                format!("Permission denied to grant capability {}", capability_id)
            );
            return Err("Permission denied");
        }

        if let Some(context) = self.security_contexts.get_mut(&context_id) {
            context.add_capability(capability_id);

            self.log_security_event(
                SecurityEventType::CapabilityGrant,
                context.user_id, 0,
                "capability".to_string(),
                "grant_capability".to_string(),
                SecurityResult::Success,
                format!("Capability {} granted to context {}", capability_id, context_id)
            );

            Ok(())
        } else {
            Err("Security context not found")
        }
    }

    /// Check if context can grant a capability
    fn can_grant_capability(&self, granter_context_id: SecurityContextId, capability_id: CapabilityId) -> bool {
        if granter_context_id == 0 {
            return true; // Root can grant anything
        }

        if let Some(context) = self.security_contexts.get(&granter_context_id) {
            // Context must have the capability to grant it
            context.has_capability(capability_id)
        } else {
            false
        }
    }

    /// Check access permission for a resource
    pub fn check_access(
        &self,
        context_id: SecurityContextId,
        resource: &str,
        requested_permissions: PermissionFlags,
    ) -> Result<bool, &'static str> {
        let context = self.security_contexts.get(&context_id)
            .ok_or("Security context not found")?;

        // Check if resource has an ACL
        if let Some(acl) = self.access_control_lists.get(resource) {
            let access_granted = acl.check_access(context, requested_permissions);

            self.log_security_event(
                SecurityEventType::Authorization,
                context.user_id, 0,
                resource.to_string(),
                "check_access".to_string(),
                if access_granted { SecurityResult::Success } else { SecurityResult::Failure },
                format!("Access {} for resource {}",
                       if access_granted { "granted" } else { "denied" }, resource)
            );

            return Ok(access_granted);
        }

        // Default deny
        self.log_security_event(
            SecurityEventType::AccessDenied,
            context.user_id, 0,
            resource.to_string(),
            "check_access".to_string(),
            SecurityResult::Blocked,
            format!("No ACL found for resource {}", resource)
        );

        Ok(false)
    }

    /// Validate system call access
    pub fn validate_syscall(
        &self,
        context_id: SecurityContextId,
        syscall_number: u32,
    ) -> Result<bool, &'static str> {
        let context = self.security_contexts.get(&context_id)
            .ok_or("Security context not found")?;

        // Check if process is sandboxed
        if context.sandbox_enabled {
            if let Some(sandbox) = self.sandbox_configs.get(&(context_id)) {
                let allowed = sandbox.allowed_syscalls.contains(&syscall_number);

                self.log_security_event(
                    SecurityEventType::SystemCall,
                    context.user_id, context_id,
                    "syscall".to_string(),
                    format!("syscall_{}", syscall_number),
                    if allowed { SecurityResult::Success } else { SecurityResult::Blocked },
                    format!("Syscall {} {} in sandbox", syscall_number,
                           if allowed { "allowed" } else { "blocked" })
                );

                return Ok(allowed);
            }
        }

        // Non-sandboxed processes allowed by default
        Ok(true)
    }

    /// Enable sandbox for a security context
    pub fn enable_sandbox(
        &mut self,
        context_id: SecurityContextId,
        config: SandboxConfig,
    ) -> Result<(), &'static str> {
        if let Some(context) = self.security_contexts.get_mut(&context_id) {
            context.sandbox_enabled = true;
            self.sandbox_configs.insert(context_id, config);

            self.log_security_event(
                SecurityEventType::ConfigChange,
                context.user_id, context_id,
                "sandbox".to_string(),
                "enable_sandbox".to_string(),
                SecurityResult::Success,
                format!("Sandbox enabled for context {}", context_id)
            );

            Ok(())
        } else {
            Err("Security context not found")
        }
    }

    /// Create ACL for a resource
    pub fn create_acl(
        &mut self,
        resource: String,
        owner: UserId,
        group: GroupId,
        permissions: PermissionFlags,
    ) -> Result<(), &'static str> {
        let mut acl = AccessControlList::new(owner, group);
        acl.default_permissions = permissions;

        self.access_control_lists.insert(resource.clone(), acl);

        self.log_security_event(
            SecurityEventType::ConfigChange,
            owner, 0,
            resource,
            "create_acl".to_string(),
            SecurityResult::Success,
            "ACL created".to_string()
        );

        Ok(())
    }

    /// Log security event
    fn log_security_event(
        &mut self,
        event_type: SecurityEventType,
        user_id: UserId,
        process_id: u32,
        resource: String,
        action: String,
        result: SecurityResult,
        details: String,
    ) {
        let entry = AuditLogEntry {
            timestamp: crate::time::uptime_ms(),
            event_type,
            user_id,
            process_id,
            resource,
            action,
            result,
            details,
        };

        self.audit_log.push(entry);

        // Keep audit log size manageable
        if self.audit_log.len() > 10000 {
            self.audit_log.drain(0..1000);
        }
    }

    /// Get security statistics
    pub fn get_security_stats(&self) -> SecurityStatistics {
        let mut stats = SecurityStatistics {
            total_contexts: self.security_contexts.len() as u32,
            total_capabilities: self.capabilities.len() as u32,
            total_acls: self.access_control_lists.len() as u32,
            audit_entries: self.audit_log.len() as u32,
            sandboxed_contexts: 0,
            failed_auth_attempts: 0,
            security_violations: 0,
        };

        // Count sandboxed contexts
        for context in self.security_contexts.values() {
            if context.sandbox_enabled {
                stats.sandboxed_contexts += 1;
            }
        }

        // Count failed auth attempts
        for &failures in self.failed_auth_attempts.values() {
            stats.failed_auth_attempts += failures;
        }

        // Count security violations
        for entry in &self.audit_log {
            match entry.result {
                SecurityResult::Failure | SecurityResult::Blocked => {
                    stats.security_violations += 1;
                }
                _ => {}
            }
        }

        stats
    }

    /// Get audit log entries
    pub fn get_audit_log(&self, max_entries: usize) -> &[AuditLogEntry] {
        let start = self.audit_log.len().saturating_sub(max_entries);
        &self.audit_log[start..]
    }
}

/// Security statistics structure
#[derive(Debug, Clone)]
pub struct SecurityStatistics {
    pub total_contexts: u32,
    pub total_capabilities: u32,
    pub total_acls: u32,
    pub audit_entries: u32,
    pub sandboxed_contexts: u32,
    pub failed_auth_attempts: u32,
    pub security_violations: u32,
}

/// Global security manager
lazy_static! {
    pub static ref SECURITY_MANAGER: Mutex<SecurityManager> = Mutex::new(SecurityManager::new());
}

/// Initialize security system
pub fn init() -> Result<(), &'static str> {
    SECURITY_MANAGER.lock().init()?;

    crate::status::register_subsystem("Security", crate::status::SystemStatus::Running,
                                     "Security framework operational");
    Ok(())
}

/// Create security context for process
pub fn create_security_context(user_id: UserId, group_id: GroupId) -> Result<SecurityContextId, &'static str> {
    SECURITY_MANAGER.lock().create_security_context(user_id, group_id)
}

/// Check access to resource
pub fn check_access(
    context_id: SecurityContextId,
    resource: &str,
    permissions: PermissionFlags,
) -> Result<bool, &'static str> {
    SECURITY_MANAGER.lock().check_access(context_id, resource, permissions)
}

/// Validate system call
pub fn validate_syscall(context_id: SecurityContextId, syscall_number: u32) -> Result<bool, &'static str> {
    SECURITY_MANAGER.lock().validate_syscall(context_id, syscall_number)
}

/// Enable sandbox for process
pub fn enable_sandbox(context_id: SecurityContextId, config: SandboxConfig) -> Result<(), &'static str> {
    SECURITY_MANAGER.lock().enable_sandbox(context_id, config)
}

/// Create capability
pub fn create_capability(
    capability_type: CapabilityType,
    target_resource: Option<String>,
    permissions: PermissionFlags,
    owner: UserId,
    group: GroupId,
) -> Result<CapabilityId, &'static str> {
    SECURITY_MANAGER.lock().create_capability(capability_type, target_resource, permissions, owner, group)
}

/// Grant capability to context
pub fn grant_capability(
    context_id: SecurityContextId,
    capability_id: CapabilityId,
    granter_context_id: SecurityContextId,
) -> Result<(), &'static str> {
    SECURITY_MANAGER.lock().grant_capability(context_id, capability_id, granter_context_id)
}

/// Get security statistics
pub fn get_security_statistics() -> SecurityStatistics {
    SECURITY_MANAGER.lock().get_security_stats()
}

/// Create ACL for resource
pub fn create_acl(
    resource: String,
    owner: UserId,
    group: GroupId,
    permissions: PermissionFlags,
) -> Result<(), &'static str> {
    SECURITY_MANAGER.lock().create_acl(resource, owner, group, permissions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_permission_flags() {
        let perms = PermissionFlags::read_write();
        assert!(perms.read);
        assert!(perms.write);
        assert!(!perms.execute);

        let full_perms = PermissionFlags::full();
        assert!(full_perms.admin);
    }

    #[test_case]
    fn test_security_context() {
        let mut context = SecurityContext::new(1, 100, 200);
        assert_eq!(context.user_id, 100);
        assert_eq!(context.group_id, 200);
        assert!(!context.has_capability(1000));

        context.add_capability(1000);
        assert!(context.has_capability(1000));
    }

    #[test_case]
    fn test_acl_creation() {
        let acl = AccessControlList::new(0, 0);
        assert_eq!(acl.owner, 0);
        assert_eq!(acl.group, 0);
    }

    #[test_case]
    fn test_sandbox_config() {
        let config = SandboxConfig::default();
        assert!(!config.allow_network);
        assert_eq!(config.memory_limit, 64 * 1024 * 1024);
        assert_eq!(config.max_processes, 1);
    }
}

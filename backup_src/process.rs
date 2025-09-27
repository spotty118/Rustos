//! Process Management System for RustOS
//!
//! This module provides comprehensive process management including:
//! - Process creation, execution, and termination
//! - Process scheduling and context switching
//! - Process memory management integration
//! - Inter-process communication support
//! - Process synchronization primitives
//! - Parent-child process relationships
//! - Resource tracking and cleanup

use alloc::{
    boxed::Box,
    collections::{BTreeMap, VecDeque},
    string::{String, ToString},
    vec::Vec,
};
use core::{
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    fmt,
};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use x86_64::VirtAddr;

use crate::{
    memory::{MemoryRegionType, MemoryProtection, VirtualMemoryRegion},
    task::{TaskId, TaskPriority},
};

/// Unique process identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessId(u64);

impl ProcessId {
    pub fn new() -> Self {
        static NEXT_PID: AtomicU64 = AtomicU64::new(1);
        ProcessId(NEXT_PID.fetch_add(1, Ordering::Relaxed))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }

    pub fn kernel() -> Self {
        ProcessId(0)
    }

    pub fn from_u64(val: u64) -> Self {
        ProcessId(val)
    }
}

impl fmt::Display for ProcessId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PID:{}", self.0)
    }
}

/// Process state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    /// Process is ready to run
    Ready,
    /// Process is currently running
    Running,
    /// Process is blocked waiting for I/O or other resource
    Blocked,
    /// Process is suspended (can be resumed)
    Suspended,
    /// Process has terminated normally
    Terminated,
    /// Process has been killed
    Killed,
    /// Process is being created
    Creating,
}

impl fmt::Display for ProcessState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcessState::Ready => write!(f, "READY"),
            ProcessState::Running => write!(f, "RUNNING"),
            ProcessState::Blocked => write!(f, "BLOCKED"),
            ProcessState::Suspended => write!(f, "SUSPENDED"),
            ProcessState::Terminated => write!(f, "TERMINATED"),
            ProcessState::Killed => write!(f, "KILLED"),
            ProcessState::Creating => write!(f, "CREATING"),
        }
    }
}

/// Process priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProcessPriority {
    Idle = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    RealTime = 4,
    Kernel = 5,
}

impl Default for ProcessPriority {
    fn default() -> Self {
        ProcessPriority::Normal
    }
}

/// Process execution context
#[derive(Debug, Clone)]
pub struct ProcessContext {
    /// CPU register state (simplified)
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub rsp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub rflags: u64,
}

impl Default for ProcessContext {
    fn default() -> Self {
        Self {
            rax: 0, rbx: 0, rcx: 0, rdx: 0,
            rsi: 0, rdi: 0, rbp: 0, rsp: 0,
            r8: 0, r9: 0, r10: 0, r11: 0,
            r12: 0, r13: 0, r14: 0, r15: 0,
            rip: 0, rflags: 0x200, // Enable interrupts by default
        }
    }
}

/// Process resource limits
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_memory: usize,
    pub max_open_files: usize,
    pub max_cpu_time: u64,
    pub max_children: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory: 64 * 1024 * 1024, // 64MB default
            max_open_files: 1024,
            max_cpu_time: u64::MAX,
            max_children: 32,
        }
    }
}

/// Process statistics
#[derive(Debug, Clone, Default)]
pub struct ProcessStats {
    pub cpu_time_user: u64,
    pub cpu_time_system: u64,
    pub memory_usage: usize,
    pub page_faults: u64,
    pub context_switches: u64,
    pub started_at: u64,
    pub last_scheduled: u64,
}

/// File descriptor table entry
#[derive(Debug, Clone)]
pub struct FileDescriptor {
    pub file_id: u64,
    pub flags: u32,
    pub position: u64,
}

/// Process Control Block (PCB)
#[derive(Debug)]
pub struct Process {
    /// Process identifier
    pub pid: ProcessId,
    /// Parent process ID
    pub parent_pid: Option<ProcessId>,
    /// Process name
    pub name: String,
    /// Current process state
    pub state: ProcessState,
    /// Process priority
    pub priority: ProcessPriority,
    /// CPU execution context
    pub context: ProcessContext,
    /// Virtual memory regions
    pub memory_regions: Vec<VirtualMemoryRegion>,
    /// File descriptor table
    pub file_descriptors: BTreeMap<u32, FileDescriptor>,
    /// Child processes
    pub children: Vec<ProcessId>,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Process statistics
    pub stats: ProcessStats,
    /// Exit code (if terminated)
    pub exit_code: Option<i32>,
    /// Working directory
    pub working_directory: String,
    /// Environment variables
    pub environment: BTreeMap<String, String>,
    /// Associated task ID for async execution
    pub task_id: Option<TaskId>,
    /// Process creation time
    pub created_at: u64,
    /// Last CPU time update
    pub last_cpu_update: u64,
}

impl Process {
    /// Create a new process
    pub fn new(name: String, parent_pid: Option<ProcessId>) -> Self {
        let pid = ProcessId::new();
        let created_at = crate::time::get_ticks();

        Self {
            pid,
            parent_pid,
            name,
            state: ProcessState::Creating,
            priority: ProcessPriority::default(),
            context: ProcessContext::default(),
            memory_regions: Vec::new(),
            file_descriptors: BTreeMap::new(),
            children: Vec::new(),
            limits: ResourceLimits::default(),
            stats: ProcessStats {
                started_at: created_at,
                ..Default::default()
            },
            exit_code: None,
            working_directory: "/".to_string(),
            environment: BTreeMap::new(),
            task_id: None,
            created_at,
            last_cpu_update: created_at,
        }
    }

    /// Check if process can allocate more memory
    pub fn can_allocate_memory(&self, size: usize) -> bool {
        self.stats.memory_usage + size <= self.limits.max_memory
    }

    /// Add memory region to process
    pub fn add_memory_region(&mut self, region: VirtualMemoryRegion) -> Result<(), &'static str> {
        if !self.can_allocate_memory(region.size) {
            return Err("Memory limit exceeded");
        }

        self.memory_regions.push(region);
        self.stats.memory_usage += region.size;
        Ok(())
    }

    /// Remove memory region from process
    pub fn remove_memory_region(&mut self, start_addr: VirtAddr) -> Result<(), &'static str> {
        if let Some(pos) = self.memory_regions.iter().position(|r| r.start == start_addr) {
            let region = self.memory_regions.remove(pos);
            self.stats.memory_usage = self.stats.memory_usage.saturating_sub(region.size);
            Ok(())
        } else {
            Err("Memory region not found")
        }
    }

    /// Allocate file descriptor
    pub fn allocate_fd(&mut self) -> Result<u32, &'static str> {
        if self.file_descriptors.len() >= self.limits.max_open_files {
            return Err("File descriptor limit exceeded");
        }

        // Find lowest available FD number
        for fd in 3..u32::MAX {
            if !self.file_descriptors.contains_key(&fd) {
                return Ok(fd);
            }
        }

        Err("No file descriptors available")
    }

    /// Add file descriptor
    pub fn add_file_descriptor(&mut self, file_id: u64, flags: u32) -> Result<u32, &'static str> {
        let fd = self.allocate_fd()?;
        self.file_descriptors.insert(fd, FileDescriptor {
            file_id,
            flags,
            position: 0,
        });
        Ok(fd)
    }

    /// Remove file descriptor
    pub fn remove_file_descriptor(&mut self, fd: u32) -> Option<FileDescriptor> {
        self.file_descriptors.remove(&fd)
    }

    /// Add child process
    pub fn add_child(&mut self, child_pid: ProcessId) -> Result<(), &'static str> {
        if self.children.len() >= self.limits.max_children {
            return Err("Child process limit exceeded");
        }

        self.children.push(child_pid);
        Ok(())
    }

    /// Remove child process
    pub fn remove_child(&mut self, child_pid: ProcessId) {
        self.children.retain(|&pid| pid != child_pid);
    }

    /// Update CPU time statistics
    pub fn update_cpu_time(&mut self, user_time: u64, system_time: u64) {
        self.stats.cpu_time_user += user_time;
        self.stats.cpu_time_system += system_time;
        self.last_cpu_update = crate::time::get_ticks();
    }

    /// Get total CPU time
    pub fn total_cpu_time(&self) -> u64 {
        self.stats.cpu_time_user + self.stats.cpu_time_system
    }

    /// Check if process has exceeded CPU time limit
    pub fn cpu_limit_exceeded(&self) -> bool {
        self.total_cpu_time() > self.limits.max_cpu_time
    }

    /// Get process age in ticks
    pub fn age(&self) -> u64 {
        crate::time::get_ticks() - self.created_at
    }
}

/// Process scheduler
#[derive(Debug)]
pub struct ProcessScheduler {
    /// Ready queues for each priority level
    ready_queues: [VecDeque<ProcessId>; 6],
    /// Currently running process
    current_process: Option<ProcessId>,
    /// Scheduling statistics
    context_switches: AtomicU64,
    /// Time slice for each priority (in ticks)
    time_slices: [u64; 6],
    /// Current process remaining time slice
    current_time_slice: u64,
}

impl ProcessScheduler {
    pub fn new() -> Self {
        Self {
            ready_queues: [
                VecDeque::new(), // Idle
                VecDeque::new(), // Low
                VecDeque::new(), // Normal
                VecDeque::new(), // High
                VecDeque::new(), // RealTime
                VecDeque::new(), // Kernel
            ],
            current_process: None,
            context_switches: AtomicU64::new(0),
            time_slices: [100, 50, 20, 10, 5, 2], // Ticks per time slice
            current_time_slice: 0,
        }
    }

    /// Add process to ready queue
    pub fn schedule_process(&mut self, pid: ProcessId, priority: ProcessPriority) {
        let queue_index = priority as usize;
        self.ready_queues[queue_index].push_back(pid);
    }

    /// Remove process from ready queues
    pub fn unschedule_process(&mut self, pid: ProcessId) {
        for queue in &mut self.ready_queues {
            queue.retain(|&p| p != pid);
        }
    }

    /// Get next process to run
    pub fn next_process(&mut self) -> Option<ProcessId> {
        // Check higher priority queues first
        for (priority, queue) in self.ready_queues.iter_mut().enumerate().rev() {
            if let Some(pid) = queue.pop_front() {
                self.current_process = Some(pid);
                self.current_time_slice = self.time_slices[priority];
                self.context_switches.fetch_add(1, Ordering::Relaxed);
                return Some(pid);
            }
        }

        None
    }

    /// Handle timer tick for preemptive scheduling
    pub fn tick(&mut self) -> Option<ProcessId> {
        if self.current_time_slice > 0 {
            self.current_time_slice -= 1;
            return self.current_process;
        }

        // Time slice expired, preempt current process
        if let Some(current_pid) = self.current_process.take() {
            // Get current process priority and re-queue it
            if let Some(process_manager) = PROCESS_MANAGER.read().as_ref() {
                if let Some(process) = process_manager.processes.read().get(&current_pid) {
                    if process.state == ProcessState::Running {
                        self.schedule_process(current_pid, process.priority);
                    }
                }
            }
        }

        // Select next process
        self.next_process()
    }

    /// Get current running process
    pub fn current_process(&self) -> Option<ProcessId> {
        self.current_process
    }

    /// Get context switch count
    pub fn context_switches(&self) -> u64 {
        self.context_switches.load(Ordering::Relaxed)
    }
}

/// Process manager
#[derive(Debug)]
pub struct ProcessManager {
    /// All processes in the system
    pub processes: RwLock<BTreeMap<ProcessId, Process>>,
    /// Process scheduler
    pub scheduler: Mutex<ProcessScheduler>,
    /// Process creation counter
    process_count: AtomicUsize,
    /// Orphaned processes waiting for cleanup
    orphaned_processes: Mutex<Vec<ProcessId>>,
}

impl ProcessManager {
    pub fn new() -> Self {
        Self {
            processes: RwLock::new(BTreeMap::new()),
            scheduler: Mutex::new(ProcessScheduler::new()),
            process_count: AtomicUsize::new(0),
            orphaned_processes: Mutex::new(Vec::new()),
        }
    }

    /// Create a new process
    pub fn create_process(
        &self,
        name: String,
        parent_pid: Option<ProcessId>,
        priority: ProcessPriority,
    ) -> Result<ProcessId, &'static str> {
        let mut process = Process::new(name, parent_pid);
        let pid = process.pid;

        // Set up standard file descriptors (stdin, stdout, stderr)
        process.file_descriptors.insert(0, FileDescriptor { file_id: 0, flags: 0, position: 0 });
        process.file_descriptors.insert(1, FileDescriptor { file_id: 1, flags: 0, position: 0 });
        process.file_descriptors.insert(2, FileDescriptor { file_id: 2, flags: 0, position: 0 });

        process.priority = priority;
        process.state = ProcessState::Ready;

        // Add to parent's children list
        if let Some(parent_pid) = parent_pid {
            if let Some(parent) = self.processes.write().get_mut(&parent_pid) {
                parent.add_child(pid)?;
            }
        }

        // Insert process and schedule it
        self.processes.write().insert(pid, process);
        self.scheduler.lock().schedule_process(pid, priority);
        self.process_count.fetch_add(1, Ordering::Relaxed);

        crate::println!("Created process {} with PID {}", name, pid);
        Ok(pid)
    }

    /// Terminate a process
    pub fn terminate_process(&self, pid: ProcessId, exit_code: i32) -> Result<(), &'static str> {
        let mut processes = self.processes.write();

        if let Some(process) = processes.get_mut(&pid) {
            if process.state == ProcessState::Terminated || process.state == ProcessState::Killed {
                return Err("Process already terminated");
            }

            // Update process state
            process.state = ProcessState::Terminated;
            process.exit_code = Some(exit_code);

            // Remove from scheduler
            self.scheduler.lock().unschedule_process(pid);

            // Handle children - make them orphaned
            let children = process.children.clone();
            for child_pid in children {
                if let Some(child) = processes.get_mut(&child_pid) {
                    child.parent_pid = None;
                    self.orphaned_processes.lock().push(child_pid);
                }
            }

            // Remove from parent's children list
            if let Some(parent_pid) = process.parent_pid {
                if let Some(parent) = processes.get_mut(&parent_pid) {
                    parent.remove_child(pid);
                }
            }

            crate::println!("Process {} terminated with exit code {}", pid, exit_code);
            Ok(())
        } else {
            Err("Process not found")
        }
    }

    /// Kill a process forcefully
    pub fn kill_process(&self, pid: ProcessId) -> Result<(), &'static str> {
        self.terminate_process(pid, -9)
    }

    /// Get process by PID
    pub fn get_process(&self, pid: ProcessId) -> Option<Process> {
        self.processes.read().get(&pid).cloned()
    }

    /// List all processes
    pub fn list_processes(&self) -> Vec<(ProcessId, String, ProcessState, ProcessPriority)> {
        self.processes.read()
            .values()
            .map(|p| (p.pid, p.name.clone(), p.state, p.priority))
            .collect()
    }

    /// Schedule next process
    pub fn schedule(&self) -> Option<ProcessId> {
        let next_pid = self.scheduler.lock().next_process();

        if let Some(pid) = next_pid {
            if let Some(process) = self.processes.write().get_mut(&pid) {
                process.state = ProcessState::Running;
                process.stats.last_scheduled = crate::time::get_ticks();
            }
        }

        next_pid
    }

    /// Handle timer tick for scheduling
    pub fn timer_tick(&self) -> Option<ProcessId> {
        self.scheduler.lock().tick()
    }

    /// Block a process
    pub fn block_process(&self, pid: ProcessId) -> Result<(), &'static str> {
        if let Some(process) = self.processes.write().get_mut(&pid) {
            if process.state == ProcessState::Running {
                process.state = ProcessState::Blocked;
                self.scheduler.lock().unschedule_process(pid);
                Ok(())
            } else {
                Err("Process is not running")
            }
        } else {
            Err("Process not found")
        }
    }

    /// Unblock a process
    pub fn unblock_process(&self, pid: ProcessId) -> Result<(), &'static str> {
        if let Some(process) = self.processes.write().get_mut(&pid) {
            if process.state == ProcessState::Blocked {
                process.state = ProcessState::Ready;
                self.scheduler.lock().schedule_process(pid, process.priority);
                Ok(())
            } else {
                Err("Process is not blocked")
            }
        } else {
            Err("Process not found")
        }
    }

    /// Get process statistics
    pub fn get_system_stats(&self) -> ProcessSystemStats {
        let processes = self.processes.read();
        let scheduler = self.scheduler.lock();

        let mut stats = ProcessSystemStats::default();
        stats.total_processes = processes.len();
        stats.context_switches = scheduler.context_switches();

        for process in processes.values() {
            match process.state {
                ProcessState::Running => stats.running_processes += 1,
                ProcessState::Ready => stats.ready_processes += 1,
                ProcessState::Blocked => stats.blocked_processes += 1,
                ProcessState::Suspended => stats.suspended_processes += 1,
                ProcessState::Terminated => stats.terminated_processes += 1,
                ProcessState::Killed => stats.killed_processes += 1,
                ProcessState::Creating => stats.creating_processes += 1,
            }

            stats.total_memory_usage += process.stats.memory_usage;
            stats.total_cpu_time += process.total_cpu_time();
        }

        stats
    }

    /// Cleanup terminated processes
    pub fn cleanup_terminated_processes(&self) {
        let mut processes = self.processes.write();
        let pids_to_remove: Vec<ProcessId> = processes
            .values()
            .filter(|p| p.state == ProcessState::Terminated || p.state == ProcessState::Killed)
            .map(|p| p.pid)
            .collect();

        for pid in pids_to_remove {
            processes.remove(&pid);
            self.process_count.fetch_sub(1, Ordering::Relaxed);
            crate::println!("Cleaned up process {}", pid);
        }

        // Also cleanup orphaned processes
        let mut orphaned = self.orphaned_processes.lock();
        orphaned.retain(|&pid| processes.contains_key(&pid));
    }

    /// Fork a new process (simplified implementation)
    pub fn fork_process(&self, parent_pid: ProcessId) -> Result<ProcessId, &'static str> {
        let parent = self.get_process(parent_pid).ok_or("Parent process not found")?;

        let child_name = format!("{}-child", parent.name);
        let child_pid = self.create_process(child_name, Some(parent_pid), parent.priority)?;

        // Copy parent's memory regions (simplified - in real OS this would be copy-on-write)
        if let Some(child) = self.processes.write().get_mut(&child_pid) {
            for region in &parent.memory_regions {
                let _ = child.add_memory_region(region.clone());
            }
            child.environment = parent.environment.clone();
            child.working_directory = parent.working_directory.clone();
        }

        Ok(child_pid)
    }
}

/// System-wide process statistics
#[derive(Debug, Default)]
pub struct ProcessSystemStats {
    pub total_processes: usize,
    pub running_processes: usize,
    pub ready_processes: usize,
    pub blocked_processes: usize,
    pub suspended_processes: usize,
    pub terminated_processes: usize,
    pub killed_processes: usize,
    pub creating_processes: usize,
    pub context_switches: u64,
    pub total_memory_usage: usize,
    pub total_cpu_time: u64,
}

lazy_static! {
    static ref PROCESS_MANAGER: RwLock<Option<ProcessManager>> = RwLock::new(None);
}

/// Initialize the process management system
pub fn init() {
    let process_manager = ProcessManager::new();

    // Create kernel process
    let _ = process_manager.create_process(
        "kernel".to_string(),
        None,
        ProcessPriority::Kernel,
    );

    *PROCESS_MANAGER.write() = Some(process_manager);
    crate::println!("Process management system initialized");
}

/// Get the global process manager
pub fn get_process_manager() -> Option<&'static ProcessManager> {
    unsafe {
        PROCESS_MANAGER.read().as_ref().map(|pm| core::mem::transmute(pm))
    }
}

/// Create a new process (high-level interface)
pub fn create_process(
    name: String,
    parent_pid: Option<ProcessId>,
    priority: ProcessPriority,
) -> Result<ProcessId, &'static str> {
    get_process_manager()
        .ok_or("Process manager not initialized")?
        .create_process(name, parent_pid, priority)
}

/// Get current running process ID
pub fn current_process_id() -> Option<ProcessId> {
    get_process_manager()?.scheduler.lock().current_process()
}

/// Schedule next process
pub fn schedule_next() -> Option<ProcessId> {
    get_process_manager()?.schedule()
}

/// Handle timer tick for process scheduling
pub fn handle_timer_tick() -> Option<ProcessId> {
    get_process_manager()?.timer_tick()
}

/// Terminate current process
pub fn exit_current_process(exit_code: i32) -> Result<(), &'static str> {
    let pid = current_process_id().ok_or("No current process")?;
    get_process_manager()
        .ok_or("Process manager not initialized")?
        .terminate_process(pid, exit_code)
}

/// Get process statistics
pub fn get_process_stats() -> Option<ProcessSystemStats> {
    get_process_manager().map(|pm| pm.get_system_stats())
}

/// Demonstrate process management
pub fn demonstrate_process_management() {
    crate::println!("=== Process Management Demonstration ===");

    if let Some(pm) = get_process_manager() {
        // Create some test processes
        let _ = pm.create_process("test_process_1".to_string(), None, ProcessPriority::Normal);
        let _ = pm.create_process("test_process_2".to_string(), None, ProcessPriority::High);
        let _ = pm.create_process("background_task".to_string(), None, ProcessPriority::Low);

        // Show process list
        crate::println!("Active processes:");
        for (pid, name, state, priority) in pm.list_processes() {
            crate::println!("  {} | {} | {} | {:?}", pid, name, state, priority);
        }

        // Show system statistics
        let stats = pm.get_system_stats();
        crate::println!("System Stats: {} total, {} running, {} ready",
                       stats.total_processes, stats.running_processes, stats.ready_processes);

        crate::println!("Context switches: {}", stats.context_switches);

        // Cleanup demonstration
        pm.cleanup_terminated_processes();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_process_creation() {
        let mut process = Process::new("test_process".to_string(), None);
        assert_eq!(process.state, ProcessState::Creating);
        assert_eq!(process.name, "test_process");
        assert!(process.parent_pid.is_none());
    }

    #[test_case]
    fn test_process_scheduler() {
        let mut scheduler = ProcessScheduler::new();
        let pid1 = ProcessId::new();
        let pid2 = ProcessId::new();

        scheduler.schedule_process(pid1, ProcessPriority::Normal);
        scheduler.schedule_process(pid2, ProcessPriority::High);

        // High priority process should be scheduled first
        assert_eq!(scheduler.next_process(), Some(pid2));
        assert_eq!(scheduler.next_process(), Some(pid1));
    }

    #[test_case]
    fn test_file_descriptor_management() {
        let mut process = Process::new("test".to_string(), None);

        let fd = process.add_file_descriptor(123, 0).expect("Failed to add FD");
        assert!(fd >= 3); // Should skip stdin, stdout, stderr

        let removed_fd = process.remove_file_descriptor(fd);
        assert!(removed_fd.is_some());
        assert_eq!(removed_fd.unwrap().file_id, 123);
    }
}

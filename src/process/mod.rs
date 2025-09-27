//! Process Management Module
//!
//! This module provides comprehensive process management functionality for RustOS,
//! including process control blocks, scheduling, system calls, and context switching.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use crate::{String, ToString};
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use spin::{Mutex, RwLock};

pub mod scheduler;
pub mod syscalls;
pub mod context;
pub mod sync;
pub mod integration;

/// Process ID type
pub type Pid = u32;

/// Maximum number of processes that can exist simultaneously
pub const MAX_PROCESSES: usize = 1024;

/// Process states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    /// Process is ready to run
    Ready,
    /// Process is currently running
    Running,
    /// Process is blocked waiting for I/O or resources
    Blocked,
    /// Process has terminated but PCB still exists (waiting for parent to collect exit status)
    Zombie,
    /// Process has been completely cleaned up
    Dead,
}

/// Process priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Real-time priority (highest)
    RealTime = 0,
    /// High priority
    High = 1,
    /// Normal priority (default)
    Normal = 2,
    /// Low priority
    Low = 3,
    /// Idle priority (lowest)
    Idle = 4,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

/// CPU register state for context switching
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CpuContext {
    // General purpose registers
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

    // Control registers
    pub rip: u64,
    pub rflags: u64,

    // Segment registers
    pub cs: u16,
    pub ds: u16,
    pub es: u16,
    pub fs: u16,
    pub gs: u16,
    pub ss: u16,
}

impl Default for CpuContext {
    fn default() -> Self {
        Self {
            rax: 0, rbx: 0, rcx: 0, rdx: 0,
            rsi: 0, rdi: 0, rbp: 0, rsp: 0,
            r8: 0, r9: 0, r10: 0, r11: 0,
            r12: 0, r13: 0, r14: 0, r15: 0,
            rip: 0, rflags: 0x202, // Enable interrupts by default
            cs: 0x08, ds: 0x10, es: 0x10, fs: 0x10, gs: 0x10, ss: 0x10,
        }
    }
}

/// Memory management information for a process
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Page directory physical address
    pub page_directory: u64,
    /// Virtual memory start address
    pub vm_start: u64,
    /// Virtual memory size
    pub vm_size: u64,
    /// Heap start address
    pub heap_start: u64,
    /// Heap size
    pub heap_size: u64,
    /// Stack start address
    pub stack_start: u64,
    /// Stack size
    pub stack_size: u64,
}

impl Default for MemoryInfo {
    fn default() -> Self {
        Self {
            page_directory: 0,
            vm_start: 0x400000,  // 4MB
            vm_size: 0x100000,   // 1MB default
            heap_start: 0x500000, // 5MB
            heap_size: 0x100000,  // 1MB
            stack_start: 0x7FFFFF000, // Near top of user space
            stack_size: 0x2000,   // 8KB default stack
        }
    }
}

/// Process Control Block (PCB)
#[derive(Debug, Clone)]
pub struct ProcessControlBlock {
    /// Process ID
    pub pid: Pid,
    /// Parent process ID
    pub parent_pid: Option<Pid>,
    /// Process state
    pub state: ProcessState,
    /// Process priority
    pub priority: Priority,
    /// CPU context for context switching
    pub context: CpuContext,
    /// Memory management information
    pub memory: MemoryInfo,
    /// Process name
    pub name: [u8; 32],
    /// CPU time used (in ticks)
    pub cpu_time: u64,
    /// Time when process was created
    pub creation_time: u64,
    /// Exit status (valid only when state is Zombie)
    pub exit_status: Option<i32>,
    /// File descriptor table
    pub fd_table: BTreeMap<u32, FileDescriptor>,
    /// Next file descriptor number
    pub next_fd: u32,
    /// Process scheduling information
    pub sched_info: SchedulingInfo,
}

/// File descriptor information
#[derive(Debug, Clone)]
pub struct FileDescriptor {
    pub fd_type: FileDescriptorType,
    pub flags: u32,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub enum FileDescriptorType {
    StandardInput,
    StandardOutput,
    StandardError,
    File { path: [u8; 256] },
    Socket { socket_id: u32 },
    Pipe { pipe_id: u32 },
}

/// Scheduling-specific information
#[derive(Debug, Clone)]
pub struct SchedulingInfo {
    /// Time slice remaining (for round-robin)
    pub time_slice: u32,
    /// Default time slice for this process
    pub default_time_slice: u32,
    /// Number of times process has been scheduled
    pub schedule_count: u64,
    /// Last time process was scheduled
    pub last_scheduled: u64,
    /// CPU affinity mask
    pub cpu_affinity: u64,
}

impl ProcessControlBlock {
    /// Create a new PCB with the given PID and parent
    pub fn new(pid: Pid, parent_pid: Option<Pid>, name: &str) -> Self {
        let mut pcb = Self {
            pid,
            parent_pid,
            state: ProcessState::Ready,
            priority: Priority::default(),
            context: CpuContext::default(),
            memory: MemoryInfo::default(),
            name: [0; 32],
            cpu_time: 0,
            creation_time: get_system_time(),
            exit_status: None,
            fd_table: BTreeMap::new(),
            next_fd: 3, // 0, 1, 2 reserved for stdin, stdout, stderr
            sched_info: SchedulingInfo {
                time_slice: 10, // 10ms default
                default_time_slice: 10,
                schedule_count: 0,
                last_scheduled: 0,
                cpu_affinity: 0xFFFFFFFFFFFFFFFF, // All CPUs
            },
        };

        // Set process name
        let name_bytes = name.as_bytes();
        let copy_len = core::cmp::min(name_bytes.len(), 31);
        pcb.name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

        // Initialize standard file descriptors
        pcb.fd_table.insert(0, FileDescriptor {
            fd_type: FileDescriptorType::StandardInput,
            flags: 0,
            offset: 0,
        });
        pcb.fd_table.insert(1, FileDescriptor {
            fd_type: FileDescriptorType::StandardOutput,
            flags: 0,
            offset: 0,
        });
        pcb.fd_table.insert(2, FileDescriptor {
            fd_type: FileDescriptorType::StandardError,
            flags: 0,
            offset: 0,
        });

        pcb
    }

    /// Get process name as string
    pub fn name_str(&self) -> &str {
        let name_len = self.name.iter().position(|&x| x == 0).unwrap_or(32);
        core::str::from_utf8(&self.name[..name_len]).unwrap_or("invalid")
    }

    /// Set process state
    pub fn set_state(&mut self, state: ProcessState) {
        self.state = state;
    }

    /// Check if process is runnable
    pub fn is_runnable(&self) -> bool {
        matches!(self.state, ProcessState::Ready)
    }

    /// Allocate a new file descriptor
    pub fn allocate_fd(&mut self, fd_type: FileDescriptorType) -> u32 {
        let fd = self.next_fd;
        self.fd_table.insert(fd, FileDescriptor {
            fd_type,
            flags: 0,
            offset: 0,
        });
        self.next_fd += 1;
        fd
    }

    /// Close a file descriptor
    pub fn close_fd(&mut self, fd: u32) -> Result<(), &'static str> {
        if fd < 3 {
            return Err("Cannot close standard file descriptors");
        }
        self.fd_table.remove(&fd).ok_or("Invalid file descriptor")?;
        Ok(())
    }
}

/// Process Manager - central coordinator for all process operations
pub struct ProcessManager {
    /// All processes in the system
    processes: RwLock<BTreeMap<Pid, ProcessControlBlock>>,
    /// Currently running process ID
    current_process: AtomicU32,
    /// Next PID to allocate
    next_pid: AtomicU32,
    /// Process count
    process_count: AtomicUsize,
    /// Scheduler instance
    scheduler: Mutex<scheduler::Scheduler>,
    /// System call dispatcher
    syscall_dispatcher: Mutex<syscalls::SyscallDispatcher>,
}

impl ProcessManager {
    /// Create a new process manager
    pub const fn new() -> Self {
        Self {
            processes: RwLock::new(BTreeMap::new()),
            current_process: AtomicU32::new(0),
            next_pid: AtomicU32::new(1),
            process_count: AtomicUsize::new(0),
            scheduler: Mutex::new(scheduler::Scheduler::new()),
            syscall_dispatcher: Mutex::new(syscalls::SyscallDispatcher::new()),
        }
    }

    /// Initialize the process manager with kernel process
    pub fn init(&self) -> Result<(), &'static str> {
        // Create kernel process (PID 0)
        let kernel_pcb = ProcessControlBlock::new(0, None, "kernel");

        {
            let mut processes = self.processes.write();
            processes.insert(0, kernel_pcb);
        }

        self.process_count.store(1, Ordering::SeqCst);
        self.current_process.store(0, Ordering::SeqCst);

        // Initialize scheduler
        {
            let mut scheduler = self.scheduler.lock();
            scheduler.init()?;
            scheduler.add_process(0, Priority::RealTime)?;
        }

        Ok(())
    }

    /// Create a new process
    pub fn create_process(&self, name: &str, parent_pid: Option<Pid>, priority: Priority) -> Result<Pid, &'static str> {
        let pid = self.next_pid.fetch_add(1, Ordering::SeqCst);

        if self.process_count.load(Ordering::SeqCst) >= MAX_PROCESSES {
            return Err("Maximum process count exceeded");
        }

        let mut pcb = ProcessControlBlock::new(pid, parent_pid, name);
        pcb.priority = priority;

        {
            let mut processes = self.processes.write();
            processes.insert(pid, pcb);
        }

        self.process_count.fetch_add(1, Ordering::SeqCst);

        // Add to scheduler
        {
            let mut scheduler = self.scheduler.lock();
            scheduler.add_process(pid, priority)?;
        }

        Ok(pid)
    }

    /// Terminate a process
    pub fn terminate_process(&self, pid: Pid, exit_status: i32) -> Result<(), &'static str> {
        {
            let mut processes = self.processes.write();
            if let Some(pcb) = processes.get_mut(&pid) {
                pcb.set_state(ProcessState::Zombie);
                pcb.exit_status = Some(exit_status);
            } else {
                return Err("Process not found");
            }
        }

        // Remove from scheduler
        {
            let mut scheduler = self.scheduler.lock();
            scheduler.remove_process(pid)?;
        }

        Ok(())
    }

    /// Get process information
    pub fn get_process(&self, pid: Pid) -> Option<ProcessControlBlock> {
        let processes = self.processes.read();
        processes.get(&pid).cloned()
    }

    /// Get current running process ID
    pub fn current_process(&self) -> Pid {
        self.current_process.load(Ordering::SeqCst)
    }

    /// Get process count
    pub fn process_count(&self) -> usize {
        self.process_count.load(Ordering::SeqCst)
    }

    /// Schedule next process (called by timer interrupt)
    pub fn schedule(&self) -> Result<Option<Pid>, &'static str> {
        let mut scheduler = self.scheduler.lock();
        scheduler.schedule()
    }

    /// Update current process
    pub fn set_current_process(&self, pid: Pid) {
        self.current_process.store(pid, Ordering::SeqCst);
    }

    /// Handle system call
    pub fn handle_syscall(&self, syscall_number: u64, args: &[u64]) -> Result<u64, &'static str> {
        let mut dispatcher = self.syscall_dispatcher.lock();
        dispatcher.dispatch(syscall_number, args, self)
    }

    /// Block current process
    pub fn block_process(&self, pid: Pid) -> Result<(), &'static str> {
        {
            let mut processes = self.processes.write();
            if let Some(pcb) = processes.get_mut(&pid) {
                pcb.set_state(ProcessState::Blocked);
            } else {
                return Err("Process not found");
            }
        }

        // Remove from scheduler ready queue
        {
            let mut scheduler = self.scheduler.lock();
            scheduler.block_process(pid)?;
        }

        Ok(())
    }

    /// Unblock a process
    pub fn unblock_process(&self, pid: Pid) -> Result<(), &'static str> {
        {
            let mut processes = self.processes.write();
            if let Some(pcb) = processes.get_mut(&pid) {
                pcb.set_state(ProcessState::Ready);
            } else {
                return Err("Process not found");
            }
        }

        // Add back to scheduler
        {
            let mut scheduler = self.scheduler.lock();
            let priority = {
                let processes = self.processes.read();
                processes.get(&pid).map(|p| p.priority).unwrap_or(Priority::Normal)
            };
            scheduler.add_process(pid, priority)?;
        }

        Ok(())
    }

    /// List all processes
    pub fn list_processes(&self) -> Vec<(Pid, String, ProcessState, Priority)> {
        let processes = self.processes.read();
        processes.iter().map(|(&pid, pcb)| {
            (pid, pcb.name_str().to_string(), pcb.state, pcb.priority)
        }).collect()
    }
}

/// Global process manager instance
static PROCESS_MANAGER: ProcessManager = ProcessManager::new();

/// Get the global process manager
pub fn get_process_manager() -> &'static ProcessManager {
    &PROCESS_MANAGER
}

/// Initialize the process management system
pub fn init() -> Result<(), &'static str> {
    // Initialize core process management
    PROCESS_MANAGER.init()?;

    // Initialize integration with other kernel systems
    integration::init()?;

    Ok(())
}

/// Simple system time counter (should be replaced with proper timer)
static SYSTEM_TIME: AtomicU64 = AtomicU64::new(0);

/// Get current system time in ticks
pub fn get_system_time() -> u64 {
    SYSTEM_TIME.load(Ordering::SeqCst)
}

/// Increment system time (called by timer interrupt)
pub fn tick_system_time() {
    SYSTEM_TIME.fetch_add(1, Ordering::SeqCst);
}

use core::sync::atomic::AtomicU64;

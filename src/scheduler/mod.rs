//! Preemptive Scheduler for RustOS
//!
//! This module implements a sophisticated preemptive scheduler with:
//! - Priority-based scheduling with multiple priority levels
//! - Time slicing for fair CPU allocation
//! - SMP support for multi-core systems
//! - Real-time task support
//! - Load balancing across CPU cores

use alloc::{collections::VecDeque, vec::Vec};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use x86_64::VirtAddr;

/// Process ID type
pub type Pid = u32;

/// Thread ID type
pub type Tid = u64;

/// CPU ID type
pub type CpuId = u8;

/// Process priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
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

impl Priority {
    /// Get time slice duration in milliseconds for this priority
    pub fn time_slice_ms(&self) -> u64 {
        match self {
            Priority::RealTime => 100,  // 100ms for real-time
            Priority::High => 50,       // 50ms for high priority
            Priority::Normal => 20,     // 20ms for normal
            Priority::Low => 10,        // 10ms for low priority
            Priority::Idle => 5,        // 5ms for idle
        }
    }

    /// Get the number of priority levels
    pub const fn count() -> usize {
        5
    }
}

/// Process state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    /// Process is ready to run
    Ready,
    /// Process is currently running
    Running,
    /// Process is blocked waiting for I/O or event
    Blocked,
    /// Process is sleeping
    Sleeping,
    /// Process has terminated
    Terminated,
    /// Process is being created
    Creating,
}

/// CPU registers state for context switching
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CpuState {
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
    pub cs: u64,
    pub ss: u64,
    
    // Segment registers
    pub ds: u64,
    pub es: u64,
    pub fs: u64,
    pub gs: u64,
}

impl Default for CpuState {
    fn default() -> Self {
        Self {
            rax: 0, rbx: 0, rcx: 0, rdx: 0,
            rsi: 0, rdi: 0, rbp: 0, rsp: 0,
            r8: 0, r9: 0, r10: 0, r11: 0,
            r12: 0, r13: 0, r14: 0, r15: 0,
            rip: 0, rflags: 0x200, // Enable interrupts
            cs: 0x08, ss: 0x10,    // Kernel code/data segments
            ds: 0x10, es: 0x10, fs: 0x10, gs: 0x10,
        }
    }
}

/// Process Control Block (PCB)
#[derive(Debug)]
pub struct Process {
    /// Process ID
    pub pid: Pid,
    /// Parent process ID
    pub parent_pid: Option<Pid>,
    /// Process priority
    pub priority: Priority,
    /// Current state
    pub state: ProcessState,
    /// CPU state for context switching
    pub cpu_state: CpuState,
    /// Virtual memory space base
    pub memory_base: VirtAddr,
    /// Stack pointer
    pub stack_pointer: VirtAddr,
    /// Stack size in bytes
    pub stack_size: usize,
    /// Time when process was created
    pub creation_time: u64,
    /// Total CPU time used (in microseconds)
    pub cpu_time_used: u64,
    /// Last time this process was scheduled
    pub last_scheduled: u64,
    /// CPU affinity mask (which CPUs this process can run on)
    pub cpu_affinity: u64,
    /// Current CPU this process is running on
    pub current_cpu: Option<CpuId>,
    /// Process name
    pub name: [u8; 32],
}

impl Process {
    /// Create a new process
    pub fn new(pid: Pid, parent_pid: Option<Pid>, priority: Priority, name: &str) -> Self {
        let mut process_name = [0u8; 32];
        let name_bytes = name.as_bytes();
        let copy_len = core::cmp::min(name_bytes.len(), 31);
        process_name[..copy_len].copy_from_slice(&name_bytes[..copy_len]);

        Self {
            pid,
            parent_pid,
            priority,
            state: ProcessState::Creating,
            cpu_state: CpuState::default(),
            memory_base: VirtAddr::new(0),
            stack_pointer: VirtAddr::new(0),
            stack_size: 0,
            creation_time: get_system_time(),
            cpu_time_used: 0,
            last_scheduled: 0,
            cpu_affinity: u64::MAX, // Can run on any CPU by default
            current_cpu: None,
            name: process_name,
        }
    }

    /// Get process name as string
    pub fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(self.name.len());
        core::str::from_utf8(&self.name[..end]).unwrap_or("<invalid>")
    }

    /// Check if process can run on the given CPU
    pub fn can_run_on_cpu(&self, cpu_id: CpuId) -> bool {
        if cpu_id >= 64 {
            return false;
        }
        (self.cpu_affinity & (1 << cpu_id)) != 0
    }

    /// Set CPU affinity
    pub fn set_cpu_affinity(&mut self, cpu_mask: u64) {
        self.cpu_affinity = cpu_mask;
    }
}

/// Per-CPU scheduler state
#[derive(Debug)]
pub struct CpuScheduler {
    /// CPU ID
    pub cpu_id: CpuId,
    /// Currently running process
    pub current_process: Option<Pid>,
    /// Ready queues for each priority level
    pub ready_queues: [VecDeque<Pid>; Priority::count()],
    /// Time slice remaining for current process (in microseconds)
    pub time_slice_remaining: u64,
    /// Total processes scheduled on this CPU
    pub total_scheduled: u64,
    /// CPU utilization percentage (0-100)
    pub utilization: u8,
    /// Idle time in microseconds
    pub idle_time: u64,
}

impl CpuScheduler {
    /// Create a new CPU scheduler
    pub fn new(cpu_id: CpuId) -> Self {
        Self {
            cpu_id,
            current_process: None,
            ready_queues: [
                VecDeque::new(), VecDeque::new(), VecDeque::new(),
                VecDeque::new(), VecDeque::new()
            ],
            time_slice_remaining: 0,
            total_scheduled: 0,
            utilization: 0,
            idle_time: 0,
        }
    }

    /// Add a process to the ready queue
    pub fn enqueue_process(&mut self, pid: Pid, priority: Priority) {
        self.ready_queues[priority as usize].push_back(pid);
    }

    /// Get the next process to run
    pub fn dequeue_next_process(&mut self) -> Option<(Pid, Priority)> {
        // Check each priority level from highest to lowest
        for (priority_idx, queue) in self.ready_queues.iter_mut().enumerate() {
            if let Some(pid) = queue.pop_front() {
                let priority = match priority_idx {
                    0 => Priority::RealTime,
                    1 => Priority::High,
                    2 => Priority::Normal,
                    3 => Priority::Low,
                    4 => Priority::Idle,
                    _ => Priority::Normal,
                };
                return Some((pid, priority));
            }
        }
        None
    }

    /// Get the number of ready processes
    pub fn ready_process_count(&self) -> usize {
        self.ready_queues.iter().map(|q| q.len()).sum()
    }

    /// Update CPU utilization
    pub fn update_utilization(&mut self, active_time: u64, total_time: u64) {
        if total_time > 0 {
            self.utilization = ((active_time * 100) / total_time) as u8;
        }
    }
}

/// Global scheduler state
pub struct GlobalScheduler {
    /// All processes in the system
    pub processes: RwLock<Vec<Process>>,
    /// Per-CPU schedulers
    pub cpu_schedulers: Vec<Mutex<CpuScheduler>>,
    /// Next process ID to assign
    pub next_pid: AtomicU64,
    /// Total number of processes
    pub process_count: AtomicUsize,
    /// System boot time
    pub boot_time: u64,
    /// Load balancing enabled
    pub load_balancing_enabled: bool,
}

impl GlobalScheduler {
    /// Create a new global scheduler
    pub fn new(num_cpus: usize) -> Self {
        let mut cpu_schedulers = Vec::with_capacity(num_cpus);
        for cpu_id in 0..num_cpus {
            cpu_schedulers.push(Mutex::new(CpuScheduler::new(cpu_id as CpuId)));
        }

        Self {
            processes: RwLock::new(Vec::new()),
            cpu_schedulers,
            next_pid: AtomicU64::new(1),
            process_count: AtomicUsize::new(0),
            boot_time: get_system_time(),
            load_balancing_enabled: true,
        }
    }

    /// Create a new process
    pub fn create_process(&self, parent_pid: Option<Pid>, priority: Priority, name: &str) -> Result<Pid, &'static str> {
        let pid = self.next_pid.fetch_add(1, Ordering::SeqCst) as Pid;
        let mut process = Process::new(pid, parent_pid, priority, name);
        process.state = ProcessState::Ready;

        // Add to process table
        {
            let mut processes = self.processes.write();
            processes.push(process);
        }

        // Schedule on least loaded CPU
        let cpu_id = self.find_least_loaded_cpu();
        {
            let mut cpu_scheduler = self.cpu_schedulers[cpu_id as usize].lock();
            cpu_scheduler.enqueue_process(pid, priority);
        }

        self.process_count.fetch_add(1, Ordering::SeqCst);
        println!("Created process {} '{}' with priority {:?} on CPU {}", pid, name, priority, cpu_id);

        Ok(pid)
    }

    /// Schedule the next process on the given CPU
    pub fn schedule(&self, cpu_id: CpuId) -> Option<Pid> {
        if cpu_id as usize >= self.cpu_schedulers.len() {
            return None;
        }

        let mut cpu_scheduler = self.cpu_schedulers[cpu_id as usize].lock();
        
        // If current process time slice expired, move it back to ready queue
        if let Some(current_pid) = cpu_scheduler.current_process {
            if cpu_scheduler.time_slice_remaining == 0 {
                if let Some(process) = self.find_process_mut(current_pid) {
                    if process.state == ProcessState::Running {
                        process.state = ProcessState::Ready;
                        cpu_scheduler.enqueue_process(current_pid, process.priority);
                    }
                }
                cpu_scheduler.current_process = None;
            }
        }

        // Get next process to run
        if cpu_scheduler.current_process.is_none() {
            if let Some((next_pid, priority)) = cpu_scheduler.dequeue_next_process() {
                // Update process state
                if let Some(process) = self.find_process_mut(next_pid) {
                    process.state = ProcessState::Running;
                    process.current_cpu = Some(cpu_id);
                    process.last_scheduled = get_system_time();
                }

                // Set time slice
                cpu_scheduler.current_process = Some(next_pid);
                cpu_scheduler.time_slice_remaining = priority.time_slice_ms() * 1000; // Convert to microseconds
                cpu_scheduler.total_scheduled += 1;

                return Some(next_pid);
            }
        }

        cpu_scheduler.current_process
    }

    /// Handle timer tick for scheduling
    pub fn timer_tick(&self, cpu_id: CpuId, elapsed_us: u64) {
        if cpu_id as usize >= self.cpu_schedulers.len() {
            return;
        }

        let mut cpu_scheduler = self.cpu_schedulers[cpu_id as usize].lock();
        
        if let Some(current_pid) = cpu_scheduler.current_process {
            // Update process CPU time
            if let Some(process) = self.find_process_mut(current_pid) {
                process.cpu_time_used += elapsed_us;
            }

            // Decrement time slice
            if cpu_scheduler.time_slice_remaining > elapsed_us {
                cpu_scheduler.time_slice_remaining -= elapsed_us;
            } else {
                cpu_scheduler.time_slice_remaining = 0;
            }
        } else {
            // CPU is idle
            cpu_scheduler.idle_time += elapsed_us;
        }

        // Update CPU utilization
        let total_time = cpu_scheduler.idle_time + elapsed_us;
        cpu_scheduler.update_utilization(elapsed_us, total_time);
    }

    /// Find the least loaded CPU
    fn find_least_loaded_cpu(&self) -> CpuId {
        let mut min_load = usize::MAX;
        let mut best_cpu = 0;

        for (cpu_id, cpu_scheduler_mutex) in self.cpu_schedulers.iter().enumerate() {
            let cpu_scheduler = cpu_scheduler_mutex.lock();
            let load = cpu_scheduler.ready_process_count();
            if load < min_load {
                min_load = load;
                best_cpu = cpu_id;
            }
        }

        best_cpu as CpuId
    }

    /// Find a process by PID (mutable reference)
    fn find_process_mut(&self, pid: Pid) -> Option<&mut Process> {
        // This is unsafe but necessary for interior mutability
        // In a real implementation, we'd use proper synchronization
        unsafe {
            let processes_ptr = self.processes.data_ptr() as *mut Vec<Process>;
            (*processes_ptr).iter_mut().find(|p| p.pid == pid)
        }
    }

    /// Get scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        let processes = self.processes.read();
        let process_count = processes.len();
        
        let mut stats_by_state = [0usize; 6];
        let mut stats_by_priority = [0usize; 5];
        
        for process in processes.iter() {
            let state_idx = match process.state {
                ProcessState::Ready => 0,
                ProcessState::Running => 1,
                ProcessState::Blocked => 2,
                ProcessState::Sleeping => 3,
                ProcessState::Terminated => 4,
                ProcessState::Creating => 5,
            };
            stats_by_state[state_idx] += 1;
            stats_by_priority[process.priority as usize] += 1;
        }

        let mut cpu_utilizations = Vec::new();
        for cpu_scheduler_mutex in &self.cpu_schedulers {
            let cpu_scheduler = cpu_scheduler_mutex.lock();
            cpu_utilizations.push(cpu_scheduler.utilization);
        }

        SchedulerStats {
            total_processes: process_count,
            ready_processes: stats_by_state[0],
            running_processes: stats_by_state[1],
            blocked_processes: stats_by_state[2],
            sleeping_processes: stats_by_state[3],
            terminated_processes: stats_by_state[4],
            creating_processes: stats_by_state[5],
            realtime_processes: stats_by_priority[0],
            high_priority_processes: stats_by_priority[1],
            normal_priority_processes: stats_by_priority[2],
            low_priority_processes: stats_by_priority[3],
            idle_priority_processes: stats_by_priority[4],
            cpu_utilizations,
            uptime_seconds: (get_system_time() - self.boot_time) / 1_000_000,
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub total_processes: usize,
    pub ready_processes: usize,
    pub running_processes: usize,
    pub blocked_processes: usize,
    pub sleeping_processes: usize,
    pub terminated_processes: usize,
    pub creating_processes: usize,
    pub realtime_processes: usize,
    pub high_priority_processes: usize,
    pub normal_priority_processes: usize,
    pub low_priority_processes: usize,
    pub idle_priority_processes: usize,
    pub cpu_utilizations: Vec<u8>,
    pub uptime_seconds: u64,
}

lazy_static! {
    static ref GLOBAL_SCHEDULER: GlobalScheduler = {
        // Detect number of CPUs from ACPI MADT
        let num_cpus = if let Some(madt) = crate::acpi::madt() {
            core::cmp::max(1, madt.processors.len())
        } else {
            1 // Single CPU fallback
        };
        
        println!("Initializing scheduler for {} CPU(s)", num_cpus);
        GlobalScheduler::new(num_cpus)
    };
}

/// Initialize the scheduler subsystem
pub fn init() -> Result<(), &'static str> {
    // Force initialization of the global scheduler
    lazy_static::initialize(&GLOBAL_SCHEDULER);
    
    // Create init process (PID 1)
    GLOBAL_SCHEDULER.create_process(None, Priority::High, "init")?;
    
    println!("✓ Scheduler initialized with {} CPU(s)", GLOBAL_SCHEDULER.cpu_schedulers.len());
    Ok(())
}

/// Create a new process
pub fn create_process(parent_pid: Option<Pid>, priority: Priority, name: &str) -> Result<Pid, &'static str> {
    GLOBAL_SCHEDULER.create_process(parent_pid, priority, name)
}

/// Schedule the next process on the current CPU
pub fn schedule() -> Option<Pid> {
    let cpu_id = get_current_cpu_id();
    GLOBAL_SCHEDULER.schedule(cpu_id)
}

/// Handle timer tick for scheduling
pub fn timer_tick(elapsed_us: u64) {
    let cpu_id = get_current_cpu_id();
    GLOBAL_SCHEDULER.timer_tick(cpu_id, elapsed_us);
}

/// Get scheduler statistics
pub fn get_scheduler_stats() -> SchedulerStats {
    GLOBAL_SCHEDULER.get_stats()
}

/// Get current CPU ID (placeholder implementation)
fn get_current_cpu_id() -> CpuId {
    // TODO: Implement proper CPU ID detection
    // For now, return CPU 0
    0
}

/// Get system time in microseconds (placeholder implementation)
fn get_system_time() -> u64 {
    // TODO: Implement proper system time
    // For now, return a placeholder value
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1000, Ordering::SeqCst)
}

/// Context switch between processes (assembly stub)
#[unsafe(naked)]
pub unsafe extern "C" fn context_switch(old_state: *mut CpuState, new_state: *const CpuState) {
    core::arch::asm!(
        // Save current context
        "mov [rdi + 0x00], rax",
        "mov [rdi + 0x08], rbx", 
        "mov [rdi + 0x10], rcx",
        "mov [rdi + 0x18], rdx",
        "mov [rdi + 0x20], rsi",
        "mov [rdi + 0x28], rdi",
        "mov [rdi + 0x30], rbp",
        "mov [rdi + 0x38], rsp",
        "mov [rdi + 0x40], r8",
        "mov [rdi + 0x48], r9",
        "mov [rdi + 0x50], r10",
        "mov [rdi + 0x58], r11",
        "mov [rdi + 0x60], r12",
        "mov [rdi + 0x68], r13",
        "mov [rdi + 0x70], r14",
        "mov [rdi + 0x78], r15",
        
        // Save RIP (return address)
        "mov rax, [rsp]",
        "mov [rdi + 0x80], rax",
        
        // Save RFLAGS
        "pushfq",
        "pop rax",
        "mov [rdi + 0x88], rax",
        
        // Load new context
        "mov rax, [rsi + 0x00]",
        "mov rbx, [rsi + 0x08]",
        "mov rcx, [rsi + 0x10]",
        "mov rdx, [rsi + 0x18]",
        "mov rbp, [rsi + 0x30]",
        "mov rsp, [rsi + 0x38]",
        "mov r8,  [rsi + 0x40]",
        "mov r9,  [rsi + 0x48]",
        "mov r10, [rsi + 0x50]",
        "mov r11, [rsi + 0x58]",
        "mov r12, [rsi + 0x60]",
        "mov r13, [rsi + 0x68]",
        "mov r14, [rsi + 0x70]",
        "mov r15, [rsi + 0x78]",
        
        // Load RFLAGS
        "mov rdi, [rsi + 0x88]",
        "push rdi",
        "popfq",
        
        // Load RIP and jump
        "mov rdi, [rsi + 0x80]",
        "jmp rdi",
        options(noreturn)
    );
}

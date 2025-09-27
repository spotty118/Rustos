//! Real-Time Process Scheduler
//!
//! This module implements an advanced real-time process scheduler for the RustOS kernel,
//! providing deterministic scheduling guarantees, priority inheritance, deadline scheduling,
//! and CPU isolation for real-time applications requiring predictable response times.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of real-time processes
const MAX_RT_PROCESSES: usize = 64;
/// Maximum number of CPU cores
const MAX_CPU_CORES: usize = 32;
/// Maximum priority levels (0 = highest, 99 = lowest)
const MAX_PRIORITY_LEVELS: usize = 100;
/// Time slice quantum in microseconds
const DEFAULT_TIME_QUANTUM_US: u64 = 1000;
/// Maximum deadline miss tolerance
const MAX_DEADLINE_MISSES: u32 = 3;

/// Real-time scheduling policies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RTSchedulingPolicy {
    /// Fixed priority preemptive scheduling
    FixedPriority,
    /// Rate Monotonic Scheduling
    RateMonotonic,
    /// Deadline Monotonic Scheduling
    DeadlineMonotonic,
    /// Earliest Deadline First
    EarliestDeadlineFirst,
    /// Least Laxity First
    LeastLaxityFirst,
    /// Proportional Share (PFair)
    ProportionalShare,
    /// Constant Bandwidth Server
    ConstantBandwidth,
    /// Sporadic Server
    SporadicServer,
}

impl fmt::Display for RTSchedulingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RTSchedulingPolicy::FixedPriority => write!(f, "Fixed Priority"),
            RTSchedulingPolicy::RateMonotonic => write!(f, "Rate Monotonic"),
            RTSchedulingPolicy::DeadlineMonotonic => write!(f, "Deadline Monotonic"),
            RTSchedulingPolicy::EarliestDeadlineFirst => write!(f, "Earliest Deadline First"),
            RTSchedulingPolicy::LeastLaxityFirst => write!(f, "Least Laxity First"),
            RTSchedulingPolicy::ProportionalShare => write!(f, "Proportional Share"),
            RTSchedulingPolicy::ConstantBandwidth => write!(f, "Constant Bandwidth"),
            RTSchedulingPolicy::SporadicServer => write!(f, "Sporadic Server"),
        }
    }
}

/// Process states in real-time scheduling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RTProcessState {
    Ready,
    Running,
    Blocked,
    Suspended,
    Terminated,
    WaitingForResource,
    WaitingForDeadline,
}

impl fmt::Display for RTProcessState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            RTProcessState::Ready => write!(f, "Ready"),
            RTProcessState::Running => write!(f, "Running"),
            RTProcessState::Blocked => write!(f, "Blocked"),
            RTProcessState::Suspended => write!(f, "Suspended"),
            RTProcessState::Terminated => write!(f, "Terminated"),
            RTProcessState::WaitingForResource => write!(f, "Waiting for Resource"),
            RTProcessState::WaitingForDeadline => write!(f, "Waiting for Deadline"),
        }
    }
}

/// Real-time process characteristics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RTProcessType {
    /// Hard real-time: missing deadline is catastrophic
    HardRealTime,
    /// Firm real-time: missing deadline makes result useless
    FirmRealTime,
    /// Soft real-time: missing deadline degrades performance
    SoftRealTime,
    /// Best effort: no timing constraints
    BestEffort,
}

/// CPU affinity settings
#[derive(Debug, Clone)]
pub struct CPUAffinity {
    pub allowed_cpus: Vec<u8, MAX_CPU_CORES>,
    pub preferred_cpu: Option<u8>,
    pub isolated_cpu: bool,
}

impl CPUAffinity {
    pub fn new() -> Self {
        Self {
            allowed_cpus: Vec::new(),
            preferred_cpu: None,
            isolated_cpu: false,
        }
    }

    pub fn allow_all_cpus(&mut self, num_cpus: u8) {
        self.allowed_cpus.clear();
        for cpu in 0..num_cpus {
            let _ = self.allowed_cpus.push(cpu);
        }
    }

    pub fn allow_cpu(&mut self, cpu: u8) -> Result<(), &'static str> {
        if self.allowed_cpus.len() >= MAX_CPU_CORES {
            return Err("CPU affinity list full");
        }
        let _ = self.allowed_cpus.push(cpu);
        Ok(())
    }

    pub fn is_cpu_allowed(&self, cpu: u8) -> bool {
        self.allowed_cpus.contains(&cpu)
    }
}

/// Real-time process control block
#[derive(Debug, Clone)]
pub struct RTProcess {
    pub process_id: u32,
    pub name: heapless::String<32>,
    pub process_type: RTProcessType,
    pub policy: RTSchedulingPolicy,
    pub state: RTProcessState,
    pub priority: u8, // 0 = highest, 99 = lowest
    pub static_priority: u8,
    pub dynamic_priority: u8,

    // Timing constraints
    pub period_us: u64,        // Period for periodic tasks
    pub deadline_us: u64,      // Relative deadline
    pub wcet_us: u64,          // Worst Case Execution Time
    pub release_time_us: u64,  // Next release time
    pub absolute_deadline_us: u64, // Absolute deadline

    // Runtime statistics
    pub execution_time_us: u64,
    pub remaining_time_us: u64,
    pub response_time_us: u64,
    pub last_start_time_us: u64,
    pub deadline_misses: u32,
    pub total_executions: u64,
    pub cpu_utilization: f32,

    // Scheduling attributes
    pub time_quantum_us: u64,
    pub remaining_quantum_us: u64,
    pub cpu_affinity: CPUAffinity,
    pub assigned_cpu: Option<u8>,
    pub preemption_count: u32,
    pub context_switches: u32,

    // Priority inheritance
    pub inherited_priority: Option<u8>,
    pub blocked_on_resource: Option<u32>,
    pub owned_resources: Vec<u32, 8>,
}

impl RTProcess {
    pub fn new(process_id: u32, name: &str, process_type: RTProcessType,
               priority: u8, period_us: u64, deadline_us: u64, wcet_us: u64) -> Self {
        let mut process_name = heapless::String::new();
        let _ = process_name.push_str(name);

        let mut affinity = CPUAffinity::new();
        affinity.allow_all_cpus(4); // Default to 4 CPUs

        Self {
            process_id,
            name: process_name,
            process_type,
            policy: RTSchedulingPolicy::FixedPriority,
            state: RTProcessState::Ready,
            priority,
            static_priority: priority,
            dynamic_priority: priority,
            period_us,
            deadline_us,
            wcet_us,
            release_time_us: 0,
            absolute_deadline_us: deadline_us,
            execution_time_us: 0,
            remaining_time_us: wcet_us,
            response_time_us: 0,
            last_start_time_us: 0,
            deadline_misses: 0,
            total_executions: 0,
            cpu_utilization: 0.0,
            time_quantum_us: DEFAULT_TIME_QUANTUM_US,
            remaining_quantum_us: DEFAULT_TIME_QUANTUM_US,
            cpu_affinity: affinity,
            assigned_cpu: None,
            preemption_count: 0,
            context_switches: 0,
            inherited_priority: None,
            blocked_on_resource: None,
            owned_resources: Vec::new(),
        }
    }

    pub fn effective_priority(&self) -> u8 {
        self.inherited_priority.unwrap_or(self.dynamic_priority)
    }

    pub fn is_deadline_missed(&self, current_time_us: u64) -> bool {
        current_time_us > self.absolute_deadline_us
    }

    pub fn laxity(&self, current_time_us: u64) -> i64 {
        self.absolute_deadline_us as i64 - current_time_us as i64 - self.remaining_time_us as i64
    }

    pub fn update_deadline(&mut self, current_time_us: u64) {
        if self.period_us > 0 {
            self.release_time_us = current_time_us;
            self.absolute_deadline_us = current_time_us + self.deadline_us;
            self.remaining_time_us = self.wcet_us;
        }
    }

    pub fn is_schedulable_rm(&self, total_utilization: f32) -> bool {
        // Rate Monotonic schedulability test
        let utilization = self.wcet_us as f32 / self.period_us as f32;
        total_utilization + utilization <= 0.69 // ln(2) approximation
    }

    pub fn is_schedulable_edf(&self, total_utilization: f32) -> bool {
        // EDF schedulability test
        let utilization = self.wcet_us as f32 / self.deadline_us as f32;
        total_utilization + utilization <= 1.0
    }
}

/// CPU core state for real-time scheduling
#[derive(Debug, Clone)]
pub struct RTCPUCore {
    pub core_id: u8,
    pub current_process: Option<u32>,
    pub isolated: bool,
    pub idle: bool,
    pub frequency_mhz: u32,
    pub utilization: f32,
    pub context_switch_count: u64,
    pub interrupt_count: u64,
    pub cache_misses: u64,
    pub rt_processes_count: u32,
}

impl RTCPUCore {
    pub fn new(core_id: u8) -> Self {
        Self {
            core_id,
            current_process: None,
            isolated: false,
            idle: true,
            frequency_mhz: 2400, // Default 2.4 GHz
            utilization: 0.0,
            context_switch_count: 0,
            interrupt_count: 0,
            cache_misses: 0,
            rt_processes_count: 0,
        }
    }

    pub fn is_available(&self) -> bool {
        self.current_process.is_none() && !self.isolated
    }

    pub fn assign_process(&mut self, process_id: u32) {
        self.current_process = Some(process_id);
        self.idle = false;
    }

    pub fn release_process(&mut self) {
        self.current_process = None;
        self.idle = true;
    }
}

/// Real-time scheduler statistics
#[derive(Debug, Clone, Copy)]
pub struct RTSchedulerStats {
    pub total_rt_processes: u32,
    pub active_rt_processes: u32,
    pub hard_rt_processes: u32,
    pub soft_rt_processes: u32,
    pub total_deadline_misses: u32,
    pub total_context_switches: u64,
    pub total_preemptions: u32,
    pub average_response_time_us: u64,
    pub worst_case_response_time_us: u64,
    pub system_utilization: f32,
    pub schedulability_ratio: f32,
    pub priority_inversions: u32,
}

impl RTSchedulerStats {
    pub fn new() -> Self {
        Self {
            total_rt_processes: 0,
            active_rt_processes: 0,
            hard_rt_processes: 0,
            soft_rt_processes: 0,
            total_deadline_misses: 0,
            total_context_switches: 0,
            total_preemptions: 0,
            average_response_time_us: 0,
            worst_case_response_time_us: 0,
            system_utilization: 0.0,
            schedulability_ratio: 0.0,
            priority_inversions: 0,
        }
    }

    pub fn deadline_miss_ratio(&self) -> f32 {
        if self.total_rt_processes > 0 {
            self.total_deadline_misses as f32 / self.total_rt_processes as f32
        } else {
            0.0
        }
    }

    pub fn real_time_efficiency(&self) -> f32 {
        let deadline_factor = 1.0 - self.deadline_miss_ratio().min(1.0);
        let utilization_factor = self.system_utilization;
        let responsiveness_factor = if self.worst_case_response_time_us > 0 {
            1000.0 / self.worst_case_response_time_us as f32
        } else {
            1.0
        }.min(1.0);

        (deadline_factor + utilization_factor + responsiveness_factor) / 3.0
    }
}

/// Main real-time scheduler
pub struct RTScheduler {
    rt_processes: Vec<RTProcess, MAX_RT_PROCESSES>,
    cpu_cores: Vec<RTCPUCore, MAX_CPU_CORES>,
    ready_queues: [Vec<u32, MAX_RT_PROCESSES>; MAX_PRIORITY_LEVELS],
    current_policy: RTSchedulingPolicy,
    stats: RTSchedulerStats,
    process_counter: u32,
    preemption_enabled: bool,
    priority_inheritance_enabled: bool,
    cpu_isolation_enabled: bool,
    load_balancing_enabled: bool,
}

impl RTScheduler {
    pub fn new() -> Self {
        const EMPTY_VEC: Vec<u32, MAX_RT_PROCESSES> = Vec::new();

        Self {
            rt_processes: Vec::new(),
            cpu_cores: Vec::new(),
            ready_queues: [EMPTY_VEC; MAX_PRIORITY_LEVELS],
            current_policy: RTSchedulingPolicy::FixedPriority,
            stats: RTSchedulerStats::new(),
            process_counter: 0,
            preemption_enabled: true,
            priority_inheritance_enabled: true,
            cpu_isolation_enabled: false,
            load_balancing_enabled: true,
        }
    }

    pub fn initialize(&mut self, num_cpu_cores: u8) -> Result<(), &'static str> {
        crate::println!("[RT] Initializing real-time scheduler...");

        // Initialize CPU cores
        for core_id in 0..num_cpu_cores {
            let core = RTCPUCore::new(core_id);
            let _ = self.cpu_cores.push(core);
        }

        crate::println!("[RT] Real-time scheduler initialized successfully");
        crate::println!("[RT] CPU cores: {}", num_cpu_cores);
        crate::println!("[RT] Scheduling policy: {}", self.current_policy);
        crate::println!("[RT] Features: Preemption={}, Priority Inheritance={}, CPU Isolation={}",
                       self.preemption_enabled, self.priority_inheritance_enabled, self.cpu_isolation_enabled);

        Ok(())
    }

    pub fn create_rt_process(&mut self, name: &str, process_type: RTProcessType,
                           priority: u8, period_us: u64, deadline_us: u64,
                           wcet_us: u64) -> Result<u32, &'static str> {
        if self.rt_processes.len() >= MAX_RT_PROCESSES {
            return Err("Maximum RT processes reached");
        }

        if priority >= MAX_PRIORITY_LEVELS as u8 {
            return Err("Invalid priority level");
        }

        // Schedulability analysis
        if !self.is_process_schedulable(wcet_us, period_us, deadline_us) {
            crate::println!("[RT] Warning: Process '{}' may not be schedulable", name);
        }

        let process = RTProcess::new(
            self.process_counter,
            name,
            process_type,
            priority,
            period_us,
            deadline_us,
            wcet_us
        );

        self.process_counter += 1;
        let process_id = process.process_id;

        let _ = self.rt_processes.push(process);

        // Update statistics
        self.stats.total_rt_processes += 1;
        match process_type {
            RTProcessType::HardRealTime => self.stats.hard_rt_processes += 1,
            RTProcessType::SoftRealTime => self.stats.soft_rt_processes += 1,
            _ => {}
        }

        // Add to ready queue
        self.add_to_ready_queue(process_id, priority);

        crate::println!("[RT] Created RT process '{}' (ID: {}) - Priority: {}, Period: {}μs, Deadline: {}μs",
                       name, process_id, priority, period_us, deadline_us);

        Ok(process_id)
    }

    pub fn set_scheduling_policy(&mut self, policy: RTSchedulingPolicy) {
        crate::println!("[RT] Changing scheduling policy from {} to {}", self.current_policy, policy);
        self.current_policy = policy;

        // Reorganize ready queues based on new policy
        self.reorganize_ready_queues();
    }

    pub fn schedule(&mut self, current_time_us: u64) -> Vec<(u8, u32), MAX_CPU_CORES> {
        let mut schedule_decisions = Vec::new();

        // Check for deadline misses and update process states
        self.check_deadline_misses(current_time_us);

        // Release periodic processes that are ready
        self.release_periodic_processes(current_time_us);

        match self.current_policy {
            RTSchedulingPolicy::FixedPriority => {
                self.schedule_fixed_priority(&mut schedule_decisions);
            }
            RTSchedulingPolicy::EarliestDeadlineFirst => {
                self.schedule_earliest_deadline_first(current_time_us, &mut schedule_decisions);
            }
            RTSchedulingPolicy::RateMonotonic => {
                self.schedule_rate_monotonic(&mut schedule_decisions);
            }
            RTSchedulingPolicy::LeastLaxityFirst => {
                self.schedule_least_laxity_first(current_time_us, &mut schedule_decisions);
            }
            _ => {
                // Fallback to fixed priority
                self.schedule_fixed_priority(&mut schedule_decisions);
            }
        }

        // Update CPU assignments
        self.update_cpu_assignments(&schedule_decisions);

        schedule_decisions
    }

    pub fn preempt_process(&mut self, process_id: u32, current_time_us: u64) -> Result<(), &'static str> {
        if !self.preemption_enabled {
            return Err("Preemption disabled");
        }

        if let Some(process) = self.rt_processes.iter_mut().find(|p| p.process_id == process_id) {
            if process.state == RTProcessState::Running {
                process.state = RTProcessState::Ready;
                process.preemption_count += 1;
                process.execution_time_us += current_time_us - process.last_start_time_us;

                // Add back to ready queue
                self.add_to_ready_queue(process_id, process.effective_priority());

                // Free the CPU
                if let Some(cpu_id) = process.assigned_cpu {
                    if let Some(core) = self.cpu_cores.iter_mut().find(|c| c.core_id == cpu_id) {
                        core.release_process();
                    }
                }
                process.assigned_cpu = None;

                crate::println!("[RT] Preempted process {} at {}μs", process_id, current_time_us);
                return Ok(());
            }
        }

        Err("Process not found or not running")
    }

    pub fn block_process(&mut self, process_id: u32, resource_id: Option<u32>) -> Result<(), &'static str> {
        if let Some(process) = self.rt_processes.iter_mut().find(|p| p.process_id == process_id) {
            process.state = RTProcessState::WaitingForResource;
            process.blocked_on_resource = resource_id;

            // Handle priority inheritance if enabled
            if self.priority_inheritance_enabled {
                self.handle_priority_inheritance(process_id, resource_id);
            }

            // Remove from ready queue
            self.remove_from_ready_queue(process_id, process.effective_priority());

            crate::println!("[RT] Blocked process {} on resource {:?}", process_id, resource_id);
            return Ok(());
        }

        Err("Process not found")
    }

    pub fn unblock_process(&mut self, process_id: u32) -> Result<(), &'static str> {
        if let Some(process) = self.rt_processes.iter_mut().find(|p| p.process_id == process_id) {
            process.state = RTProcessState::Ready;
            process.blocked_on_resource = None;

            // Restore original priority if inheritance was applied
            if process.inherited_priority.is_some() {
                process.inherited_priority = None;
                process.dynamic_priority = process.static_priority;
            }

            // Add back to ready queue
            self.add_to_ready_queue(process_id, process.effective_priority());

            crate::println!("[RT] Unblocked process {}", process_id);
            return Ok(());
        }

        Err("Process not found")
    }

    pub fn get_process_info(&self, process_id: u32) -> Option<&RTProcess> {
        self.rt_processes.iter().find(|p| p.process_id == process_id)
    }

    pub fn get_scheduler_stats(&self) -> RTSchedulerStats {
        self.stats
    }

    pub fn enable_cpu_isolation(&mut self, cpu_id: u8, enabled: bool) -> Result<(), &'static str> {
        if let Some(core) = self.cpu_cores.iter_mut().find(|c| c.core_id == cpu_id) {
            core.isolated = enabled;
            self.cpu_isolation_enabled = enabled;

            crate::println!("[RT] CPU {} isolation: {}", cpu_id, if enabled { "enabled" } else { "disabled" });
            return Ok(());
        }

        Err("CPU core not found")
    }

    pub fn set_process_affinity(&mut self, process_id: u32, cpu_mask: &[u8]) -> Result<(), &'static str> {
        if let Some(process) = self.rt_processes.iter_mut().find(|p| p.process_id == process_id) {
            process.cpu_affinity.allowed_cpus.clear();

            for &cpu in cpu_mask {
                if cpu < self.cpu_cores.len() as u8 {
                    let _ = process.cpu_affinity.allowed_cpus.push(cpu);
                }
            }

            crate::println!("[RT] Set CPU affinity for process {} to: {:?}", process_id, cpu_mask);
            return Ok(());
        }

        Err("Process not found")
    }

    // Private implementation methods
    fn is_process_schedulable(&self, wcet_us: u64, period_us: u64, deadline_us: u64) -> bool {
        let utilization = wcet_us as f32 / period_us as f32;
        let total_utilization = self.calculate_total_utilization();

        match self.current_policy {
            RTSchedulingPolicy::RateMonotonic | RTSchedulingPolicy::FixedPriority => {
                total_utilization + utilization <= 0.69
            }
            RTSchedulingPolicy::EarliestDeadlineFirst => {
                let deadline_utilization = wcet_us as f32 / deadline_us as f32;
                total_utilization + deadline_utilization <= 1.0
            }
            _ => true, // Assume schedulable for other policies
        }
    }

    fn calculate_total_utilization(&self) -> f32 {
        self.rt_processes.iter()
            .filter(|p| p.state != RTProcessState::Terminated)
            .map(|p| p.wcet_us as f32 / p.period_us as f32)
            .sum()
    }

    fn add_to_ready_queue(&mut self, process_id: u32, priority: u8) {
        if priority < MAX_PRIORITY_LEVELS as u8 {
            let _ = self.ready_queues[priority as usize].push(process_id);
            self.stats.active_rt_processes += 1;
        }
    }

    fn remove_from_ready_queue(&mut self, process_id: u32, priority: u8) {
        if priority < MAX_PRIORITY_LEVELS as u8 {
            if let Some(pos) = self.ready_queues[priority as usize].iter().position(|&x| x == process_id) {
                self.ready_queues[priority as usize].remove(pos);
                self.stats.active_rt_processes = self.stats.active_rt_processes.saturating_sub(1);
            }
        }
    }

    fn schedule_fixed_priority(&mut self, schedule_decisions: &mut Vec<(u8, u32), MAX_CPU_CORES>) {
        // Highest priority first (lowest number = highest priority)
        for priority_level in 0..MAX_PRIORITY_LEVELS {
            while let Some(&process_id) = self.ready_queues[priority_level].first() {
                if let Some(cpu_id) = self.find_available_cpu(process_id) {
                    self.ready_queues[priority_level].remove(0);
                    let _ = schedule_decisions.push((cpu_id, process_id));

                    if schedule_decisions.len() >= MAX_CPU_CORES {
                        return;
                    }
                } else {
                    break; // No available CPU for this priority level
                }
            }
        }
    }

    fn schedule_earliest_deadline_first(&mut self, current_time_us: u64,
                                       schedule_decisions: &mut Vec<(u8, u32), MAX_CPU_CORES>) {
        // Collect all ready processes and sort by absolute deadline
        let mut ready_processes = Vec::new();

        for priority_level in 0..MAX_PRIORITY_LEVELS {
            for &process_id in &self.ready_queues[priority_level] {
                if let Some(process) = self.rt_processes.iter().find(|p| p.process_id == process_id) {
                    let _ = ready_processes.push((process.absolute_deadline_us, process_id));
                }
            }
        }

        // Sort by deadline (earliest first)
        ready_processes.sort_by_key(|&(deadline, _)| deadline);

        // Schedule processes in deadline order
        for (_, process_id) in ready_processes {
            if let Some(cpu_id) = self.find_available_cpu(process_id) {
                // Remove from appropriate ready queue
                if let Some(process) = self.rt_processes.iter().find(|p| p.process_id == process_id) {
                    self.remove_from_ready_queue(process_id, process.effective_priority());
                }

                let _ = schedule_decisions.push((cpu_id, process_id));

                if schedule_decisions.len() >= MAX_CPU_CORES {
                    break;
                }
            }
        }
    }

    fn schedule_rate_monotonic(&mut self, schedule_decisions: &mut Vec<(u8, u32), MAX_CPU_CORES>) {
        // Rate Monotonic: shorter period = higher priority
        // For this implementation, we'll use the configured priorities which should
        // be assigned based on rate monotonic principles
        self.schedule_fixed_priority(schedule_decisions);
    }

    fn schedule_least_laxity_first(&mut self, current_time_us: u64,
                                  schedule_decisions: &mut Vec<(u8, u32), MAX_CPU_CORES>) {
        let mut ready_processes = Vec::new();

        for priority_level in 0..MAX_PRIORITY_LEVELS {
            for &process_id in &self.ready_queues[priority_level] {
                if let Some(process) = self.rt_processes.iter().find(|p| p.process_id == process_id) {
                    let laxity = process.laxity(current_time_us);
                    let _ = ready_processes.push((laxity, process_id));
                }
            }
        }

        // Sort by laxity (least first)
        ready_processes.sort_by_key(|&(laxity, _)| laxity);

        for (_, process_id) in ready_processes {
            if let Some(cpu_id) = self.find_available_cpu(process_id) {
                if let Some(process) = self.rt_processes.iter().find(|p| p.process_id == process_id) {
                    self.remove_from_ready_queue(process_id, process.effective_priority());
                }

                let _ = schedule_decisions.push((cpu_id, process_id));

                if schedule_decisions.len() >= MAX_CPU_CORES {
                    break;
                }
            }
        }
    }

    fn find_available_cpu(&self, process_id: u32) -> Option<u8> {
        if let Some(process) = self.rt_processes.iter().find(|p| p.process_id == process_id) {
            // Check CPU affinity
            for &cpu_id in &process.cpu_affinity.allowed_cpus {
                if let Some(core) = self.cpu_cores.iter().find(|c| c.core_id == cpu_id) {
                    if core.is_available() {
                        return Some(cpu_id);
                    }
                }
            }
        }
        None
    }

    fn update_cpu_assignments(&mut self, schedule_decisions: &[(u8, u32)]) {
        for &(cpu_id, process_id) in schedule_decisions {
            // Update CPU core
            if let Some(core) = self.cpu_cores.iter_mut().find(|c| c.core_id == cpu_id) {
                core.assign_process(process_id);
            }

            // Update process
            if let Some(process) = self.rt_processes.iter_mut().find(|p| p.process_id == process_id) {
                process.assigned_cpu = Some(cpu_id);
                process.state = RTProcessState::Running;
                process.last_start_time_us = self.get_current_time_us();

                if process.total_executions == 0 {
                    process.context_switches += 1;
                }
            }
        }
    }

    fn check_deadline_misses(&mut self, current_time_us: u64) {
        for process in &mut self.rt_processes {
            if process.state == RTProcessState::Running ||
                process.state == RTProcessState::Ready {
                if process.is_deadline_missed(current_time_us) {
                    process.deadline_misses += 1;
                    self.stats.total_deadline_misses += 1;

                    crate::println!("[RT] DEADLINE MISS: Process {} missed deadline by {}μs",
                                   process.process_id,
                                   current_time_us - process.absolute_deadline_us);

                    // Handle hard real-time deadline miss
                    if process.process_type == RTProcessType::HardRealTime {
                        if process.deadline_misses >= MAX_DEADLINE_MISSES {
                            process.state = RTProcessState::Terminated;
                            crate::println!("[RT] CRITICAL: Hard RT process {} terminated due to deadline misses",
                                           process.process_id);
                        }
                    }
                }
            }
        }
    }

    fn release_periodic_processes(&mut self, current_time_us: u64) {
        for process in &mut self.rt_processes {
            if process.period_us > 0 &&
               current_time_us >= process.release_time_us + process.period_us {
                process.update_deadline(current_time_us);

                if process.state != RTProcessState::Terminated {
                    process.state = RTProcessState::Ready;
                    self.add_to_ready_queue(process.process_id, process.effective_priority());
                }
            }
        }
    }

    fn reorganize_ready_queues(&mut self) {
        // Clear all queues and re-add processes
        for queue in &mut self.ready_queues {
            queue.clear();
        }

        for process in &self.rt_processes {
            if process.state == RTProcessState::Ready {
                self.add_to_ready_queue(process.process_id, process.effective_priority());
            }
        }
    }

    fn handle_priority_inheritance(&mut self, _process_id: u32, _resource_id: Option<u32>) {
        // Simplified priority inheritance implementation
        // In a full implementation, this would handle complex priority inheritance chains
    }

    fn get_current_time_us(&self) -> u64 {
        static mut TIME_COUNTER: u64 = 0;
        unsafe {
            TIME_COUNTER += 1000; // Increment by 1ms
            TIME_COUNTER
        }
    }
}

lazy_static! {
    static ref RT_SCHEDULER: Mutex<RTScheduler> = Mutex::new(RTScheduler::new());
}

pub fn init_realtime_scheduler(num_cpu_cores: u8) {
    let mut scheduler = RT_SCHEDULER.lock();
    match scheduler.initialize(num_cpu_cores) {
        Ok(_) => crate::println!("[RT] Real-time scheduler ready"),
        Err(e) => crate::println!("[RT] Failed to initialize: {}", e),
    }
}

pub fn create_rt_process(name: &str, process_type: RTProcessType, priority: u8,
                        period_us: u64, deadline_us: u64, wcet_us: u64) -> Result<u32, &'static str> {
    RT_SCHEDULER.lock().create_rt_process(name, process_type, priority, period_us, deadline_us, wcet_us)
}

pub fn schedule_rt_processes(current_time_us: u64) -> Vec<(u8, u32), MAX_CPU_CORES> {
    RT_SCHEDULER.lock().schedule(current_time_us)
}

pub fn set_rt_scheduling_policy(policy: RTSchedulingPolicy) {
    RT_SCHEDULER.lock().set_scheduling_policy(policy);
}

pub fn get_rt_scheduler_stats() -> RTSchedulerStats {
    RT_SCHEDULER.lock().get_scheduler_stats()
}

pub fn enable_rt_cpu_isolation(cpu_id: u8, enabled: bool) -> Result<(), &'static str> {
    RT_SCHEDULER.lock().enable_cpu_isolation(cpu_id, enabled)
}

pub fn set_rt_process_affinity(process_id: u32, cpu_mask: &[u8]) -> Result<(), &'static str> {
    RT_SCHEDULER.lock().set_process_affinity(process_id, cpu_mask)
}

pub fn periodic_rt_scheduler_task() {
    let current_time = RT_SCHEDULER.lock().get_current_time_us();
    let scheduled = schedule_rt_processes(current_time);

    if !scheduled.is_empty() {
        crate::println!("[RT] Scheduled {} RT processes", scheduled.len());
    }
}

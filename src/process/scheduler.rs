//! Task Scheduler Implementation
//!
//! This module implements multiple scheduling algorithms including round-robin
//! and priority-based scheduling for RustOS processes.

use super::{Pid, Priority, get_system_time};
use alloc::collections::{BTreeMap, VecDeque};

/// Scheduling algorithm types
#[derive(Debug, Clone, Copy)]
pub enum SchedulingAlgorithm {
    /// Round-robin scheduling
    RoundRobin,
    /// Priority-based scheduling
    Priority,
    /// Multilevel feedback queue
    MultilevelFeedback,
}

/// Process queue for scheduling
#[derive(Debug)]
struct ProcessQueue {
    /// Processes in this queue
    processes: VecDeque<Pid>,
    /// Time slice for processes in this queue (ms)
    time_slice: u32,
    /// Priority level of this queue
    priority: Priority,
}

impl ProcessQueue {
    fn new(priority: Priority, time_slice: u32) -> Self {
        Self {
            processes: VecDeque::new(),
            time_slice,
            priority,
        }
    }

    fn add_process(&mut self, pid: Pid) {
        if !self.processes.contains(&pid) {
            self.processes.push_back(pid);
        }
    }

    fn remove_process(&mut self, pid: Pid) -> bool {
        if let Some(pos) = self.processes.iter().position(|&p| p == pid) {
            self.processes.remove(pos);
            true
        } else {
            false
        }
    }

    fn next_process(&mut self) -> Option<Pid> {
        self.processes.pop_front()
    }

    fn rotate_to_back(&mut self, pid: Pid) {
        self.processes.push_back(pid);
    }

    fn is_empty(&self) -> bool {
        self.processes.is_empty()
    }

    fn len(&self) -> usize {
        self.processes.len()
    }
}

/// Process scheduling statistics
#[derive(Debug, Clone)]
pub struct SchedulingStats {
    /// Total context switches
    pub context_switches: u64,
    /// Total scheduling decisions
    pub scheduling_decisions: u64,
    /// Average wait time per process
    pub average_wait_time: f32,
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Last scheduling decision time
    pub last_schedule_time: u64,
}

impl Default for SchedulingStats {
    fn default() -> Self {
        Self {
            context_switches: 0,
            scheduling_decisions: 0,
            average_wait_time: 0.0,
            cpu_utilization: 0.0,
            last_schedule_time: 0,
        }
    }
}

/// Main scheduler implementation
pub struct Scheduler {
    /// Current scheduling algorithm
    algorithm: SchedulingAlgorithm,
    /// Process queues by priority
    queues: BTreeMap<Priority, ProcessQueue>,
    /// Currently running process
    current_process: Option<Pid>,
    /// Process information for scheduling
    process_info: BTreeMap<Pid, ProcessSchedulingInfo>,
    /// Scheduling statistics
    stats: SchedulingStats,
    /// Time slice counter for current process
    current_time_slice: u32,
    /// Maximum time slice in ms
    max_time_slice: u32,
    /// Minimum time slice in ms
    min_time_slice: u32,
}

/// Per-process scheduling information
#[derive(Debug, Clone)]
struct ProcessSchedulingInfo {
    /// Process priority
    priority: Priority,
    /// Time when process was last scheduled
    last_scheduled: u64,
    /// Total CPU time used
    total_cpu_time: u64,
    /// Time when process became ready
    ready_time: u64,
    /// Number of times process has been scheduled
    schedule_count: u64,
    /// Process is currently blocked
    blocked: bool,
}

impl Scheduler {
    /// Create a new scheduler
    pub const fn new() -> Self {
        Self {
            algorithm: SchedulingAlgorithm::MultilevelFeedback,
            queues: BTreeMap::new(),
            current_process: None,
            process_info: BTreeMap::new(),
            stats: SchedulingStats {
                context_switches: 0,
                scheduling_decisions: 0,
                average_wait_time: 0.0,
                cpu_utilization: 0.0,
                last_schedule_time: 0,
            },
            current_time_slice: 0,
            max_time_slice: 100, // 100ms
            min_time_slice: 5,   // 5ms
        }
    }

    /// Initialize the scheduler
    pub fn init(&mut self) -> Result<(), &'static str> {
        // Initialize priority queues
        self.queues.insert(Priority::RealTime, ProcessQueue::new(Priority::RealTime, 50));
        self.queues.insert(Priority::High, ProcessQueue::new(Priority::High, 25));
        self.queues.insert(Priority::Normal, ProcessQueue::new(Priority::Normal, 10));
        self.queues.insert(Priority::Low, ProcessQueue::new(Priority::Low, 5));
        self.queues.insert(Priority::Idle, ProcessQueue::new(Priority::Idle, 1));

        self.stats.last_schedule_time = get_system_time();
        Ok(())
    }

    /// Add a process to the scheduler
    pub fn add_process(&mut self, pid: Pid, priority: Priority) -> Result<(), &'static str> {
        // Add process info
        self.process_info.insert(pid, ProcessSchedulingInfo {
            priority,
            last_scheduled: 0,
            total_cpu_time: 0,
            ready_time: get_system_time(),
            schedule_count: 0,
            blocked: false,
        });

        // Add to appropriate queue
        if let Some(queue) = self.queues.get_mut(&priority) {
            queue.add_process(pid);
        } else {
            return Err("Invalid priority level");
        }

        Ok(())
    }

    /// Remove a process from the scheduler
    pub fn remove_process(&mut self, pid: Pid) -> Result<(), &'static str> {
        // Remove from process info
        if let Some(info) = self.process_info.remove(&pid) {
            // Remove from queue
            if let Some(queue) = self.queues.get_mut(&info.priority) {
                queue.remove_process(pid);
            }

            // If this was the current process, clear it
            if self.current_process == Some(pid) {
                self.current_process = None;
                self.current_time_slice = 0;
            }
        } else {
            return Err("Process not found in scheduler");
        }

        Ok(())
    }

    /// Block a process (remove from ready queue but keep info)
    pub fn block_process(&mut self, pid: Pid) -> Result<(), &'static str> {
        if let Some(info) = self.process_info.get_mut(&pid) {
            info.blocked = true;

            // Remove from queue
            if let Some(queue) = self.queues.get_mut(&info.priority) {
                queue.remove_process(pid);
            }

            // If this was the current process, clear it
            if self.current_process == Some(pid) {
                self.current_process = None;
                self.current_time_slice = 0;
            }
        } else {
            return Err("Process not found in scheduler");
        }

        Ok(())
    }

    /// Unblock a process (add back to ready queue)
    pub fn unblock_process(&mut self, pid: Pid) -> Result<(), &'static str> {
        if let Some(info) = self.process_info.get_mut(&pid) {
            info.blocked = false;
            info.ready_time = get_system_time();

            // Add back to queue
            if let Some(queue) = self.queues.get_mut(&info.priority) {
                queue.add_process(pid);
            }
        } else {
            return Err("Process not found in scheduler");
        }

        Ok(())
    }

    /// Perform scheduling decision
    pub fn schedule(&mut self) -> Result<Option<Pid>, &'static str> {
        self.stats.scheduling_decisions += 1;
        let current_time = get_system_time();

        // Check if current process should be preempted
        let should_preempt = self.should_preempt(current_time);

        if !should_preempt && self.current_process.is_some() {
            // Continue with current process
            return Ok(self.current_process);
        }

        // If we're preempting, put current process back in queue
        if let Some(current_pid) = self.current_process {
            if let Some(info) = self.process_info.get(&current_pid) {
                if !info.blocked {
                    if let Some(queue) = self.queues.get_mut(&info.priority) {
                        queue.rotate_to_back(current_pid);
                    }
                }
            }
        }

        // Select next process based on algorithm
        let next_process = match self.algorithm {
            SchedulingAlgorithm::RoundRobin => self.round_robin_schedule(),
            SchedulingAlgorithm::Priority => self.priority_schedule(),
            SchedulingAlgorithm::MultilevelFeedback => self.multilevel_feedback_schedule(),
        };

        // Update scheduling info for selected process
        if let Some(pid) = next_process {
            let mut wait_info = None;

            if let Some(info) = self.process_info.get_mut(&pid) {
                info.last_scheduled = current_time;
                info.schedule_count += 1;

                wait_info = Some((
                    current_time.saturating_sub(info.ready_time),
                    info.priority,
                ));
            }

            if let Some((wait_time, priority)) = wait_info {
                self.update_average_wait_time(wait_time as f32);
                self.current_time_slice = self.queues.get(&priority)
                    .map(|q| q.time_slice)
                    .unwrap_or(self.min_time_slice);
            }

            // Update context switch count if process changed
            if self.current_process != next_process {
                self.stats.context_switches += 1;
            }
        }

        self.current_process = next_process;
        self.stats.last_schedule_time = current_time;

        Ok(next_process)
    }

    /// Check if current process should be preempted
    fn should_preempt(&self, current_time: u64) -> bool {
        if self.current_process.is_none() {
            return true;
        }

        // Time slice expired
        let time_since_schedule = current_time.saturating_sub(self.stats.last_schedule_time);
        if time_since_schedule >= self.current_time_slice as u64 {
            return true;
        }

        // Higher priority process available
        if let Some(current_pid) = self.current_process {
            if let Some(current_info) = self.process_info.get(&current_pid) {
                // Check if any higher priority queue has processes
                for (&priority, queue) in &self.queues {
                    if priority < current_info.priority && !queue.is_empty() {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Round-robin scheduling
    fn round_robin_schedule(&mut self) -> Option<Pid> {
        // Find first non-empty queue
        for (_, queue) in self.queues.iter_mut() {
            if !queue.is_empty() {
                return queue.next_process();
            }
        }
        None
    }

    /// Priority-based scheduling
    fn priority_schedule(&mut self) -> Option<Pid> {
        // Start from highest priority queue
        for (_, queue) in self.queues.iter_mut() {
            if !queue.is_empty() {
                return queue.next_process();
            }
        }
        None
    }

    /// Multilevel feedback queue scheduling
    fn multilevel_feedback_schedule(&mut self) -> Option<Pid> {
        // Same as priority for now, but could implement aging
        self.priority_schedule()
    }

    /// Update average wait time
    fn update_average_wait_time(&mut self, new_wait_time: f32) {
        let total_decisions = self.stats.scheduling_decisions as f32;
        self.stats.average_wait_time =
            (self.stats.average_wait_time * (total_decisions - 1.0) + new_wait_time) / total_decisions;
    }

    /// Get current process
    pub fn current_process(&self) -> Option<Pid> {
        self.current_process
    }

    /// Get scheduling statistics
    pub fn get_stats(&self) -> &SchedulingStats {
        &self.stats
    }

    /// Set scheduling algorithm
    pub fn set_algorithm(&mut self, algorithm: SchedulingAlgorithm) {
        self.algorithm = algorithm;
    }

    /// Get ready queue length
    pub fn ready_queue_length(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }

    /// Update process priority (for priority inheritance, etc.)
    pub fn update_process_priority(&mut self, pid: Pid, new_priority: Priority) -> Result<(), &'static str> {
        if let Some(info) = self.process_info.get_mut(&pid) {
            let old_priority = info.priority;
            info.priority = new_priority;

            // Move process between queues if not blocked
            if !info.blocked {
                // Remove from old queue
                if let Some(old_queue) = self.queues.get_mut(&old_priority) {
                    old_queue.remove_process(pid);
                }

                // Add to new queue
                if let Some(new_queue) = self.queues.get_mut(&new_priority) {
                    new_queue.add_process(pid);
                }
            }

            Ok(())
        } else {
            Err("Process not found")
        }
    }

    /// Tick the scheduler (called by timer interrupt)
    pub fn tick(&mut self) {
        // Decrement current time slice
        if self.current_time_slice > 0 {
            self.current_time_slice -= 1;
        }

        // Update CPU utilization
        let current_time = get_system_time();
        let time_diff = current_time.saturating_sub(self.stats.last_schedule_time);
        if time_diff > 0 {
            let utilization = if self.current_process.is_some() { 100.0 } else { 0.0 };
            self.stats.cpu_utilization =
                (self.stats.cpu_utilization * 0.9) + (utilization * 0.1);
        }

        // Update process CPU time
        if let Some(current_pid) = self.current_process {
            if let Some(info) = self.process_info.get_mut(&current_pid) {
                info.total_cpu_time += 1;
            }
        }
    }
}

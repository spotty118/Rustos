//! Task Management System for RustOS
//!
//! This module provides:
//! - Task structure and lifecycle management
//! - Basic cooperative scheduler
//! - Task switching and context management
//! - Integration with AI system for performance optimization

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, Waker};
use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;
use lazy_static::lazy_static;

/// Unique task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskId(u64);

impl TaskId {
    fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        TaskId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Task state enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskState {
    Ready,
    Running,
    Blocked,
    Terminated,
}

/// Task priority levels for scheduler optimization
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum TaskPriority {
    Idle = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

/// Task performance metrics for AI optimization
#[derive(Debug, Clone, Default)]
pub struct TaskMetrics {
    pub execution_time: u64,       // Microseconds
    pub cpu_cycles: u64,           // CPU cycles consumed
    pub context_switches: u32,     // Number of context switches
    pub memory_usage: usize,       // Peak memory usage in bytes
    pub io_operations: u32,        // I/O operations performed
    pub cache_misses: u32,         // Cache misses during execution
}

/// Represents a task in the system
pub struct Task {
    id: TaskId,
    future: Pin<Box<dyn Future<Output = ()>>>,
    state: TaskState,
    priority: TaskPriority,
    metrics: TaskMetrics,
    created_at: u64,               // Timestamp when task was created
    last_scheduled: u64,           // Last time task was scheduled
}

impl Task {
    /// Create a new task with the given future
    pub fn new(future: impl Future<Output = ()> + 'static) -> Task {
        Task {
            id: TaskId::new(),
            future: Box::pin(future),
            state: TaskState::Ready,
            priority: TaskPriority::default(),
            metrics: TaskMetrics::default(),
            created_at: crate::arch::get_timestamp(),
            last_scheduled: 0,
        }
    }

    /// Create a new task with specified priority
    pub fn new_with_priority(
        future: impl Future<Output = ()> + 'static,
        priority: TaskPriority,
    ) -> Task {
        Task {
            id: TaskId::new(),
            future: Box::pin(future),
            state: TaskState::Ready,
            priority,
            metrics: TaskMetrics::default(),
            created_at: crate::arch::get_timestamp(),
            last_scheduled: 0,
        }
    }

    /// Get task ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Get current task state
    pub fn state(&self) -> TaskState {
        self.state
    }

    /// Get task priority
    pub fn priority(&self) -> TaskPriority {
        self.priority
    }

    /// Get task performance metrics
    pub fn metrics(&self) -> &TaskMetrics {
        &self.metrics
    }

    /// Update task metrics
    pub fn update_metrics(&mut self, execution_time: u64, cpu_cycles: u64) {
        self.metrics.execution_time += execution_time;
        self.metrics.cpu_cycles += cpu_cycles;
        self.metrics.context_switches += 1;
    }

    /// Set task priority (used by AI scheduler optimization)
    pub fn set_priority(&mut self, priority: TaskPriority) {
        self.priority = priority;
    }

    /// Poll the task's future
    fn poll(&mut self, context: &mut Context) -> Poll<()> {
        self.state = TaskState::Running;
        let start_time = crate::arch::get_timestamp();
        let start_cycles = crate::arch::get_cpu_cycles();

        let result = self.future.as_mut().poll(context);

        let end_time = crate::arch::get_timestamp();
        let end_cycles = crate::arch::get_cpu_cycles();

        // Update metrics
        self.update_metrics(
            end_time.saturating_sub(start_time),
            end_cycles.saturating_sub(start_cycles),
        );

        match result {
            Poll::Ready(()) => {
                self.state = TaskState::Terminated;
            }
            Poll::Pending => {
                self.state = TaskState::Ready;
            }
        }

        result
    }
}

/// Task queue for different priority levels
#[derive(Default)]
struct PriorityQueue {
    critical: VecDeque<Task>,
    high: VecDeque<Task>,
    normal: VecDeque<Task>,
    low: VecDeque<Task>,
    idle: VecDeque<Task>,
}

impl PriorityQueue {
    fn push(&mut self, task: Task) {
        match task.priority {
            TaskPriority::Critical => self.critical.push_back(task),
            TaskPriority::High => self.high.push_back(task),
            TaskPriority::Normal => self.normal.push_back(task),
            TaskPriority::Low => self.low.push_back(task),
            TaskPriority::Idle => self.idle.push_back(task),
        }
    }

    fn pop(&mut self) -> Option<Task> {
        self.critical.pop_front()
            .or_else(|| self.high.pop_front())
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.low.pop_front())
            .or_else(|| self.idle.pop_front())
    }

    fn is_empty(&self) -> bool {
        self.critical.is_empty()
            && self.high.is_empty()
            && self.normal.is_empty()
            && self.low.is_empty()
            && self.idle.is_empty()
    }

    fn len(&self) -> usize {
        self.critical.len() + self.high.len() + self.normal.len() + self.low.len() + self.idle.len()
    }
}

/// Simple executor for running tasks
pub struct Executor {
    task_queue: PriorityQueue,
    waker_cache: Vec<(TaskId, Waker)>,
    scheduler_stats: SchedulerStats,
}

/// Scheduler performance statistics for AI optimization
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub total_tasks_executed: u64,
    pub total_execution_time: u64,
    pub average_task_time: u64,
    pub context_switches: u64,
    pub priority_adjustments: u32,
    pub scheduler_overhead: u64,
}

impl Executor {
    pub fn new() -> Self {
        Executor {
            task_queue: PriorityQueue::default(),
            waker_cache: Vec::new(),
            scheduler_stats: SchedulerStats::default(),
        }
    }

    /// Spawn a new task
    pub fn spawn(&mut self, future: impl Future<Output = ()> + 'static) -> TaskId {
        let task = Task::new(future);
        let task_id = task.id();
        self.task_queue.push(task);
        task_id
    }

    /// Spawn a new task with specified priority
    pub fn spawn_with_priority(
        &mut self,
        future: impl Future<Output = ()> + 'static,
        priority: TaskPriority,
    ) -> TaskId {
        let task = Task::new_with_priority(future, priority);
        let task_id = task.id();
        self.task_queue.push(task);
        task_id
    }

    /// Run the executor until all tasks are complete
    pub fn run(&mut self) -> Result<(), &'static str> {
        while !self.task_queue.is_empty() {
            self.run_ready_tasks()?;

            // Allow AI system to optimize scheduler if available
            self.ai_optimize_scheduler();

            // Yield to allow other system operations
            crate::arch::cpu_relax();
        }
        Ok(())
    }

    /// Run a single iteration of the scheduler
    pub fn run_once(&mut self) -> Result<bool, &'static str> {
        if self.task_queue.is_empty() {
            return Ok(false);
        }

        self.run_ready_tasks()?;
        self.ai_optimize_scheduler();
        Ok(!self.task_queue.is_empty())
    }

    /// Execute all ready tasks
    fn run_ready_tasks(&mut self) -> Result<(), &'static str> {
        let start_time = crate::arch::get_timestamp();
        let mut tasks_processed = 0;

        // Process tasks from priority queue
        let mut pending_tasks = VecDeque::new();

        while let Some(mut task) = self.task_queue.pop() {
            let waker = self.create_waker(task.id());
            let mut context = Context::from_waker(&waker);

            match task.poll(&mut context) {
                Poll::Ready(()) => {
                    // Task completed
                    self.scheduler_stats.total_tasks_executed += 1;
                    tasks_processed += 1;
                }
                Poll::Pending => {
                    // Task not ready, put it back in queue
                    pending_tasks.push_back(task);
                }
            }
        }

        // Put pending tasks back in queue
        for task in pending_tasks {
            self.task_queue.push(task);
        }

        // Update scheduler statistics
        let end_time = crate::arch::get_timestamp();
        let execution_time = end_time.saturating_sub(start_time);

        self.scheduler_stats.scheduler_overhead += execution_time;
        self.scheduler_stats.context_switches += tasks_processed;

        if self.scheduler_stats.total_tasks_executed > 0 {
            self.scheduler_stats.average_task_time =
                self.scheduler_stats.total_execution_time / self.scheduler_stats.total_tasks_executed;
        }

        Ok(())
    }

    /// Create a waker for the given task
    fn create_waker(&mut self, task_id: TaskId) -> Waker {
        // Simple waker implementation - in a real system this would
        // integrate with interrupt handling and async I/O
        use core::task::RawWaker;
        use core::task::RawWakerVTable;

        fn raw_waker_clone(data: *const ()) -> RawWaker {
            RawWaker::new(data, &VTABLE)
        }

        fn raw_waker_wake(_data: *const ()) {
            // In a real implementation, this would wake up the executor
        }

        fn raw_waker_wake_by_ref(_data: *const ()) {
            // In a real implementation, this would wake up the executor
        }

        fn raw_waker_drop(_data: *const ()) {
            // Cleanup if needed
        }

        static VTABLE: RawWakerVTable = RawWakerVTable::new(
            raw_waker_clone,
            raw_waker_wake,
            raw_waker_wake_by_ref,
            raw_waker_drop,
        );

        let raw_waker = RawWaker::new(core::ptr::null(), &VTABLE);
        unsafe { Waker::from_raw(raw_waker) }
    }

    /// AI-driven scheduler optimization
    fn ai_optimize_scheduler(&mut self) {
        // Collect scheduler metrics for AI analysis
        let scheduler_metrics = crate::ai::learning::SchedulerMetrics {
            task_count: self.task_queue.len() as u32,
            average_execution_time: self.scheduler_stats.average_task_time,
            context_switches: self.scheduler_stats.context_switches as u32,
            scheduler_overhead: self.scheduler_stats.scheduler_overhead,
        };

        // Let AI system analyze and potentially adjust priorities
        if let Some(optimization) = crate::ai::learning::analyze_scheduler_performance(&scheduler_metrics) {
            match optimization {
                crate::ai::learning::SchedulerOptimization::IncreaseQuantum => {
                    // Allow tasks to run longer before context switching
                },
                crate::ai::learning::SchedulerOptimization::DecreaseQuantum => {
                    // Switch tasks more frequently for better responsiveness
                },
                crate::ai::learning::SchedulerOptimization::AdjustPriorities => {
                    // AI suggests priority adjustments - implement as needed
                    self.scheduler_stats.priority_adjustments += 1;
                },
            }
        }
    }

    /// Get current scheduler statistics
    pub fn get_stats(&self) -> &SchedulerStats {
        &self.scheduler_stats
    }

    /// Get number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        self.task_queue.len()
    }
}

/// Global task executor
lazy_static! {
    pub static ref EXECUTOR: Mutex<Executor> = Mutex::new(Executor::new());
}

/// Spawn a new task on the global executor
pub fn spawn(future: impl Future<Output = ()> + 'static) -> TaskId {
    EXECUTOR.lock().spawn(future)
}

/// Spawn a task with specified priority
pub fn spawn_with_priority(
    future: impl Future<Output = ()> + 'static,
    priority: TaskPriority,
) -> TaskId {
    EXECUTOR.lock().spawn_with_priority(future, priority)
}

/// Run the global executor for one iteration
pub fn run_executor_once() -> Result<bool, &'static str> {
    EXECUTOR.lock().run_once()
}

/// Get executor statistics
pub fn get_executor_stats() -> SchedulerStats {
    EXECUTOR.lock().get_stats().clone()
}

/// Scheduler tick handler called by timer interrupt
pub fn scheduler_tick() {
    // Update scheduler timing and potentially trigger context switches
    let _ = run_executor_once();
}

/// Yield control back to scheduler
pub async fn yield_now() {
    YieldNow::new().await;
}

/// Initialize task management system
pub fn init() {
    crate::println!("[TASK] Task management system initialized");
    crate::println!("[TASK] Cooperative scheduler with AI optimization ready");

    // Spawn a test task to verify the system works
    spawn(async {
        crate::println!("[TASK] Test task executed successfully");
    });
}

/// Example async function for demonstration
pub async fn example_task() {
    crate::println!("[TASK] Example async task started");

    // Simulate some work
    for i in 0..5 {
        crate::println!("[TASK] Example task iteration {}", i);
        // Yield control back to scheduler
        YieldNow::new().await;
    }

    crate::println!("[TASK] Example async task completed");
}

/// Simple yield future for cooperative multitasking
pub struct YieldNow {
    yielded: bool,
}

impl YieldNow {
    pub fn new() -> Self {
        YieldNow { yielded: false }
    }
}

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_task_creation() {
        let task = Task::new(async {});
        assert_eq!(task.state(), TaskState::Ready);
        assert_eq!(task.priority(), TaskPriority::Normal);
    }

    #[test_case]
    fn test_task_priority() {
        let task = Task::new_with_priority(async {}, TaskPriority::High);
        assert_eq!(task.priority(), TaskPriority::High);
    }

    #[test_case]
    fn test_executor_creation() {
        let executor = Executor::new();
        assert_eq!(executor.pending_tasks(), 0);
    }
}

//! SMP (Symmetric Multi-Processing) Support for RustOS
//!
//! This module provides:
//! - CPU core detection and initialization
//! - Per-CPU data structures and management
//! - CPU topology discovery
//! - Load balancing across cores
//! - Inter-processor interrupts (IPI)
//! - CPU hotplug support
//! - NUMA-aware scheduling
//! - Spinlock and synchronization primitives
//! - Core-local storage and caching

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use lazy_static::lazy_static;
use spin::Mutex;
use x86_64::VirtAddr;

/// Maximum number of CPUs supported
pub const MAX_CPUS: usize = 256;

/// CPU states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CpuState {
    Offline,   // CPU is not available
    Booting,   // CPU is being brought online
    Online,    // CPU is active and scheduling tasks
    Idle,      // CPU is idle but available
    Shutdown,  // CPU is being shut down
    HotUnplug, // CPU is being hot-unplugged
}

/// CPU topology information
#[derive(Debug, Clone)]
pub struct CpuTopology {
    pub physical_id: u32, // Physical CPU package ID
    pub core_id: u32,     // Core ID within package
    pub thread_id: u32,   // Thread ID within core (for hyperthreading)
    pub numa_node: u32,   // NUMA node this CPU belongs to
    pub cache_levels: Vec<CacheInfo>,
}

/// CPU cache information
#[derive(Debug, Clone)]
pub struct CacheInfo {
    pub level: u8,             // Cache level (1, 2, 3, etc.)
    pub cache_type: CacheType, // Instruction, Data, or Unified
    pub size_kb: u32,          // Cache size in KB
    pub line_size: u32,        // Cache line size in bytes
    pub associativity: u32,    // Cache associativity
    pub shared_cpus: Vec<u32>, // CPUs sharing this cache
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheType {
    Instruction,
    Data,
    Unified,
}

/// Per-CPU data structure
#[repr(C)]
pub struct CpuData {
    pub cpu_id: u32,
    pub state: AtomicU32, // CpuState as u32
    pub topology: CpuTopology,
    pub current_task: Option<u32>, // Current running task ID
    pub idle_time: AtomicU64,
    pub run_queue_size: AtomicU32,
    pub load_average: AtomicU32, // Load * 1000 for fixed point
    pub frequency_mhz: AtomicU32,
    pub temperature_celsius: AtomicU32,
    pub power_state: AtomicU32,
    pub interrupt_count: AtomicU64,
    pub context_switches: AtomicU64,
    pub cache_misses: AtomicU64,
    pub instructions_retired: AtomicU64,
    pub local_timer_ticks: AtomicU64,
    pub ipi_received: AtomicU64,
    pub ipi_sent: AtomicU64,
}

impl CpuData {
    pub fn new(cpu_id: u32) -> Self {
        Self {
            cpu_id,
            state: AtomicU32::new(CpuState::Offline as u32),
            topology: CpuTopology {
                physical_id: 0,
                core_id: 0,
                thread_id: 0,
                numa_node: 0,
                cache_levels: Vec::new(),
            },
            current_task: None,
            idle_time: AtomicU64::new(0),
            run_queue_size: AtomicU32::new(0),
            load_average: AtomicU32::new(0),
            frequency_mhz: AtomicU32::new(0),
            temperature_celsius: AtomicU32::new(0),
            power_state: AtomicU32::new(0),
            interrupt_count: AtomicU64::new(0),
            context_switches: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            instructions_retired: AtomicU64::new(0),
            local_timer_ticks: AtomicU64::new(0),
            ipi_received: AtomicU64::new(0),
            ipi_sent: AtomicU64::new(0),
        }
    }

    pub fn get_state(&self) -> CpuState {
        match self.state.load(Ordering::Acquire) {
            0 => CpuState::Offline,
            1 => CpuState::Booting,
            2 => CpuState::Online,
            3 => CpuState::Idle,
            4 => CpuState::Shutdown,
            5 => CpuState::HotUnplug,
            _ => CpuState::Offline,
        }
    }

    pub fn set_state(&self, state: CpuState) {
        self.state.store(state as u32, Ordering::Release);
    }

    pub fn update_load_average(&self, new_load: f32) {
        let fixed_load = (new_load * 1000.0) as u32;
        let current = self.load_average.load(Ordering::Relaxed);
        // Exponential moving average: new_avg = 0.9 * old_avg + 0.1 * new_load
        let new_avg = ((current as u64 * 9 + fixed_load as u64) / 10) as u32;
        self.load_average.store(new_avg, Ordering::Relaxed);
    }

    pub fn get_load_average(&self) -> f32 {
        self.load_average.load(Ordering::Relaxed) as f32 / 1000.0
    }
}

/// Inter-Processor Interrupt (IPI) types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IpiType {
    Reschedule,   // Force reschedule on target CPU
    TlbFlush,     // Flush TLB on target CPU
    Halt,         // Halt target CPU
    FunctionCall, // Execute function on target CPU
    Timer,        // Timer interrupt
    Wakeup,       // Wake up idle CPU
}

/// IPI message structure
#[derive(Debug, Clone)]
pub struct IpiMessage {
    pub ipi_type: IpiType,
    pub sender_cpu: u32,
    pub data: u64,
    pub callback: Option<fn(u64)>,
}

/// CPU-local storage
pub struct CpuLocal<T> {
    data: UnsafeCell<Vec<T>>,
}

impl<T> CpuLocal<T> {
    pub const fn new() -> Self {
        Self {
            data: UnsafeCell::new(Vec::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: UnsafeCell::new(Vec::with_capacity(capacity)),
        }
    }

    /// Get reference to current CPU's data
    pub fn get(&self, cpu_id: u32) -> Option<&T> {
        unsafe {
            let vec = &*self.data.get();
            vec.get(cpu_id as usize)
        }
    }

    /// Get mutable reference to current CPU's data
    pub fn get_mut(&self, cpu_id: u32) -> Option<&mut T> {
        unsafe {
            let vec = &mut *self.data.get();
            vec.get_mut(cpu_id as usize)
        }
    }

    /// Set data for specific CPU
    pub fn set(&self, cpu_id: u32, value: T) {
        unsafe {
            let vec = &mut *self.data.get();
            if vec.len() <= cpu_id as usize {
                vec.resize_with(cpu_id as usize + 1, || panic!("Cannot resize CpuLocal"));
            }
            if cpu_id < vec.len() as u32 {
                vec[cpu_id as usize] = value;
            }
        }
    }
}

unsafe impl<T: Send> Sync for CpuLocal<T> {}
unsafe impl<T: Send> Send for CpuLocal<T> {}

/// SMP management system
pub struct SmpManager {
    cpu_data: [Option<Box<CpuData>>; MAX_CPUS],
    online_cpus: AtomicU32,
    total_cpus: u32,
    boot_cpu_id: u32,
    cpu_topology_map: BTreeMap<u32, CpuTopology>,
    numa_nodes: BTreeMap<u32, Vec<u32>>, // NUMA node -> CPU list
    ipi_queues: [Mutex<Vec<IpiMessage>>; MAX_CPUS],
    load_balancer_enabled: AtomicBool,
    power_management_enabled: AtomicBool,
}

impl SmpManager {
    pub fn new() -> Self {
        // Initialize IPI queues array safely
        let mut ipi_queues = Vec::with_capacity(MAX_CPUS);
        for _ in 0..MAX_CPUS {
            ipi_queues.push(Mutex::new(Vec::new()));
        }

        // Convert Vec to array - this is a workaround for const generic limitations
        let mut ipi_array: [Mutex<Vec<IpiMessage>>; MAX_CPUS] =
            unsafe { core::mem::transmute_copy(&ipi_queues) };
        core::mem::forget(ipi_queues);

        Self {
            cpu_data: [const { None }; MAX_CPUS],
            online_cpus: AtomicU32::new(1), // Boot CPU is online
            total_cpus: 1,
            boot_cpu_id: 0,
            cpu_topology_map: BTreeMap::new(),
            numa_nodes: BTreeMap::new(),
            ipi_queues: ipi_array,
            load_balancer_enabled: AtomicBool::new(true),
            power_management_enabled: AtomicBool::new(true),
        }
    }

    /// Initialize SMP system
    pub fn init(&mut self) -> Result<(), &'static str> {
        crate::println!("[SMP] Initializing SMP system...");

        // Detect CPUs
        self.detect_cpus()?;

        // Initialize boot CPU
        self.init_boot_cpu()?;

        // Discover CPU topology
        self.discover_topology()?;

        // Start application processors
        self.start_application_processors()?;

        crate::println!("[SMP] SMP initialization complete");
        crate::println!(
            "[SMP] Online CPUs: {}/{}",
            self.online_cpus.load(Ordering::Acquire),
            self.total_cpus
        );

        Ok(())
    }

    /// Detect available CPUs in the system
    fn detect_cpus(&mut self) -> Result<(), &'static str> {
        // In a real implementation, this would:
        // 1. Parse ACPI MADT (Multiple APIC Description Table)
        // 2. Read MP (Multi-Processor) configuration table
        // 3. Use CPUID to detect cores and threads

        // For now, simulate detection
        self.total_cpus = self.get_cpu_count_from_cpuid();

        crate::println!("[SMP] Detected {} CPUs", self.total_cpus);

        // Initialize CPU data structures
        for cpu_id in 0..self.total_cpus.min(MAX_CPUS as u32) {
            let cpu_data = Box::new(CpuData::new(cpu_id));
            self.cpu_data[cpu_id as usize] = Some(cpu_data);
        }

        Ok(())
    }

    /// Get CPU count using CPUID instruction
    fn get_cpu_count_from_cpuid(&self) -> u32 {
        // This would use actual CPUID instruction in real implementation
        // For now, return a reasonable default for testing
        4 // Simulate quad-core system
    }

    /// Initialize the boot CPU (BSP - Bootstrap Processor)
    fn init_boot_cpu(&mut self) -> Result<(), &'static str> {
        if let Some(cpu_data) = &self.cpu_data[self.boot_cpu_id as usize] {
            cpu_data.set_state(CpuState::Online);
            crate::println!("[SMP] Boot CPU {} initialized", self.boot_cpu_id);
        }
        Ok(())
    }

    /// Discover CPU topology and cache hierarchy
    fn discover_topology(&mut self) -> Result<(), &'static str> {
        for cpu_id in 0..self.total_cpus.min(MAX_CPUS as u32) {
            if let Some(cpu_data) = &mut self.cpu_data[cpu_id as usize] {
                // Detect topology using CPUID
                let topology = self.detect_cpu_topology(cpu_id);
                cpu_data.topology = topology.clone();
                self.cpu_topology_map.insert(cpu_id, topology.clone());

                // Group CPUs by NUMA node
                self.numa_nodes
                    .entry(topology.numa_node)
                    .or_insert_with(Vec::new)
                    .push(cpu_id);
            }
        }

        crate::println!("[SMP] CPU topology discovered");
        crate::println!("[SMP] NUMA nodes: {}", self.numa_nodes.len());

        Ok(())
    }

    /// Detect CPU topology for a specific CPU
    fn detect_cpu_topology(&self, cpu_id: u32) -> CpuTopology {
        // In real implementation, this would use CPUID extensively
        // For now, simulate realistic topology

        let physical_id = cpu_id / 2; // 2 cores per package
        let core_id = cpu_id % 2;
        let thread_id = 0; // No hyperthreading in this simulation
        let numa_node = cpu_id / 4; // 4 CPUs per NUMA node

        // Simulate cache hierarchy
        let mut cache_levels = Vec::new();

        // L1 cache (per core)
        cache_levels.push(CacheInfo {
            level: 1,
            cache_type: CacheType::Data,
            size_kb: 32,
            line_size: 64,
            associativity: 8,
            shared_cpus: vec![cpu_id],
        });

        // L2 cache (per core)
        cache_levels.push(CacheInfo {
            level: 2,
            cache_type: CacheType::Unified,
            size_kb: 256,
            line_size: 64,
            associativity: 8,
            shared_cpus: vec![cpu_id],
        });

        // L3 cache (shared across package)
        let package_cpus: Vec<u32> = (0..self.total_cpus)
            .filter(|&id| id / 2 == physical_id)
            .collect();

        cache_levels.push(CacheInfo {
            level: 3,
            cache_type: CacheType::Unified,
            size_kb: 8192,
            line_size: 64,
            associativity: 16,
            shared_cpus: package_cpus,
        });

        CpuTopology {
            physical_id,
            core_id,
            thread_id,
            numa_node,
            cache_levels,
        }
    }

    /// Start Application Processors (APs)
    fn start_application_processors(&mut self) -> Result<(), &'static str> {
        for cpu_id in 1..self.total_cpus.min(MAX_CPUS as u32) {
            if let Err(e) = self.start_cpu(cpu_id) {
                crate::println!("[SMP] Failed to start CPU {}: {}", cpu_id, e);
            } else {
                self.online_cpus.fetch_add(1, Ordering::AcqRel);
            }
        }
        Ok(())
    }

    /// Start a specific CPU
    fn start_cpu(&mut self, cpu_id: u32) -> Result<(), &'static str> {
        if let Some(cpu_data) = &self.cpu_data[cpu_id as usize] {
            cpu_data.set_state(CpuState::Booting);

            // In real implementation:
            // 1. Send INIT IPI
            // 2. Send SIPI (Startup IPI) with trampoline code address
            // 3. Wait for CPU to signal ready
            // 4. Initialize CPU-specific structures (GDT, IDT, etc.)

            // Simulate CPU startup
            crate::println!("[SMP] Starting CPU {}...", cpu_id);

            // Simulate successful startup
            cpu_data.set_state(CpuState::Online);
            crate::println!("[SMP] CPU {} online", cpu_id);

            return Ok(());
        }

        Err("CPU data not found")
    }

    /// Send Inter-Processor Interrupt
    pub fn send_ipi(&self, target_cpu: u32, ipi: IpiMessage) -> Result<(), &'static str> {
        if target_cpu >= MAX_CPUS as u32 {
            return Err("Invalid CPU ID");
        }

        if let Some(cpu_data) = &self.cpu_data[target_cpu as usize] {
            if cpu_data.get_state() != CpuState::Online {
                return Err("Target CPU not online");
            }

            // Queue the IPI message
            {
                let mut queue = self.ipi_queues[target_cpu as usize].lock();
                queue.push(ipi);
            }

            // In real implementation, would send hardware IPI here
            // For now, just update statistics
            cpu_data.ipi_received.fetch_add(1, Ordering::Relaxed);

            if let Some(sender_data) = &self.cpu_data[ipi.sender_cpu as usize] {
                sender_data.ipi_sent.fetch_add(1, Ordering::Relaxed);
            }

            Ok(())
        } else {
            Err("Target CPU not found")
        }
    }

    /// Process IPIs for current CPU
    pub fn process_ipis(&self, cpu_id: u32) {
        if cpu_id >= MAX_CPUS as u32 {
            return;
        }

        let mut queue = self.ipi_queues[cpu_id as usize].lock();
        while let Some(ipi) = queue.pop() {
            match ipi.ipi_type {
                IpiType::Reschedule => {
                    // Trigger reschedule on this CPU
                    crate::process::schedule_next();
                }
                IpiType::TlbFlush => {
                    // Flush TLB
                    self.flush_tlb();
                }
                IpiType::Halt => {
                    // Halt this CPU
                    if let Some(cpu_data) = &self.cpu_data[cpu_id as usize] {
                        cpu_data.set_state(CpuState::Shutdown);
                    }
                }
                IpiType::FunctionCall => {
                    // Execute callback function
                    if let Some(callback) = ipi.callback {
                        callback(ipi.data);
                    }
                }
                IpiType::Timer => {
                    // Handle timer interrupt
                    crate::time::timer_interrupt_handler();
                }
                IpiType::Wakeup => {
                    // Wake up from idle
                    if let Some(cpu_data) = &self.cpu_data[cpu_id as usize] {
                        if cpu_data.get_state() == CpuState::Idle {
                            cpu_data.set_state(CpuState::Online);
                        }
                    }
                }
            }
        }
    }

    /// Flush TLB (Translation Lookaside Buffer)
    fn flush_tlb(&self) {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                core::arch::asm!("mov {}, cr3; mov cr3, {}", out(reg) _, in(reg) _);
            }
        }
    }

    /// Get current CPU ID
    pub fn current_cpu_id(&self) -> u32 {
        // In real implementation, this would use CPU-local storage or APIC ID
        // For now, always return boot CPU
        self.boot_cpu_id
    }

    /// Get CPU data for specific CPU
    pub fn get_cpu_data(&self, cpu_id: u32) -> Option<&CpuData> {
        if cpu_id < MAX_CPUS as u32 {
            self.cpu_data[cpu_id as usize].as_ref().map(|b| b.as_ref())
        } else {
            None
        }
    }

    /// Get online CPU count
    pub fn online_cpu_count(&self) -> u32 {
        self.online_cpus.load(Ordering::Acquire)
    }

    /// Get total CPU count
    pub fn total_cpu_count(&self) -> u32 {
        self.total_cpus
    }

    /// Load balancing across CPUs
    pub fn balance_load(&self) {
        if !self.load_balancer_enabled.load(Ordering::Acquire) {
            return;
        }

        // Find most and least loaded CPUs
        let mut max_load = 0.0f32;
        let mut min_load = f32::MAX;
        let mut max_cpu = 0u32;
        let mut min_cpu = 0u32;

        for cpu_id in 0..self.total_cpus.min(MAX_CPUS as u32) {
            if let Some(cpu_data) = &self.cpu_data[cpu_id as usize] {
                if cpu_data.get_state() == CpuState::Online {
                    let load = cpu_data.get_load_average();
                    if load > max_load {
                        max_load = load;
                        max_cpu = cpu_id;
                    }
                    if load < min_load {
                        min_load = load;
                        min_cpu = cpu_id;
                    }
                }
            }
        }

        // If load difference is significant, migrate tasks
        if max_load - min_load > 0.5 {
            crate::println!(
                "[SMP] Load balancing: CPU {} ({:.2}) -> CPU {} ({:.2})",
                max_cpu,
                max_load,
                min_cpu,
                min_load
            );

            // Send IPI to trigger task migration
            let ipi = IpiMessage {
                ipi_type: IpiType::Reschedule,
                sender_cpu: self.current_cpu_id(),
                data: min_cpu as u64,
                callback: None,
            };

            let _ = self.send_ipi(max_cpu, ipi);
        }
    }

    /// Update CPU statistics
    pub fn update_cpu_stats(&self, cpu_id: u32) {
        if let Some(cpu_data) = &self.cpu_data[cpu_id as usize] {
            // Update load average based on run queue size
            let queue_size = cpu_data.run_queue_size.load(Ordering::Relaxed);
            let load = queue_size as f32 / 10.0; // Normalize
            cpu_data.update_load_average(load);

            // Update tick counter
            cpu_data.local_timer_ticks.fetch_add(1, Ordering::Relaxed);

            // Simulate performance counter updates
            cpu_data
                .instructions_retired
                .fetch_add(1000000, Ordering::Relaxed);
            cpu_data.cache_misses.fetch_add(100, Ordering::Relaxed);
        }
    }

    /// Power management - put idle CPUs to sleep
    pub fn manage_power(&self) {
        if !self.power_management_enabled.load(Ordering::Acquire) {
            return;
        }

        for cpu_id in 0..self.total_cpus.min(MAX_CPUS as u32) {
            if let Some(cpu_data) = &self.cpu_data[cpu_id as usize] {
                if cpu_data.get_state() == CpuState::Online {
                    let load = cpu_data.get_load_average();
                    let queue_size = cpu_data.run_queue_size.load(Ordering::Relaxed);

                    // If CPU is idle and has no tasks, put it to idle state
                    if load < 0.1 && queue_size == 0 && cpu_id != self.boot_cpu_id {
                        cpu_data.set_state(CpuState::Idle);
                        crate::println!("[SMP] CPU {} entering idle state", cpu_id);
                    }
                }
            }
        }
    }

    /// NUMA-aware memory allocation hint
    pub fn get_numa_node(&self, cpu_id: u32) -> u32 {
        if let Some(cpu_data) = &self.cpu_data[cpu_id as usize] {
            cpu_data.topology.numa_node
        } else {
            0 // Default to node 0
        }
    }

    /// Get CPUs in the same NUMA node
    pub fn get_numa_cpus(&self, numa_node: u32) -> Vec<u32> {
        self.numa_nodes
            .get(&numa_node)
            .cloned()
            .unwrap_or_else(Vec::new)
    }

    /// Get comprehensive SMP statistics
    pub fn get_smp_stats(&self) -> SmpStatistics {
        let mut stats = SmpStatistics {
            total_cpus: self.total_cpus,
            online_cpus: self.online_cpus.load(Ordering::Acquire),
            idle_cpus: 0,
            total_interrupts: 0,
            total_context_switches: 0,
            total_ipi_sent: 0,
            total_ipi_received: 0,
            average_load: 0.0,
            numa_nodes: self.numa_nodes.len() as u32,
            cache_miss_rate: 0.0,
        };

        let mut total_load = 0.0f32;
        let mut online_count = 0u32;

        for cpu_id in 0..self.total_cpus.min(MAX_CPUS as u32) {
            if let Some(cpu_data) = &self.cpu_data[cpu_id as usize] {
                match cpu_data.get_state() {
                    CpuState::Online => {
                        online_count += 1;
                        total_load += cpu_data.get_load_average();
                    }
                    CpuState::Idle => {
                        stats.idle_cpus += 1;
                        online_count += 1;
                    }
                    _ => {}
                }

                stats.total_interrupts += cpu_data.interrupt_count.load(Ordering::Relaxed);
                stats.total_context_switches += cpu_data.context_switches.load(Ordering::Relaxed);
                stats.total_ipi_sent += cpu_data.ipi_sent.load(Ordering::Relaxed);
                stats.total_ipi_received += cpu_data.ipi_received.load(Ordering::Relaxed);

                // Calculate cache miss rate
                let instructions = cpu_data.instructions_retired.load(Ordering::Relaxed);
                let misses = cpu_data.cache_misses.load(Ordering::Relaxed);
                if instructions > 0 {
                    stats.cache_miss_rate += (misses as f32 / instructions as f32) * 100.0;
                }
            }
        }

        if online_count > 0 {
            stats.average_load = total_load / online_count as f32;
            stats.cache_miss_rate /= online_count as f32;
        }

        stats
    }
}

/// SMP statistics structure
#[derive(Debug, Clone)]
pub struct SmpStatistics {
    pub total_cpus: u32,
    pub online_cpus: u32,
    pub idle_cpus: u32,
    pub total_interrupts: u64,
    pub total_context_switches: u64,
    pub total_ipi_sent: u64,
    pub total_ipi_received: u64,
    pub average_load: f32,
    pub numa_nodes: u32,
    pub cache_miss_rate: f32,
}

/// Global SMP manager
lazy_static! {
    pub static ref SMP_MANAGER: Mutex<SmpManager> = Mutex::new(SmpManager::new());
}

/// Initialize SMP system
pub fn init() -> Result<(), &'static str> {
    let mut smp = SMP_MANAGER.lock();
    smp.init()?;

    crate::status::register_subsystem(
        "SMP",
        crate::status::SystemStatus::Running,
        "SMP system operational",
    );
    Ok(())
}

/// Get current CPU ID
pub fn current_cpu() -> u32 {
    SMP_MANAGER.lock().current_cpu_id()
}

/// Get online CPU count
pub fn online_cpus() -> u32 {
    SMP_MANAGER.lock().online_cpu_count()
}

/// Send IPI to specific CPU
pub fn send_ipi(target_cpu: u32, ipi_type: IpiType, data: u64) -> Result<(), &'static str> {
    let current_cpu = current_cpu();
    let ipi = IpiMessage {
        ipi_type,
        sender_cpu: current_cpu,
        data,
        callback: None,
    };

    SMP_MANAGER.lock().send_ipi(target_cpu, ipi)
}

/// Send function call IPI
pub fn send_function_ipi(
    target_cpu: u32,
    callback: fn(u64),
    data: u64,
) -> Result<(), &'static str> {
    let current_cpu = current_cpu();
    let ipi = IpiMessage {
        ipi_type: IpiType::FunctionCall,
        sender_cpu: current_cpu,
        data,
        callback: Some(callback),
    };

    SMP_MANAGER.lock().send_ipi(target_cpu, ipi)
}

/// Process IPIs for current CPU (called from interrupt handler)
pub fn process_ipis() {
    let cpu_id = current_cpu();
    SMP_MANAGER.lock().process_ipis(cpu_id);
}

/// Trigger load balancing
pub fn balance_load() {
    SMP_MANAGER.lock().balance_load();
}

/// Update CPU performance counters
pub fn update_cpu_stats() {
    let cpu_id = current_cpu();
    SMP_MANAGER.lock().update_cpu_stats(cpu_id);
}

/// SMP periodic maintenance task
pub async fn smp_maintenance_task() {
    loop {
        // Process IPIs
        process_ipis();

        // Update statistics
        update_cpu_stats();

        // Load balancing
        balance_load();

        // Power management
        SMP_MANAGER.lock().manage_power();

        // Sleep for 100ms
        crate::time::sleep_ms(100).await;
    }
}

/// Get SMP statistics
pub fn get_smp_statistics() -> SmpStatistics {
    SMP_MANAGER.lock().get_smp_stats()
}

/// Get NUMA node for current CPU
pub fn current_numa_node() -> u32 {
    let cpu_id = current_cpu();
    SMP_MANAGER.lock().get_numa_node(cpu_id)
}

/// Get CPUs in same NUMA node as current CPU
pub fn numa_local_cpus() -> Vec<u32> {
    let numa_node = current_numa_node();
    SMP_MANAGER.lock().get_numa_cpus(numa_node)
}

/// Broadcast IPI to all CPUs except current
pub fn broadcast_ipi(ipi_type: IpiType, data: u64) -> Result<u32, &'static str> {
    let current_cpu = current_cpu();
    let smp = SMP_MANAGER.lock();
    let mut sent_count = 0;

    for cpu_id in 0..smp.total_cpu_count() {
        if cpu_id != current_cpu {
            let ipi = IpiMessage {
                ipi_type,
                sender_cpu: current_cpu,
                data,
                callback: None,
            };

            if smp.send_ipi(cpu_id, ipi).is_ok() {
                sent_count += 1;
            }
        }
    }

    Ok(sent_count)
}

/// CPU hotplug - bring CPU online
pub fn cpu_online(cpu_id: u32) -> Result<(), &'static str> {
    let mut smp = SMP_MANAGER.lock();
    smp.start_cpu(cpu_id)?;
    smp.online_cpus.fetch_add(1, Ordering::AcqRel);
    Ok(())
}

/// CPU hotplug - take CPU offline
pub fn cpu_offline(cpu_id: u32) -> Result<(), &'static str> {
    if cpu_id == 0 {
        return Err("Cannot offline boot CPU");
    }

    // Send halt IPI
    let ipi = IpiMessage {
        ipi_type: IpiType::Halt,
        sender_cpu: current_cpu(),
        data: 0,
        callback: None,
    };

    let smp = SMP_MANAGER.lock();
    smp.send_ipi(cpu_id, ipi)?;
    smp.online_cpus.fetch_sub(1, Ordering::AcqRel);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_cpu_data_creation() {
        let cpu_data = CpuData::new(0);
        assert_eq!(cpu_data.cpu_id, 0);
        assert_eq!(cpu_data.get_state(), CpuState::Offline);
        assert_eq!(cpu_data.get_load_average(), 0.0);
    }

    #[test_case]
    fn test_cpu_state_transitions() {
        let cpu_data = CpuData::new(1);
        cpu_data.set_state(CpuState::Online);
        assert_eq!(cpu_data.get_state(), CpuState::Online);

        cpu_data.set_state(CpuState::Idle);
        assert_eq!(cpu_data.get_state(), CpuState::Idle);
    }

    #[test_case]
    fn test_load_average_calculation() {
        let cpu_data = CpuData::new(0);
        cpu_data.update_load_average(1.0);
        assert!((cpu_data.get_load_average() - 0.1).abs() < 0.001);

        cpu_data.update_load_average(0.5);
        let avg = cpu_data.get_load_average();
        assert!(avg > 0.1 && avg < 0.5);
    }

    #[test_case]
    fn test_ipi_message_creation() {
        let ipi = IpiMessage {
            ipi_type: IpiType::Reschedule,
            sender_cpu: 0,
            data: 42,
            callback: None,
        };

        assert_eq!(ipi.ipi_type, IpiType::Reschedule);
        assert_eq!(ipi.sender_cpu, 0);
        assert_eq!(ipi.data, 42);
    }

    #[test_case]
    fn test_cpu_local_storage() {
        let cpu_local: CpuLocal<u32> = CpuLocal::new();
        cpu_local.set(0, 100);
        cpu_local.set(1, 200);

        assert_eq!(cpu_local.get(0), Some(&100));
        assert_eq!(cpu_local.get(1), Some(&200));
        assert_eq!(cpu_local.get(2), None);
    }

    #[test_case]
    fn test_cache_info() {
        let cache = CacheInfo {
            level: 1,
            cache_type: CacheType::Data,
            size_kb: 32,
            line_size: 64,
            associativity: 8,
            shared_cpus: vec![0, 1],
        };

        assert_eq!(cache.level, 1);
        assert_eq!(cache.cache_type, CacheType::Data);
        assert_eq!(cache.shared_cpus.len(), 2);
    }

    #[test_case]
    fn test_smp_manager_creation() {
        let smp = SmpManager::new();
        assert_eq!(smp.total_cpus, 1);
        assert_eq!(smp.boot_cpu_id, 0);
        assert_eq!(smp.online_cpus.load(Ordering::Acquire), 1);
    }
}

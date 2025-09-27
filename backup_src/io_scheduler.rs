//! High-Performance I/O Scheduler
//!
//! This module implements advanced I/O scheduling algorithms for the RustOS kernel,
//! providing optimal disk scheduling, request merging, I/O prioritization,
//! and adaptive scheduling based on workload patterns for maximum throughput
//! and minimal latency.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of I/O requests in queue
const MAX_IO_REQUESTS: usize = 512;
/// Maximum number of I/O queues
const MAX_IO_QUEUES: usize = 16;
/// Maximum number of storage devices
const MAX_STORAGE_DEVICES: usize = 8;
/// I/O batch size for grouped operations
const IO_BATCH_SIZE: usize = 32;
/// Maximum I/O request merge distance
const MAX_MERGE_DISTANCE: u64 = 1024 * 1024; // 1MB

/// I/O request types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IORequestType {
    Read,
    Write,
    Sync,
    Trim,
    Flush,
    Barrier,
    Discard,
    WriteZeros,
}

impl fmt::Display for IORequestType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IORequestType::Read => write!(f, "Read"),
            IORequestType::Write => write!(f, "Write"),
            IORequestType::Sync => write!(f, "Sync"),
            IORequestType::Trim => write!(f, "Trim"),
            IORequestType::Flush => write!(f, "Flush"),
            IORequestType::Barrier => write!(f, "Barrier"),
            IORequestType::Discard => write!(f, "Discard"),
            IORequestType::WriteZeros => write!(f, "WriteZeros"),
        }
    }
}

/// I/O scheduling algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IOSchedulingAlgorithm {
    /// First-Come, First-Served
    FCFS,
    /// Shortest Seek Time First
    SSTF,
    /// SCAN (Elevator) algorithm
    SCAN,
    /// C-SCAN (Circular SCAN)
    CSCAN,
    /// LOOK algorithm
    LOOK,
    /// C-LOOK algorithm
    CLOOK,
    /// Deadline scheduler
    Deadline,
    /// Completely Fair Queuing
    CFQ,
    /// NOOP (No-op) scheduler
    NOOP,
    /// Budget Fair Queuing
    BFQ,
    /// Kyber multiqueue scheduler
    Kyber,
    /// Multi-Queue Block I/O
    MQDeadline,
}

impl fmt::Display for IOSchedulingAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IOSchedulingAlgorithm::FCFS => write!(f, "FCFS"),
            IOSchedulingAlgorithm::SSTF => write!(f, "SSTF"),
            IOSchedulingAlgorithm::SCAN => write!(f, "SCAN"),
            IOSchedulingAlgorithm::CSCAN => write!(f, "C-SCAN"),
            IOSchedulingAlgorithm::LOOK => write!(f, "LOOK"),
            IOSchedulingAlgorithm::CLOOK => write!(f, "C-LOOK"),
            IOSchedulingAlgorithm::Deadline => write!(f, "Deadline"),
            IOSchedulingAlgorithm::CFQ => write!(f, "CFQ"),
            IOSchedulingAlgorithm::NOOP => write!(f, "NOOP"),
            IOSchedulingAlgorithm::BFQ => write!(f, "BFQ"),
            IOSchedulingAlgorithm::Kyber => write!(f, "Kyber"),
            IOSchedulingAlgorithm::MQDeadline => write!(f, "MQ-Deadline"),
        }
    }
}

/// I/O priority levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum IOPriority {
    IdleClass,
    BestEffort0,
    BestEffort1,
    BestEffort2,
    BestEffort3,
    BestEffort4,
    BestEffort5,
    BestEffort6,
    BestEffort7,
    RealTime0,
    RealTime1,
    RealTime2,
    RealTime3,
    RealTime4,
    RealTime5,
    RealTime6,
    RealTime7,
}

impl IOPriority {
    pub fn to_numeric(&self) -> u8 {
        match self {
            IOPriority::IdleClass => 0,
            IOPriority::BestEffort0 => 1,
            IOPriority::BestEffort1 => 2,
            IOPriority::BestEffort2 => 3,
            IOPriority::BestEffort3 => 4,
            IOPriority::BestEffort4 => 5,
            IOPriority::BestEffort5 => 6,
            IOPriority::BestEffort6 => 7,
            IOPriority::BestEffort7 => 8,
            IOPriority::RealTime0 => 9,
            IOPriority::RealTime1 => 10,
            IOPriority::RealTime2 => 11,
            IOPriority::RealTime3 => 12,
            IOPriority::RealTime4 => 13,
            IOPriority::RealTime5 => 14,
            IOPriority::RealTime6 => 15,
            IOPriority::RealTime7 => 16,
        }
    }

    pub fn is_real_time(&self) -> bool {
        matches!(self,
            IOPriority::RealTime0 | IOPriority::RealTime1 | IOPriority::RealTime2 | IOPriority::RealTime3 |
            IOPriority::RealTime4 | IOPriority::RealTime5 | IOPriority::RealTime6 | IOPriority::RealTime7
        )
    }
}

/// I/O request status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IORequestStatus {
    Queued,
    Dispatched,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Merged,
}

impl fmt::Display for IORequestStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IORequestStatus::Queued => write!(f, "Queued"),
            IORequestStatus::Dispatched => write!(f, "Dispatched"),
            IORequestStatus::InProgress => write!(f, "In Progress"),
            IORequestStatus::Completed => write!(f, "Completed"),
            IORequestStatus::Failed => write!(f, "Failed"),
            IORequestStatus::Cancelled => write!(f, "Cancelled"),
            IORequestStatus::Merged => write!(f, "Merged"),
        }
    }
}

/// I/O request descriptor
#[derive(Debug, Clone)]
pub struct IORequest {
    pub request_id: u64,
    pub request_type: IORequestType,
    pub device_id: u8,
    pub sector_start: u64,
    pub sector_count: u32,
    pub data_size: u32,
    pub priority: IOPriority,
    pub status: IORequestStatus,
    pub process_id: u32,
    pub submit_time_us: u64,
    pub dispatch_time_us: Option<u64>,
    pub completion_time_us: Option<u64>,
    pub deadline_us: Option<u64>,
    pub merged_count: u16,
    pub retry_count: u8,
}

impl IORequest {
    pub fn new(request_id: u64, request_type: IORequestType, device_id: u8,
               sector_start: u64, sector_count: u32, priority: IOPriority,
               process_id: u32, current_time_us: u64) -> Self {
        let deadline_us = if priority.is_real_time() {
            Some(current_time_us + 50_000) // 50ms deadline for RT
        } else {
            Some(current_time_us + 500_000) // 500ms deadline for BE
        };

        Self {
            request_id,
            request_type,
            device_id,
            sector_start,
            sector_count,
            data_size: sector_count * 512, // Assume 512-byte sectors
            priority,
            status: IORequestStatus::Queued,
            process_id,
            submit_time_us: current_time_us,
            dispatch_time_us: None,
            completion_time_us: None,
            deadline_us,
            merged_count: 0,
            retry_count: 0,
        }
    }

    pub fn sector_end(&self) -> u64 {
        self.sector_start + self.sector_count as u64
    }

    pub fn is_adjacent(&self, other: &IORequest) -> bool {
        self.device_id == other.device_id &&
        self.request_type == other.request_type &&
        (self.sector_end() == other.sector_start || other.sector_end() == self.sector_start)
    }

    pub fn can_merge(&self, other: &IORequest) -> bool {
        self.device_id == other.device_id &&
        self.request_type == other.request_type &&
        self.priority == other.priority &&
        self.is_adjacent(other) &&
        (self.data_size + other.data_size) <= MAX_MERGE_DISTANCE as u32
    }

    pub fn merge_with(&mut self, other: &IORequest) -> bool {
        if !self.can_merge(other) {
            return false;
        }

        // Merge the requests
        if other.sector_start < self.sector_start {
            self.sector_start = other.sector_start;
        }

        self.sector_count += other.sector_count;
        self.data_size += other.data_size;
        self.merged_count += 1 + other.merged_count;

        // Keep the earlier deadline
        if let (Some(self_deadline), Some(other_deadline)) = (self.deadline_us, other.deadline_us) {
            self.deadline_us = Some(self_deadline.min(other_deadline));
        }

        true
    }

    pub fn is_expired(&self, current_time_us: u64) -> bool {
        if let Some(deadline) = self.deadline_us {
            current_time_us > deadline
        } else {
            false
        }
    }

    pub fn wait_time_us(&self, current_time_us: u64) -> u64 {
        current_time_us - self.submit_time_us
    }

    pub fn service_time_us(&self) -> Option<u64> {
        if let (Some(dispatch), Some(completion)) = (self.dispatch_time_us, self.completion_time_us) {
            Some(completion - dispatch)
        } else {
            None
        }
    }
}

/// Storage device characteristics
#[derive(Debug, Clone)]
pub struct StorageDevice {
    pub device_id: u8,
    pub device_name: heapless::String<32>,
    pub is_rotational: bool,
    pub sector_size: u32,
    pub total_sectors: u64,
    pub queue_depth: u16,
    pub max_sectors_per_request: u32,
    pub current_head_position: u64,
    pub seek_time_per_track_ns: u32,
    pub rotation_latency_ns: u32,
    pub transfer_rate_mbps: u32,
    pub active_requests: u16,
    pub total_requests_completed: u64,
    pub total_bytes_transferred: u64,
    pub average_response_time_us: u64,
}

impl StorageDevice {
    pub fn new_ssd(device_id: u8, name: &str, capacity_gb: u32) -> Self {
        let mut device_name = heapless::String::new();
        let _ = device_name.push_str(name);

        Self {
            device_id,
            device_name,
            is_rotational: false,
            sector_size: 512,
            total_sectors: (capacity_gb as u64) * 1024 * 1024 * 1024 / 512,
            queue_depth: 32,
            max_sectors_per_request: 256,
            current_head_position: 0,
            seek_time_per_track_ns: 0, // SSDs have no seek time
            rotation_latency_ns: 0, // SSDs have no rotation
            transfer_rate_mbps: 500, // 500 MB/s for typical SSD
            active_requests: 0,
            total_requests_completed: 0,
            total_bytes_transferred: 0,
            average_response_time_us: 100, // 0.1ms typical SSD latency
        }
    }

    pub fn new_hdd(device_id: u8, name: &str, capacity_gb: u32) -> Self {
        let mut device_name = heapless::String::new();
        let _ = device_name.push_str(name);

        Self {
            device_id,
            device_name,
            is_rotational: true,
            sector_size: 512,
            total_sectors: (capacity_gb as u64) * 1024 * 1024 * 1024 / 512,
            queue_depth: 16,
            max_sectors_per_request: 128,
            current_head_position: 0,
            seek_time_per_track_ns: 9_000_000, // 9ms average seek time
            rotation_latency_ns: 4_200_000, // 7200 RPM = ~4.2ms average rotation
            transfer_rate_mbps: 150, // 150 MB/s for typical HDD
            active_requests: 0,
            total_requests_completed: 0,
            total_bytes_transferred: 0,
            average_response_time_us: 10_000, // 10ms typical HDD latency
        }
    }

    pub fn can_accept_request(&self) -> bool {
        self.active_requests < self.queue_depth
    }

    pub fn estimated_service_time(&self, request: &IORequest) -> u64 {
        if !self.is_rotational {
            // SSD: constant latency + transfer time
            let transfer_time_us = (request.data_size as u64 * 1_000_000) / (self.transfer_rate_mbps as u64 * 1_000_000);
            return 100 + transfer_time_us; // 0.1ms base latency + transfer
        }

        // HDD: seek + rotation + transfer
        let seek_distance = (request.sector_start as i64 - self.current_head_position as i64).abs() as u64;
        let seek_time_us = (seek_distance * self.seek_time_per_track_ns as u64) / (1000 * 1000);
        let rotation_time_us = self.rotation_latency_ns as u64 / 1000;
        let transfer_time_us = (request.data_size as u64 * 1_000_000) / (self.transfer_rate_mbps as u64 * 1_000_000);

        seek_time_us + rotation_time_us + transfer_time_us
    }

    pub fn utilization_percent(&self) -> f32 {
        if self.queue_depth > 0 {
            (self.active_requests as f32 / self.queue_depth as f32) * 100.0
        } else {
            0.0
        }
    }
}

/// I/O queue for specific priority class
#[derive(Debug, Clone)]
pub struct IOQueue {
    pub queue_id: u8,
    pub priority_class: IOPriority,
    pub requests: Vec<IORequest, MAX_IO_REQUESTS>,
    pub algorithm: IOSchedulingAlgorithm,
    pub bandwidth_quota_mbps: u32,
    pub current_bandwidth_usage: u32,
    pub time_slice_us: u64,
    pub last_dispatch_time_us: u64,
    pub total_dispatched: u64,
    pub total_completed: u64,
    pub average_wait_time_us: u64,
}

impl IOQueue {
    pub fn new(queue_id: u8, priority_class: IOPriority, algorithm: IOSchedulingAlgorithm) -> Self {
        let bandwidth_quota = if priority_class.is_real_time() {
            100 // 100 MB/s for RT queues
        } else {
            50  // 50 MB/s for BE queues
        };

        let time_slice = if priority_class.is_real_time() {
            10_000 // 10ms time slice for RT
        } else {
            50_000 // 50ms time slice for BE
        };

        Self {
            queue_id,
            priority_class,
            requests: Vec::new(),
            algorithm,
            bandwidth_quota_mbps: bandwidth_quota,
            current_bandwidth_usage: 0,
            time_slice_us: time_slice,
            last_dispatch_time_us: 0,
            total_dispatched: 0,
            total_completed: 0,
            average_wait_time_us: 0,
        }
    }

    pub fn add_request(&mut self, request: IORequest) -> Result<(), &'static str> {
        if self.requests.len() >= MAX_IO_REQUESTS {
            return Err("Queue full");
        }

        // Try to merge with existing requests
        for existing_request in &mut self.requests {
            if existing_request.status == IORequestStatus::Queued &&
               existing_request.can_merge(&request) {
                if existing_request.merge_with(&request) {
                    return Ok(());
                }
            }
        }

        // Add new request if no merge possible
        let _ = self.requests.push(request);
        Ok(())
    }

    pub fn get_next_request(&mut self, current_head_position: u64) -> Option<usize> {
        if self.requests.is_empty() {
            return None;
        }

        match self.algorithm {
            IOSchedulingAlgorithm::FCFS => self.fcfs_schedule(),
            IOSchedulingAlgorithm::SSTF => self.sstf_schedule(current_head_position),
            IOSchedulingAlgorithm::SCAN => self.scan_schedule(current_head_position),
            IOSchedulingAlgorithm::LOOK => self.look_schedule(current_head_position),
            IOSchedulingAlgorithm::Deadline => self.deadline_schedule(),
            IOSchedulingAlgorithm::CFQ => self.cfq_schedule(),
            IOSchedulingAlgorithm::NOOP => self.noop_schedule(),
            _ => self.fcfs_schedule(), // Fallback
        }
    }

    pub fn remove_request(&mut self, index: usize) -> Option<IORequest> {
        if index < self.requests.len() {
            Some(self.requests.remove(index))
        } else {
            None
        }
    }

    pub fn expire_requests(&mut self, current_time_us: u64) -> Vec<IORequest, 32> {
        let mut expired = Vec::new();
        let mut i = 0;

        while i < self.requests.len() {
            if self.requests[i].is_expired(current_time_us) {
                if let Some(request) = self.remove_request(i) {
                    let _ = expired.push(request);
                }
            } else {
                i += 1;
            }
        }

        expired
    }

    pub fn queue_length(&self) -> usize {
        self.requests.len()
    }

    pub fn average_priority(&self) -> f32 {
        if self.requests.is_empty() {
            return 0.0;
        }

        let total_priority: u32 = self.requests.iter()
            .map(|r| r.priority.to_numeric() as u32)
            .sum();

        total_priority as f32 / self.requests.len() as f32
    }

    // Scheduling algorithms implementation
    fn fcfs_schedule(&mut self) -> Option<usize> {
        // First-Come, First-Served: just return the first queued request
        self.requests.iter().position(|r| r.status == IORequestStatus::Queued)
    }

    fn sstf_schedule(&mut self, current_head_position: u64) -> Option<usize> {
        // Shortest Seek Time First: find request with minimum seek distance
        let mut best_index = None;
        let mut shortest_distance = u64::MAX;

        for (index, request) in self.requests.iter().enumerate() {
            if request.status == IORequestStatus::Queued {
                let distance = (request.sector_start as i64 - current_head_position as i64).abs() as u64;
                if distance < shortest_distance {
                    shortest_distance = distance;
                    best_index = Some(index);
                }
            }
        }

        best_index
    }

    fn scan_schedule(&mut self, current_head_position: u64) -> Option<usize> {
        // SCAN: service requests in one direction, then reverse
        static mut SCAN_DIRECTION_UP: bool = true;

        let mut best_index = None;
        let mut best_distance = u64::MAX;

        unsafe {
            for (index, request) in self.requests.iter().enumerate() {
                if request.status == IORequestStatus::Queued {
                    let distance = (request.sector_start as i64 - current_head_position as i64).abs() as u64;

                    if SCAN_DIRECTION_UP && request.sector_start >= current_head_position {
                        if distance < best_distance {
                            best_distance = distance;
                            best_index = Some(index);
                        }
                    } else if !SCAN_DIRECTION_UP && request.sector_start <= current_head_position {
                        if distance < best_distance {
                            best_distance = distance;
                            best_index = Some(index);
                        }
                    }
                }
            }

            // If no request found in current direction, change direction
            if best_index.is_none() {
                SCAN_DIRECTION_UP = !SCAN_DIRECTION_UP;
                return self.scan_schedule(current_head_position);
            }
        }

        best_index
    }

    fn look_schedule(&mut self, current_head_position: u64) -> Option<usize> {
        // LOOK: like SCAN but only goes as far as the last request
        static mut LOOK_DIRECTION_UP: bool = true;

        let mut best_index = None;
        let mut best_sector = if unsafe { LOOK_DIRECTION_UP } { u64::MAX } else { 0 };

        unsafe {
            for (index, request) in self.requests.iter().enumerate() {
                if request.status == IORequestStatus::Queued {
                    if LOOK_DIRECTION_UP && request.sector_start >= current_head_position {
                        if request.sector_start < best_sector {
                            best_sector = request.sector_start;
                            best_index = Some(index);
                        }
                    } else if !LOOK_DIRECTION_UP && request.sector_start <= current_head_position {
                        if request.sector_start > best_sector {
                            best_sector = request.sector_start;
                            best_index = Some(index);
                        }
                    }
                }
            }

            if best_index.is_none() {
                LOOK_DIRECTION_UP = !LOOK_DIRECTION_UP;
                return self.look_schedule(current_head_position);
            }
        }

        best_index
    }

    fn deadline_schedule(&mut self) -> Option<usize> {
        // Deadline scheduler: prioritize expired requests, then by deadline
        let current_time = self.get_current_time_us();
        let mut best_index = None;
        let mut earliest_deadline = u64::MAX;

        for (index, request) in self.requests.iter().enumerate() {
            if request.status == IORequestStatus::Queued {
                if let Some(deadline) = request.deadline_us {
                    if deadline < earliest_deadline {
                        earliest_deadline = deadline;
                        best_index = Some(index);
                    }
                }
            }
        }

        best_index
    }

    fn cfq_schedule(&mut self) -> Option<usize> {
        // Completely Fair Queuing: round-robin with time slices
        // Simplified version - just return first queued request
        self.requests.iter().position(|r| r.status == IORequestStatus::Queued)
    }

    fn noop_schedule(&mut self) -> Option<usize> {
        // NOOP: simple FIFO with merging
        self.requests.iter().position(|r| r.status == IORequestStatus::Queued)
    }

    fn get_current_time_us(&self) -> u64 {
        static mut TIME_COUNTER: u64 = 0;
        unsafe {
            TIME_COUNTER += 1000; // Increment by 1ms
            TIME_COUNTER
        }
    }
}

/// I/O scheduler statistics
#[derive(Debug, Clone, Copy)]
pub struct IOSchedulerStats {
    pub total_requests_submitted: u64,
    pub total_requests_completed: u64,
    pub total_requests_merged: u32,
    pub total_bytes_read: u64,
    pub total_bytes_written: u64,
    pub average_request_size_kb: f32,
    pub average_queue_depth: f32,
    pub average_response_time_us: u64,
    pub average_throughput_mbps: f32,
    pub queue_utilization_percent: f32,
    pub merge_ratio: f32,
    pub deadline_miss_rate: f32,
}

impl IOSchedulerStats {
    pub fn new() -> Self {
        Self {
            total_requests_submitted: 0,
            total_requests_completed: 0,
            total_requests_merged: 0,
            total_bytes_read: 0,
            total_bytes_written: 0,
            average_request_size_kb: 0.0,
            average_queue_depth: 0.0,
            average_response_time_us: 0,
            average_throughput_mbps: 0.0,
            queue_utilization_percent: 0.0,
            merge_ratio: 0.0,
            deadline_miss_rate: 0.0,
        }
    }

    pub fn calculate_efficiency(&self) -> f32 {
        let throughput_factor = (self.average_throughput_mbps / 500.0).min(1.0);
        let latency_factor = (10_000.0 / self.average_response_time_us as f32).min(1.0);
        let merge_factor = self.merge_ratio;

        (throughput_factor + latency_factor + merge_factor) / 3.0
    }
}

/// Main I/O scheduler system
pub struct IOScheduler {
    io_queues: Vec<IOQueue, MAX_IO_QUEUES>,
    storage_devices: Vec<StorageDevice, MAX_STORAGE_DEVICES>,
    default_algorithm: IOSchedulingAlgorithm,
    stats: IOSchedulerStats,
    request_counter: u64,
    queue_counter: u8,
    adaptive_scheduling: bool,
    request_batching: bool,
    deadline_enforcement: bool,
    bandwidth_throttling: bool,
}

impl IOScheduler {
    pub fn new() -> Self {
        Self {
            io_queues: Vec::new(),
            storage_devices: Vec::new(),
            default_algorithm: IOSchedulingAlgorithm::Deadline,
            stats: IOSchedulerStats::new(),
            request_counter: 0,
            queue_counter: 0,
            adaptive_scheduling: true,
            request_batching: true,
            deadline_enforcement: true,
            bandwidth_throttling: false,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        crate::println!("[I/O] Initializing high-performance I/O scheduler...");

        // Create default I/O queues for different priority classes
        self.create_default_queues()?;

        // Add default storage devices
        self.add_default_devices()?;

        crate::println!("[I/O] I/O scheduler initialized successfully");
        crate::println!("[I/O] Default algorithm: {}", self.default_algorithm);
        crate::println!("[I/O] Queues: {}, Devices: {}", self.io_queues.len(), self.storage_devices.len());
        crate::println!("[I/O] Features: Adaptive={}, Batching={}, Deadlines={}",
                       self.adaptive_scheduling, self.request_batching, self.deadline_enforcement);

        Ok(())
    }

    pub fn submit_request(&mut self, request_type: IORequestType, device_id: u8,
                         sector_start: u64, sector_count: u32, priority: IOPriority,
                         process_id: u32) -> Result<u64, &'static str> {

        // Validate device exists
        if !self.storage_devices.iter().any(|d| d.device_id == device_id) {
            return Err("Invalid device ID");
        }

        let current_time = self.get_current_time_us();
        let request = IORequest::new(
            self.request_counter,
            request_type,
            device_id,
            sector_start,
            sector_count,
            priority,
            process_id,
            current_time
        );

        self.request_counter += 1;
        let request_id = request.request_id;

        // Find appropriate queue for this priority
        let queue_index = self.find_queue_for_priority(priority)?;

        // Add request to queue
        self.io_queues[queue_index].add_request(request)?;

        // Update statistics
        self.stats.total_requests_submitted += 1;

        if request_type == IORequestType::Read {
            self.stats.total_bytes_read += sector_count as u64 * 512;
        } else if request_type == IORequestType::Write {
            self.stats.total_bytes_written += sector_count as u64 * 512;
        }

        Ok(request_id)
    }

    pub fn schedule_requests(&mut self) -> Vec<IORequest, IO_BATCH_SIZE> {
        let mut scheduled_requests = Vec::new();
        let current_time = self.get_current_time_us();

        // Process expired requests first if deadline enforcement is enabled
        if self.deadline_enforcement {
            self.handle_expired_requests(current_time);
        }

        // Schedule requests from queues based on priority
        for queue in &mut self.io_queues {
            if scheduled_requests.len() >= IO_BATCH_SIZE {
                break;
            }

            // Get device for head position
            if let Some(device_id) = self.get_primary_device_for_queue(queue.queue_id) {
                if let Some(device) = self.storage_devices.iter().find(|d| d.device_id == device_id) {
                    if let Some(request_index) = queue.get_next_request(device.current_head_position) {
                        if let Some(mut request) = queue.remove_request(request_index) {
                            request.status = IORequestStatus::Dispatched;
                            request.dispatch_time_us = Some(current_time);
                            let _ = scheduled_requests.push(request);
                        }
                    }
                }
            }
        }

        scheduled_requests
    }

    pub fn get_scheduler_stats(&self) -> IOSchedulerStats {
        self.stats
    }

    pub fn set_scheduling_algorithm(&mut self, algorithm: IOSchedulingAlgorithm) {
        self.default_algorithm = algorithm;
        for queue in &mut self.io_queues {
            queue.algorithm = algorithm;
        }
        crate::println!("[I/O] Scheduling algorithm changed to: {}", algorithm);
    }

    fn create_default_queues(&mut self) -> Result<(), &'static str> {
        let queue_configs = [
            (IOPriority::RealTime7, IOSchedulingAlgorithm::Deadline),
            (IOPriority::RealTime0, IOSchedulingAlgorithm::Deadline),
            (IOPriority::BestEffort7, IOSchedulingAlgorithm::CFQ),
            (IOPriority::BestEffort0, IOSchedulingAlgorithm::CFQ),
            (IOPriority::IdleClass, IOSchedulingAlgorithm::FCFS),
        ];

        for &(priority, algorithm) in &queue_configs {
            let queue = IOQueue::new(self.queue_counter, priority, algorithm);
            self.queue_counter += 1;
            let _ = self.io_queues.push(queue);
        }

        crate::println!("[I/O] Created {} default I/O queues", self.io_queues.len());
        Ok(())
    }

    fn add_default_devices(&mut self) -> Result<(), &'static str> {
        let ssd = StorageDevice::new_ssd(0, "nvme0n1", 512);
        let hdd = StorageDevice::new_hdd(1, "sda", 1024);

        let _ = self.storage_devices.push(ssd);
        let _ = self.storage_devices.push(hdd);

        crate::println!("[I/O] Added default storage devices: SSD (512GB), HDD (1TB)");
        Ok(())
    }

    fn find_queue_for_priority(&self, priority: IOPriority) -> Result<usize, &'static str> {
        self.io_queues.iter()
            .position(|q| q.priority_class == priority)
            .or_else(|| self.io_queues.iter().position(|q| q.priority_class == IOPriority::BestEffort0))
            .ok_or("No suitable queue found")
    }

    fn get_primary_device_for_queue(&self, _queue_id: u8) -> Option<u8> {
        self.storage_devices.first().map(|d| d.device_id)
    }

    fn handle_expired_requests(&mut self, current_time: u64) {
        for queue in &mut self.io_queues {
            let expired = queue.expire_requests(current_time);
            for request in expired {
                crate::println!("[I/O] Request {} expired after {}μs",
                               request.request_id, request.wait_time_us(current_time));
            }
        }
    }

    fn get_current_time_us(&self) -> u64 {
        static mut TIME_COUNTER: u64 = 0;
        unsafe {
            TIME_COUNTER += 1000;
            TIME_COUNTER
        }
    }
}

lazy_static! {
    static ref IO_SCHEDULER: Mutex<IOScheduler> = Mutex::new(IOScheduler::new());
}

pub fn init_io_scheduler() {
    let mut scheduler = IO_SCHEDULER.lock();
    match scheduler.initialize() {
        Ok(_) => crate::println!("[I/O] High-performance I/O scheduler ready"),
        Err(e) => crate::println!("[I/O] Failed to initialize: {}", e),
    }
}

pub fn submit_io_request(request_type: IORequestType, device_id: u8,
                        sector_start: u64, sector_count: u32, priority: IOPriority,
                        process_id: u32) -> Result<u64, &'static str> {
    IO_SCHEDULER.lock().submit_request(request_type, device_id, sector_start,
                                      sector_count, priority, process_id)
}

pub fn schedule_io_requests() -> Vec<IORequest, IO_BATCH_SIZE> {
    IO_SCHEDULER.lock().schedule_requests()
}

pub fn set_io_scheduling_algorithm(algorithm: IOSchedulingAlgorithm) {
    IO_SCHEDULER.lock().set_scheduling_algorithm(algorithm);
}

pub fn get_io_scheduler_stats() -> IOSchedulerStats {
    IO_SCHEDULER.lock().get_scheduler_stats()
}

pub fn periodic_io_scheduler_task() {
    let scheduled = schedule_io_requests();
    if !scheduled.is_empty() {
        crate::println!("[I/O] Scheduled {} I/O requests", scheduled.len());
    }
}

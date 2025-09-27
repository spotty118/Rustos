//! Advanced Storage Performance Analytics and Optimization System
//!
//! This module provides comprehensive storage performance monitoring, analysis,
//! and optimization capabilities for the RustOS kernel. It integrates with the
//! AI system to provide intelligent storage tuning and predictive optimization.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of storage devices to monitor
const MAX_STORAGE_DEVICES: usize = 32;
/// Maximum number of I/O operations to track
const MAX_IO_OPERATIONS: usize = 1024;
/// Maximum number of storage optimization rules
const MAX_STORAGE_RULES: usize = 64;
/// Storage sampling interval in milliseconds
const STORAGE_SAMPLING_INTERVAL_MS: u64 = 100;
/// Maximum file system cache entries
const MAX_CACHE_ENTRIES: usize = 512;

/// Storage device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageDeviceType {
    HDD,        // Traditional hard disk drive
    SSD,        // Solid state drive
    NVMe,       // NVMe SSD
    eMMC,       // Embedded MultiMediaCard
    SD,         // SD Card
    USB,        // USB storage
    RAM,        // RAM disk
    Network,    // Network attached storage
    Optical,    // CD/DVD/Blu-ray
    Tape,       // Tape storage
    Unknown,
}

impl fmt::Display for StorageDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StorageDeviceType::HDD => write!(f, "HDD"),
            StorageDeviceType::SSD => write!(f, "SSD"),
            StorageDeviceType::NVMe => write!(f, "NVMe"),
            StorageDeviceType::eMMC => write!(f, "eMMC"),
            StorageDeviceType::SD => write!(f, "SD Card"),
            StorageDeviceType::USB => write!(f, "USB Storage"),
            StorageDeviceType::RAM => write!(f, "RAM Disk"),
            StorageDeviceType::Network => write!(f, "Network Storage"),
            StorageDeviceType::Optical => write!(f, "Optical"),
            StorageDeviceType::Tape => write!(f, "Tape"),
            StorageDeviceType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Storage interface types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageInterface {
    SATA,
    IDE,
    SCSI,
    NVMe,
    USB,
    MMC,
    Network,
    Virtual,
}

/// I/O operation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IOOperationType {
    Read,
    Write,
    Sync,
    Flush,
    Trim,
    Format,
    Verify,
}

/// Storage optimization strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageOptimizationStrategy {
    /// Maximize read/write throughput
    HighThroughput,
    /// Minimize access latency
    LowLatency,
    /// Balance performance and longevity
    Balanced,
    /// Minimize power consumption
    PowerEfficient,
    /// Optimize for database workloads
    DatabaseOptimized,
    /// Optimize for streaming media
    StreamingOptimized,
    /// AI-driven adaptive optimization
    AIAdaptive,
    /// Optimize for data integrity
    IntegrityFocused,
}

impl fmt::Display for StorageOptimizationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StorageOptimizationStrategy::HighThroughput => write!(f, "High Throughput"),
            StorageOptimizationStrategy::LowLatency => write!(f, "Low Latency"),
            StorageOptimizationStrategy::Balanced => write!(f, "Balanced"),
            StorageOptimizationStrategy::PowerEfficient => write!(f, "Power Efficient"),
            StorageOptimizationStrategy::DatabaseOptimized => write!(f, "Database Optimized"),
            StorageOptimizationStrategy::StreamingOptimized => write!(f, "Streaming Optimized"),
            StorageOptimizationStrategy::AIAdaptive => write!(f, "AI Adaptive"),
            StorageOptimizationStrategy::IntegrityFocused => write!(f, "Integrity Focused"),
        }
    }
}

/// Storage device performance metrics
#[derive(Debug, Clone)]
pub struct StorageDeviceMetrics {
    pub device_id: u32,
    pub device_name: &'static str,
    pub device_type: StorageDeviceType,
    pub interface: StorageInterface,
    pub capacity_gb: u64,
    pub available_gb: u64,

    // Performance metrics
    pub read_throughput_mbps: f32,
    pub write_throughput_mbps: f32,
    pub read_latency_ms: f32,
    pub write_latency_ms: f32,
    pub iops_read: u32,
    pub iops_write: u32,
    pub queue_depth: u32,

    // I/O statistics
    pub total_reads: u64,
    pub total_writes: u64,
    pub total_bytes_read: u64,
    pub total_bytes_written: u64,
    pub read_errors: u32,
    pub write_errors: u32,

    // Health metrics
    pub temperature_celsius: f32,
    pub power_on_hours: u64,
    pub wear_level_percent: f32,
    pub bad_blocks: u32,
    pub health_score: f32,  // 0.0 to 1.0

    // Cache and buffering
    pub cache_hit_rate: f32,
    pub write_cache_enabled: bool,
    pub read_ahead_enabled: bool,
    pub buffer_utilization: f32,

    // Advanced metrics
    pub fragmentation_level: f32,
    pub compression_ratio: f32,
    pub encryption_enabled: bool,
    pub smart_status: bool,
}

impl StorageDeviceMetrics {
    pub fn new(id: u32, name: &'static str, device_type: StorageDeviceType, interface: StorageInterface) -> Self {
        Self {
            device_id: id,
            device_name: name,
            device_type,
            interface,
            capacity_gb: 1000, // Default 1TB
            available_gb: 800,

            read_throughput_mbps: 0.0,
            write_throughput_mbps: 0.0,
            read_latency_ms: 1.0,
            write_latency_ms: 1.0,
            iops_read: 0,
            iops_write: 0,
            queue_depth: 1,

            total_reads: 0,
            total_writes: 0,
            total_bytes_read: 0,
            total_bytes_written: 0,
            read_errors: 0,
            write_errors: 0,

            temperature_celsius: 35.0,
            power_on_hours: 1000,
            wear_level_percent: 5.0,
            bad_blocks: 0,
            health_score: 1.0,

            cache_hit_rate: 85.0,
            write_cache_enabled: true,
            read_ahead_enabled: true,
            buffer_utilization: 30.0,

            fragmentation_level: 10.0,
            compression_ratio: 1.2,
            encryption_enabled: false,
            smart_status: true,
        }
    }
}

/// Individual I/O operation tracking
#[derive(Debug, Clone)]
pub struct IOOperation {
    pub operation_id: u64,
    pub device_id: u32,
    pub operation_type: IOOperationType,
    pub sector_start: u64,
    pub sector_count: u32,
    pub size_bytes: u32,
    pub timestamp_start: u64,
    pub timestamp_end: u64,
    pub latency_us: u32,
    pub priority: u8,
    pub completed: bool,
    pub error_code: u32,
}

impl IOOperation {
    pub fn new(id: u64, device_id: u32, op_type: IOOperationType, sector: u64, count: u32) -> Self {
        Self {
            operation_id: id,
            device_id,
            operation_type: op_type,
            sector_start: sector,
            sector_count: count,
            size_bytes: count * 512, // Assume 512-byte sectors
            timestamp_start: 0,
            timestamp_end: 0,
            latency_us: 0,
            priority: 3, // Normal priority
            completed: false,
            error_code: 0,
        }
    }
}

/// File system cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub block_id: u64,
    pub device_id: u32,
    pub size_bytes: u32,
    pub access_count: u32,
    pub last_access: u64,
    pub dirty: bool,
    pub locked: bool,
    pub priority: u8,
}

/// Storage optimization rule
#[derive(Debug, Clone)]
pub struct StorageOptimizationRule {
    pub rule_id: u32,
    pub name: &'static str,
    pub condition: StorageCondition,
    pub action: StorageAction,
    pub priority: u8,
    pub enabled: bool,
    pub trigger_count: u32,
    pub device_types: Vec<StorageDeviceType, 8>,
}

#[derive(Debug, Clone)]
pub enum StorageCondition {
    ReadLatencyAbove(f32),
    WriteLatencyAbove(f32),
    ThroughputBelow(f32),
    QueueDepthAbove(u32),
    CacheHitRateBelow(f32),
    TemperatureAbove(f32),
    ErrorRateAbove(f32),
    FragmentationAbove(f32),
    WearLevelAbove(f32),
    HealthScoreBelow(f32),
}

#[derive(Debug, Clone)]
pub enum StorageAction {
    EnableWriteCache,
    DisableWriteCache,
    EnableReadAhead,
    DisableReadAhead,
    IncreaseQueueDepth(u32),
    DecreaseQueueDepth(u32),
    EnableCompression,
    DisableCompression,
    TriggerDefragmentation,
    TriggerTrim,
    ReducePowerMode,
    MaxPerformanceMode,
    AlertMaintenanceRequired,
    EnableEncryption,
    DisableEncryption,
}

/// Comprehensive storage system statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_devices: u32,
    pub healthy_devices: u32,
    pub total_capacity_gb: u64,
    pub used_capacity_gb: u64,
    pub available_capacity_gb: u64,

    // Aggregate performance
    pub total_read_throughput_mbps: f32,
    pub total_write_throughput_mbps: f32,
    pub average_read_latency_ms: f32,
    pub average_write_latency_ms: f32,
    pub total_iops: u32,

    // System health
    pub overall_health_score: f32,
    pub critical_devices: u32,
    pub warning_devices: u32,
    pub average_temperature: f32,

    // I/O statistics
    pub total_io_operations: u64,
    pub completed_operations: u64,
    pub failed_operations: u64,
    pub pending_operations: u32,

    // Cache performance
    pub cache_utilization: f32,
    pub overall_cache_hit_rate: f32,
    pub cache_efficiency_score: f32,

    // Optimization metrics
    pub optimizations_applied: u64,
    pub defragmentations_performed: u32,
    pub trim_operations: u32,
    pub compression_savings_gb: f32,
    pub ai_optimizations: u64,
}

impl Default for StorageStats {
    fn default() -> Self {
        Self {
            total_devices: 0,
            healthy_devices: 0,
            total_capacity_gb: 0,
            used_capacity_gb: 0,
            available_capacity_gb: 0,

            total_read_throughput_mbps: 0.0,
            total_write_throughput_mbps: 0.0,
            average_read_latency_ms: 1.0,
            average_write_latency_ms: 1.0,
            total_iops: 0,

            overall_health_score: 1.0,
            critical_devices: 0,
            warning_devices: 0,
            average_temperature: 35.0,

            total_io_operations: 0,
            completed_operations: 0,
            failed_operations: 0,
            pending_operations: 0,

            cache_utilization: 30.0,
            overall_cache_hit_rate: 85.0,
            cache_efficiency_score: 0.85,

            optimizations_applied: 0,
            defragmentations_performed: 0,
            trim_operations: 0,
            compression_savings_gb: 0.0,
            ai_optimizations: 0,
        }
    }
}

/// Main storage analytics engine
pub struct StorageAnalytics {
    // Device management
    devices: FnvIndexMap<u32, StorageDeviceMetrics, MAX_STORAGE_DEVICES>,
    io_operations: Vec<IOOperation, MAX_IO_OPERATIONS>,
    cache_entries: FnvIndexMap<u64, CacheEntry, MAX_CACHE_ENTRIES>,
    optimization_rules: Vec<StorageOptimizationRule, MAX_STORAGE_RULES>,

    // Configuration
    active_strategy: StorageOptimizationStrategy,
    monitoring_enabled: bool,
    ai_optimization_enabled: bool,
    predictive_caching_enabled: bool,
    compression_enabled: bool,

    // State
    stats: StorageStats,
    last_analysis_timestamp: u64,
    next_device_id: u32,
    next_operation_id: u64,
    next_rule_id: u32,

    // Performance baselines
    baseline_read_latency: f32,
    baseline_write_latency: f32,
    baseline_throughput: f32,
    baseline_iops: u32,
}

impl StorageAnalytics {
    pub fn new() -> Self {
        Self {
            devices: FnvIndexMap::new(),
            io_operations: Vec::new(),
            cache_entries: FnvIndexMap::new(),
            optimization_rules: Vec::new(),

            active_strategy: StorageOptimizationStrategy::Balanced,
            monitoring_enabled: false,
            ai_optimization_enabled: true,
            predictive_caching_enabled: true,
            compression_enabled: false,

            stats: StorageStats::default(),
            last_analysis_timestamp: 0,
            next_device_id: 1,
            next_operation_id: 1,
            next_rule_id: 1,

            baseline_read_latency: 1.0,
            baseline_write_latency: 1.0,
            baseline_throughput: 100.0,
            baseline_iops: 1000,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        crate::println!("[STORAGE] Initializing storage analytics system...");

        // Discover storage devices
        self.discover_storage_devices()?;

        // Load default optimization rules
        self.load_default_optimization_rules()?;

        // Initialize file system cache
        self.initialize_cache()?;

        // Establish performance baselines
        self.establish_baselines()?;

        // Enable monitoring
        self.monitoring_enabled = true;

        crate::println!("[STORAGE] Storage analytics system initialized");
        crate::println!("[STORAGE] Found {} storage devices", self.devices.len());
        crate::println!("[STORAGE] Loaded {} optimization rules", self.optimization_rules.len());
        crate::println!("[STORAGE] Cache initialized with {} entries capacity", MAX_CACHE_ENTRIES);
        crate::println!("[STORAGE] Active strategy: {}", self.active_strategy);

        Ok(())
    }

    fn discover_storage_devices(&mut self) -> Result<(), &'static str> {
        crate::println!("[STORAGE] Discovering storage devices...");

        // Simulate storage device discovery
        let devices_to_add = [
            ("sda", StorageDeviceType::SSD, StorageInterface::SATA, 1000),
            ("nvme0n1", StorageDeviceType::NVMe, StorageInterface::NVMe, 500),
            ("sdb", StorageDeviceType::HDD, StorageInterface::SATA, 4000),
            ("mmcblk0", StorageDeviceType::eMMC, StorageInterface::MMC, 64),
        ];

        for (name, device_type, interface, capacity_gb) in &devices_to_add {
            let device_id = self.next_device_id;
            self.next_device_id += 1;

            let mut device = StorageDeviceMetrics::new(device_id, name, *device_type, *interface);
            device.capacity_gb = *capacity_gb;
            device.available_gb = (*capacity_gb as f32 * 0.8) as u64; // 80% available

            // Set device-specific defaults
            match device_type {
                StorageDeviceType::NVMe => {
                    device.read_throughput_mbps = 3500.0;
                    device.write_throughput_mbps = 3000.0;
                    device.read_latency_ms = 0.1;
                    device.write_latency_ms = 0.1;
                    device.iops_read = 500000;
                    device.iops_write = 400000;
                },
                StorageDeviceType::SSD => {
                    device.read_throughput_mbps = 550.0;
                    device.write_throughput_mbps = 500.0;
                    device.read_latency_ms = 0.3;
                    device.write_latency_ms = 0.4;
                    device.iops_read = 100000;
                    device.iops_write = 90000;
                },
                StorageDeviceType::HDD => {
                    device.read_throughput_mbps = 150.0;
                    device.write_throughput_mbps = 140.0;
                    device.read_latency_ms = 8.0;
                    device.write_latency_ms = 10.0;
                    device.iops_read = 150;
                    device.iops_write = 120;
                    device.temperature_celsius = 40.0;
                },
                StorageDeviceType::eMMC => {
                    device.read_throughput_mbps = 100.0;
                    device.write_throughput_mbps = 50.0;
                    device.read_latency_ms = 1.0;
                    device.write_latency_ms = 2.0;
                    device.iops_read = 10000;
                    device.iops_write = 5000;
                },
                _ => {} // Use defaults
            }

            self.devices.insert(device_id, device)
                .map_err(|_| "Failed to add storage device")?;

            crate::println!("[STORAGE] Discovered {} {}: {} ({} GB)",
                           device_type, name, interface, capacity_gb);
        }

        Ok(())
    }

    fn load_default_optimization_rules(&mut self) -> Result<(), &'static str> {
        let default_rules = [
            ("High Read Latency", StorageCondition::ReadLatencyAbove(5.0), StorageAction::EnableReadAhead, 8),
            ("High Write Latency", StorageCondition::WriteLatencyAbove(10.0), StorageAction::EnableWriteCache, 8),
            ("Low Throughput", StorageCondition::ThroughputBelow(50.0), StorageAction::IncreaseQueueDepth(32), 7),
            ("High Temperature", StorageCondition::TemperatureAbove(70.0), StorageAction::ReducePowerMode, 9),
            ("Low Cache Hit Rate", StorageCondition::CacheHitRateBelow(70.0), StorageAction::EnableReadAhead, 6),
            ("High Fragmentation", StorageCondition::FragmentationAbove(50.0), StorageAction::TriggerDefragmentation, 5),
            ("High Wear Level", StorageCondition::WearLevelAbove(80.0), StorageAction::AlertMaintenanceRequired, 10),
            ("Low Health Score", StorageCondition::HealthScoreBelow(0.7), StorageAction::AlertMaintenanceRequired, 10),
        ];

        for (name, condition, action, priority) in &default_rules {
            let rule = StorageOptimizationRule {
                rule_id: self.next_rule_id,
                name,
                condition: condition.clone(),
                action: action.clone(),
                priority: *priority,
                enabled: true,
                trigger_count: 0,
                device_types: Vec::new(), // Apply to all device types by default
            };

            self.next_rule_id += 1;

            self.optimization_rules.push(rule)
                .map_err(|_| "Failed to add optimization rule")?;
        }

        Ok(())
    }

    fn initialize_cache(&mut self) -> Result<(), &'static str> {
        crate::println!("[STORAGE] Initializing file system cache...");

        // Pre-populate cache with some frequently accessed blocks
        for block_id in 0..10 {
            let cache_entry = CacheEntry {
                block_id,
                device_id: 1, // Primary storage device
                size_bytes: 4096,
                access_count: 1,
                last_access: crate::time::get_current_timestamp_ms(),
                dirty: false,
                locked: false,
                priority: 5,
            };

            self.cache_entries.insert(block_id, cache_entry)
                .map_err(|_| "Failed to initialize cache entry")?;
        }

        Ok(())
    }

    fn establish_baselines(&mut self) -> Result<(), &'static str> {
        crate::println!("[STORAGE] Establishing storage performance baselines...");

        // Collect initial metrics
        self.collect_storage_metrics()?;

        // Calculate baselines from current device metrics
        let mut total_read_latency = 0.0;
        let mut total_write_latency = 0.0;
        let mut total_throughput = 0.0;
        let mut total_iops = 0;
        let mut device_count = 0;

        for (_, device) in &self.devices {
            total_read_latency += device.read_latency_ms;
            total_write_latency += device.write_latency_ms;
            total_throughput += device.read_throughput_mbps;
            total_iops += device.iops_read;
            device_count += 1;
        }

        if device_count > 0 {
            self.baseline_read_latency = total_read_latency / device_count as f32;
            self.baseline_write_latency = total_write_latency / device_count as f32;
            self.baseline_throughput = total_throughput / device_count as f32;
            self.baseline_iops = total_iops / device_count;
        }

        crate::println!("[STORAGE] Baselines established - Read: {:.1}ms, Write: {:.1}ms, Throughput: {:.1}MB/s, IOPS: {}",
                       self.baseline_read_latency, self.baseline_write_latency, self.baseline_throughput, self.baseline_iops);

        Ok(())
    }

    pub fn collect_storage_metrics(&mut self) -> Result<(), &'static str> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        let timestamp = crate::time::get_current_timestamp_ms();

        // Update device metrics
        for (_, device) in &mut self.devices {
            self.update_device_metrics(device, timestamp);
        }

        // Update I/O operation metrics
        self.update_io_metrics(timestamp)?;

        // Update cache metrics
        self.update_cache_metrics(timestamp)?;

        // Calculate aggregate statistics
        self.calculate_aggregate_stats();

        // Analyze storage health
        self.analyze_storage_health();

        Ok(())
    }

    fn update_device_metrics(&self, device: &mut StorageDeviceMetrics, timestamp: u64) {
        let time_factor = (timestamp % 20000) as f32 / 20000.0;
        let load_factor = 0.5 + (time_factor * 0.5); // Simulate varying load

        // Update performance metrics based on device type and current load
        match device.device_type {
            StorageDeviceType::NVMe => {
                device.read_throughput_mbps = 3500.0 * (1.0 - load_factor * 0.3);
                device.write_throughput_mbps = 3000.0 * (1.0 - load_factor * 0.4);
                device.read_latency_ms = 0.1 * (1.0 + load_factor);
                device.write_latency_ms = 0.1 * (1.0 + load_factor * 1.5);
                device.queue_depth = (16.0 * load_factor) as u32;
                device.temperature_celsius = 35.0 + (load_factor * 15.0);
            },
            StorageDeviceType::SSD => {
                device.read_throughput_mbps = 550.0 * (1.0 - load_factor * 0.2);
                device.write_throughput_mbps = 500.0 * (1.0 - load_factor * 0.3);
                device.read_latency_ms = 0.3 * (1.0 + load_factor * 2.0);
                device.write_latency_ms = 0.4 * (1.0 + load_factor * 2.5);
                device.queue_depth = (8.0 * load_factor) as u32;
                device.temperature_celsius = 38.0 + (load_factor * 12.0);
            },
            StorageDeviceType::HDD => {
                device.read_throughput_mbps = 150.0 * (1.0 - load_factor * 0.4);
                device.write_throughput_mbps = 140.0 * (1.0 - load_factor * 0.5);
                device.read_latency_ms = 8.0 * (1.0 + load_factor * 3.0);
                device.write_latency_ms = 10.0 * (1.0 + load_factor * 4.0);
                device.queue_depth = (4.0 * load_factor) as u32;
                device.temperature_celsius = 40.0 + (load_factor * 20.0);
                device.fragmentation_level += time_factor * 0.1; // HDDs fragment over time
            },
            _ => {
                // Default update for other device types
                device.read_throughput_mbps *= 1.0 - load_factor * 0.1;
                device.write_throughput_mbps *= 1.0 - load_factor * 0.2;
                device.read_latency_ms *= 1.0 + load_factor;
                device.write_latency_ms *= 1.0 + load_factor * 1.2;
            }
        }

        // Update I/O counters
        let io_operations = (load_factor * 1000.0) as u64;
        device.total_reads += io_operations / 2;
        device.total_writes += io_operations / 2;
        device.total_bytes_read += io_operations * 4096; // 4KB average
        device.total_bytes_written += io_operations * 4096;

        // Update IOPS based on current throughput
        device.iops_read = (device.read_throughput_mbps * 1000000.0 / 4096.0) as u32;
        device.iops_write = (device.write_throughput_mbps * 1000000.0 / 4096.0) as u32;

        // Simulate cache hit rate changes
        device.cache_hit_rate = 85.0 + (time_factor - 0.5) * 20.0;
        device.cache_hit_rate = device.cache_hit_rate.max(50.0).min(98.0);

        // Update health metrics
        device.wear_level_percent += time_factor * 0.001; // Gradual wear
        device.health_score = (1.0 - device.wear_level_percent / 100.0).max(0.0);

        // Simulate occasional errors
        if time_factor > 0.95 {
            device.read_errors += 1;
        }
        if time_factor > 0.98 {
            device.write_errors += 1;
        }
    }

    fn update_io_metrics(&mut self, timestamp: u64) -> Result<(), &'static str> {
        // Remove completed operations older than 10 seconds
        let mut operations_to_remove = Vec::<usize, 64>::new();

        for (index, operation) in self.io_operations.iter_mut().enumerate() {
            if !operation.completed && timestamp - operation.timestamp_start > 1000 {
                // Complete pending operations
                operation.completed = true;
                operation.timestamp_end = timestamp;
                operation.latency_us = ((timestamp - operation.timestamp_start) * 1000) as u32;
            }

            if operation.completed && timestamp - operation.timestamp_end > 10000 {
                // Mark for removal
                if operations_to_remove.push(index).is_err() {
                    break; // Vector full
                }
            }
        }

        // Remove old operations (in reverse order to maintain indices)
        for &index in operations_to_remove.iter().rev() {
            if index < self.io_operations.len() {
                self.io_operations.remove(index);
            }
        }

        // Simulate new I/O operations
        if self.io_operations.len() < MAX_IO_OPERATIONS - 10 {
            for _ in 0..5 {
                let operation_id = self.next_operation_id;
                self.next_operation_id += 1;

                let device_id = if self.devices.len() > 0 { 1 } else { 0 };
                let op_type = if timestamp % 3 == 0 { IOOperationType::Read } else { IOOperationType::Write };
                let sector = (timestamp % 1000000) as u64;
                let count = 8; // 4KB operation

                let mut operation = IOOperation::new(operation_id, device_id, op_type, sector, count);
                operation.timestamp_start = timestamp;

                if self.io_operations.push(operation).is_err() {
                    break; // Vector full
                }
            }
        }

        Ok(())
    }

    fn update_cache_metrics(&mut self, timestamp: u64) -> Result<(), &'static str> {
        // Update cache access patterns
        for (_, cache_entry) in &mut self.cache_entries {
            // Age cache entries
            cache_entry.access_count += 1;
            cache_entry.last_access = timestamp;

            // Mark some entries as dirty occasionally
            if timestamp % 1000 == 0 {
                cache_entry.dirty = true;
            }
        }

        // Simulate cache eviction for old entries
        let mut entries_to_remove = Vec::<u64, 32>::new();
        for (block_id, cache_entry) in &self.cache_entries {
            if timestamp - cache_entry.last_access > 60000 { // 60 seconds
                if entries_to_remove.push(*block_id).is_err() {
                    break;
                }
            }
        }

        for block_id in entries_to_remove {
            self.cache_entries.remove(&block_id);
        }

        Ok(())
    }

    fn calculate_aggregate_stats(&mut self) {
        self.stats.total_devices = self.devices.len() as u32;
        self.stats.healthy_devices = self.devices.iter()
            .filter(|(_, device)| device.health_score > 0.8)
            .count() as u32;

        // Calculate capacity statistics
        let mut total_capacity = 0u64;
        let mut used_capacity = 0u64;
        let mut total_read_throughput = 0.0;
        let mut total_write_throughput = 0.0;
        let mut total_read_latency = 0.0;
        let mut total_write_latency = 0.0;
        let mut total_iops = 0u32;
        let mut total_temperature = 0.0;
        let mut device_count = 0;

        for (_, device) in &self.devices {
            total_capacity += device.capacity_gb;
            used_capacity += device.capacity_gb - device.available_gb;
            total_read_throughput += device.read_throughput_mbps;
            total_write_throughput += device.write_throughput_mbps;
            total_read_latency += device.read_latency_ms;
            total_write_latency += device.write_latency_ms;
            total_iops += device.iops_read;
            total_temperature += device.temperature_celsius;
            device_count += 1;
        }

        self.stats.total_capacity_gb = total_capacity;
        self.stats.used_capacity_gb = used_capacity;
        self.stats.available_capacity_gb = total_capacity - used_capacity;
        self.stats.total_read_throughput_mbps = total_read_throughput;
        self.stats.total_write_throughput_mbps = total_write_throughput;
        self.stats.total_iops = total_iops;

        if device_count > 0 {
            self.stats.average_read_latency_ms = total_read_latency / device_count as f32;
            self.stats.average_write_latency_ms = total_write_latency / device_count as f32;
            self.stats.average_temperature = total_temperature / device_count as f32;
        }

        // Update I/O operation statistics
        self.stats.total_io_operations = self.next_operation_id - 1;
        self.stats.completed_operations = self.io_operations.iter()
            .filter(|op| op.completed)
            .count() as u64;
        self.stats.pending_operations = (self.stats.total_io_operations - self.stats.completed_operations) as u32;
        self.stats.failed_operations = self.io_operations.iter()
            .filter(|op| op.error_code != 0)
            .count() as u64;

        // Update cache statistics
        self.stats.cache_utilization = (self.cache_entries.len() as f32 / MAX_CACHE_ENTRIES as f32) * 100.0;

        let total_cache_accesses: u32 = self.cache_entries.iter()
            .map(|(_, entry)| entry.access_count)
            .sum();

        if total_cache_accesses > 0 {
            self.stats.overall_cache_hit_rate = (self.cache_entries.len() as f32 / total_cache_accesses as f32) * 100.0;
            self.stats.cache_efficiency_score = (self.stats.overall_cache_hit_rate / 100.0).min(1.0);
        }
    }

    fn analyze_storage_health(&mut self) {
        let mut critical_devices = 0;
        let mut warning_devices = 0;
        let mut total_health = 0.0;

        for (_, device) in &self.devices {
            if device.health_score < 0.5 {
                critical_devices += 1;
            } else if device.health_score < 0.8 {
                warning_devices += 1;
            }
            total_health += device.health_score;
        }

        self.stats.critical_devices = critical_devices;
        self.stats.warning_devices = warning_devices;

        if self.devices.len() > 0 {
            self.stats.overall_health_score = total_health / self.devices.len() as f32;
        }
    }

    pub fn analyze_and_optimize(&mut self) -> Result<(), &'static str> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        // Apply optimization rules
        self.apply_optimization_rules()?;

        // Run AI-driven optimizations if enabled
        if self.ai_optimization_enabled {
            self.run_ai_optimizations()?;
        }

        // Perform predictive caching if enabled
        if self.predictive_caching_enabled {
            self.optimize_cache_strategy()?;
        }

        // Strategy-specific optimizations
        self.apply_strategy_optimizations()?;

        Ok(())
    }

    fn apply_optimization_rules(&mut self) -> Result<(), &'static str> {
        let mut rules_applied = 0;

        // Apply rules in priority order
        for priority in (0..=10u8).rev() {
            for rule in &mut self.optimization_rules {
                if !rule.enabled || rule.priority != priority {
                    continue;
                }

                // Check rule conditions against each applicable device
                for (_, device) in &self.devices {
                    // Check if rule applies to this device type
                    if !rule.device_types.is_empty() &&
                       !rule.device_types.contains(&device.device_type) {
                        continue;
                    }

                    let should_trigger = match &rule.condition {
                        StorageCondition::ReadLatencyAbove(threshold) => {
                            device.read_latency_ms > *threshold
                        },
                        StorageCondition::WriteLatencyAbove(threshold) => {
                            device.write_latency_ms > *threshold
                        },
                        StorageCondition::ThroughputBelow(threshold) => {
                            device.read_throughput_mbps < *threshold
                        },
                        StorageCondition::QueueDepthAbove(threshold) => {
                            device.queue_depth > *threshold
                        },
                        StorageCondition::CacheHitRateBelow(threshold) => {
                            device.cache_hit_rate < *threshold
                        },
                        StorageCondition::TemperatureAbove(threshold) => {
                            device.temperature_celsius > *threshold
                        },
                        StorageCondition::ErrorRateAbove(threshold) => {
                            let error_rate = (device.read_errors + device.write_errors) as f32 /
                                           (device.total_reads + device.total_writes).max(1) as f32;
                            error_rate > *threshold
                        },
                        StorageCondition::FragmentationAbove(threshold) => {
                            device.fragmentation_level > *threshold
                        },
                        StorageCondition::WearLevelAbove(threshold) => {
                            device.wear_level_percent > *threshold
                        },
                        StorageCondition::HealthScoreBelow(threshold) => {
                            device.health_score < *threshold
                        },
                    };

                    if should_trigger {
                        crate::println!("[STORAGE] Triggering rule '{}' for device {}",
                                       rule.name, device.device_name);
                        self.execute_storage_action(&rule.action, device.device_id)?;
                        rule.trigger_count += 1;
                        rules_applied += 1;
                        self.stats.optimizations_applied += 1;
                        break; // Apply rule only once per cycle
                    }
                }
            }
        }

        if rules_applied > 0 {
            crate::println!("[STORAGE] Applied {} optimization rules", rules_applied);
        }

        Ok(())
    }

    fn execute_storage_action(&mut self, action: &StorageAction, device_id: u32) -> Result<(), &'static str> {
        match action {
            StorageAction::EnableWriteCache => {
                crate::println!("[STORAGE] Enabling write cache for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.write_cache_enabled = true;
                }
            },
            StorageAction::DisableWriteCache => {
                crate::println!("[STORAGE] Disabling write cache for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.write_cache_enabled = false;
                }
            },
            StorageAction::EnableReadAhead => {
                crate::println!("[STORAGE] Enabling read-ahead for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.read_ahead_enabled = true;
                    device.cache_hit_rate += 5.0; // Improve cache hit rate
                }
            },
            StorageAction::DisableReadAhead => {
                crate::println!("[STORAGE] Disabling read-ahead for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.read_ahead_enabled = false;
                }
            },
            StorageAction::IncreaseQueueDepth(depth) => {
                crate::println!("[STORAGE] Increasing queue depth to {} for device {}", depth, device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.queue_depth = *depth;
                }
            },
            StorageAction::DecreaseQueueDepth(depth) => {
                crate::println!("[STORAGE] Decreasing queue depth to {} for device {}", depth, device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.queue_depth = *depth;
                }
            },
            StorageAction::EnableCompression => {
                crate::println!("[STORAGE] Enabling compression for device {}", device_id);
                self.compression_enabled = true;
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.compression_ratio = 1.5;
                    self.stats.compression_savings_gb += (device.used_capacity_gb() as f32 * 0.3);
                }
            },
            StorageAction::DisableCompression => {
                crate::println!("[STORAGE] Disabling compression for device {}", device_id);
                self.compression_enabled = false;
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.compression_ratio = 1.0;
                }
            },
            StorageAction::TriggerDefragmentation => {
                crate::println!("[STORAGE] Triggering defragmentation for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.fragmentation_level = device.fragmentation_level * 0.3;
                    self.stats.defragmentations_performed += 1;
                }
            },
            StorageAction::TriggerTrim => {
                crate::println!("[STORAGE] Triggering TRIM for device {}", device_id);
                self.stats.trim_operations += 1;
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.available_gb = (device.available_gb as f32 * 1.05) as u64;
                }
            },
            StorageAction::ReducePowerMode => {
                crate::println!("[STORAGE] Enabling power saving mode for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.temperature_celsius *= 0.9; // Reduce temperature
                }
            },
            StorageAction::MaxPerformanceMode => {
                crate::println!("[STORAGE] Enabling maximum performance mode for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.read_throughput_mbps *= 1.1;
                    device.write_throughput_mbps *= 1.1;
                }
            },
            StorageAction::AlertMaintenanceRequired => {
                crate::println!("[STORAGE] MAINTENANCE ALERT: Device {} requires attention", device_id);
                if let Some(device) = self.devices.get(&device_id) {
                    crate::println!("[STORAGE] Device: {}, Health: {:.1}%, Wear: {:.1}%",
                                   device.device_name, device.health_score * 100.0, device.wear_level_percent);
                }
            },
            StorageAction::EnableEncryption => {
                crate::println!("[STORAGE] Enabling encryption for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.encryption_enabled = true;
                    device.write_latency_ms *= 1.2; // Encryption overhead
                }
            },
            StorageAction::DisableEncryption => {
                crate::println!("[STORAGE] Disabling encryption for device {}", device_id);
                if let Some(device) = self.devices.get_mut(&device_id) {
                    device.encryption_enabled = false;
                }
            },
        }

        Ok(())
    }

    fn run_ai_optimizations(&mut self) -> Result<(), &'static str> {
        // Convert storage metrics to AI input format
        let storage_ai_metrics = crate::ai::learning::HardwareMetrics {
            cpu_usage: 15, // Storage processing load
            memory_usage: 25,
            io_operations: (self.stats.total_iops / 10) as u32,
            interrupt_count: self.stats.pending_operations * 10,
            context_switches: 200,
            cache_misses: (100.0 - self.stats.overall_cache_hit_rate) as u32,
            thermal_state: self.stats.average_temperature as u32,
            power_efficiency: 80,
            gpu_usage: 0, // Storage doesn't use GPU
            gpu_memory_usage: 0,
            gpu_temperature: 40,
        };

        // Send metrics to AI system for analysis
        crate::ai::process_hardware_metrics(storage_ai_metrics);

        // AI-driven storage optimizations
        if self.stats.overall_health_score < 0.8 {
            crate::println!("[STORAGE] AI detected storage performance issues, applying optimizations");

            // Switch optimization strategy based on AI analysis
            if self.stats.average_read_latency_ms > self.baseline_read_latency * 2.0 {
                self.set_optimization_strategy(StorageOptimizationStrategy::LowLatency)?;
            } else if self.stats.total_read_throughput_mbps < self.baseline_throughput * 0.7 {
                self.set_optimization_strategy(StorageOptimizationStrategy::HighThroughput)?;
            }

            self.stats.ai_optimizations += 1;
        }

        Ok(())
    }

    fn optimize_cache_strategy(&mut self) -> Result<(), &'static str> {
        // Predictive caching based on access patterns
        let current_time = crate::time::get_current_timestamp_ms();

        // Identify frequently accessed blocks
        let mut hot_blocks = Vec::<u64, 32>::new();
        for (block_id, entry) in &self.cache_entries {
            if entry.access_count > 5 && current_time - entry.last_access < 30000 {
                if hot_blocks.push(*block_id).is_err() {
                    break;
                }
            }
        }

        // Pre-load adjacent blocks for hot blocks
        for &hot_block in &hot_blocks {
            for offset in 1..4 { // Pre-load next 3 blocks
                let adjacent_block = hot_block + offset;
                if !self.cache_entries.contains_key(&adjacent_block) && !self.cache_entries.is_full() {
                    let cache_entry = CacheEntry {
                        block_id: adjacent_block,
                        device_id: 1, // Primary device
                        size_bytes: 4096,
                        access_count: 1,
                        last_access: current_time,
                        dirty: false,
                        locked: false,
                        priority: 3, // Lower priority for predictive entries
                    };

                    if self.cache_entries.insert(adjacent_block, cache_entry).is_err() {
                        break;
                    }
                }
            }
        }

        if !hot_blocks.is_empty() {
            crate::println!("[STORAGE] Predictive caching: pre-loaded {} adjacent blocks",
                           hot_blocks.len() * 3);
        }

        Ok(())
    }

    fn apply_strategy_optimizations(&mut self) -> Result<(), &'static str> {
        match self.active_strategy {
            StorageOptimizationStrategy::HighThroughput => {
                // Optimize for maximum throughput
                for (_, device) in &mut self.devices {
                    if device.queue_depth < 32 {
                        device.queue_depth = 32;
                    }
                    device.write_cache_enabled = true;
                    device.read_ahead_enabled = true;
                }
                crate::println!("[STORAGE] Applied high-throughput optimizations");
            },
            StorageOptimizationStrategy::LowLatency => {
                // Optimize for minimum latency
                for (_, device) in &mut self.devices {
                    if device.queue_depth > 1 {
                        device.queue_depth = 1;
                    }
                    device.write_cache_enabled = false; // Reduce latency variance
                }
                crate::println!("[STORAGE] Applied low-latency optimizations");
            },
            StorageOptimizationStrategy::PowerEfficient => {
                // Optimize for power savings
                for (_, device) in &mut self.devices {
                    device.temperature_celsius *= 0.95; // Reduce power/heat
                }
                crate::println!("[STORAGE] Applied power-efficient optimizations");
            },
            StorageOptimizationStrategy::DatabaseOptimized => {
                // Optimize for database workloads
                for (_, device) in &mut self.devices {
                    device.queue_depth = 16; // Good balance for databases
                    device.write_cache_enabled = true;
                    device.read_ahead_enabled = false; // Random access patterns
                }
                crate::println!("[STORAGE] Applied database-optimized settings");
            },
            StorageOptimizationStrategy::StreamingOptimized => {
                // Optimize for streaming media
                for (_, device) in &mut self.devices {
                    device.queue_depth = 8;
                    device.read_ahead_enabled = true; // Sequential access
                    device.write_cache_enabled = true;
                }
                crate::println!("[STORAGE] Applied streaming-optimized settings");
            },
            StorageOptimizationStrategy::IntegrityFocused => {
                // Prioritize data integrity
                for (_, device) in &mut self.devices {
                    device.write_cache_enabled = false; // Ensure immediate writes
                    device.encryption_enabled = true;
                }
                crate::println!("[STORAGE] Applied integrity-focused optimizations");
            },
            StorageOptimizationStrategy::AIAdaptive => {
                // Already handled in run_ai_optimizations
            },
            StorageOptimizationStrategy::Balanced => {
                // Maintain balance - default behavior
            },
        }

        Ok(())
    }

    pub fn set_optimization_strategy(&mut self, strategy: StorageOptimizationStrategy) -> Result<(), &'static str> {
        if self.active_strategy != strategy {
            crate::println!("[STORAGE] Switching storage optimization strategy: {} -> {}",
                           self.active_strategy, strategy);
            self.active_strategy = strategy;
            self.last_analysis_timestamp = crate::time::get_current_timestamp_ms();
        }
        Ok(())
    }

    pub fn get_stats(&self) -> &StorageStats {
        &self.stats
    }

    pub fn get_active_strategy(&self) -> StorageOptimizationStrategy {
        self.active_strategy
    }

    pub fn generate_storage_report(&self) -> Result<(), &'static str> {
        crate::println!("=== Storage Performance Report ===");
        crate::println!("Active Strategy: {}", self.active_strategy);
        crate::println!("Monitoring: {}", if self.monitoring_enabled { "Enabled" } else { "Disabled" });
        crate::println!("AI Optimization: {}", if self.ai_optimization_enabled { "Enabled" } else { "Disabled" });
        crate::println!("Predictive Caching: {}", if self.predictive_caching_enabled { "Enabled" } else { "Disabled" });
        crate::println!();

        crate::println!("Storage Capacity:");
        crate::println!("  Total: {} GB", self.stats.total_capacity_gb);
        crate::println!("  Used: {} GB ({:.1}%)", self.stats.used_capacity_gb,
                       (self.stats.used_capacity_gb as f32 / self.stats.total_capacity_gb as f32) * 100.0);
        crate::println!("  Available: {} GB", self.stats.available_capacity_gb);
        crate::println!();

        crate::println!("Storage Devices ({} total, {} healthy):", self.stats.total_devices, self.stats.healthy_devices);
        for (_, device) in &self.devices {
            crate::println!("  {}: {} {} - {:.1}/{:.1} MB/s, {:.1}/{:.1}ms latency, {:.1}°C",
                           device.device_name,
                           device.device_type,
                           device.interface,
                           device.read_throughput_mbps,
                           device.write_throughput_mbps,
                           device.read_latency_ms,
                           device.write_latency_ms,
                           device.temperature_celsius);
        }
        crate::println!();

        crate::println!("Performance Metrics:");
        crate::println!("  Total Read Throughput: {:.1} MB/s", self.stats.total_read_throughput_mbps);
        crate::println!("  Total Write Throughput: {:.1} MB/s", self.stats.total_write_throughput_mbps);
        crate::println!("  Average Read Latency: {:.1} ms", self.stats.average_read_latency_ms);
        crate::println!("  Average Write Latency: {:.1} ms", self.stats.average_write_latency_ms);
        crate::println!("  Total IOPS: {}", self.stats.total_iops);
        crate::println!("  Overall Health Score: {:.1}%", self.stats.overall_health_score * 100.0);
        crate::println!();

        crate::println!("Cache Performance:");
        crate::println!("  Cache Utilization: {:.1}%", self.stats.cache_utilization);
        crate::println!("  Cache Hit Rate: {:.1}%", self.stats.overall_cache_hit_rate);
        crate::println!("  Cache Efficiency: {:.1}%", self.stats.cache_efficiency_score * 100.0);
        crate::println!();

        crate::println!("I/O Operations:");
        crate::println!("  Total Operations: {}", self.stats.total_io_operations);
        crate::println!("  Completed: {}", self.stats.completed_operations);
        crate::println!("  Pending: {}", self.stats.pending_operations);
        crate::println!("  Failed: {}", self.stats.failed_operations);
        crate::println!();

        crate::println!("Optimizations:");
        crate::println!("  Rules Applied: {}", self.stats.optimizations_applied);
        crate::println!("  Defragmentations: {}", self.stats.defragmentations_performed);
        crate::println!("  TRIM Operations: {}", self.stats.trim_operations);
        crate::println!("  Compression Savings: {:.1} GB", self.stats.compression_savings_gb);
        crate::println!("  AI Optimizations: {}", self.stats.ai_optimizations);

        if self.stats.critical_devices > 0 {
            crate::println!();
            crate::println!("WARNINGS:");
            crate::println!("  Critical Devices: {}", self.stats.critical_devices);
            crate::println!("  Warning Devices: {}", self.stats.warning_devices);
        }

        Ok(())
    }
}

// Add extension method for StorageDeviceMetrics
impl StorageDeviceMetrics {
    pub fn used_capacity_gb(&self) -> u64 {
        self.capacity_gb - self.available_gb
    }
}

lazy_static! {
    static ref STORAGE_ANALYTICS: Mutex<StorageAnalytics> = Mutex::new(StorageAnalytics::new());
}

/// Initialize storage analytics system
pub fn init_storage_analytics() -> Result<(), &'static str> {
    let mut analytics = STORAGE_ANALYTICS.lock();
    analytics.initialize()
}

/// Collect storage performance metrics
pub fn collect_storage_metrics() -> Result<(), &'static str> {
    let mut analytics = STORAGE_ANALYTICS.lock();
    analytics.collect_storage_metrics()
}

/// Analyze storage performance and apply optimizations
pub fn analyze_and_optimize_storage() -> Result<(), &'static str> {
    let mut analytics = STORAGE_ANALYTICS.lock();
    analytics.analyze_and_optimize()
}

/// Set storage optimization strategy
pub fn set_storage_strategy(strategy: StorageOptimizationStrategy) -> Result<(), &'static str> {
    let mut analytics = STORAGE_ANALYTICS.lock();
    analytics.set_optimization_strategy(strategy)
}

/// Get current storage statistics
pub fn get_storage_stats() -> StorageStats {
    let analytics = STORAGE_ANALYTICS.lock();
    analytics.get_stats().clone()
}

/// Generate and display storage performance report
pub fn generate_storage_report() -> Result<(), &'static str> {
    let analytics = STORAGE_ANALYTICS.lock();
    analytics.generate_storage_report()
}

/// Storage analytics task (to be called periodically)
pub fn storage_analytics_task() {
    if collect_storage_metrics().is_ok() {
        let _ = analyze_and_optimize_storage();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_storage_analytics_creation() {
        let analytics = StorageAnalytics::new();
        assert_eq!(analytics.active_strategy, StorageOptimizationStrategy::Balanced);
        assert!(!analytics.monitoring_enabled);
    }

    #[test_case]
    fn test_device_metrics() {
        let metrics = StorageDeviceMetrics::new(1, "sda", StorageDeviceType::SSD, StorageInterface::SATA);
        assert_eq!(metrics.device_id, 1);
        assert_eq!(metrics.device_type, StorageDeviceType::SSD);
        assert_eq!(metrics.device_name, "sda");
    }

    #[test_case]
    fn test_io_operation() {
        let operation = IOOperation::new(1, 1, IOOperationType::Read, 1000, 8);
        assert_eq!(operation.operation_id, 1);
        assert_eq!(operation.operation_type, IOOperationType::Read);
        assert_eq!(operation.size_bytes, 4096); // 8 sectors * 512 bytes
    }
}

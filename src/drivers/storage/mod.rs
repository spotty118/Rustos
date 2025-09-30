//! # Storage Drivers Module
//!
//! This module provides comprehensive storage device drivers for RustOS,
//! including AHCI SATA, NVMe, IDE/PATA, and USB Mass Storage support.

pub mod ahci;
pub mod nvme;
pub mod ide;
pub mod usb_mass_storage;
pub mod filesystem_interface;
pub mod detection;
pub mod pci_scan;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::fmt;
use spin::RwLock;

/// Storage device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDeviceType {
    /// SATA Hard Drive
    SataHdd,
    /// SATA Solid State Drive
    SataSsd,
    /// NVMe SSD
    NvmeSsd,
    /// IDE/PATA Drive
    IdeDrive,
    /// USB Mass Storage
    UsbMassStorage,
    /// RAID Array
    RaidArray,
    /// Optical Drive
    OpticalDrive,
    /// Unknown storage device
    Unknown,
}

impl fmt::Display for StorageDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageDeviceType::SataHdd => write!(f, "SATA HDD"),
            StorageDeviceType::SataSsd => write!(f, "SATA SSD"),
            StorageDeviceType::NvmeSsd => write!(f, "NVMe SSD"),
            StorageDeviceType::IdeDrive => write!(f, "IDE Drive"),
            StorageDeviceType::UsbMassStorage => write!(f, "USB Mass Storage"),
            StorageDeviceType::RaidArray => write!(f, "RAID Array"),
            StorageDeviceType::OpticalDrive => write!(f, "Optical Drive"),
            StorageDeviceType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Storage device state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDeviceState {
    /// Device is offline/not detected
    Offline,
    /// Device is initializing
    Initializing,
    /// Device is ready for I/O
    Ready,
    /// Device is in sleep/standby mode
    Standby,
    /// Device has encountered an error
    Error,
    /// Device is being reset
    Resetting,
}

impl fmt::Display for StorageDeviceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageDeviceState::Offline => write!(f, "OFFLINE"),
            StorageDeviceState::Initializing => write!(f, "INITIALIZING"),
            StorageDeviceState::Ready => write!(f, "READY"),
            StorageDeviceState::Standby => write!(f, "STANDBY"),
            StorageDeviceState::Error => write!(f, "ERROR"),
            StorageDeviceState::Resetting => write!(f, "RESETTING"),
        }
    }
}

/// Storage device capabilities
#[derive(Debug, Clone)]
pub struct StorageCapabilities {
    /// Device capacity in bytes
    pub capacity_bytes: u64,
    /// Sector size in bytes
    pub sector_size: u32,
    /// Maximum transfer size in bytes
    pub max_transfer_size: u32,
    /// Supports TRIM/UNMAP commands
    pub supports_trim: bool,
    /// Supports NCQ (Native Command Queuing)
    pub supports_ncq: bool,
    /// Supports SMART monitoring
    pub supports_smart: bool,
    /// Maximum queue depth
    pub max_queue_depth: u16,
    /// Read speed in MB/s (estimated)
    pub read_speed_mbps: u32,
    /// Write speed in MB/s (estimated)
    pub write_speed_mbps: u32,
    /// Device is removable
    pub is_removable: bool,
}

impl Default for StorageCapabilities {
    fn default() -> Self {
        Self {
            capacity_bytes: 0,
            sector_size: 512,
            max_transfer_size: 65536,
            supports_trim: false,
            supports_ncq: false,
            supports_smart: false,
            max_queue_depth: 1,
            read_speed_mbps: 100,
            write_speed_mbps: 100,
            is_removable: false,
        }
    }
}

/// Storage device error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageError {
    /// Device not found
    DeviceNotFound,
    /// Invalid sector address
    InvalidSector,
    /// Transfer too large
    TransferTooLarge,
    /// Device timeout
    Timeout,
    /// Hardware error
    HardwareError,
    /// Media error
    MediaError,
    /// Device busy
    DeviceBusy,
    /// Permission denied
    PermissionDenied,
    /// Not supported
    NotSupported,
    /// Buffer too small
    BufferTooSmall,
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::DeviceNotFound => write!(f, "Device not found"),
            StorageError::InvalidSector => write!(f, "Invalid sector address"),
            StorageError::TransferTooLarge => write!(f, "Transfer too large"),
            StorageError::Timeout => write!(f, "Device timeout"),
            StorageError::HardwareError => write!(f, "Hardware error"),
            StorageError::MediaError => write!(f, "Media error"),
            StorageError::DeviceBusy => write!(f, "Device busy"),
            StorageError::PermissionDenied => write!(f, "Permission denied"),
            StorageError::NotSupported => write!(f, "Not supported"),
            StorageError::BufferTooSmall => write!(f, "Buffer too small"),
        }
    }
}

/// Storage device statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total read operations
    pub reads_total: u64,
    /// Total write operations
    pub writes_total: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Read errors
    pub read_errors: u64,
    /// Write errors
    pub write_errors: u64,
    /// Average read latency in microseconds
    pub avg_read_latency_us: u32,
    /// Average write latency in microseconds
    pub avg_write_latency_us: u32,
    /// Device uptime in seconds
    pub uptime_seconds: u64,
}

/// Storage driver interface
pub trait StorageDriver: Send + Sync + core::fmt::Debug {
    /// Get driver name
    fn name(&self) -> &str;

    /// Get device type
    fn device_type(&self) -> StorageDeviceType;

    /// Get device state
    fn state(&self) -> StorageDeviceState;

    /// Get device capabilities
    fn capabilities(&self) -> StorageCapabilities;

    /// Initialize the device
    fn init(&mut self) -> Result<(), StorageError>;

    /// Read sectors from the device
    fn read_sectors(&mut self, start_sector: u64, buffer: &mut [u8]) -> Result<usize, StorageError>;

    /// Write sectors to the device
    fn write_sectors(&mut self, start_sector: u64, buffer: &[u8]) -> Result<usize, StorageError>;

    /// Flush any pending writes
    fn flush(&mut self) -> Result<(), StorageError>;

    /// Get device statistics
    fn get_stats(&self) -> StorageStats;

    /// Reset the device
    fn reset(&mut self) -> Result<(), StorageError>;

    /// Put device in standby mode
    fn standby(&mut self) -> Result<(), StorageError>;

    /// Wake device from standby
    fn wake(&mut self) -> Result<(), StorageError>;

    /// Execute vendor-specific command
    fn vendor_command(&mut self, command: u8, data: &[u8]) -> Result<Vec<u8>, StorageError>;

    /// Get SMART data (if supported)
    fn get_smart_data(&self) -> Result<Vec<u8>, StorageError>;
}

/// Storage device descriptor
#[derive(Debug)]
pub struct StorageDevice {
    /// Device ID
    pub id: u32,
    /// Device driver
    pub driver: Box<dyn StorageDriver>,
    /// Device model
    pub model: String,
    /// Device serial number
    pub serial: String,
    /// Firmware version
    pub firmware: String,
    /// Registration timestamp
    pub registered_at: u64,
    /// Last access timestamp
    pub last_access: u64,
}

impl StorageDevice {
    pub fn new(
        id: u32,
        driver: Box<dyn StorageDriver>,
        model: String,
        serial: String,
        firmware: String,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            driver,
            model,
            serial,
            firmware,
            registered_at: timestamp,
            last_access: timestamp,
        }
    }

    /// Update last access time
    pub fn update_access(&mut self, timestamp: u64) {
        self.last_access = timestamp;
    }

    /// Get device information
    pub fn get_info(&self) -> StorageDeviceInfo {
        StorageDeviceInfo {
            id: self.id,
            name: self.driver.name().into(),
            device_type: self.driver.device_type(),
            state: self.driver.state(),
            capabilities: self.driver.capabilities(),
            stats: self.driver.get_stats(),
            model: self.model.clone(),
            serial: self.serial.clone(),
            firmware: self.firmware.clone(),
            registered_at: self.registered_at,
            last_access: self.last_access,
        }
    }
}

/// Device information structure
#[derive(Debug, Clone)]
pub struct StorageDeviceInfo {
    pub id: u32,
    pub name: String,
    pub device_type: StorageDeviceType,
    pub state: StorageDeviceState,
    pub capabilities: StorageCapabilities,
    pub stats: StorageStats,
    pub model: String,
    pub serial: String,
    pub firmware: String,
    pub registered_at: u64,
    pub last_access: u64,
}

/// Storage driver manager
#[derive(Debug)]
pub struct StorageDriverManager {
    /// Registered devices
    devices: BTreeMap<u32, StorageDevice>,
    /// Next device ID
    next_id: u32,
    /// Manager statistics
    stats: StorageManagerStats,
}

impl StorageDriverManager {
    pub fn new() -> Self {
        Self {
            devices: BTreeMap::new(),
            next_id: 1,
            stats: StorageManagerStats::default(),
        }
    }

    /// Register a storage device
    pub fn register_device(
        &mut self,
        driver: Box<dyn StorageDriver>,
        model: String,
        serial: String,
        firmware: String,
        timestamp: u64,
    ) -> Result<u32, StorageError> {
        let id = self.next_id;
        self.next_id += 1;

        let device = StorageDevice::new(id, driver, model, serial, firmware, timestamp);
        self.devices.insert(id, device);

        self.stats.devices_registered += 1;

        Ok(id)
    }

    /// Unregister a storage device
    pub fn unregister_device(&mut self, id: u32) -> Result<(), StorageError> {
        if self.devices.remove(&id).is_some() {
            self.stats.devices_unregistered += 1;
            Ok(())
        } else {
            Err(StorageError::DeviceNotFound)
        }
    }

    /// Get device by ID
    pub fn get_device(&self, id: u32) -> Option<&StorageDevice> {
        self.devices.get(&id)
    }

    /// Get mutable device by ID
    pub fn get_device_mut(&mut self, id: u32) -> Option<&mut StorageDevice> {
        self.devices.get_mut(&id)
    }

    /// Get all device information
    pub fn get_all_device_info(&self) -> Vec<StorageDeviceInfo> {
        self.devices
            .values()
            .map(|device| device.get_info())
            .collect()
    }

    /// Get device count
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Initialize all devices
    pub fn init_all_devices(&mut self) -> Result<(), StorageError> {
        for device in self.devices.values_mut() {
            device.driver.init()?;
        }
        Ok(())
    }

    /// Read from device
    pub fn read_sectors(
        &mut self,
        device_id: u32,
        start_sector: u64,
        buffer: &mut [u8],
    ) -> Result<usize, StorageError> {
        let device = self.devices.get_mut(&device_id)
            .ok_or(StorageError::DeviceNotFound)?;

        let result = device.driver.read_sectors(start_sector, buffer);
        device.update_access(crate::time::get_system_time_ms());

        if result.is_ok() {
            self.stats.total_reads += 1;
        }

        result
    }

    /// Write to device
    pub fn write_sectors(
        &mut self,
        device_id: u32,
        start_sector: u64,
        buffer: &[u8],
    ) -> Result<usize, StorageError> {
        let device = self.devices.get_mut(&device_id)
            .ok_or(StorageError::DeviceNotFound)?;

        let result = device.driver.write_sectors(start_sector, buffer);
        device.update_access(crate::time::get_system_time_ms());

        if result.is_ok() {
            self.stats.total_writes += 1;
        }

        result
    }

    /// Get manager statistics
    pub fn get_stats(&self) -> &StorageManagerStats {
        &self.stats
    }
}

/// Storage manager statistics
#[derive(Debug, Default, Clone)]
pub struct StorageManagerStats {
    pub devices_registered: u64,
    pub devices_unregistered: u64,
    pub total_reads: u64,
    pub total_writes: u64,
    pub errors: u64,
}

/// Global storage driver manager
static STORAGE_MANAGER: RwLock<Option<StorageDriverManager>> = RwLock::new(None);

/// Initialize global storage driver manager
pub fn init_storage_manager() {
    *STORAGE_MANAGER.write() = Some(StorageDriverManager::new());
}

/// Get reference to global storage driver manager
pub fn with_storage_manager<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut StorageDriverManager) -> R,
{
    STORAGE_MANAGER.write().as_mut().map(f)
}

/// High-level storage management functions
pub fn get_storage_device_list() -> Vec<StorageDeviceInfo> {
    with_storage_manager(|manager| manager.get_all_device_info()).unwrap_or_default()
}

pub fn read_storage_sectors(
    device_id: u32,
    start_sector: u64,
    buffer: &mut [u8],
) -> Result<usize, StorageError> {
    with_storage_manager(|manager| manager.read_sectors(device_id, start_sector, buffer))
        .ok_or(StorageError::DeviceNotFound)?
}

pub fn write_storage_sectors(
    device_id: u32,
    start_sector: u64,
    buffer: &[u8],
) -> Result<usize, StorageError> {
    with_storage_manager(|manager| manager.write_sectors(device_id, start_sector, buffer))
        .ok_or(StorageError::DeviceNotFound)?
}

/// Initialize storage subsystem during kernel boot
pub fn init_storage_subsystem() -> Result<detection::DetectionResults, StorageError> {
    detection::detect_and_initialize_storage()
}
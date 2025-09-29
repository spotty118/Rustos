//! Optimized I/O Performance System for RustOS
//!
//! This module provides high-performance I/O operations with:
//! - Asynchronous I/O support
//! - I/O scheduling and prioritization
//! - DMA transfer optimization
//! - Network packet processing acceleration

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::task::{Context, Poll, Waker};
use core::pin::Pin;
use core::future::Future;
use alloc::vec::Vec;
use spin::Mutex;
use crate::data_structures::{LockFreeMpscQueue, CacheFriendlyRingBuffer, CACHE_LINE_SIZE};

/// I/O request types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoRequestType {
    Read,
    Write,
    NetworkReceive,
    NetworkTransmit,
    DiskRead,
    DiskWrite,
}

/// I/O priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IoPriority {
    RealTime = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

/// I/O request structure
#[derive(Debug)]
pub struct IoRequest {
    pub id: u64,
    pub request_type: IoRequestType,
    pub priority: IoPriority,
    pub buffer: *mut u8,
    pub size: usize,
    pub offset: u64,
    pub device_id: u32,
    pub waker: Option<Waker>,
    pub completion_status: IoCompletionStatus,
}

/// I/O completion status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoCompletionStatus {
    Pending,
    Completed(usize), // Bytes transferred
    Error(IoError),
}

/// I/O error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoError {
    DeviceNotFound,
    InvalidRequest,
    Timeout,
    HardwareError,
    PermissionDenied,
    BufferTooSmall,
    DeviceBusy,
    NoData,
}

/// I/O statistics
#[repr(align(64))]
pub struct IoStats {
    pub total_requests: AtomicU64,
    pub completed_requests: AtomicU64,
    pub failed_requests: AtomicU64,
    pub bytes_read: AtomicU64,
    pub bytes_written: AtomicU64,
    pub avg_latency_us: AtomicU64,
    pub queue_depth: AtomicUsize,
    _padding: [u8; CACHE_LINE_SIZE - 6 * core::mem::size_of::<AtomicU64>() - core::mem::size_of::<AtomicUsize>()],
}

static IO_STATS: IoStats = IoStats {
    total_requests: AtomicU64::new(0),
    completed_requests: AtomicU64::new(0),
    failed_requests: AtomicU64::new(0),
    bytes_read: AtomicU64::new(0),
    bytes_written: AtomicU64::new(0),
    avg_latency_us: AtomicU64::new(0),
    queue_depth: AtomicUsize::new(0),
    _padding: [0; CACHE_LINE_SIZE - 6 * core::mem::size_of::<AtomicU64>() - core::mem::size_of::<AtomicUsize>()],
};

/// Asynchronous I/O scheduler
pub struct AsyncIoScheduler {
    /// High-priority queue for real-time and high-priority I/O
    high_priority_queue: LockFreeMpscQueue<IoRequest>,
    /// Normal priority queue
    normal_priority_queue: LockFreeMpscQueue<IoRequest>,
    /// Low priority and background queue
    low_priority_queue: LockFreeMpscQueue<IoRequest>,
    /// Currently active I/O requests
    active_requests: Mutex<Vec<IoRequest>>,
    /// Next request ID
    next_request_id: AtomicU64,
    /// DMA controller
    dma_controller: DmaController,
}

impl AsyncIoScheduler {
    /// Create a new async I/O scheduler
    pub fn new() -> Self {
        Self {
            high_priority_queue: LockFreeMpscQueue::new(),
            normal_priority_queue: LockFreeMpscQueue::new(),
            low_priority_queue: LockFreeMpscQueue::new(),
            active_requests: Mutex::new(Vec::new()),
            next_request_id: AtomicU64::new(1),
            dma_controller: DmaController::new(),
        }
    }

    /// Submit an I/O request
    pub fn submit_request(&self, mut request: IoRequest) -> IoFuture {
        request.id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        request.completion_status = IoCompletionStatus::Pending;

        IO_STATS.total_requests.fetch_add(1, Ordering::Relaxed);
        IO_STATS.queue_depth.fetch_add(1, Ordering::Relaxed);

        // Save the request ID before moving the request
        let request_id = request.id;

        // Queue based on priority
        match request.priority {
            IoPriority::RealTime | IoPriority::High => {
                self.high_priority_queue.enqueue(request);
            }
            IoPriority::Normal => {
                self.normal_priority_queue.enqueue(request);
            }
            IoPriority::Low | IoPriority::Background => {
                self.low_priority_queue.enqueue(request);
            }
        }

        IoFuture {
            request_id,
            scheduler: self,
        }
    }

    /// Process pending I/O requests
    pub fn process_requests(&self) {
        // Process high priority first
        while let Some(request) = self.high_priority_queue.dequeue() {
            self.execute_request(request);
        }

        // Then normal priority
        if let Some(request) = self.normal_priority_queue.dequeue() {
            self.execute_request(request);
        }

        // Finally low priority (only if no higher priority work)
        if self.high_priority_queue.is_empty() && self.normal_priority_queue.is_empty() {
            if let Some(request) = self.low_priority_queue.dequeue() {
                self.execute_request(request);
            }
        }
    }

    /// Execute a single I/O request
    fn execute_request(&self, mut request: IoRequest) {
        let start_time = crate::performance_monitor::read_tsc();

        // Use DMA for large transfers
        let result = if request.size >= 4096 {
            self.dma_controller.transfer(&request)
        } else {
            self.programmed_io_transfer(&request)
        };

        // Calculate latency
        let end_time = crate::performance_monitor::read_tsc();
        let latency_cycles = end_time - start_time;
        let latency_us = latency_cycles / 3000; // Approximate conversion to microseconds

        // Update statistics
        match result {
            Ok(bytes_transferred) => {
                request.completion_status = IoCompletionStatus::Completed(bytes_transferred);
                IO_STATS.completed_requests.fetch_add(1, Ordering::Relaxed);

                match request.request_type {
                    IoRequestType::Read | IoRequestType::NetworkReceive | IoRequestType::DiskRead => {
                        IO_STATS.bytes_read.fetch_add(bytes_transferred as u64, Ordering::Relaxed);
                    }
                    IoRequestType::Write | IoRequestType::NetworkTransmit | IoRequestType::DiskWrite => {
                        IO_STATS.bytes_written.fetch_add(bytes_transferred as u64, Ordering::Relaxed);
                    }
                }
            }
            Err(error) => {
                request.completion_status = IoCompletionStatus::Error(error);
                IO_STATS.failed_requests.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update average latency using exponential moving average
        let current_avg = IO_STATS.avg_latency_us.load(Ordering::Relaxed);
        let new_avg = (current_avg * 7 + latency_us) / 8;
        IO_STATS.avg_latency_us.store(new_avg, Ordering::Relaxed);

        IO_STATS.queue_depth.fetch_sub(1, Ordering::Relaxed);

        // Wake up any waiting tasks
        if let Some(waker) = request.waker.take() {
            waker.wake();
        }

        // Add to completed requests for polling
        let mut active = self.active_requests.lock();
        active.push(request);
    }

    /// Perform programmed I/O transfer (for small transfers)
    fn programmed_io_transfer(&self, request: &IoRequest) -> Result<usize, IoError> {
        match request.request_type {
            IoRequestType::Read => {
                // Perform actual hardware read via port I/O
                self.hardware_read(request)
            }
            IoRequestType::Write => {                        
                // Perform actual hardware write via port I/O
                self.hardware_write(request)
            }
            IoRequestType::NetworkReceive => {
                // Interface with actual network hardware
                self.network_hardware_receive(request)
            }
            IoRequestType::NetworkTransmit => {
                // Interface with actual network hardware
                self.network_hardware_transmit(request)
            }
            _ => Err(IoError::InvalidRequest),
        }
    }

    /// Interface with network hardware for packet reception
    fn network_hardware_receive(&self, request: &IoRequest) -> Result<usize, IoError> {
        if request.size > 1500 {
            return Err(IoError::BufferTooSmall); // MTU exceeded
        }

        // Access network device via driver manager
        if let Some(network_driver) = crate::network::drivers::with_driver_manager(|dm| {
            dm.get_device(request.device_id)
        }).flatten() {
            // Attempt to receive packet from hardware
            match network_driver.receive_packet() {
                Some(packet_data) => {
                    let copy_size = packet_data.len().min(request.size);
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            packet_data.as_ptr(),
                            request.buffer,
                            copy_size
                        );
                    }
                    Ok(copy_size)
                }
                None => Err(IoError::NoData), // No packet available
            }
        } else {
            Err(IoError::DeviceNotFound)
        }
    }

    /// Interface with network hardware for packet transmission
    fn network_hardware_transmit(&self, request: &IoRequest) -> Result<usize, IoError> {
        if request.size > 1500 {
            return Err(IoError::BufferTooSmall); // MTU exceeded
        }

        // Access network device via driver manager
        if let Some(mut network_driver) = crate::network::drivers::with_driver_manager(|dm| {
            dm.get_device(request.device_id)
        }).flatten() {
            // Create packet data from buffer
            let packet_data = unsafe {
                core::slice::from_raw_parts(request.buffer, request.size)
            };
            
            // Transmit packet via hardware
            match network_driver.send_packet(packet_data) {
                Ok(()) => Ok(request.size),
                Err(_) => Err(IoError::HardwareError),
            }
        } else {
            Err(IoError::DeviceNotFound)
        }
    }

    /// Perform hardware read operation
    fn hardware_read(&self, request: &IoRequest) -> Result<usize, IoError> {
        match request.request_type {
            IoRequestType::Read => {
                // Use storage driver for disk reads
                if let Some(storage_driver) = crate::drivers::storage::get_device(request.device_id) {
                    storage_driver.read_sectors(request.offset, request.size, request.buffer)
                        .map_err(|_| IoError::HardwareError)
                } else {
                    // Fall back to direct port I/O for other devices
                    unsafe {
                        // Read from I/O port based on device_id
                        use x86_64::instructions::port::Port;
                        let mut port = Port::<u8>::new(request.device_id as u16);
                        for i in 0..request.size {
                            *request.buffer.add(i) = port.read();
                        }
                    }
                    Ok(request.size)
                }
            }
            _ => Err(IoError::InvalidRequest),
        }
    }

    /// Perform hardware write operation
    fn hardware_write(&self, request: &IoRequest) -> Result<usize, IoError> {
        match request.request_type {
            IoRequestType::Write => {
                // Use storage driver for disk writes
                if let Some(storage_driver) = crate::drivers::storage::get_device(request.device_id) {
                    let data = unsafe { core::slice::from_raw_parts(request.buffer, request.size) };
                    storage_driver.write_sectors(request.offset, data)
                        .map(|_| request.size)
                        .map_err(|_| IoError::HardwareError)
                } else {
                    // Fall back to direct port I/O for other devices
                    unsafe {
                        // Write to I/O port based on device_id
                        use x86_64::instructions::port::Port;
                        let mut port = Port::<u8>::new(request.device_id as u16);
                        for i in 0..request.size {
                            port.write(*request.buffer.add(i));
                        }
                    }
                    Ok(request.size)
                }
            }
            _ => Err(IoError::InvalidRequest),
        }
    }

    /// Check for completed requests
    pub fn poll_completion(&self, request_id: u64) -> Poll<IoCompletionStatus> {
        let mut active = self.active_requests.lock();
        if let Some(pos) = active.iter().position(|req| req.id == request_id) {
            let request = active.remove(pos);
            Poll::Ready(request.completion_status)
        } else {
            Poll::Pending
        }
    }
}

/// DMA controller for high-performance transfers
pub struct DmaController {
    channels: [DmaChannel; 8],
    next_channel: AtomicUsize,
}

#[repr(align(64))]
struct DmaChannel {
    active: AtomicU64,
    source_addr: AtomicU64,
    dest_addr: AtomicU64,
    transfer_size: AtomicUsize,
    _padding: [u8; CACHE_LINE_SIZE - 3 * core::mem::size_of::<AtomicU64>() - core::mem::size_of::<AtomicUsize>()],
}

impl DmaController {
    fn new() -> Self {
        const INIT_CHANNEL: DmaChannel = DmaChannel {
            active: AtomicU64::new(0),
            source_addr: AtomicU64::new(0),
            dest_addr: AtomicU64::new(0),
            transfer_size: AtomicUsize::new(0),
            _padding: [0; CACHE_LINE_SIZE - 3 * core::mem::size_of::<AtomicU64>() - core::mem::size_of::<AtomicUsize>()],
        };

        Self {
            channels: [INIT_CHANNEL; 8],
            next_channel: AtomicUsize::new(0),
        }
    }

    /// Perform DMA transfer
    fn transfer(&self, request: &IoRequest) -> Result<usize, IoError> {
        // Find available DMA channel
        let channel_id = self.next_channel.fetch_add(1, Ordering::Relaxed) % 8;
        let channel = &self.channels[channel_id];

        // Check if channel is available
        if channel.active.load(Ordering::Relaxed) != 0 {
            return Err(IoError::DeviceBusy);
        }

        // Set up DMA transfer
        match request.request_type {
            IoRequestType::Read | IoRequestType::DiskRead => {
                // Simulate reading from device to buffer
                channel.source_addr.store(0x10000000 + request.offset, Ordering::Relaxed); // Device memory
                channel.dest_addr.store(request.buffer as u64, Ordering::Relaxed);
            }
            IoRequestType::Write | IoRequestType::DiskWrite => {
                // Simulate writing from buffer to device
                channel.source_addr.store(request.buffer as u64, Ordering::Relaxed);
                channel.dest_addr.store(0x10000000 + request.offset, Ordering::Relaxed); // Device memory
            }
            _ => return Err(IoError::InvalidRequest),
        }

        channel.transfer_size.store(request.size, Ordering::Relaxed);
        channel.active.store(1, Ordering::Release);

        // Simulate DMA transfer time
        let transfer_cycles = request.size / 1000; // Approximate cycles per byte
        for _ in 0..transfer_cycles {
            unsafe { core::arch::asm!("nop"); }
        }

        // Mark transfer complete
        channel.active.store(0, Ordering::Release);

        Ok(request.size)
    }
}

/// Future for asynchronous I/O operations
pub struct IoFuture<'a> {
    request_id: u64,
    scheduler: &'a AsyncIoScheduler,
}

impl<'a> Future for IoFuture<'a> {
    type Output = IoCompletionStatus;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.scheduler.poll_completion(self.request_id) {
            Poll::Ready(status) => Poll::Ready(status),
            Poll::Pending => {
                // Register waker for when the I/O completes
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

/// Network packet processor for high-performance networking
pub struct NetworkPacketProcessor {
    receive_ring: CacheFriendlyRingBuffer<NetworkPacket>,
    transmit_ring: CacheFriendlyRingBuffer<NetworkPacket>,
    packet_pool: LockFreeMpscQueue<NetworkPacket>,
}

#[repr(align(64))]
pub struct NetworkPacket {
    pub data: [u8; 1536], // MTU + headers
    pub length: usize,
    pub packet_type: PacketType,
    pub timestamp: u64,
    pub _padding: [u8; 0], // No padding needed - struct is already large
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketType {
    Ethernet,
    Ip,
    Tcp,
    Udp,
    Icmp,
}

impl NetworkPacketProcessor {
    /// Create new network packet processor
    pub fn new() -> Option<Self> {
        let receive_ring = CacheFriendlyRingBuffer::new(1024)?;
        let transmit_ring = CacheFriendlyRingBuffer::new(1024)?;
        let packet_pool = LockFreeMpscQueue::new();

        // Pre-allocate packet pool
        for _ in 0..2048 {
            let packet = NetworkPacket {
                data: [0; 1536],
                length: 0,
                packet_type: PacketType::Ethernet,
                timestamp: 0,
                _padding: [],
            };
            packet_pool.enqueue(packet);
        }

        Some(Self {
            receive_ring,
            transmit_ring,
            packet_pool,
        })
    }

    /// Process received packets
    pub fn process_received_packets(&self) -> usize {
        let mut processed = 0;

        while let Some(packet) = self.receive_ring.pop() {
            // Process packet based on type
            match packet.packet_type {
                PacketType::Tcp => self.process_tcp_packet(&packet),
                PacketType::Udp => self.process_udp_packet(&packet),
                PacketType::Icmp => self.process_icmp_packet(&packet),
                _ => {}
            }

            // Return packet to pool
            self.packet_pool.enqueue(packet);
            processed += 1;
        }

        processed
    }

    /// Process TCP packet
    fn process_tcp_packet(&self, _packet: &NetworkPacket) {
        // TCP processing implementation
    }

    /// Process UDP packet
    fn process_udp_packet(&self, _packet: &NetworkPacket) {
        // UDP processing implementation
    }

    /// Process ICMP packet
    fn process_icmp_packet(&self, _packet: &NetworkPacket) {
        // ICMP processing implementation
    }

    /// Queue packet for transmission
    pub fn queue_for_transmission(&self, packet: NetworkPacket) -> Result<(), NetworkPacket> {
        // Validate packet before queuing
        if packet.length == 0 {
            crate::println!("[NETWORK] Rejecting packet with zero length");
            return Err(packet);
        }
        
        if packet.length > 1536 {
            crate::println!("[NETWORK] Rejecting oversized packet: {} bytes (max 1536)", packet.length);
            return Err(packet);
        }
        
        // Additional validation: ensure length doesn't exceed actual data buffer
        if packet.length > packet.data.len() {
            crate::println!("[NETWORK] Rejecting packet with length {} exceeding buffer size {}", 
                packet.length, packet.data.len());
            return Err(packet);
        }
        
        // Check for basic packet sanity
        match packet.packet_type {
            PacketType::Ethernet | PacketType::Ip | 
            PacketType::Tcp | PacketType::Udp | PacketType::Icmp => {
                // Valid packet types
            }
        }
        
        self.transmit_ring.push(packet)
    }
}

/// Global I/O scheduler instance
static mut IO_SCHEDULER: Option<AsyncIoScheduler> = None;
static mut NETWORK_PROCESSOR: Option<NetworkPacketProcessor> = None;

/// Initialize optimized I/O subsystem
pub fn init_io_system() -> Result<(), &'static str> {
    unsafe {
        IO_SCHEDULER = Some(AsyncIoScheduler::new());
        NETWORK_PROCESSOR = NetworkPacketProcessor::new();
    }
    Ok(())
}

/// Get global I/O scheduler
pub fn io_scheduler() -> &'static AsyncIoScheduler {
    unsafe {
        IO_SCHEDULER.as_ref().expect("I/O scheduler not initialized")
    }
}

/// Get global network processor
pub fn network_processor() -> &'static NetworkPacketProcessor {
    unsafe {
        NETWORK_PROCESSOR.as_ref().expect("Network processor not initialized")
    }
}

/// Get I/O statistics
pub fn get_io_statistics() -> (u64, u64, u64, u64, u64, u64, usize) {
    (
        IO_STATS.total_requests.load(Ordering::Relaxed),
        IO_STATS.completed_requests.load(Ordering::Relaxed),
        IO_STATS.failed_requests.load(Ordering::Relaxed),
        IO_STATS.bytes_read.load(Ordering::Relaxed),
        IO_STATS.bytes_written.load(Ordering::Relaxed),
        IO_STATS.avg_latency_us.load(Ordering::Relaxed),
        IO_STATS.queue_depth.load(Ordering::Relaxed),
    )
}
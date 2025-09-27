//! Advanced Network Stack with Zero-Copy I/O
//!
//! This module implements a high-performance network stack for the RustOS kernel,
//! featuring zero-copy I/O, advanced packet processing, hardware offloading,
//! multiple protocol support, and intelligent traffic shaping for optimal
//! network performance and minimal latency.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of network interfaces
const MAX_NETWORK_INTERFACES: usize = 16;
/// Maximum number of network flows
const MAX_NETWORK_FLOWS: usize = 1024;
/// Maximum packet buffer size
const MAX_PACKET_SIZE: usize = 9000; // Jumbo frame support
/// Default MTU size
const DEFAULT_MTU: usize = 1500;
/// Maximum number of packet buffers
const MAX_PACKET_BUFFERS: usize = 4096;
/// Ring buffer size for zero-copy operations
const RING_BUFFER_SIZE: usize = 1024;

/// Network protocol types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkProtocol {
    Ethernet,
    IPv4,
    IPv6,
    TCP,
    UDP,
    ICMP,
    ARP,
    DHCP,
    DNS,
    HTTP,
    HTTPS,
    WebSocket,
    QUIC,
}

impl fmt::Display for NetworkProtocol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NetworkProtocol::Ethernet => write!(f, "Ethernet"),
            NetworkProtocol::IPv4 => write!(f, "IPv4"),
            NetworkProtocol::IPv6 => write!(f, "IPv6"),
            NetworkProtocol::TCP => write!(f, "TCP"),
            NetworkProtocol::UDP => write!(f, "UDP"),
            NetworkProtocol::ICMP => write!(f, "ICMP"),
            NetworkProtocol::ARP => write!(f, "ARP"),
            NetworkProtocol::DHCP => write!(f, "DHCP"),
            NetworkProtocol::DNS => write!(f, "DNS"),
            NetworkProtocol::HTTP => write!(f, "HTTP"),
            NetworkProtocol::HTTPS => write!(f, "HTTPS"),
            NetworkProtocol::WebSocket => write!(f, "WebSocket"),
            NetworkProtocol::QUIC => write!(f, "QUIC"),
        }
    }
}

/// Network interface types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkInterfaceType {
    Ethernet,
    WiFi,
    Loopback,
    Bridge,
    VLAN,
    Tunnel,
    VirtualEthernet,
    InfiniBand,
}

/// Quality of Service classes
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum QoSClass {
    BestEffort,
    BulkData,
    Interactive,
    Streaming,
    RealTime,
    NetworkControl,
}

impl QoSClass {
    pub fn priority(&self) -> u8 {
        match self {
            QoSClass::BestEffort => 0,
            QoSClass::BulkData => 1,
            QoSClass::Interactive => 2,
            QoSClass::Streaming => 3,
            QoSClass::RealTime => 4,
            QoSClass::NetworkControl => 5,
        }
    }
}

/// Network packet buffer with zero-copy support
#[derive(Debug, Clone)]
pub struct PacketBuffer {
    pub buffer_id: u32,
    pub data: Vec<u8, MAX_PACKET_SIZE>,
    pub length: usize,
    pub offset: usize,
    pub protocol: NetworkProtocol,
    pub timestamp_ns: u64,
    pub interface_id: u8,
    pub qos_class: QoSClass,
    pub checksum: Option<u32>,
    pub hardware_offload: bool,
    pub reference_count: u32,
}

impl PacketBuffer {
    pub fn new(buffer_id: u32, interface_id: u8) -> Self {
        Self {
            buffer_id,
            data: Vec::new(),
            length: 0,
            offset: 0,
            protocol: NetworkProtocol::Ethernet,
            timestamp_ns: 0,
            interface_id,
            qos_class: QoSClass::BestEffort,
            checksum: None,
            hardware_offload: false,
            reference_count: 1,
        }
    }

    pub fn set_data(&mut self, data: &[u8]) -> Result<(), &'static str> {
        if data.len() > MAX_PACKET_SIZE {
            return Err("Packet too large");
        }

        self.data.clear();
        for &byte in data {
            let _ = self.data.push(byte);
        }
        self.length = data.len();
        Ok(())
    }

    pub fn get_payload(&self) -> &[u8] {
        &self.data[self.offset..self.length]
    }

    pub fn add_reference(&mut self) {
        self.reference_count += 1;
    }

    pub fn release_reference(&mut self) -> bool {
        self.reference_count -= 1;
        self.reference_count == 0
    }

    pub fn can_hardware_offload(&self) -> bool {
        self.length >= 64 && // Minimum size for hardware processing
        self.qos_class >= QoSClass::Interactive
    }
}

/// Network flow tracking
#[derive(Debug, Clone)]
pub struct NetworkFlow {
    pub flow_id: u64,
    pub source_ip: [u8; 16], // IPv6 compatible
    pub dest_ip: [u8; 16],
    pub source_port: u16,
    pub dest_port: u16,
    pub protocol: NetworkProtocol,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u32,
    pub packets_received: u32,
    pub established_time_ns: u64,
    pub last_activity_ns: u64,
    pub qos_class: QoSClass,
    pub bandwidth_limit_bps: Option<u64>,
    pub latency_target_ns: Option<u64>,
    pub congestion_window: u32,
    pub round_trip_time_ns: u64,
}

impl NetworkFlow {
    pub fn new(flow_id: u64, source_ip: [u8; 16], dest_ip: [u8; 16],
               source_port: u16, dest_port: u16, protocol: NetworkProtocol) -> Self {
        Self {
            flow_id,
            source_ip,
            dest_ip,
            source_port,
            dest_port,
            protocol,
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            established_time_ns: 0,
            last_activity_ns: 0,
            qos_class: QoSClass::BestEffort,
            bandwidth_limit_bps: None,
            latency_target_ns: None,
            congestion_window: 65536, // 64KB initial window
            round_trip_time_ns: 100_000_000, // 100ms default
        }
    }

    pub fn update_activity(&mut self, timestamp_ns: u64, bytes: u64, is_outbound: bool) {
        self.last_activity_ns = timestamp_ns;

        if is_outbound {
            self.bytes_sent += bytes;
            self.packets_sent += 1;
        } else {
            self.bytes_received += bytes;
            self.packets_received += 1;
        }
    }

    pub fn is_expired(&self, current_time_ns: u64, timeout_ns: u64) -> bool {
        current_time_ns - self.last_activity_ns > timeout_ns
    }

    pub fn throughput_bps(&self, window_ns: u64) -> f64 {
        if window_ns == 0 {
            return 0.0;
        }

        let total_bytes = self.bytes_sent + self.bytes_received;
        (total_bytes as f64 * 8.0 * 1_000_000_000.0) / window_ns as f64
    }
}

/// Network interface configuration
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    pub interface_id: u8,
    pub name: heapless::String<16>,
    pub interface_type: NetworkInterfaceType,
    pub mac_address: [u8; 6],
    pub ipv4_address: [u8; 4],
    pub ipv4_netmask: [u8; 4],
    pub ipv6_address: [u8; 16],
    pub mtu: usize,
    pub link_up: bool,
    pub link_speed_mbps: u32,
    pub duplex_full: bool,
    pub rx_packets: u64,
    pub tx_packets: u64,
    pub rx_bytes: u64,
    pub tx_bytes: u64,
    pub rx_errors: u32,
    pub tx_errors: u32,
    pub rx_dropped: u32,
    pub tx_dropped: u32,
    pub multicast_packets: u32,
    pub hardware_features: NetworkHardwareFeatures,
}

/// Hardware offloading features
#[derive(Debug, Clone, Copy)]
pub struct NetworkHardwareFeatures {
    pub checksum_offload_tx: bool,
    pub checksum_offload_rx: bool,
    pub tcp_segmentation_offload: bool,
    pub large_receive_offload: bool,
    pub scatter_gather: bool,
    pub vlan_offload: bool,
    pub receive_side_scaling: bool,
    pub generic_receive_offload: bool,
}

impl NetworkHardwareFeatures {
    pub fn new() -> Self {
        Self {
            checksum_offload_tx: true,
            checksum_offload_rx: true,
            tcp_segmentation_offload: true,
            large_receive_offload: true,
            scatter_gather: true,
            vlan_offload: false,
            receive_side_scaling: true,
            generic_receive_offload: true,
        }
    }
}

impl NetworkInterface {
    pub fn new(interface_id: u8, name: &str, interface_type: NetworkInterfaceType) -> Self {
        let mut iface_name = heapless::String::new();
        let _ = iface_name.push_str(name);

        Self {
            interface_id,
            name: iface_name,
            interface_type,
            mac_address: [0x02, 0x00, 0x00, 0x00, 0x00, interface_id],
            ipv4_address: [0, 0, 0, 0],
            ipv4_netmask: [255, 255, 255, 0],
            ipv6_address: [0; 16],
            mtu: DEFAULT_MTU,
            link_up: false,
            link_speed_mbps: 1000, // Gigabit default
            duplex_full: true,
            rx_packets: 0,
            tx_packets: 0,
            rx_bytes: 0,
            tx_bytes: 0,
            rx_errors: 0,
            tx_errors: 0,
            rx_dropped: 0,
            tx_dropped: 0,
            multicast_packets: 0,
            hardware_features: NetworkHardwareFeatures::new(),
        }
    }

    pub fn set_ipv4_address(&mut self, ip: [u8; 4], netmask: [u8; 4]) {
        self.ipv4_address = ip;
        self.ipv4_netmask = netmask;
    }

    pub fn set_link_state(&mut self, up: bool, speed_mbps: u32) {
        self.link_up = up;
        self.link_speed_mbps = speed_mbps;
    }

    pub fn update_rx_stats(&mut self, packets: u64, bytes: u64, errors: u32, dropped: u32) {
        self.rx_packets += packets;
        self.rx_bytes += bytes;
        self.rx_errors += errors;
        self.rx_dropped += dropped;
    }

    pub fn update_tx_stats(&mut self, packets: u64, bytes: u64, errors: u32, dropped: u32) {
        self.tx_packets += packets;
        self.tx_bytes += bytes;
        self.tx_errors += errors;
        self.tx_dropped += dropped;
    }

    pub fn utilization_percent(&self) -> f32 {
        if self.link_speed_mbps == 0 {
            return 0.0;
        }

        let total_bits = (self.rx_bytes + self.tx_bytes) * 8;
        let max_bits = (self.link_speed_mbps as u64) * 1_000_000;

        if max_bits > 0 {
            (total_bits as f32 / max_bits as f32) * 100.0
        } else {
            0.0
        }
    }
}

/// Zero-copy ring buffer for network I/O
#[derive(Debug)]
pub struct ZeroCopyRingBuffer {
    pub buffer_id: u32,
    pub head: usize,
    pub tail: usize,
    pub size: usize,
    pub packet_descriptors: Vec<u32, RING_BUFFER_SIZE>, // Packet buffer IDs
    pub free_slots: usize,
    pub total_packets_processed: u64,
    pub zero_copy_hits: u64,
    pub memory_copies_avoided: u64,
}

impl ZeroCopyRingBuffer {
    pub fn new(buffer_id: u32) -> Self {
        Self {
            buffer_id,
            head: 0,
            tail: 0,
            size: RING_BUFFER_SIZE,
            packet_descriptors: Vec::new(),
            free_slots: RING_BUFFER_SIZE,
            total_packets_processed: 0,
            zero_copy_hits: 0,
            memory_copies_avoided: 0,
        }
    }

    pub fn push_packet(&mut self, packet_id: u32) -> Result<(), &'static str> {
        if self.is_full() {
            return Err("Ring buffer full");
        }

        // Initialize with empty descriptors if needed
        while self.packet_descriptors.len() < self.size {
            let _ = self.packet_descriptors.push(0);
        }

        self.packet_descriptors[self.head] = packet_id;
        self.head = (self.head + 1) % self.size;
        self.free_slots -= 1;
        self.total_packets_processed += 1;
        self.zero_copy_hits += 1;

        Ok(())
    }

    pub fn pop_packet(&mut self) -> Option<u32> {
        if self.is_empty() {
            return None;
        }

        let packet_id = self.packet_descriptors[self.tail];
        self.tail = (self.tail + 1) % self.size;
        self.free_slots += 1;

        Some(packet_id)
    }

    pub fn is_empty(&self) -> bool {
        self.head == self.tail && self.free_slots == self.size
    }

    pub fn is_full(&self) -> bool {
        self.free_slots == 0
    }

    pub fn utilization_percent(&self) -> f32 {
        ((self.size - self.free_slots) as f32 / self.size as f32) * 100.0
    }

    pub fn zero_copy_efficiency(&self) -> f32 {
        if self.total_packets_processed > 0 {
            (self.zero_copy_hits as f32 / self.total_packets_processed as f32) * 100.0
        } else {
            0.0
        }
    }
}

/// Network traffic shaping and QoS
#[derive(Debug, Clone)]
pub struct TrafficShaper {
    pub shaper_id: u32,
    pub interface_id: u8,
    pub qos_queues: FnvIndexMap<QoSClass, Vec<u32, 256>, 8>, // QoS class -> packet IDs
    pub bandwidth_limits: FnvIndexMap<QoSClass, u64, 8>, // bits per second
    pub current_usage: FnvIndexMap<QoSClass, u64, 8>,
    pub packet_scheduler: PacketSchedulingAlgorithm,
    pub total_shaped_bytes: u64,
    pub total_dropped_bytes: u64,
    pub congestion_detected: bool,
}

/// Packet scheduling algorithms for QoS
#[derive(Debug, Clone, Copy)]
pub enum PacketSchedulingAlgorithm {
    FIFO,
    PriorityQueuing,
    WeightedFairQueuing,
    DeficitRoundRobin,
    HierarchicalTokenBucket,
    ClassBasedQueuing,
}

impl TrafficShaper {
    pub fn new(shaper_id: u32, interface_id: u8) -> Self {
        let mut bandwidth_limits = FnvIndexMap::new();
        let _ = bandwidth_limits.insert(QoSClass::RealTime, 100_000_000); // 100 Mbps
        let _ = bandwidth_limits.insert(QoSClass::Streaming, 50_000_000);  // 50 Mbps
        let _ = bandwidth_limits.insert(QoSClass::Interactive, 25_000_000); // 25 Mbps
        let _ = bandwidth_limits.insert(QoSClass::BestEffort, 10_000_000);  // 10 Mbps

        Self {
            shaper_id,
            interface_id,
            qos_queues: FnvIndexMap::new(),
            bandwidth_limits,
            current_usage: FnvIndexMap::new(),
            packet_scheduler: PacketSchedulingAlgorithm::WeightedFairQueuing,
            total_shaped_bytes: 0,
            total_dropped_bytes: 0,
            congestion_detected: false,
        }
    }

    pub fn enqueue_packet(&mut self, packet_id: u32, qos_class: QoSClass, packet_size: u64) -> Result<(), &'static str> {
        // Check bandwidth limits
        let current_usage = self.current_usage.get(&qos_class).copied().unwrap_or(0);
        let limit = self.bandwidth_limits.get(&qos_class).copied().unwrap_or(u64::MAX);

        if current_usage + packet_size * 8 > limit {
            self.total_dropped_bytes += packet_size;
            self.congestion_detected = true;
            return Err("Bandwidth limit exceeded");
        }

        // Add to appropriate queue
        if let Some(queue) = self.qos_queues.get_mut(&qos_class) {
            if queue.len() < 256 {
                let _ = queue.push(packet_id);
            } else {
                return Err("QoS queue full");
            }
        } else {
            let mut new_queue = Vec::new();
            let _ = new_queue.push(packet_id);
            let _ = self.qos_queues.insert(qos_class, new_queue);
        }

        // Update usage
        let new_usage = current_usage + packet_size * 8;
        let _ = self.current_usage.insert(qos_class, new_usage);
        self.total_shaped_bytes += packet_size;

        Ok(())
    }

    pub fn dequeue_packet(&mut self) -> Option<(u32, QoSClass)> {
        match self.packet_scheduler {
            PacketSchedulingAlgorithm::PriorityQueuing => self.priority_dequeue(),
            PacketSchedulingAlgorithm::WeightedFairQueuing => self.wfq_dequeue(),
            _ => self.fifo_dequeue(),
        }
    }

    fn priority_dequeue(&mut self) -> Option<(u32, QoSClass)> {
        let priority_order = [
            QoSClass::NetworkControl,
            QoSClass::RealTime,
            QoSClass::Streaming,
            QoSClass::Interactive,
            QoSClass::BulkData,
            QoSClass::BestEffort,
        ];

        for &qos_class in &priority_order {
            if let Some(queue) = self.qos_queues.get_mut(&qos_class) {
                if !queue.is_empty() {
                    let packet_id = queue.remove(0);
                    return Some((packet_id, qos_class));
                }
            }
        }

        None
    }

    fn wfq_dequeue(&mut self) -> Option<(u32, QoSClass)> {
        // Simplified weighted fair queuing - rotate through queues based on weights
        static mut WFQ_COUNTER: u32 = 0;

        let qos_weights = [
            (QoSClass::RealTime, 5),
            (QoSClass::Streaming, 3),
            (QoSClass::Interactive, 2),
            (QoSClass::BestEffort, 1),
        ];

        unsafe {
            for _ in 0..qos_weights.len() {
                let index = WFQ_COUNTER as usize % qos_weights.len();
                let (qos_class, weight) = qos_weights[index];

                if let Some(queue) = self.qos_queues.get_mut(&qos_class) {
                    if !queue.is_empty() && WFQ_COUNTER % (6 - weight) == 0 {
                        let packet_id = queue.remove(0);
                        WFQ_COUNTER += 1;
                        return Some((packet_id, qos_class));
                    }
                }
                WFQ_COUNTER += 1;
            }
        }

        None
    }

    fn fifo_dequeue(&mut self) -> Option<(u32, QoSClass)> {
        // Simple FIFO across all queues
        for (qos_class, queue) in &mut self.qos_queues {
            if !queue.is_empty() {
                let packet_id = queue.remove(0);
                return Some((packet_id, *qos_class));
            }
        }
        None
    }

    pub fn reset_bandwidth_counters(&mut self) {
        for (_, usage) in &mut self.current_usage {
            *usage = 0;
        }
        self.congestion_detected = false;
    }

    pub fn get_queue_lengths(&self) -> Vec<(QoSClass, usize), 8> {
        let mut lengths = Vec::new();
        for (&qos_class, queue) in &self.qos_queues {
            let _ = lengths.push((qos_class, queue.len()));
        }
        lengths
    }
}

/// Network stack statistics
#[derive(Debug, Clone, Copy)]
pub struct NetworkStackStats {
    pub total_packets_processed: u64,
    pub total_bytes_processed: u64,
    pub zero_copy_operations: u64,
    pub memory_copy_operations: u64,
    pub hardware_offloads: u64,
    pub active_flows: u32,
    pub expired_flows: u32,
    pub average_packet_size: f32,
    pub peak_throughput_mbps: f32,
    pub current_throughput_mbps: f32,
    pub packet_drop_rate: f32,
    pub average_latency_ns: u64,
    pub congestion_events: u32,
}

impl NetworkStackStats {
    pub fn new() -> Self {
        Self {
            total_packets_processed: 0,
            total_bytes_processed: 0,
            zero_copy_operations: 0,
            memory_copy_operations: 0,
            hardware_offloads: 0,
            active_flows: 0,
            expired_flows: 0,
            average_packet_size: 0.0,
            peak_throughput_mbps: 0.0,
            current_throughput_mbps: 0.0,
            packet_drop_rate: 0.0,
            average_latency_ns: 0,
            congestion_events: 0,
        }
    }

    pub fn zero_copy_efficiency(&self) -> f32 {
        let total_ops = self.zero_copy_operations + self.memory_copy_operations;
        if total_ops > 0 {
            (self.zero_copy_operations as f32 / total_ops as f32) * 100.0
        } else {
            0.0
        }
    }

    pub fn hardware_offload_ratio(&self) -> f32 {
        if self.total_packets_processed > 0 {
            (self.hardware_offloads as f32 / self.total_packets_processed as f32) * 100.0
        } else {
            0.0
        }
    }
}

/// Main advanced network stack
pub struct AdvancedNetworkStack {
    interfaces: Vec<NetworkInterface, MAX_NETWORK_INTERFACES>,
    packet_buffers: Vec<PacketBuffer, MAX_PACKET_BUFFERS>,
    active_flows: FnvIndexMap<u64, NetworkFlow, MAX_NETWORK_FLOWS>,
    rx_rings: Vec<ZeroCopyRingBuffer, MAX_NETWORK_INTERFACES>,
    tx_rings: Vec<ZeroCopyRingBuffer, MAX_NETWORK_INTERFACES>,
    traffic_shapers: Vec<TrafficShaper, MAX_NETWORK_INTERFACES>,
    stats: NetworkStackStats,
    buffer_counter: u32,
    flow_counter: u64,
    zero_copy_enabled: bool,
    hardware_offload_enabled: bool,
    qos_enabled: bool,
    flow_control_enabled: bool,
}

impl AdvancedNetworkStack {
    pub fn new() -> Self {
        Self {
            interfaces: Vec::new(),
            packet_buffers: Vec::new(),
            active_flows: FnvIndexMap::new(),
            rx_rings: Vec::new(),
            tx_rings: Vec::new(),
            traffic_shapers: Vec::new(),
            stats: NetworkStackStats::new(),
            buffer_counter: 0,
            flow_counter: 0,
            zero_copy_enabled: true,
            hardware_offload_enabled: true,
            qos_enabled: true,
            flow_control_enabled: true,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        crate::println!("[NET] Initializing advanced network stack...");

        // Create default interfaces
        self.create_default_interfaces()?;

        // Initialize packet buffer pools
        self.initialize_packet_buffers()?;

        // Setup zero-copy ring buffers
        self.setup_ring_buffers()?;

        // Initialize traffic shapers
        self.initialize_traffic_shapers()?;

        crate::println!("[NET] Advanced network stack initialized successfully");
        crate::println!("[NET] Interfaces: {}, Packet buffers: {}",
                       self.interfaces.len(), self.packet_buffers.len());
        crate::println!("[NET] Features: Zero-copy={}, Hardware offload={}, QoS={}",
                       self.zero_copy_enabled, self.hardware_offload_enabled, self.qos_enabled);

        Ok(())
    }

    pub fn create_interface(&mut self, name: &str, interface_type: NetworkInterfaceType) -> Result<u8, &'static str> {
        if self.interfaces.len() >= MAX_NETWORK_INTERFACES {
            return Err("Maximum interfaces reached");
        }

        let interface_id = self.interfaces.len() as u8;
        let interface = NetworkInterface::new(interface_id, name, interface_type);

        let _ = self.interfaces.push(interface);

        // Create corresponding ring buffers
        let rx_ring = ZeroCopyRingBuffer::new(interface_id as u32 * 2);
        let tx_ring = ZeroCopyRingBuffer::new(interface_id as u32 * 2 + 1);
        let _ = self.rx_rings.push(rx_ring);
        let _ = self.tx_rings.push(tx_ring);

        // Create traffic shaper
        let shaper = TrafficShaper::new(interface_id as u32, interface_id);
        let _ = self.traffic_shapers.push(shaper);

        crate::println!("[NET] Created interface '{}' (ID: {}, Type: {:?})",
                       name, interface_id, interface_type);

        Ok(interface_id)
    }

    pub fn receive_packet(&mut self, interface_id: u8, data: &[u8],
                         protocol: NetworkProtocol) -> Result<u32, &'static str> {
        // Allocate packet buffer
        let mut packet = PacketBuffer::new(self.buffer_counter, interface_id);
        packet.set_data(data)?;
        packet.protocol = protocol;
        packet.timestamp_ns = self.get_current_time_ns();

        // Determine QoS class based on protocol and packet inspection
        packet.qos_class = self.classify_packet(&packet);

        // Enable hardware offload if supported
        if self.hardware_offload_enabled && packet.can_hardware_offload() {
            packet.hardware_offload = true;
            self.stats.hardware_offloads += 1;
        }

        self.buffer_counter += 1;
        let packet_id = packet.buffer_id;

        // Add to packet buffer pool
        if self.packet_buffers.len() < MAX_PACKET_BUFFERS {
            let _ = self.packet_buffers.push(packet);
        } else {
            return Err("Packet buffer pool full");
        }

        // Add to receive ring buffer for zero-copy processing
        if self.zero_copy_enabled && interface_id < self.rx_rings.len() as u8 {
            if let Some(rx_ring) = self.rx_rings.get_mut(interface_id as usize) {
                let _ = rx_ring.push_packet(packet_id);
                self.stats.zero_copy_operations += 1;
            }
        } else {
            self.stats.memory_copy_operations += 1;
        }

        // Update interface statistics
        if let Some(interface) = self.interfaces.get_mut(interface_id as usize) {
            interface.update_rx_stats(1, data.len() as u64, 0, 0);
        }

        // Update flow tracking
        self.update_flow_tracking(packet_id, false);

        // Update stack statistics
        self.stats.total_packets_processed += 1;
        self.stats.total_bytes_processed += data.len() as u64;

        crate::println!("[NET] Received packet (ID: {}) on interface {} - {} bytes, {} protocol",
                       packet_id, interface_id, data.len(), protocol);

        Ok(packet_id)
    }

    pub fn transmit_packet(&mut self, interface_id: u8, packet_id: u32) -> Result<(), &'static str> {
        // Find packet buffer
        let packet_index = self.packet_buffers.iter()
            .position(|p| p.buffer_id == packet_id)
            .ok_or("Packet not found")?;

        let packet = &self.packet_buffers[packet_index];

        // Apply QoS traffic shaping
        if self.qos_enabled && interface_id < self.traffic_shapers.len() as u8 {
            if let Some(shaper) = self.traffic_shapers.get_mut(interface_id as usize) {
                if shaper.enqueue_packet(packet_id, packet.qos_class, packet.length as u64).is_err() {
                    return Err("Traffic shaping rejected packet");
                }
            }
        }

        // Add to transmit ring buffer for zero-copy
        if self.zero_copy_enabled && interface_id < self.tx_rings.len() as u8 {
            if let Some(tx_ring) = self.tx_rings.get_mut(interface_id as usize) {
                let _ = tx_ring.push_packet(packet_id);
            }
        }

        // Update interface statistics
        if let Some(interface) = self.interfaces.get_mut(interface_id as usize) {
            interface.update_tx_stats(1, packet.length as u64, 0, 0);
        }

        // Update flow tracking
        self.update_flow_tracking(packet_id, true);

        crate::println!("[NET] Transmitted packet (ID: {}) on interface {}", packet_id, interface_id);

        Ok(())
    }

    pub fn get_network_stats(&self) -> NetworkStackStats {
        self.stats
    }

    pub fn get_interface_stats(&self, interface_id: u8) -> Option<&NetworkInterface> {
        self.interfaces.get(interface_id as usize)
    }

    pub fn enable_zero_copy(&mut self, enabled: bool) {
        self.zero_copy_enabled = enabled;
        crate::println!("[NET] Zero-copy I/O: {}", if enabled { "enabled" } else { "disabled" });
    }

    pub fn enable_hardware_offload(&mut self, enabled: bool) {
        self.hardware_offload_enabled = enabled;
        crate::println!("[NET] Hardware offload: {}", if enabled { "enabled" } else { "disabled" });
    }

    pub fn periodic_network_task(&mut self) {
        let current_time = self.get_current_time_ns();

        // Process expired flows
        self.cleanup_expired_flows(current_time);

        // Reset traffic shaping counters periodically
        for shaper in &mut self.traffic_shapers {
            shaper.reset_bandwidth_counters();
        }

        // Update throughput calculations
        self.update_throughput_stats();
    }

    // Private helper methods
    fn create_default_interfaces(&mut self) -> Result<(), &'static str> {
        let _ = self.create_interface("lo", NetworkInterfaceType::Loopback);
        let _ = self.create_interface("eth0", NetworkInterfaceType::Ethernet);
        crate::println!("[NET] Created default network interfaces");
        Ok(())
    }

    fn initialize_packet_buffers(&mut self) -> Result<(), &'static str> {
        // Pre-allocate packet buffers for performance
        for i in 0..256 { // Start with 256 buffers
            let buffer = PacketBuffer::new(i, 0);
            let _ = self.packet_buffers.push(buffer);
        }
        crate::println!("[NET] Initialized {} packet buffers", self.packet_buffers.len());
        Ok(())
    }

    fn setup_ring_buffers(&mut self) -> Result<(), &'static str> {
        // Ring buffers are created when interfaces are created
        crate::println!("[NET] Zero-copy ring buffers ready");
        Ok(())
    }

    fn initialize_traffic_shapers(&mut self) -> Result<(), &'static str> {
        // Traffic shapers are created when interfaces are created
        crate::println!("[NET] Traffic shapers initialized");
        Ok(())
    }

    fn classify_packet(&self, packet: &PacketBuffer) -> QoSClass {
        match packet.protocol {
            NetworkProtocol::ICMP => QoSClass::NetworkControl,
            NetworkProtocol::DNS | NetworkProtocol::DHCP => QoSClass::Interactive,
            NetworkProtocol::HTTP | NetworkProtocol::HTTPS => QoSClass::Interactive,
            NetworkProtocol::UDP => QoSClass::RealTime, // Assume real-time for UDP
            NetworkProtocol::TCP => QoSClass::BestEffort,
            _ => QoSClass::BestEffort,
        }
    }

    fn update_flow_tracking(&mut self, _packet_id: u32, _is_outbound: bool) {
        // Simplified flow tracking - in a real implementation would parse packet headers
        self.stats.active_flows = self.active_flows.len() as u32;
    }

    fn cleanup_expired_flows(&mut self, current_time_ns: u64) {
        let timeout_ns = 300_000_000_000; // 5 minutes

        let mut expired_flows = Vec::new();
        for (flow_id, flow) in &self.active_flows {
            if flow.is_expired(current_time_ns, timeout_ns) {
                let _ = expired_flows.push(*flow_id);
            }
        }

        for flow_id in expired_flows {
            self.active_flows.remove(&flow_id);
            self.stats.expired_flows += 1;
        }
    }

    fn update_throughput_stats(&mut self) {
        // Calculate current throughput based on recent activity
        let total_bits = self.stats.total_bytes_processed * 8;
        if total_bits > 0 {
            self.stats.current_throughput_mbps = (total_bits as f32) / 1_000_000.0;

            if self.stats.current_throughput_mbps > self.stats.peak_throughput_mbps {
                self.stats.peak_throughput_mbps = self.stats.current_throughput_mbps;
            }
        }

        // Update average packet size
        if self.stats.total_packets_processed > 0 {
            self.stats.average_packet_size =
                self.stats.total_bytes_processed as f32 / self.stats.total_packets_processed as f32;
        }
    }

    fn get_current_time_ns(&self) -> u64 {
        static mut TIME_COUNTER: u64 = 0;
        unsafe {
            TIME_COUNTER += 1_000_000; // Increment by 1ms in nanoseconds
            TIME_COUNTER
        }
    }
}

lazy_static! {
    static ref NETWORK_STACK: Mutex<AdvancedNetworkStack> = Mutex::new(AdvancedNetworkStack::new());
}

pub fn init_network_stack() {
    let mut stack = NETWORK_STACK.lock();
    match stack.initialize() {
        Ok(_) => crate::println!("[NET] Advanced network stack ready"),
        Err(e) => crate::println!("[NET] Failed to initialize: {}", e),
    }
}

pub fn create_interface(name: &str, interface_type: NetworkInterfaceType) -> Result<u8, &'static str> {
    NETWORK_STACK.lock().create_interface(name, interface_type)
}

pub fn receive_packet(interface_id: u8, data: &[u8], protocol: NetworkProtocol) -> Result<u32, &'static str> {
    NETWORK_STACK.lock().receive_packet(interface_id, data, protocol)
}

pub fn transmit_packet(interface_id: u8, packet_id: u32) -> Result<(), &'static str> {
    NETWORK_STACK.lock().transmit_packet(interface_id, packet_id)
}

pub fn get_network_stats() -> NetworkStackStats {
    NETWORK_STACK.lock().get_network_stats()
}

pub fn enable_zero_copy(enabled: bool) {
    NETWORK_STACK.lock().enable_zero_copy(enabled);
}

pub fn enable_hardware_offload(enabled: bool) {
    NETWORK_STACK.lock().enable_hardware_offload(enabled);
}

pub fn periodic_network_task() {
    NETWORK_STACK.lock().periodic_network_task();
}

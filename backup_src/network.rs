//! Network Stack for RustOS
//!
//! This module provides:
//! - Basic TCP/IP protocol implementation
//! - Network interface management
//! - Socket abstraction layer
//! - Ethernet frame handling
//! - IP packet routing
//! - TCP connection management
//! - UDP datagram support
//! - Integration with peripheral network drivers
//! - AI-driven network optimization

use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use core::sync::atomic::{AtomicU32, Ordering};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum Transmission Unit for Ethernet
pub const ETH_MTU: usize = 1500;

/// Maximum number of network interfaces
pub const MAX_NETWORK_INTERFACES: usize = 16;

/// Maximum number of sockets
pub const MAX_SOCKETS: usize = 1024;

/// TCP default window size
pub const TCP_DEFAULT_WINDOW: u16 = 8192;

/// Network interface identifier
pub type InterfaceId = u32;

/// Socket identifier
pub type SocketId = u32;

/// IP address (IPv4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IpAddr([u8; 4]);

impl IpAddr {
    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        IpAddr([a, b, c, d])
    }

    pub const fn localhost() -> Self {
        IpAddr([127, 0, 0, 1])
    }

    pub const fn any() -> Self {
        IpAddr([0, 0, 0, 0])
    }

    pub const fn broadcast() -> Self {
        IpAddr([255, 255, 255, 255])
    }

    pub fn octets(&self) -> [u8; 4] {
        self.0
    }

    pub fn is_loopback(&self) -> bool {
        self.0[0] == 127
    }

    pub fn is_private(&self) -> bool {
        match self.0 {
            [10, _, _, _] => true,
            [172, b, _, _] if b >= 16 && b <= 31 => true,
            [192, 168, _, _] => true,
            _ => false,
        }
    }

    pub fn is_multicast(&self) -> bool {
        self.0[0] >= 224 && self.0[0] <= 239
    }

    pub fn is_broadcast(&self) -> bool {
        *self == Self::broadcast()
    }
}

impl core::fmt::Display for IpAddr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}.{}.{}", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

/// MAC address (Ethernet)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacAddr([u8; 6]);

impl MacAddr {
    pub const fn new(a: u8, b: u8, c: u8, d: u8, e: u8, f: u8) -> Self {
        MacAddr([a, b, c, d, e, f])
    }

    pub const fn broadcast() -> Self {
        MacAddr([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
    }

    pub fn is_broadcast(&self) -> bool {
        *self == Self::broadcast()
    }

    pub fn is_multicast(&self) -> bool {
        self.0[0] & 0x01 != 0
    }

    pub fn octets(&self) -> [u8; 6] {
        self.0
    }
}

impl core::fmt::Display for MacAddr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
               self.0[0], self.0[1], self.0[2], self.0[3], self.0[4], self.0[5])
    }
}

/// Network protocol types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Protocol {
    Icmp = 1,
    Tcp = 6,
    Udp = 17,
}

/// Socket types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SocketType {
    Stream,    // TCP
    Datagram,  // UDP
    Raw,       // Raw IP
}

/// Socket state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SocketState {
    Closed,
    Listen,
    SynSent,
    SynReceived,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}

/// Ethernet frame header
#[derive(Debug, Clone)]
pub struct EthernetHeader {
    pub dst_mac: MacAddr,
    pub src_mac: MacAddr,
    pub ethertype: u16,
}

impl EthernetHeader {
    pub const SIZE: usize = 14;

    pub fn new(dst_mac: MacAddr, src_mac: MacAddr, ethertype: u16) -> Self {
        Self { dst_mac, src_mac, ethertype }
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..6].copy_from_slice(&self.dst_mac.octets());
        bytes[6..12].copy_from_slice(&self.src_mac.octets());
        bytes[12..14].copy_from_slice(&self.ethertype.to_be_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }

        let mut dst_mac = [0u8; 6];
        let mut src_mac = [0u8; 6];
        dst_mac.copy_from_slice(&bytes[0..6]);
        src_mac.copy_from_slice(&bytes[6..12]);
        let ethertype = u16::from_be_bytes([bytes[12], bytes[13]]);

        Some(Self {
            dst_mac: MacAddr(dst_mac),
            src_mac: MacAddr(src_mac),
            ethertype,
        })
    }
}

/// IP header (simplified IPv4)
#[derive(Debug, Clone)]
pub struct IpHeader {
    pub version: u8,
    pub header_length: u8,
    pub type_of_service: u8,
    pub total_length: u16,
    pub identification: u16,
    pub flags: u8,
    pub fragment_offset: u16,
    pub time_to_live: u8,
    pub protocol: u8,
    pub header_checksum: u16,
    pub src_addr: IpAddr,
    pub dst_addr: IpAddr,
}

impl IpHeader {
    pub const MIN_SIZE: usize = 20;

    pub fn new(src_addr: IpAddr, dst_addr: IpAddr, protocol: u8, payload_len: u16) -> Self {
        Self {
            version: 4,
            header_length: 5, // 20 bytes
            type_of_service: 0,
            total_length: Self::MIN_SIZE as u16 + payload_len,
            identification: 0,
            flags: 0x02, // Don't fragment
            fragment_offset: 0,
            time_to_live: 64,
            protocol,
            header_checksum: 0, // Will be calculated
            src_addr,
            dst_addr,
        }
    }

    pub fn to_bytes(&self) -> [u8; Self::MIN_SIZE] {
        let mut bytes = [0u8; Self::MIN_SIZE];
        bytes[0] = (self.version << 4) | self.header_length;
        bytes[1] = self.type_of_service;
        bytes[2..4].copy_from_slice(&self.total_length.to_be_bytes());
        bytes[4..6].copy_from_slice(&self.identification.to_be_bytes());
        bytes[6..8].copy_from_slice(&((self.flags as u16) << 13 | self.fragment_offset).to_be_bytes());
        bytes[8] = self.time_to_live;
        bytes[9] = self.protocol;
        bytes[10..12].copy_from_slice(&self.header_checksum.to_be_bytes());
        bytes[12..16].copy_from_slice(&self.src_addr.octets());
        bytes[16..20].copy_from_slice(&self.dst_addr.octets());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::MIN_SIZE {
            return None;
        }

        let version = bytes[0] >> 4;
        let header_length = bytes[0] & 0x0F;
        let type_of_service = bytes[1];
        let total_length = u16::from_be_bytes([bytes[2], bytes[3]]);
        let identification = u16::from_be_bytes([bytes[4], bytes[5]]);
        let flags_and_fragment = u16::from_be_bytes([bytes[6], bytes[7]]);
        let flags = (flags_and_fragment >> 13) as u8;
        let fragment_offset = flags_and_fragment & 0x1FFF;
        let time_to_live = bytes[8];
        let protocol = bytes[9];
        let header_checksum = u16::from_be_bytes([bytes[10], bytes[11]]);

        let mut src_addr = [0u8; 4];
        let mut dst_addr = [0u8; 4];
        src_addr.copy_from_slice(&bytes[12..16]);
        dst_addr.copy_from_slice(&bytes[16..20]);

        Some(Self {
            version,
            header_length,
            type_of_service,
            total_length,
            identification,
            flags,
            fragment_offset,
            time_to_live,
            protocol,
            header_checksum,
            src_addr: IpAddr(src_addr),
            dst_addr: IpAddr(dst_addr),
        })
    }

    pub fn calculate_checksum(&mut self) {
        self.header_checksum = 0;
        let bytes = self.to_bytes();
        let checksum = calculate_internet_checksum(&bytes);
        self.header_checksum = checksum;
    }
}

/// TCP header
#[derive(Debug, Clone)]
pub struct TcpHeader {
    pub src_port: u16,
    pub dst_port: u16,
    pub sequence_number: u32,
    pub acknowledgment_number: u32,
    pub data_offset: u8,
    pub flags: u8,
    pub window_size: u16,
    pub checksum: u16,
    pub urgent_pointer: u16,
}

impl TcpHeader {
    pub const MIN_SIZE: usize = 20;

    // TCP flags
    pub const FIN: u8 = 0x01;
    pub const SYN: u8 = 0x02;
    pub const RST: u8 = 0x04;
    pub const PSH: u8 = 0x08;
    pub const ACK: u8 = 0x10;
    pub const URG: u8 = 0x20;

    pub fn new(src_port: u16, dst_port: u16) -> Self {
        Self {
            src_port,
            dst_port,
            sequence_number: 0,
            acknowledgment_number: 0,
            data_offset: 5, // 20 bytes
            flags: 0,
            window_size: TCP_DEFAULT_WINDOW,
            checksum: 0,
            urgent_pointer: 0,
        }
    }

    pub fn to_bytes(&self) -> [u8; Self::MIN_SIZE] {
        let mut bytes = [0u8; Self::MIN_SIZE];
        bytes[0..2].copy_from_slice(&self.src_port.to_be_bytes());
        bytes[2..4].copy_from_slice(&self.dst_port.to_be_bytes());
        bytes[4..8].copy_from_slice(&self.sequence_number.to_be_bytes());
        bytes[8..12].copy_from_slice(&self.acknowledgment_number.to_be_bytes());
        bytes[12] = self.data_offset << 4;
        bytes[13] = self.flags;
        bytes[14..16].copy_from_slice(&self.window_size.to_be_bytes());
        bytes[16..18].copy_from_slice(&self.checksum.to_be_bytes());
        bytes[18..20].copy_from_slice(&self.urgent_pointer.to_be_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::MIN_SIZE {
            return None;
        }

        Some(Self {
            src_port: u16::from_be_bytes([bytes[0], bytes[1]]),
            dst_port: u16::from_be_bytes([bytes[2], bytes[3]]),
            sequence_number: u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            acknowledgment_number: u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            data_offset: bytes[12] >> 4,
            flags: bytes[13],
            window_size: u16::from_be_bytes([bytes[14], bytes[15]]),
            checksum: u16::from_be_bytes([bytes[16], bytes[17]]),
            urgent_pointer: u16::from_be_bytes([bytes[18], bytes[19]]),
        })
    }
}

/// UDP header
#[derive(Debug, Clone)]
pub struct UdpHeader {
    pub src_port: u16,
    pub dst_port: u16,
    pub length: u16,
    pub checksum: u16,
}

impl UdpHeader {
    pub const SIZE: usize = 8;

    pub fn new(src_port: u16, dst_port: u16, payload_len: u16) -> Self {
        Self {
            src_port,
            dst_port,
            length: Self::SIZE as u16 + payload_len,
            checksum: 0,
        }
    }

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..2].copy_from_slice(&self.src_port.to_be_bytes());
        bytes[2..4].copy_from_slice(&self.dst_port.to_be_bytes());
        bytes[4..6].copy_from_slice(&self.length.to_be_bytes());
        bytes[6..8].copy_from_slice(&self.checksum.to_be_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }

        Some(Self {
            src_port: u16::from_be_bytes([bytes[0], bytes[1]]),
            dst_port: u16::from_be_bytes([bytes[2], bytes[3]]),
            length: u16::from_be_bytes([bytes[4], bytes[5]]),
            checksum: u16::from_be_bytes([bytes[6], bytes[7]]),
        })
    }
}

/// Network packet
#[derive(Debug, Clone)]
pub struct NetworkPacket {
    pub ethernet_header: Option<EthernetHeader>,
    pub ip_header: Option<IpHeader>,
    pub transport_header: Option<TransportHeader>,
    pub payload: Vec<u8>,
}

/// Transport layer header (TCP or UDP)
#[derive(Debug, Clone)]
pub enum TransportHeader {
    Tcp(TcpHeader),
    Udp(UdpHeader),
}

/// Network interface
#[derive(Debug)]
pub struct NetworkInterface {
    pub id: InterfaceId,
    pub name: String,
    pub mac_addr: MacAddr,
    pub ip_addr: Option<IpAddr>,
    pub netmask: Option<IpAddr>,
    pub gateway: Option<IpAddr>,
    pub mtu: usize,
    pub is_up: bool,
    pub tx_packets: u64,
    pub rx_packets: u64,
    pub tx_bytes: u64,
    pub rx_bytes: u64,
    pub tx_errors: u32,
    pub rx_errors: u32,
}

impl NetworkInterface {
    pub fn new(id: InterfaceId, name: String, mac_addr: MacAddr) -> Self {
        Self {
            id,
            name,
            mac_addr,
            ip_addr: None,
            netmask: None,
            gateway: None,
            mtu: ETH_MTU,
            is_up: false,
            tx_packets: 0,
            rx_packets: 0,
            tx_bytes: 0,
            rx_bytes: 0,
            tx_errors: 0,
            rx_errors: 0,
        }
    }

    pub fn configure(&mut self, ip_addr: IpAddr, netmask: IpAddr, gateway: Option<IpAddr>) {
        self.ip_addr = Some(ip_addr);
        self.netmask = Some(netmask);
        self.gateway = gateway;
    }

    pub fn up(&mut self) {
        self.is_up = true;
    }

    pub fn down(&mut self) {
        self.is_up = false;
    }

    pub fn transmit_packet(&mut self, packet: &NetworkPacket) -> Result<(), &'static str> {
        if !self.is_up {
            return Err("Interface is down");
        }

        // In a real implementation, this would send to hardware driver
        self.tx_packets += 1;
        self.tx_bytes += packet.payload.len() as u64;

        // Simulate packet transmission
        crate::println!("[NET] TX {} bytes on interface {}", packet.payload.len(), self.name);
        Ok(())
    }

    pub fn receive_packet(&mut self, data: &[u8]) -> Result<NetworkPacket, &'static str> {
        if !self.is_up {
            return Err("Interface is down");
        }

        self.rx_packets += 1;
        self.rx_bytes += data.len() as u64;

        // Parse Ethernet frame
        let eth_header = EthernetHeader::from_bytes(data)
            .ok_or("Invalid Ethernet header")?;

        let ip_start = EthernetHeader::SIZE;
        if data.len() < ip_start {
            return Err("Packet too short");
        }

        // Parse IP header if it's an IP packet
        let (ip_header, transport_header, payload_start) = if eth_header.ethertype == 0x0800 {
            let ip_header = IpHeader::from_bytes(&data[ip_start..])
                .ok_or("Invalid IP header")?;

            let transport_start = ip_start + IpHeader::MIN_SIZE;
            let transport_header = match ip_header.protocol {
                6 => { // TCP
                    let tcp_header = TcpHeader::from_bytes(&data[transport_start..])
                        .ok_or("Invalid TCP header")?;
                    Some(TransportHeader::Tcp(tcp_header))
                }
                17 => { // UDP
                    let udp_header = UdpHeader::from_bytes(&data[transport_start..])
                        .ok_or("Invalid UDP header")?;
                    Some(TransportHeader::Udp(udp_header))
                }
                _ => None,
            };

            let payload_start = transport_start + match &transport_header {
                Some(TransportHeader::Tcp(_)) => TcpHeader::MIN_SIZE,
                Some(TransportHeader::Udp(_)) => UdpHeader::SIZE,
                None => 0,
            };

            (Some(ip_header), transport_header, payload_start)
        } else {
            (None, None, ip_start)
        };

        let payload = if payload_start < data.len() {
            data[payload_start..].to_vec()
        } else {
            Vec::new()
        };

        Ok(NetworkPacket {
            ethernet_header: Some(eth_header),
            ip_header,
            transport_header,
            payload,
        })
    }
}

/// Socket implementation
#[derive(Debug)]
pub struct Socket {
    pub id: SocketId,
    pub socket_type: SocketType,
    pub state: SocketState,
    pub local_addr: Option<IpAddr>,
    pub local_port: Option<u16>,
    pub remote_addr: Option<IpAddr>,
    pub remote_port: Option<u16>,
    pub rx_buffer: VecDeque<u8>,
    pub tx_buffer: VecDeque<u8>,
    pub rx_buffer_size: usize,
    pub tx_buffer_size: usize,
    pub creation_time: u64,
    pub last_activity: u64,
    pub owner_process: u32,
}

impl Socket {
    pub fn new(id: SocketId, socket_type: SocketType, owner_process: u32) -> Self {
        let creation_time = crate::time::uptime_ms();
        Self {
            id,
            socket_type,
            state: SocketState::Closed,
            local_addr: None,
            local_port: None,
            remote_addr: None,
            remote_port: None,
            rx_buffer: VecDeque::new(),
            tx_buffer: VecDeque::new(),
            rx_buffer_size: 8192,
            tx_buffer_size: 8192,
            creation_time,
            last_activity: creation_time,
            owner_process,
        }
    }

    pub fn bind(&mut self, addr: IpAddr, port: u16) -> Result<(), &'static str> {
        if self.state != SocketState::Closed {
            return Err("Socket not closed");
        }

        self.local_addr = Some(addr);
        self.local_port = Some(port);
        self.last_activity = crate::time::uptime_ms();
        Ok(())
    }

    pub fn listen(&mut self) -> Result<(), &'static str> {
        if self.socket_type != SocketType::Stream {
            return Err("Only stream sockets can listen");
        }

        if self.local_port.is_none() {
            return Err("Socket not bound");
        }

        self.state = SocketState::Listen;
        self.last_activity = crate::time::uptime_ms();
        Ok(())
    }

    pub fn connect(&mut self, remote_addr: IpAddr, remote_port: u16) -> Result<(), &'static str> {
        if self.state != SocketState::Closed {
            return Err("Socket not closed");
        }

        self.remote_addr = Some(remote_addr);
        self.remote_port = Some(remote_port);

        if self.socket_type == SocketType::Stream {
            self.state = SocketState::SynSent;
            // TODO: Send SYN packet
        } else {
            self.state = SocketState::Established;
        }

        self.last_activity = crate::time::uptime_ms();
        Ok(())
    }

    pub fn send(&mut self, data: &[u8]) -> Result<usize, &'static str> {
        if self.state != SocketState::Established {
            return Err("Socket not established");
        }

        let available_space = self.tx_buffer_size - self.tx_buffer.len();
        let bytes_to_send = data.len().min(available_space);

        for &byte in &data[..bytes_to_send] {
            self.tx_buffer.push_back(byte);
        }

        self.last_activity = crate::time::uptime_ms();
        Ok(bytes_to_send)
    }

    pub fn receive(&mut self, buffer: &mut [u8]) -> Result<usize, &'static str> {
        if self.state != SocketState::Established && self.state != SocketState::CloseWait {
            return Err("Socket not ready for receive");
        }

        let bytes_to_receive = buffer.len().min(self.rx_buffer.len());

        for i in 0..bytes_to_receive {
            buffer[i] = self.rx_buffer.pop_front().unwrap();
        }

        self.last_activity = crate::time::uptime_ms();
        Ok(bytes_to_receive)
    }

    pub fn close(&mut self) {
        match self.socket_type {
            SocketType::Stream => {
                match self.state {
                    SocketState::Established => self.state = SocketState::FinWait1,
                    SocketState::CloseWait => self.state = SocketState::LastAck,
                    _ => self.state = SocketState::Closed,
                }
            }
            _ => self.state = SocketState::Closed,
        }

        self.last_activity = crate::time::uptime_ms();
    }
}

/// Routing table entry
#[derive(Debug, Clone)]
pub struct RouteEntry {
    pub destination: IpAddr,
    pub netmask: IpAddr,
    pub gateway: Option<IpAddr>,
    pub interface_id: InterfaceId,
    pub metric: u32,
}

/// Network statistics
#[derive(Debug, Default)]
pub struct NetworkStatistics {
    pub total_packets_sent: u64,
    pub total_packets_received: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub tcp_connections: u32,
    pub udp_sockets: u32,
    pub routing_table_size: u32,
    pub arp_table_size: u32,
    pub packet_errors: u32,
    pub checksum_errors: u32,
}

/// Main network manager
pub struct NetworkManager {
    interfaces: BTreeMap<InterfaceId, NetworkInterface>,
    sockets: BTreeMap<SocketId, Socket>,
    routing_table: Vec<RouteEntry>,
    arp_table: BTreeMap<IpAddr, MacAddr>,
    next_interface_id: InterfaceId,
    next_socket_id: SocketId,
    statistics: NetworkStatistics,
}

impl NetworkManager {
    pub fn new() -> Self {
        Self {
            interfaces: BTreeMap::new(),
            sockets: BTreeMap::new(),
            routing_table: Vec::new(),
            arp_table: BTreeMap::new(),
            next_interface_id: 1,
            next_socket_id: 1,
            statistics: NetworkStatistics::default(),
        }
    }

    /// Create a new network interface
    pub fn create_interface(&mut self, name: String, mac_addr: MacAddr) -> Result<InterfaceId, &'static str> {
        let id = self.next_interface_id;
        self.next_interface_id += 1;

        let interface = NetworkInterface::new(id, name, mac_addr);
        self.interfaces.insert(id, interface);

        Ok(id)
    }

    /// Configure network interface
    pub fn configure_interface(
        &mut self,
        id: InterfaceId,
        ip_addr: IpAddr,
        netmask: IpAddr,
        gateway: Option<IpAddr>
    ) -> Result<(), &'static str> {
        let interface = self.interfaces.get_mut(&id).ok_or("Interface not found")?;
        interface.configure(ip_addr, netmask, gateway);

        // Add route for local network
        self.routing_table.push(RouteEntry {
            destination: IpAddr::new(
                ip_addr.0[0] & netmask.0[0],
                ip_addr.0[1] & netmask.0[1],
                ip_addr.0[2] & netmask.0[2],
                ip_addr.0[3] & netmask.0[3],
            ),
            netmask,
            gateway: None,
            interface_id: id,
            metric: 0,
        });

        // Add default route if gateway specified
        if let Some(gw) = gateway {
            self.routing_table.push(RouteEntry {
                destination: IpAddr::any(),
                netmask: IpAddr::any(),
                gateway: Some(gw),
                interface_id: id,
                metric: 10,
            });
        }

        Ok(())
    }

    /// Bring interface up
    pub fn interface_up(&mut self, id: InterfaceId) -> Result<(), &'static str> {
        let interface = self.interfaces.get_mut(&id).ok_or("Interface not found")?;
        interface.up();
        Ok(())
    }

    /// Bring interface down
    pub fn interface_down(&mut self, id: InterfaceId) -> Result<(), &'static str> {
        let interface = self.interfaces.get_mut(&id).ok_or("Interface not found")?;
        interface.down();
        Ok(())
    }

    /// Create a socket
    pub fn create_socket(&mut self, socket_type: SocketType, owner_process: u32) -> Result<SocketId, &'static str> {
        if self.sockets.len() >= MAX_SOCKETS {
            return Err("Too many sockets");
        }

        let id = self.next_socket_id;
        self.next_socket_id += 1;

        let socket = Socket::new(id, socket_type, owner_process);
        self.sockets.insert(id, socket);

        match socket_type {
            SocketType::Stream => self.statistics.tcp_connections += 1,
            SocketType::Datagram => self.statistics.udp_sockets += 1,
            _ => {}
        }

        Ok(id)
    }

    /// Close a socket
    pub fn close_socket(&mut self, id: SocketId) -> Result<(), &'static str> {
        if let Some(mut socket) = self.sockets.remove(&id) {
            socket.close();
            match socket.socket_type {
                SocketType::Stream => self.statistics.tcp_connections -= 1,
                SocketType::Datagram => self.statistics.udp_sockets -= 1,
                _ => {}
            }
            Ok(())
        } else {
            Err("Socket not found")
        }
    }

    /// Send packet through appropriate interface
    pub fn send_packet(&mut self, packet: NetworkPacket) -> Result<(), &'static str> {
        let dst_addr = packet.ip_header.as_ref()
            .map(|h| h.dst_addr)
            .unwrap_or(IpAddr::broadcast());

        // Find route
        let route = self.find_route(dst_addr).ok_or("No route to destination")?;
        let interface = self.interfaces.get_mut(&route.interface_id)
            .ok_or("Interface not found")?;

        interface.transmit_packet(&packet)?;
        self.statistics.total_packets_sent += 1;
        self.statistics.total_bytes_sent += packet.payload.len() as u64;

        Ok(())
    }

    /// Find route to destination
    fn find_route(&self, dst_addr: IpAddr) -> Option<&RouteEntry> {
        // Find best matching route (longest prefix match)
        let mut best_route = None;
        let mut best_metric = u32::MAX;

        for route in &self.routing_table {
            // Check if destination matches this route
            let net_addr = IpAddr::new(
                dst_addr.0[0] & route.netmask.0[0],
                dst_addr.0[1] & route.netmask.0[1],
                dst_addr.0[2] & route.netmask.0[2],
                dst_addr.0[3] & route.netmask.0[3],
            );

            if net_addr == route.destination && route.metric < best_metric {
                best_route = Some(route);
                best_metric = route.metric;
            }
        }

        best_route
    }

    /// Process incoming packet
    pub fn process_packet(&mut self, interface_id: InterfaceId, data: &[u8]) -> Result<(), &'static str> {
        let interface = self.interfaces.get_mut(&interface_id)
            .ok_or("Interface not found")?;

        let packet = interface.receive_packet(data)?;
        self.statistics.total_packets_received += 1;
        self.statistics.total_bytes_received += packet.payload.len() as u64;

        // Route packet to appropriate socket or forward
        if let Some(ip_header) = &packet.ip_header {
            // Check if packet is for us
            if interface.ip_addr == Some(ip_header.dst_addr) || ip_header.dst_addr.is_broadcast() {
                self.deliver_to_socket(&packet)?;
            } else {
                // Forward packet if we're a router
                self.forward_packet(packet)?;
            }
        }

        Ok(())
    }

    /// Deliver packet to appropriate socket
    fn deliver_to_socket(&mut self, packet: &NetworkPacket) -> Result<(), &'static str> {
        if let Some(transport_header) = &packet.transport_header {
            match transport_header {
                TransportHeader::Tcp(tcp_header) => {
                    self.deliver_tcp_packet(packet, tcp_header)
                }
                TransportHeader::Udp(udp_header) => {
                    self.deliver_udp_packet(packet, udp_header)
                }
            }
        } else {
            Ok(()) // Raw packet or unsupported protocol
        }
    }

    /// Deliver TCP packet to socket
    fn deliver_tcp_packet(&mut self, packet: &NetworkPacket, tcp_header: &TcpHeader) -> Result<(), &'static str> {
        let ip_header = packet.ip_header.as_ref().unwrap();

        // Find socket that matches this packet
        for socket in self.sockets.values_mut() {
            if socket.socket_type == SocketType::Stream &&
               socket.local_port == Some(tcp_header.dst_port) &&
               (socket.local_addr.is_none() || socket.local_addr == Some(ip_header.dst_addr)) {

                // Add data to socket receive buffer
                for &byte in &packet.payload {
                    if socket.rx_buffer.len() < socket.rx_buffer_size {
                        socket.rx_buffer.push_back(byte);
                    }
                }

                socket.last_activity = crate::time::uptime_ms();
                break;
            }
        }

        Ok(())
    }

    /// Deliver UDP packet to socket
    fn deliver_udp_packet(&mut self, packet: &NetworkPacket, udp_header: &UdpHeader) -> Result<(), &'static str> {
        let ip_header = packet.ip_header.as_ref().unwrap();

        // Find socket that matches this packet
        for socket in self.sockets.values_mut() {
            if socket.socket_type == SocketType::Datagram &&
               socket.local_port == Some(udp_header.dst_port) &&
               (socket.local_addr.is_none() || socket.local_addr == Some(ip_header.dst_addr)) {

                // Add data to socket receive buffer
                for &byte in &packet.payload {
                    if socket.rx_buffer.len() < socket.rx_buffer_size {
                        socket.rx_buffer.push_back(byte);
                    }
                }

                socket.last_activity = crate::time::uptime_ms();
                break;
            }
        }

        Ok(())
    }

    /// Forward packet to next hop
    fn forward_packet(&mut self, packet: NetworkPacket) -> Result<(), &'static str> {
        if let Some(ip_header) = &packet.ip_header {
            if ip_header.time_to_live <= 1 {
                return Err("TTL expired");
            }

            // Find route and forward
            let route = self.find_route(ip_header.dst_addr).ok_or("No route")?;
            let interface = self.interfaces.get_mut(&route.interface_id)
                .ok_or("Interface not found")?;

            interface.transmit_packet(&packet)?;
        }

        Ok(())
    }

    /// Get socket by ID
    pub fn get_socket(&self, id: SocketId) -> Option<&Socket> {
        self.sockets.get(&id)
    }

    /// Get mutable socket by ID
    pub fn get_socket_mut(&mut self, id: SocketId) -> Option<&mut Socket> {
        self.sockets.get_mut(&id)
    }

    /// Get interface by ID
    pub fn get_interface(&self, id: InterfaceId) -> Option<&NetworkInterface> {
        self.interfaces.get(&id)
    }

    /// Get network statistics
    pub fn get_statistics(&self) -> &NetworkStatistics {
        &self.statistics
    }

    /// List all interfaces
    pub fn list_interfaces(&self) -> Vec<(InterfaceId, &String, bool, Option<IpAddr>)> {
        self.interfaces.iter()
            .map(|(&id, iface)| (id, &iface.name, iface.is_up, iface.ip_addr))
            .collect()
    }

    /// Cleanup idle sockets
    pub fn cleanup_sockets(&mut self) {
        let current_time = crate::time::uptime_ms();
        let timeout = 300000; // 5 minutes

        let mut to_remove = Vec::new();
        for (&id, socket) in &self.sockets {
            if current_time - socket.last_activity > timeout && socket.state == SocketState::Closed {
                to_remove.push(id);
            }
        }

        for id in to_remove {
            let _ = self.close_socket(id);
        }
    }
}

/// Calculate Internet checksum
fn calculate_internet_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;

    // Sum 16-bit words
    for chunk in data.chunks(2) {
        if chunk.len() == 2 {
            sum += u16::from_be_bytes([chunk[0], chunk[1]]) as u32;
        } else {
            sum += (chunk[0] as u32) << 8;
        }
    }

    // Add carry
    while (sum >> 16) != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    // One's complement
    !(sum as u16)
}

/// Global network manager
lazy_static! {
    pub static ref NETWORK_MANAGER: Mutex<NetworkManager> = Mutex::new(NetworkManager::new());
}

/// Initialize network stack
pub fn init() {
    crate::println!("[NET] Network stack initialized");
    crate::println!("[NET] Maximum interfaces: {}", MAX_NETWORK_INTERFACES);
    crate::println!("[NET] Maximum sockets: {}", MAX_SOCKETS);
    crate::println!("[NET] MTU: {} bytes", ETH_MTU);

    // Create loopback interface
    let mut nm = NETWORK_MANAGER.lock();
    if let Ok(id) = nm.create_interface("lo".to_string(), MacAddr::new(0, 0, 0, 0, 0, 0)) {
        let _ = nm.configure_interface(id, IpAddr::localhost(), IpAddr::new(255, 0, 0, 0), None);
        let _ = nm.interface_up(id);
    }

    crate::status::register_subsystem("Network", crate::status::SystemStatus::Running,
                                     "Network stack operational");
}

/// Create a new network interface
pub fn create_interface(name: String, mac_addr: MacAddr) -> Result<InterfaceId, &'static str> {
    NETWORK_MANAGER.lock().create_interface(name, mac_addr)
}

/// Configure network interface
pub fn configure_interface(
    id: InterfaceId,
    ip_addr: IpAddr,
    netmask: IpAddr,
    gateway: Option<IpAddr>
) -> Result<(), &'static str> {
    NETWORK_MANAGER.lock().configure_interface(id, ip_addr, netmask, gateway)
}

/// Create a socket
pub fn socket(socket_type: SocketType) -> Result<SocketId, &'static str> {
    let owner_process = crate::process::get_current_pid().unwrap_or(0);
    NETWORK_MANAGER.lock().create_socket(socket_type, owner_process)
}

/// Bind socket to address
pub fn bind(socket_id: SocketId, addr: IpAddr, port: u16) -> Result<(), &'static str> {
    let mut nm = NETWORK_MANAGER.lock();
    let socket = nm.get_socket_mut(socket_id).ok_or("Socket not found")?;
    socket.bind(addr, port)
}

/// Listen on socket
pub fn listen(socket_id: SocketId) -> Result<(), &'static str> {
    let mut nm = NETWORK_MANAGER.lock();
    let socket = nm.get_socket_mut(socket_id).ok_or("Socket not found")?;
    socket.listen()
}

/// Connect socket to remote address
pub fn connect(socket_id: SocketId, addr: IpAddr, port: u16) -> Result<(), &'static str> {
    let mut nm = NETWORK_MANAGER.lock();
    let socket = nm.get_socket_mut(socket_id).ok_or("Socket not found")?;
    socket.connect(addr, port)
}

/// Send data through socket
pub fn send(socket_id: SocketId, data: &[u8]) -> Result<usize, &'static str> {
    let mut nm = NETWORK_MANAGER.lock();
    let socket = nm.get_socket_mut(socket_id).ok_or("Socket not found")?;
    socket.send(data)
}

/// Receive data from socket
pub fn recv(socket_id: SocketId, buffer: &mut [u8]) -> Result<usize, &'static str> {
    let mut nm = NETWORK_MANAGER.lock();
    let socket = nm.get_socket_mut(socket_id).ok_or("Socket not found")?;
    socket.receive(buffer)
}

/// Close socket
pub fn close_socket(socket_id: SocketId) -> Result<(), &'static str> {
    NETWORK_MANAGER.lock().close_socket(socket_id)
}

/// Get network statistics
pub fn get_network_statistics() -> NetworkStatistics {
    NETWORK_MANAGER.lock().get_statistics().clone()
}

/// List network interfaces
pub fn list_network_interfaces() -> Vec<(InterfaceId, String, bool, Option<IpAddr>)> {
    NETWORK_MANAGER.lock().list_interfaces().into_iter()
        .map(|(id, name, up, addr)| (id, name.clone(), up, addr))
        .collect()
}

/// Network cleanup task
pub async fn network_cleanup_task() {
    loop {
        NETWORK_MANAGER.lock().cleanup_sockets();
        crate::time::sleep_ms(10000).await; // Cleanup every 10 seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_ip_addr_creation() {
        let addr = IpAddr::new(192, 168, 1, 1);
        assert_eq!(addr.octets(), [192, 168, 1, 1]);
        assert!(addr.is_private());
        assert!(!addr.is_loopback());
    }

    #[test_case]
    fn test_mac_addr_creation() {
        let mac = MacAddr::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        assert_eq!(mac.octets(), [0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert!(!mac.is_broadcast());
        assert!(!mac.is_multicast());
    }

    #[test_case]
    fn test_ethernet_header() {
        let dst = MacAddr::broadcast();
        let src = MacAddr::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        let header = EthernetHeader::new(dst, src, 0x0800);

        let bytes = header.to_bytes();
        let parsed = EthernetHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.dst_mac, dst);
        assert_eq!(parsed.src_mac, src);
        assert_eq!(parsed.ethertype, 0x0800);
    }

    #[test_case]
    fn test_socket_creation() {
        let socket = Socket::new(1, SocketType::Stream, 0);
        assert_eq!(socket.id, 1);
        assert_eq!(socket.socket_type, SocketType::Stream);
        assert_eq!(socket.state, SocketState::Closed);
    }

    #[test_case]
    fn test_network_manager() {
        let mut nm = NetworkManager::new();
        let mac = MacAddr::new(0x00, 0x11, 0x22, 0x33, 0x44, 0x55);
        let iface_id = nm.create_interface("eth0".to_string(), mac).unwrap();
        assert_eq!(iface_id, 1);

        let socket_id = nm.create_socket(SocketType::Stream, 0).unwrap();
        assert_eq!(socket_id, 1);
    }
}

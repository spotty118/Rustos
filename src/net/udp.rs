//! UDP (User Datagram Protocol) implementation
//!
//! This module provides UDP packet processing and socket operations
//! for connectionless datagram communication.

use super::{NetworkAddress, NetworkResult, NetworkError, PacketBuffer, NetworkStack};
use alloc::{vec::Vec, collections::BTreeMap};
use spin::RwLock;

/// UDP header size
pub const UDP_HEADER_SIZE: usize = 8;

/// UDP header
#[derive(Debug, Clone)]
pub struct UdpHeader {
    pub source_port: u16,
    pub dest_port: u16,
    pub length: u16,
    pub checksum: u16,
}

impl UdpHeader {
    /// Parse UDP header from packet buffer
    pub fn parse(buffer: &mut PacketBuffer) -> NetworkResult<Self> {
        if buffer.remaining() < UDP_HEADER_SIZE {
            return Err(NetworkError::InvalidPacket);
        }

        let src_port_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let source_port = u16::from_be_bytes([src_port_bytes[0], src_port_bytes[1]]);

        let dst_port_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let dest_port = u16::from_be_bytes([dst_port_bytes[0], dst_port_bytes[1]]);

        let length_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let length = u16::from_be_bytes([length_bytes[0], length_bytes[1]]);

        let checksum_bytes = buffer.read(2).ok_or(NetworkError::InvalidPacket)?;
        let checksum = u16::from_be_bytes([checksum_bytes[0], checksum_bytes[1]]);

        Ok(UdpHeader {
            source_port,
            dest_port,
            length,
            checksum,
        })
    }

    /// Calculate UDP checksum
    pub fn calculate_checksum(&self, src_ip: &NetworkAddress, dst_ip: &NetworkAddress, payload: &[u8]) -> u16 {
        let mut sum = 0u32;

        // Pseudo-header
        match (src_ip, dst_ip) {
            (NetworkAddress::IPv4(src), NetworkAddress::IPv4(dst)) => {
                sum += ((src[0] as u32) << 8) | (src[1] as u32);
                sum += ((src[2] as u32) << 8) | (src[3] as u32);
                sum += ((dst[0] as u32) << 8) | (dst[1] as u32);
                sum += ((dst[2] as u32) << 8) | (dst[3] as u32);
                sum += 17; // Protocol (UDP)
                sum += self.length as u32;
            }
            _ => return 0, // IPv6 not implemented yet
        }

        // UDP header
        sum += self.source_port as u32;
        sum += self.dest_port as u32;
        sum += self.length as u32;
        // Skip checksum field

        // Payload
        for chunk in payload.chunks(2) {
            if chunk.len() == 2 {
                sum += ((chunk[0] as u32) << 8) | (chunk[1] as u32);
            } else {
                sum += (chunk[0] as u32) << 8;
            }
        }

        // Fold 32-bit sum to 16 bits
        while (sum >> 16) != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        let result = !sum as u16;
        if result == 0 { 0xFFFF } else { result } // UDP checksum of 0 means no checksum
    }

    /// Serialize UDP header to buffer
    pub fn serialize(&self, buffer: &mut PacketBuffer) -> NetworkResult<()> {
        buffer.write(&self.source_port.to_be_bytes())?;
        buffer.write(&self.dest_port.to_be_bytes())?;
        buffer.write(&self.length.to_be_bytes())?;
        buffer.write(&self.checksum.to_be_bytes())?;
        Ok(())
    }
}

/// UDP socket
#[derive(Debug, Clone)]
pub struct UdpSocket {
    pub local_addr: NetworkAddress,
    pub local_port: u16,
    pub remote_addr: Option<NetworkAddress>,
    pub remote_port: Option<u16>,
    pub recv_buffer: Vec<UdpDatagram>,
    pub broadcast: bool,
    pub multicast_groups: Vec<NetworkAddress>,
}

/// UDP datagram
#[derive(Debug, Clone)]
pub struct UdpDatagram {
    pub source_addr: NetworkAddress,
    pub source_port: u16,
    pub data: Vec<u8>,
    pub timestamp: u64,
}

impl UdpSocket {
    pub fn new(local_addr: NetworkAddress, local_port: u16) -> Self {
        Self {
            local_addr,
            local_port,
            remote_addr: None,
            remote_port: None,
            recv_buffer: Vec::new(),
            broadcast: false,
            multicast_groups: Vec::new(),
        }
    }

    /// Connect UDP socket (sets default destination)
    pub fn connect(&mut self, remote_addr: NetworkAddress, remote_port: u16) {
        self.remote_addr = Some(remote_addr);
        self.remote_port = Some(remote_port);
    }

    /// Disconnect UDP socket
    pub fn disconnect(&mut self) {
        self.remote_addr = None;
        self.remote_port = None;
    }

    /// Add datagram to receive buffer
    pub fn add_datagram(&mut self, datagram: UdpDatagram) {
        // Limit buffer size to prevent memory exhaustion
        const MAX_BUFFER_SIZE: usize = 100;
        if self.recv_buffer.len() >= MAX_BUFFER_SIZE {
            self.recv_buffer.remove(0); // Remove oldest datagram
        }
        self.recv_buffer.push(datagram);
    }

    /// Get next datagram from receive buffer
    pub fn get_datagram(&mut self) -> Option<UdpDatagram> {
        if !self.recv_buffer.is_empty() {
            Some(self.recv_buffer.remove(0))
        } else {
            None
        }
    }

    /// Check if socket has data available
    pub fn has_data(&self) -> bool {
        !self.recv_buffer.is_empty()
    }

    /// Join multicast group
    pub fn join_multicast(&mut self, group: NetworkAddress) -> NetworkResult<()> {
        if !group.is_multicast() {
            return Err(NetworkError::InvalidAddress);
        }
        
        if !self.multicast_groups.contains(&group) {
            self.multicast_groups.push(group);
        }
        Ok(())
    }

    /// Leave multicast group
    pub fn leave_multicast(&mut self, group: NetworkAddress) -> NetworkResult<()> {
        self.multicast_groups.retain(|&addr| addr != group);
        Ok(())
    }
}

/// UDP socket manager
pub struct UdpManager {
    sockets: RwLock<BTreeMap<(NetworkAddress, u16), UdpSocket>>,
    next_port: RwLock<u16>,
}

impl UdpManager {
    pub fn new() -> Self {
        Self {
            sockets: RwLock::new(BTreeMap::new()),
            next_port: RwLock::new(32768), // Start of dynamic port range
        }
    }

    pub fn allocate_port(&self) -> u16 {
        let mut next_port = self.next_port.write();
        let port = *next_port;
        *next_port = if port >= 65535 { 32768 } else { port + 1 };
        port
    }

    pub fn bind_socket(&self, local_addr: NetworkAddress, local_port: u16) -> NetworkResult<()> {
        let mut sockets = self.sockets.write();
        let key = (local_addr, local_port);
        
        if sockets.contains_key(&key) {
            return Err(NetworkError::AddressInUse);
        }

        let socket = UdpSocket::new(local_addr, local_port);
        sockets.insert(key, socket);
        println!("UDP socket bound to {}:{}", local_addr, local_port);
        Ok(())
    }

    pub fn unbind_socket(&self, local_addr: NetworkAddress, local_port: u16) -> NetworkResult<()> {
        let mut sockets = self.sockets.write();
        let key = (local_addr, local_port);
        
        if sockets.remove(&key).is_some() {
            println!("UDP socket unbound from {}:{}", local_addr, local_port);
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    pub fn get_socket(&self, local_addr: &NetworkAddress, local_port: u16) -> Option<UdpSocket> {
        let sockets = self.sockets.read();
        let key = (*local_addr, local_port);
        sockets.get(&key).cloned()
    }

    pub fn update_socket<F>(&self, local_addr: NetworkAddress, local_port: u16, f: F) -> NetworkResult<()>
    where
        F: FnOnce(&mut UdpSocket),
    {
        let mut sockets = self.sockets.write();
        let key = (local_addr, local_port);
        
        if let Some(socket) = sockets.get_mut(&key) {
            f(socket);
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Find sockets that should receive a datagram
    pub fn find_receiving_sockets(&self, dest_addr: &NetworkAddress, dest_port: u16) -> Vec<(NetworkAddress, u16)> {
        let sockets = self.sockets.read();
        let mut receivers = Vec::new();

        for ((addr, port), socket) in sockets.iter() {
            // Check if socket should receive this datagram
            let should_receive = if *port == dest_port {
                // Exact address match
                *addr == *dest_addr ||
                // Wildcard address (0.0.0.0)
                matches!(addr, NetworkAddress::IPv4([0, 0, 0, 0])) ||
                // Broadcast
                (socket.broadcast && dest_addr.is_broadcast()) ||
                // Multicast
                (dest_addr.is_multicast() && socket.multicast_groups.contains(dest_addr))
            } else {
                false
            };

            if should_receive {
                receivers.push((*addr, *port));
            }
        }

        receivers
    }
}

static UDP_MANAGER: UdpManager = UdpManager {
    sockets: RwLock::new(BTreeMap::new()),
    next_port: RwLock::new(32768),
};

/// Process incoming UDP packet
pub fn process_packet(
    _network_stack: &NetworkStack,
    src_ip: NetworkAddress,
    dst_ip: NetworkAddress,
    mut packet: PacketBuffer,
) -> NetworkResult<()> {
    let header = UdpHeader::parse(&mut packet)?;
    
    println!("UDP packet: {}:{} -> {}:{} (len: {})",
        src_ip, header.source_port, dst_ip, header.dest_port, header.length);

    // Validate length
    if header.length < UDP_HEADER_SIZE as u16 {
        return Err(NetworkError::InvalidPacket);
    }

    let payload_length = header.length as usize - UDP_HEADER_SIZE;
    if packet.remaining() < payload_length {
        return Err(NetworkError::InvalidPacket);
    }

    // Read payload
    let payload = packet.read(payload_length).ok_or(NetworkError::InvalidPacket)?;

    // Verify checksum (if not zero)
    if header.checksum != 0 {
        let calculated_checksum = header.calculate_checksum(&src_ip, &dst_ip, payload);
        if calculated_checksum != 0 {
            println!("UDP checksum mismatch");
            return Err(NetworkError::InvalidPacket);
        }
    }

    // Find receiving sockets
    let receivers = UDP_MANAGER.find_receiving_sockets(&dst_ip, header.dest_port);
    
    if receivers.is_empty() {
        println!("No UDP socket listening on port {}", header.dest_port);
        // TODO: Send ICMP Port Unreachable
        return Ok(());
    }

    // Deliver to all matching sockets
    for (local_addr, local_port) in receivers {
        let datagram = UdpDatagram {
            source_addr: src_ip,
            source_port: header.source_port,
            data: payload.to_vec(),
            timestamp: get_current_time(),
        };

        UDP_MANAGER.update_socket(local_addr, local_port, |socket| {
            socket.add_datagram(datagram);
        }).ok(); // Ignore errors for delivery
    }

    Ok(())
}

/// Send UDP packet
pub fn send_udp_packet(
    src_ip: NetworkAddress,
    src_port: u16,
    dst_ip: NetworkAddress,
    dst_port: u16,
    payload: &[u8],
) -> NetworkResult<()> {
    let length = UDP_HEADER_SIZE + payload.len();
    if length > u16::MAX as usize {
        return Err(NetworkError::BufferOverflow);
    }

    // Create UDP header
    let mut header = UdpHeader {
        source_port: src_port,
        dest_port: dst_port,
        length: length as u16,
        checksum: 0,
    };

    // Calculate checksum
    header.checksum = header.calculate_checksum(&src_ip, &dst_ip, payload);

    println!("Sending UDP packet: {}:{} -> {}:{} ({} bytes)",
        src_ip, src_port, dst_ip, dst_port, payload.len());

    // TODO: Send through IP layer
    // For now, just log the operation
    
    Ok(())
}

/// UDP socket operations

/// Create UDP socket
pub fn udp_socket() -> NetworkResult<(NetworkAddress, u16)> {
    let local_addr = NetworkAddress::IPv4([0, 0, 0, 0]); // Wildcard address
    let local_port = UDP_MANAGER.allocate_port();
    
    UDP_MANAGER.bind_socket(local_addr, local_port)?;
    Ok((local_addr, local_port))
}

/// Bind UDP socket to specific address
pub fn udp_bind(local_addr: NetworkAddress, local_port: u16) -> NetworkResult<()> {
    UDP_MANAGER.bind_socket(local_addr, local_port)
}

/// Connect UDP socket
pub fn udp_connect(
    local_addr: NetworkAddress,
    local_port: u16,
    remote_addr: NetworkAddress,
    remote_port: u16,
) -> NetworkResult<()> {
    UDP_MANAGER.update_socket(local_addr, local_port, |socket| {
        socket.connect(remote_addr, remote_port);
    })
}

/// Send data through UDP socket
pub fn udp_send(
    local_addr: NetworkAddress,
    local_port: u16,
    data: &[u8],
) -> NetworkResult<usize> {
    let socket = UDP_MANAGER.get_socket(&local_addr, local_port)
        .ok_or(NetworkError::InvalidAddress)?;

    if let (Some(remote_addr), Some(remote_port)) = (socket.remote_addr, socket.remote_port) {
        send_udp_packet(local_addr, local_port, remote_addr, remote_port, data)?;
        Ok(data.len())
    } else {
        Err(NetworkError::NotSupported)
    }
}

/// Send data to specific address through UDP socket
pub fn udp_send_to(
    local_addr: NetworkAddress,
    local_port: u16,
    remote_addr: NetworkAddress,
    remote_port: u16,
    data: &[u8],
) -> NetworkResult<usize> {
    // Verify socket exists
    UDP_MANAGER.get_socket(&local_addr, local_port)
        .ok_or(NetworkError::InvalidAddress)?;

    send_udp_packet(local_addr, local_port, remote_addr, remote_port, data)?;
    Ok(data.len())
}

/// Receive data from UDP socket
pub fn udp_recv(local_addr: NetworkAddress, local_port: u16) -> NetworkResult<Option<(Vec<u8>, NetworkAddress, u16)>> {
    let mut result = None;
    
    UDP_MANAGER.update_socket(local_addr, local_port, |socket| {
        if let Some(datagram) = socket.get_datagram() {
            result = Some((datagram.data, datagram.source_addr, datagram.source_port));
        }
    })?;

    Ok(result)
}

/// Check if UDP socket has data available
pub fn udp_has_data(local_addr: NetworkAddress, local_port: u16) -> bool {
    UDP_MANAGER.get_socket(&local_addr, local_port)
        .map(|socket| socket.has_data())
        .unwrap_or(false)
}

/// Close UDP socket
pub fn udp_close(local_addr: NetworkAddress, local_port: u16) -> NetworkResult<()> {
    UDP_MANAGER.unbind_socket(local_addr, local_port)
}

/// Set socket broadcast option
pub fn udp_set_broadcast(local_addr: NetworkAddress, local_port: u16, broadcast: bool) -> NetworkResult<()> {
    UDP_MANAGER.update_socket(local_addr, local_port, |socket| {
        socket.broadcast = broadcast;
    })
}

/// Join multicast group
pub fn udp_join_multicast(
    local_addr: NetworkAddress,
    local_port: u16,
    group: NetworkAddress,
) -> NetworkResult<()> {
    UDP_MANAGER.update_socket(local_addr, local_port, |socket| {
        socket.join_multicast(group).ok();
    })
}

/// Leave multicast group
pub fn udp_leave_multicast(
    local_addr: NetworkAddress,
    local_port: u16,
    group: NetworkAddress,
) -> NetworkResult<()> {
    UDP_MANAGER.update_socket(local_addr, local_port, |socket| {
        socket.leave_multicast(group).ok();
    })
}

/// Get current time (placeholder)
fn get_current_time() -> u64 {
    // TODO: Get actual system time
    1000000 // Placeholder timestamp
}

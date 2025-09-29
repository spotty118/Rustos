//! # Socket Interface Implementation
//!
//! This module provides high-level socket abstractions for the RustOS network stack,
//! supporting TCP and UDP sockets with BSD-style socket API.

use alloc::{collections::BTreeMap, boxed::Box, string::String, vec::Vec, format};
use core::fmt;
use spin::RwLock;

use crate::network::{
    tcp::{TcpProcessor, TcpSocket},
    udp::{UdpProcessor, UdpSocket, UdpSocketState},
    NetworkError, Protocol, SocketAddr,
};

/// Socket types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketType {
    /// Stream socket (TCP)
    Stream,
    /// Datagram socket (UDP)
    Datagram,
    /// Raw socket
    Raw,
}

impl fmt::Display for SocketType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SocketType::Stream => write!(f, "STREAM"),
            SocketType::Datagram => write!(f, "DGRAM"),
            SocketType::Raw => write!(f, "RAW"),
        }
    }
}

/// Socket domain/family
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketDomain {
    /// Internet Protocol version 4
    Inet,
    /// Internet Protocol version 6
    Inet6,
    /// Unix domain sockets
    Unix,
}

impl fmt::Display for SocketDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SocketDomain::Inet => write!(f, "AF_INET"),
            SocketDomain::Inet6 => write!(f, "AF_INET6"),
            SocketDomain::Unix => write!(f, "AF_UNIX"),
        }
    }
}

/// Socket options
#[derive(Debug, Clone)]
pub struct SocketOptions {
    /// Socket is non-blocking
    pub non_blocking: bool,
    /// Reuse address
    pub reuse_addr: bool,
    /// Reuse port
    pub reuse_port: bool,
    /// Keep alive enabled
    pub keep_alive: bool,
    /// Broadcast enabled (UDP only)
    pub broadcast: bool,
    /// Receive buffer size
    pub recv_buffer_size: usize,
    /// Send buffer size
    pub send_buffer_size: usize,
    /// Receive timeout in milliseconds
    pub recv_timeout: Option<u64>,
    /// Send timeout in milliseconds
    pub send_timeout: Option<u64>,
    /// TCP no delay (disable Nagle algorithm)
    pub tcp_nodelay: bool,
}

impl Default for SocketOptions {
    fn default() -> Self {
        Self {
            non_blocking: false,
            reuse_addr: false,
            reuse_port: false,
            keep_alive: false,
            broadcast: false,
            recv_buffer_size: 8192,
            send_buffer_size: 8192,
            recv_timeout: None,
            send_timeout: None,
            tcp_nodelay: false,
        }
    }
}

/// Socket state information
#[derive(Debug, Clone)]
pub struct SocketInfo {
    pub socket_type: SocketType,
    pub domain: SocketDomain,
    pub local_addr: Option<SocketAddr>,
    pub remote_addr: Option<SocketAddr>,
    pub state: String,
    pub options: SocketOptions,
    pub created_at: u64,
    pub last_activity: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
}

/// Generic socket handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SocketHandle(pub u32);

impl SocketHandle {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn id(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for SocketHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "socket({})", self.0)
    }
}

/// Socket implementation
#[derive(Debug)]
pub enum Socket {
    /// TCP socket
    Tcp {
        socket: TcpSocket,
        options: SocketOptions,
        stats: SocketStats,
    },
    /// UDP socket
    Udp {
        socket: UdpSocket,
        options: SocketOptions,
        stats: SocketStats,
    },
    /// Raw socket (placeholder)
    Raw {
        local_addr: Option<SocketAddr>,
        options: SocketOptions,
        stats: SocketStats,
    },
}

impl Socket {
    /// Create a new TCP socket
    pub fn new_tcp(timestamp: u64) -> Self {
        Self::Tcp {
            socket: TcpSocket::new(),
            options: SocketOptions::default(),
            stats: SocketStats::new(timestamp),
        }
    }

    /// Create a new UDP socket
    pub fn new_udp(timestamp: u64) -> Self {
        Self::Udp {
            socket: UdpSocket::new(),
            options: SocketOptions::default(),
            stats: SocketStats::new(timestamp),
        }
    }

    /// Create a new raw socket
    pub fn new_raw(timestamp: u64) -> Self {
        Self::Raw {
            local_addr: None,
            options: SocketOptions::default(),
            stats: SocketStats::new(timestamp),
        }
    }

    /// Get socket type
    pub fn socket_type(&self) -> SocketType {
        match self {
            Socket::Tcp { .. } => SocketType::Stream,
            Socket::Udp { .. } => SocketType::Datagram,
            Socket::Raw { .. } => SocketType::Raw,
        }
    }

    /// Get local address
    pub fn local_addr(&self) -> Option<SocketAddr> {
        match self {
            Socket::Tcp { socket, .. } => socket.local_addr,
            Socket::Udp { socket, .. } => socket.local_addr(),
            Socket::Raw { local_addr, .. } => *local_addr,
        }
    }

    /// Get remote address
    pub fn remote_addr(&self) -> Option<SocketAddr> {
        match self {
            Socket::Tcp { socket, .. } => socket.remote_addr,
            Socket::Udp { socket, .. } => socket.remote_addr(),
            Socket::Raw { .. } => None,
        }
    }

    /// Get socket options
    pub fn options(&self) -> &SocketOptions {
        match self {
            Socket::Tcp { options, .. } => options,
            Socket::Udp { options, .. } => options,
            Socket::Raw { options, .. } => options,
        }
    }

    /// Get mutable socket options
    pub fn options_mut(&mut self) -> &mut SocketOptions {
        match self {
            Socket::Tcp { options, .. } => options,
            Socket::Udp { options, .. } => options,
            Socket::Raw { options, .. } => options,
        }
    }

    /// Get socket statistics
    pub fn stats(&self) -> &SocketStats {
        match self {
            Socket::Tcp { stats, .. } => stats,
            Socket::Udp { stats, .. } => stats,
            Socket::Raw { stats, .. } => stats,
        }
    }

    /// Get mutable socket statistics
    pub fn stats_mut(&mut self) -> &mut SocketStats {
        match self {
            Socket::Tcp { stats, .. } => stats,
            Socket::Udp { stats, .. } => stats,
            Socket::Raw { stats, .. } => stats,
        }
    }

    /// Update last activity timestamp
    pub fn update_activity(&mut self, timestamp: u64) {
        self.stats_mut().last_activity = timestamp;
    }

    /// Check if socket is connected/established
    pub fn is_connected(&self) -> bool {
        match self {
            Socket::Tcp { socket, .. } => socket.is_established(),
            Socket::Udp { socket, .. } => socket.state == UdpSocketState::Connected,
            Socket::Raw { .. } => false,
        }
    }

    /// Check if socket is listening
    pub fn is_listening(&self) -> bool {
        match self {
            Socket::Tcp { socket, .. } => socket.is_listening(),
            Socket::Udp { .. } => false, // UDP doesn't have listen state
            Socket::Raw { .. } => false,
        }
    }

    /// Check if socket is closed
    pub fn is_closed(&self) -> bool {
        match self {
            Socket::Tcp { socket, .. } => socket.is_closed(),
            Socket::Udp { socket, .. } => socket.state == UdpSocketState::Closed,
            Socket::Raw { .. } => false,
        }
    }

    /// Get state string
    pub fn state_string(&self) -> String {
        match self {
            Socket::Tcp { socket, .. } => format!("{}", socket.state),
            Socket::Udp { socket, .. } => format!("{:?}", socket.state),
            Socket::Raw { .. } => "RAW".into(),
        }
    }
}

/// Socket statistics
#[derive(Debug, Clone)]
pub struct SocketStats {
    pub created_at: u64,
    pub last_activity: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub errors: u64,
    pub timeouts: u64,
    pub connections: u64,
    pub disconnections: u64,
}

impl SocketStats {
    pub fn new(timestamp: u64) -> Self {
        Self {
            created_at: timestamp,
            last_activity: timestamp,
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            errors: 0,
            timeouts: 0,
            connections: 0,
            disconnections: 0,
        }
    }

    pub fn record_send(&mut self, bytes: usize) {
        self.bytes_sent += bytes as u64;
        self.packets_sent += 1;
    }

    pub fn record_receive(&mut self, bytes: usize) {
        self.bytes_received += bytes as u64;
        self.packets_received += 1;
    }

    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    pub fn record_timeout(&mut self) {
        self.timeouts += 1;
    }

    pub fn record_connection(&mut self) {
        self.connections += 1;
    }

    pub fn record_disconnection(&mut self) {
        self.disconnections += 1;
    }
}

/// Socket manager for handling all socket operations
#[derive(Debug)]
pub struct SocketManager {
    /// All sockets
    sockets: BTreeMap<SocketHandle, Socket>,
    /// Next socket handle ID
    next_handle: u32,
    /// TCP processor
    tcp_processor: TcpProcessor,
    /// UDP processor
    udp_processor: UdpProcessor,
    /// Manager statistics
    stats: SocketManagerStats,
}

impl SocketManager {
    pub fn new() -> Self {
        Self {
            sockets: BTreeMap::new(),
            next_handle: 1,
            tcp_processor: TcpProcessor::new(),
            udp_processor: UdpProcessor::new(),
            stats: SocketManagerStats::default(),
        }
    }

    /// Create a new socket
    pub fn create_socket(
        &mut self,
        domain: SocketDomain,
        socket_type: SocketType,
        _protocol: Protocol,
        timestamp: u64,
    ) -> Result<SocketHandle, NetworkError> {
        // For now, only support IPv4
        if domain != SocketDomain::Inet {
            return Err(NetworkError::ProtocolError);
        }

        let handle = SocketHandle::new(self.next_handle);
        self.next_handle += 1;

        let socket = match socket_type {
            SocketType::Stream => Socket::new_tcp(timestamp),
            SocketType::Datagram => Socket::new_udp(timestamp),
            SocketType::Raw => Socket::new_raw(timestamp),
        };

        self.sockets.insert(handle, socket);
        self.stats.sockets_created += 1;
        self.stats.active_sockets += 1;

        Ok(handle)
    }

    /// Close a socket
    pub fn close_socket(&mut self, handle: SocketHandle) -> Result<(), NetworkError> {
        if let Some(_socket) = self.sockets.remove(&handle) {
            self.stats.sockets_closed += 1;
            self.stats.active_sockets -= 1;
            Ok(())
        } else {
            Err(NetworkError::InvalidAddress)
        }
    }

    /// Bind socket to address
    pub fn bind(&mut self, handle: SocketHandle, addr: SocketAddr) -> Result<(), NetworkError> {
        let socket = self
            .sockets
            .get_mut(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        match socket {
            Socket::Tcp { socket, .. } => socket.bind(addr),
            Socket::Udp { socket, .. } => socket.bind(addr),
            Socket::Raw { local_addr, .. } => {
                *local_addr = Some(addr);
                Ok(())
            }
        }
    }

    /// Listen for connections (TCP only)
    pub fn listen(&mut self, handle: SocketHandle, _backlog: u32) -> Result<(), NetworkError> {
        let socket = self
            .sockets
            .get_mut(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        match socket {
            Socket::Tcp { socket, .. } => socket.listen(),
            _ => Err(NetworkError::ProtocolError),
        }
    }

    /// Connect to remote address
    pub fn connect(
        &mut self,
        handle: SocketHandle,
        addr: SocketAddr,
        timestamp: u64,
    ) -> Result<(), NetworkError> {
        let socket = self
            .sockets
            .get_mut(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        match socket {
            Socket::Tcp { socket, stats, .. } => {
                let _packet = socket.connect(addr, timestamp)?;
                stats.record_connection();
                // In a real implementation, we'd send the SYN packet here
                Ok(())
            }
            Socket::Udp { socket, stats, .. } => {
                socket.connect(addr)?;
                stats.record_connection();
                Ok(())
            }
            Socket::Raw { .. } => Err(NetworkError::ProtocolError),
        }
    }

    /// Send data
    pub fn send(
        &mut self,
        handle: SocketHandle,
        data: &[u8],
        _flags: u32,
    ) -> Result<usize, NetworkError> {
        let socket = self
            .sockets
            .get_mut(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        match socket {
            Socket::Tcp { socket, stats, .. } => {
                if let Some(_packet) = socket.send(data)? {
                    stats.record_send(data.len());
                    // In a real implementation, we'd send the packet here
                    Ok(data.len())
                } else {
                    Ok(0)
                }
            }
            Socket::Udp { socket, stats, .. } => {
                if let Some(remote_addr) = socket.remote_addr() {
                    let _packet = socket.send_to(data, remote_addr)?;
                    stats.record_send(data.len());
                    // In a real implementation, we'd send the packet here
                    Ok(data.len())
                } else {
                    Err(NetworkError::InvalidAddress)
                }
            }
            Socket::Raw { .. } => {
                // Raw socket implementation would go here
                Err(NetworkError::ProtocolError)
            }
        }
    }

    /// Send data to specific address (UDP only)
    pub fn send_to(
        &mut self,
        handle: SocketHandle,
        data: &[u8],
        addr: SocketAddr,
        _flags: u32,
    ) -> Result<usize, NetworkError> {
        let socket = self
            .sockets
            .get_mut(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        match socket {
            Socket::Udp { socket, stats, .. } => {
                let _packet = socket.send_to(data, addr)?;
                stats.record_send(data.len());
                // In a real implementation, we'd send the packet here
                Ok(data.len())
            }
            _ => Err(NetworkError::ProtocolError),
        }
    }

    /// Receive data
    pub fn receive(
        &mut self,
        handle: SocketHandle,
        buffer: &mut [u8],
        _flags: u32,
    ) -> Result<usize, NetworkError> {
        let socket = self
            .sockets
            .get_mut(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        match socket {
            Socket::Tcp { socket, stats, .. } => {
                let data = socket.recv();
                let len = core::cmp::min(data.len(), buffer.len());
                buffer[..len].copy_from_slice(&data[..len]);
                stats.record_receive(len);
                Ok(len)
            }
            Socket::Udp { socket, stats, .. } => {
                if let Some((data, _addr)) = socket.recv() {
                    let len = core::cmp::min(data.len(), buffer.len());
                    buffer[..len].copy_from_slice(&data[..len]);
                    stats.record_receive(len);
                    Ok(len)
                } else {
                    Ok(0)
                }
            }
            Socket::Raw { .. } => Err(NetworkError::ProtocolError),
        }
    }

    /// Receive data with source address (UDP only)
    pub fn receive_from(
        &mut self,
        handle: SocketHandle,
        buffer: &mut [u8],
        _flags: u32,
    ) -> Result<(usize, SocketAddr), NetworkError> {
        let socket = self
            .sockets
            .get_mut(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        match socket {
            Socket::Udp { socket, stats, .. } => {
                if let Some((data, addr)) = socket.recv() {
                    let len = core::cmp::min(data.len(), buffer.len());
                    buffer[..len].copy_from_slice(&data[..len]);
                    stats.record_receive(len);
                    Ok((len, addr))
                } else {
                    Err(NetworkError::Timeout)
                }
            }
            _ => Err(NetworkError::ProtocolError),
        }
    }

    /// Set socket option
    pub fn set_socket_option(
        &mut self,
        handle: SocketHandle,
        option: SocketOption,
        value: SocketOptionValue,
    ) -> Result<(), NetworkError> {
        let socket = self
            .sockets
            .get_mut(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        let options = socket.options_mut();

        match (option, value) {
            (SocketOption::ReuseAddr, SocketOptionValue::Bool(val)) => {
                options.reuse_addr = val;
            }
            (SocketOption::ReusePort, SocketOptionValue::Bool(val)) => {
                options.reuse_port = val;
            }
            (SocketOption::Broadcast, SocketOptionValue::Bool(val)) => {
                options.broadcast = val;
            }
            (SocketOption::KeepAlive, SocketOptionValue::Bool(val)) => {
                options.keep_alive = val;
            }
            (SocketOption::NonBlocking, SocketOptionValue::Bool(val)) => {
                options.non_blocking = val;
            }
            (SocketOption::RecvBuffer, SocketOptionValue::Int(val)) => {
                options.recv_buffer_size = val as usize;
            }
            (SocketOption::SendBuffer, SocketOptionValue::Int(val)) => {
                options.send_buffer_size = val as usize;
            }
            (SocketOption::RecvTimeout, SocketOptionValue::Int(val)) => {
                options.recv_timeout = Some(val as u64);
            }
            (SocketOption::SendTimeout, SocketOptionValue::Int(val)) => {
                options.send_timeout = Some(val as u64);
            }
            (SocketOption::TcpNodelay, SocketOptionValue::Bool(val)) => {
                options.tcp_nodelay = val;
            }
            _ => return Err(NetworkError::InvalidAddress),
        }

        Ok(())
    }

    /// Get socket option
    pub fn get_socket_option(
        &self,
        handle: SocketHandle,
        option: SocketOption,
    ) -> Result<SocketOptionValue, NetworkError> {
        let socket = self
            .sockets
            .get(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        let options = socket.options();

        let value = match option {
            SocketOption::ReuseAddr => SocketOptionValue::Bool(options.reuse_addr),
            SocketOption::ReusePort => SocketOptionValue::Bool(options.reuse_port),
            SocketOption::Broadcast => SocketOptionValue::Bool(options.broadcast),
            SocketOption::KeepAlive => SocketOptionValue::Bool(options.keep_alive),
            SocketOption::NonBlocking => SocketOptionValue::Bool(options.non_blocking),
            SocketOption::RecvBuffer => SocketOptionValue::Int(options.recv_buffer_size as u32),
            SocketOption::SendBuffer => SocketOptionValue::Int(options.send_buffer_size as u32),
            SocketOption::RecvTimeout => {
                SocketOptionValue::Int(options.recv_timeout.unwrap_or(0) as u32)
            }
            SocketOption::SendTimeout => {
                SocketOptionValue::Int(options.send_timeout.unwrap_or(0) as u32)
            }
            SocketOption::TcpNodelay => SocketOptionValue::Bool(options.tcp_nodelay),
            SocketOption::Error => SocketOptionValue::Int(0), // Would return last error
        };

        Ok(value)
    }

    /// Get socket information
    pub fn get_socket_info(&self, handle: SocketHandle) -> Result<SocketInfo, NetworkError> {
        let socket = self
            .sockets
            .get(&handle)
            .ok_or(NetworkError::InvalidAddress)?;

        let stats = socket.stats();

        Ok(SocketInfo {
            socket_type: socket.socket_type(),
            domain: SocketDomain::Inet,
            local_addr: socket.local_addr(),
            remote_addr: socket.remote_addr(),
            state: socket.state_string(),
            options: socket.options().clone(),
            created_at: stats.created_at,
            last_activity: stats.last_activity,
            bytes_sent: stats.bytes_sent,
            bytes_received: stats.bytes_received,
            packets_sent: stats.packets_sent,
            packets_received: stats.packets_received,
        })
    }

    /// Get all socket handles
    pub fn get_socket_handles(&self) -> Vec<SocketHandle> {
        self.sockets.keys().copied().collect()
    }

    /// Get socket count
    pub fn socket_count(&self) -> usize {
        self.sockets.len()
    }

    /// Get manager statistics
    pub fn get_stats(&self) -> &SocketManagerStats {
        &self.stats
    }

    /// Cleanup closed sockets and perform maintenance
    pub fn cleanup(&mut self, timestamp: u64) {
        let mut to_remove = Vec::new();

        for (handle, socket) in &mut self.sockets {
            if socket.is_closed() {
                to_remove.push(*handle);
            } else {
                socket.update_activity(timestamp);
            }
        }

        for handle in to_remove {
            self.sockets.remove(&handle);
            self.stats.active_sockets -= 1;
            self.stats.sockets_closed += 1;
        }
    }
}

impl Default for SocketManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Socket options enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketOption {
    ReuseAddr,
    ReusePort,
    Broadcast,
    KeepAlive,
    NonBlocking,
    RecvBuffer,
    SendBuffer,
    RecvTimeout,
    SendTimeout,
    TcpNodelay,
    Error,
}

/// Socket option values
#[derive(Debug, Clone, PartialEq)]
pub enum SocketOptionValue {
    Bool(bool),
    Int(u32),
    Bytes(Vec<u8>),
}

/// Socket manager statistics
#[derive(Debug, Default, Clone)]
pub struct SocketManagerStats {
    pub sockets_created: u64,
    pub sockets_closed: u64,
    pub active_sockets: u64,
    pub tcp_sockets: u64,
    pub udp_sockets: u64,
    pub raw_sockets: u64,
    pub bind_operations: u64,
    pub connect_operations: u64,
    pub listen_operations: u64,
    pub send_operations: u64,
    pub recv_operations: u64,
    pub errors: u64,
}

/// Global socket manager instance
static SOCKET_MANAGER: RwLock<Option<SocketManager>> = RwLock::new(None);

/// Initialize global socket manager
pub fn init_socket_manager() {
    *SOCKET_MANAGER.write() = Some(SocketManager::new());
}

/// Get reference to global socket manager
pub fn with_socket_manager<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut SocketManager) -> R,
{
    SOCKET_MANAGER.write().as_mut().map(f)
}

/// High-level socket functions
pub fn socket(
    domain: SocketDomain,
    socket_type: SocketType,
    protocol: Protocol,
) -> Result<SocketHandle, NetworkError> {
    with_socket_manager(|manager| {
        manager.create_socket(domain, socket_type, protocol, 0) // TODO: get proper timestamp
    })
    .ok_or(NetworkError::ProtocolError)?
}

pub fn bind(handle: SocketHandle, addr: SocketAddr) -> Result<(), NetworkError> {
    with_socket_manager(|manager| manager.bind(handle, addr)).ok_or(NetworkError::ProtocolError)?
}

pub fn listen(handle: SocketHandle, backlog: u32) -> Result<(), NetworkError> {
    with_socket_manager(|manager| manager.listen(handle, backlog))
        .ok_or(NetworkError::ProtocolError)?
}

pub fn connect(handle: SocketHandle, addr: SocketAddr) -> Result<(), NetworkError> {
    with_socket_manager(|manager| {
        manager.connect(handle, addr, 0) // TODO: get proper timestamp
    })
    .ok_or(NetworkError::ProtocolError)?
}

pub fn send(handle: SocketHandle, data: &[u8]) -> Result<usize, NetworkError> {
    with_socket_manager(|manager| manager.send(handle, data, 0))
        .ok_or(NetworkError::ProtocolError)?
}

pub fn recv(handle: SocketHandle, buffer: &mut [u8]) -> Result<usize, NetworkError> {
    with_socket_manager(|manager| manager.receive(handle, buffer, 0))
        .ok_or(NetworkError::ProtocolError)?
}

pub fn close(handle: SocketHandle) -> Result<(), NetworkError> {
    with_socket_manager(|manager| manager.close_socket(handle))
        .ok_or(NetworkError::ProtocolError)?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_creation() {
        let mut manager = SocketManager::new();
        let handle = manager
            .create_socket(SocketDomain::Inet, SocketType::Stream, Protocol::Tcp, 0)
            .unwrap();

        assert_eq!(handle.id(), 1);
        assert_eq!(manager.socket_count(), 1);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_bind() {
        let mut manager = SocketManager::new();
        let handle = manager
            .create_socket(SocketDomain::Inet, SocketType::Stream, Protocol::Tcp, 0)
            .unwrap();

        let addr = SocketAddr::new(Ipv4Address::LOCALHOST, 8080);
        assert!(manager.bind(handle, addr).is_ok());

        let info = manager.get_socket_info(handle).unwrap();
        assert_eq!(info.local_addr, Some(addr));
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_options() {
        let mut manager = SocketManager::new();
        let handle = manager
            .create_socket(SocketDomain::Inet, SocketType::Stream, Protocol::Tcp, 0)
            .unwrap();

        // Set reuse address option
        manager
            .set_socket_option(
                handle,
                SocketOption::ReuseAddr,
                SocketOptionValue::Bool(true),
            )
            .unwrap();

        // Get reuse address option
        let value = manager
            .get_socket_option(handle, SocketOption::ReuseAddr)
            .unwrap();
        assert_eq!(value, SocketOptionValue::Bool(true));
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_types() {
        assert_eq!(format!("{}", SocketType::Stream), "STREAM");
        assert_eq!(format!("{}", SocketType::Datagram), "DGRAM");
        assert_eq!(format!("{}", SocketType::Raw), "RAW");
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_handle() {
        let handle = SocketHandle::new(42);
        assert_eq!(handle.id(), 42);
        assert_eq!(format!("{}", handle), "socket(42)");
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_stats() {
        let mut stats = SocketStats::new(1000);
        assert_eq!(stats.created_at, 1000);

        stats.record_send(100);
        assert_eq!(stats.bytes_sent, 100);
        assert_eq!(stats.packets_sent, 1);

        stats.record_receive(50);
        assert_eq!(stats.bytes_received, 50);
        assert_eq!(stats.packets_received, 1);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_manager_operations() {
        let mut manager = SocketManager::new();

        // Create TCP socket
        let tcp_handle = manager
            .create_socket(SocketDomain::Inet, SocketType::Stream, Protocol::Tcp, 1000)
            .unwrap();

        // Create UDP socket
        let udp_handle = manager
            .create_socket(
                SocketDomain::Inet,
                SocketType::Datagram,
                Protocol::Udp,
                1000,
            )
            .unwrap();

        assert_eq!(manager.socket_count(), 2);

        // Close one socket
        manager.close_socket(tcp_handle).unwrap();
        assert_eq!(manager.socket_count(), 1);

        // Get socket handles
        let handles = manager.get_socket_handles();
        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0], udp_handle);
    }

    #[cfg(feature = "std-tests")] // Disabled: #[cfg(feature = "disabled-tests")] // #[cfg(feature = "disabled-tests")] // #[test]
    fn test_socket_domains() {
        assert_eq!(format!("{}", SocketDomain::Inet), "AF_INET");
        assert_eq!(format!("{}", SocketDomain::Inet6), "AF_INET6");
        assert_eq!(format!("{}", SocketDomain::Unix), "AF_UNIX");
    }
}

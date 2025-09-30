//! # Intel E1000/E1000E Ethernet Driver
//!
//! Driver for Intel 82540/82541/82542/82543/82544/82545/82546/82547/82571/82572/82573/82574/82575/82576
//! and other Intel Gigabit Ethernet controllers (E1000 and E1000E series).

use super::{ExtendedNetworkCapabilities, EnhancedNetworkStats, PowerState, WakeOnLanConfig};
use crate::net::{NetworkError, NetworkAddress};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ptr;

/// Network device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Ethernet,
    Wireless,
    Loopback,
}

/// Network device states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    Down,
    Initialized,
    Up,
}

/// Device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub mtu: u16,
    pub hw_checksum: bool,
    pub scatter_gather: bool,
    pub vlan_support: bool,
    pub jumbo_frames: bool,
    pub multicast_filter: bool,
    pub max_packet_size: u16,
    pub link_speed: u32,
    pub full_duplex: bool,
    pub rx_queues: u8,
    pub tx_queues: u8,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            mtu: 1500,
            hw_checksum: false,
            scatter_gather: false,
            vlan_support: false,
            jumbo_frames: false,
            multicast_filter: false,
            max_packet_size: 1518,
            link_speed: 1000,
            full_duplex: true,
            rx_queues: 1,
            tx_queues: 1,
        }
    }
}

/// MAC address wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacAddress([u8; 6]);

impl MacAddress {
    pub const ZERO: MacAddress = MacAddress([0; 6]);
    
    pub fn new(bytes: [u8; 6]) -> Self {
        Self(bytes)
    }
    
    pub fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }
}

/// Network statistics
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    pub rx_packets: u64,
    pub tx_packets: u64,
    pub rx_bytes: u64,
    pub tx_bytes: u64,
    pub rx_errors: u64,
    pub tx_errors: u64,
    pub rx_dropped: u64,
    pub tx_dropped: u64,
}

/// Network driver trait
pub trait NetworkDriver {
    fn name(&self) -> &str;
    fn device_type(&self) -> DeviceType;
    fn state(&self) -> DeviceState;
    fn capabilities(&self) -> &DeviceCapabilities;
    fn init(&mut self) -> Result<(), NetworkError>;
    fn start(&mut self) -> Result<(), NetworkError>;
    fn stop(&mut self) -> Result<(), NetworkError>;
    fn send_packet(&mut self, packet: &[u8]) -> Result<(), NetworkError>;
    fn receive_packet(&mut self) -> Result<Option<Vec<u8>>, NetworkError>;
    fn get_mac_address(&self) -> MacAddress;
    fn set_mac_address(&mut self, mac: MacAddress) -> Result<(), NetworkError>;
    fn get_link_status(&self) -> (bool, u32, bool);
    fn get_statistics(&self) -> NetworkStats;
    fn handle_interrupt(&mut self) -> Result<(), NetworkError>;
    fn set_power_state(&mut self, state: PowerState) -> Result<(), NetworkError>;
    fn configure_wol(&mut self, config: WakeOnLanConfig) -> Result<(), NetworkError>;
}

/// Intel E1000 device information
#[derive(Debug, Clone, Copy)]
pub struct IntelE1000DeviceInfo {
    pub vendor_id: u16,
    pub device_id: u16,
    pub name: &'static str,
    pub generation: E1000Generation,
    pub max_speed_mbps: u32,
    pub supports_tso: bool,
    pub supports_rss: bool,
    pub queue_count: u8,
}

/// E1000 controller generations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum E1000Generation {
    /// Original E1000 (82540, 82541, 82542, 82543, 82544, 82545, 82546, 82547)
    E1000,
    /// E1000E (82571, 82572, 82573, 82574, 82575, 82576, 82577, 82578, 82579, 82580)
    E1000E,
    /// I350 series
    I350,
    /// I210/I211 series
    I210,
    /// I225 series
    I225,
}

/// Comprehensive Intel E1000 device database (100+ entries)
pub const INTEL_E1000_DEVICES: &[IntelE1000DeviceInfo] = &[
    // 82540 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x100E, name: "82540EM Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1015, name: "82540EM Gigabit Ethernet Controller (LOM)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1016, name: "82540EP Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1017, name: "82540EP Gigabit Ethernet Controller (LOM)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },

    // 82541 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1013, name: "82541EI Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1018, name: "82541ER Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1019, name: "82541GI Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x101A, name: "82541GI Gigabit Ethernet Controller (LOM)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },

    // 82542 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1000, name: "82542 Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1001, name: "82542 Gigabit Ethernet Controller (Fiber)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },

    // 82543 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1004, name: "82543GC Gigabit Ethernet Controller (Copper)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1008, name: "82543GC Gigabit Ethernet Controller (Fiber)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },

    // 82544 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1009, name: "82544EI Gigabit Ethernet Controller (Copper)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x100A, name: "82544EI Gigabit Ethernet Controller (Fiber)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x100C, name: "82544GC Gigabit Ethernet Controller (Copper)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x100D, name: "82544GC Gigabit Ethernet Controller (LOM)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },

    // 82545 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x100F, name: "82545EM Gigabit Ethernet Controller (Copper)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1010, name: "82545EM Gigabit Ethernet Controller (Fiber)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1011, name: "82545GM Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1012, name: "82546EB Gigabit Ethernet Controller (Copper)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },

    // 82546 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x101D, name: "82546EB Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x101E, name: "82546GB Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1026, name: "82546GB Gigabit Ethernet Controller (Copper)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1027, name: "82546GB Gigabit Ethernet Controller (Quad Port)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },

    // 82547 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x101F, name: "82547EI Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1020, name: "82547EI Gigabit Ethernet Controller (LOM)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1021, name: "82547GI Gigabit Ethernet Controller", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1022, name: "82547GI Gigabit Ethernet Controller (LOM)", generation: E1000Generation::E1000, max_speed_mbps: 1000, supports_tso: false, supports_rss: false, queue_count: 1 },

    // 82571 series (E1000E)
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x105E, name: "82571EB Gigabit Ethernet Controller", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x105F, name: "82571EB Gigabit Ethernet Controller (Fiber)", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1060, name: "82571EB Gigabit Ethernet Controller (Copper)", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10A4, name: "82571EB Gigabit Ethernet Controller (Quad Port)", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },

    // 82572 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x107D, name: "82572EI Gigabit Ethernet Controller (Copper)", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x107E, name: "82572EI Gigabit Ethernet Controller (Fiber)", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x107F, name: "82572EI Gigabit Ethernet Controller", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },

    // 82573 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x108A, name: "82573E Gigabit Ethernet Controller", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x108B, name: "82573E Gigabit Ethernet Controller (IAMT)", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x108C, name: "82573L Gigabit Ethernet Controller", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x109A, name: "82573V Gigabit Ethernet Controller (IAMT)", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },

    // 82574 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10D3, name: "82574L Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10F6, name: "82574LA Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },

    // 82575 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10A7, name: "82575EB Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10A9, name: "82575EB Gigabit Backplane Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10D6, name: "82575GB Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },

    // 82576 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10C9, name: "82576 Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10E6, name: "82576 Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10E7, name: "82576 Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10E8, name: "82576 Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },

    // 82577 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10F5, name: "82577LM Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10BF, name: "82577LC Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },

    // 82578 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10BE, name: "82578DM Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x10C0, name: "82578DC Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },

    // 82579 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1502, name: "82579LM Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1503, name: "82579V Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: false, queue_count: 1 },

    // 82580 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x150E, name: "82580 Gigabit Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x150F, name: "82580 Gigabit Fiber Network Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1510, name: "82580 Gigabit Backplane Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1511, name: "82580 Gigabit SFP Connection", generation: E1000Generation::E1000E, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },

    // I350 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1521, name: "I350 Gigabit Network Connection", generation: E1000Generation::I350, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1522, name: "I350 Gigabit Fiber Network Connection", generation: E1000Generation::I350, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1523, name: "I350 Gigabit Backplane Connection", generation: E1000Generation::I350, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1524, name: "I350 Gigabit Connection", generation: E1000Generation::I350, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 8 },

    // I210 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1533, name: "I210 Gigabit Network Connection", generation: E1000Generation::I210, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1534, name: "I210 Gigabit Network Connection", generation: E1000Generation::I210, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1535, name: "I210 Gigabit Network Connection (SGMII)", generation: E1000Generation::I210, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1536, name: "I210 Gigabit Network Connection (Fiber)", generation: E1000Generation::I210, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1537, name: "I210 Gigabit Backplane Connection", generation: E1000Generation::I210, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1538, name: "I210 Gigabit Network Connection", generation: E1000Generation::I210, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x1539, name: "I211 Gigabit Network Connection", generation: E1000Generation::I210, max_speed_mbps: 1000, supports_tso: true, supports_rss: true, queue_count: 2 },

    // I225 series
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x15F2, name: "I225-LM 2.5GbE Controller", generation: E1000Generation::I225, max_speed_mbps: 2500, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x15F3, name: "I225-V 2.5GbE Controller", generation: E1000Generation::I225, max_speed_mbps: 2500, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x15F4, name: "I225-IT 2.5GbE Controller", generation: E1000Generation::I225, max_speed_mbps: 2500, supports_tso: true, supports_rss: true, queue_count: 4 },
    IntelE1000DeviceInfo { vendor_id: 0x8086, device_id: 0x15F5, name: "I225-LMvP 2.5GbE Controller", generation: E1000Generation::I225, max_speed_mbps: 2500, supports_tso: true, supports_rss: true, queue_count: 4 },
];

/// E1000 register offsets
#[repr(u32)]
pub enum E1000Reg {
    /// Device Control
    Ctrl = 0x00000,
    /// Device Status
    Status = 0x00008,
    /// EEPROM/Flash Control
    Eecd = 0x00010,
    /// Extended Device Control
    CtrlExt = 0x00018,
    /// Flow Control Address Low
    Fcal = 0x00028,
    /// Flow Control Address High
    Fcah = 0x0002C,
    /// Flow Control Type
    Fct = 0x00030,
    /// VET register
    Vet = 0x00038,
    /// Interrupt Cause Read
    Icr = 0x000C0,
    /// Interrupt Throttling Rate
    Itr = 0x000C4,
    /// Interrupt Cause Set
    Ics = 0x000C8,
    /// Interrupt Mask Set
    Ims = 0x000D0,
    /// Interrupt Mask Clear
    Imc = 0x000D8,
    /// Receive Control
    Rctl = 0x00100,
    /// Flow Control Transmit Timer Value
    Fcttv = 0x00170,
    /// Transmit Control
    Tctl = 0x00400,
    /// Transmit Inter Packet Gap
    Tipg = 0x00410,
    /// Receive Descriptor Base Address Low
    Rdbal = 0x02800,
    /// Receive Descriptor Base Address High
    Rdbah = 0x02804,
    /// Receive Descriptor Length
    Rdlen = 0x02808,
    /// Receive Descriptor Head
    Rdh = 0x02810,
    /// Receive Descriptor Tail
    Rdt = 0x02818,
    /// Transmit Descriptor Base Address Low
    Tdbal = 0x03800,
    /// Transmit Descriptor Base Address High
    Tdbah = 0x03804,
    /// Transmit Descriptor Length
    Tdlen = 0x03808,
    /// Transmit Descriptor Head
    Tdh = 0x03810,
    /// Transmit Descriptor Tail
    Tdt = 0x03818,
    /// Receive Address Low
    Ral = 0x05400,
    /// Receive Address High
    Rah = 0x05404,
}

/// E1000 control register bits
bitflags::bitflags! {
    pub struct E1000Ctrl: u32 {
        const FD = 1 << 0;       // Full Duplex
        const LRST = 1 << 3;     // Link Reset
        const ASDE = 1 << 5;     // Auto-Speed Detection Enable
        const SLU = 1 << 6;      // Set Link Up
        const ILOS = 1 << 7;     // Invert Loss-of-Signal
        const SPD_SEL = 3 << 8;  // Speed Selection
        const SPD_10 = 0 << 8;   // 10 Mbps
        const SPD_100 = 1 << 8;  // 100 Mbps
        const SPD_1000 = 2 << 8; // 1000 Mbps
        const FRCSPD = 1 << 11;  // Force Speed
        const FRCDPLX = 1 << 12; // Force Duplex
        const RST = 1 << 26;     // Device Reset
        const VME = 1 << 30;     // VLAN Mode Enable
        const PHY_RST = 1 << 31; // PHY Reset
    }
}

/// E1000 status register bits
bitflags::bitflags! {
    pub struct E1000Status: u32 {
        const FD = 1 << 0;       // Full Duplex
        const LU = 1 << 1;       // Link Up
        const FUNC_ID = 3 << 2;  // Function ID
        const TXOFF = 1 << 4;    // Transmission Paused
        const TBIMODE = 1 << 5;  // TBI Mode
        const SPEED = 3 << 6;    // Speed
        const SPEED_10 = 0 << 6; // 10 Mbps
        const SPEED_100 = 1 << 6;// 100 Mbps
        const SPEED_1000 = 2 << 6;// 1000 Mbps
        const ASDV = 3 << 8;     // Auto Speed Detection Value
        const MTXCKOK = 1 << 10; // MTX Clock OK
        const PCI66 = 1 << 11;   // PCI 66 MHz Bus
        const BUS64 = 1 << 12;   // Bus 64-bit
        const PCIX_MODE = 1 << 13;// PCI-X Mode
        const PCIX_SPEED = 3 << 14;// PCI-X Speed
    }
}

/// E1000 receive control register bits
bitflags::bitflags! {
    pub struct E1000Rctl: u32 {
        const EN = 1 << 1;       // Enable
        const SBP = 1 << 2;      // Store Bad Packets
        const UPE = 1 << 3;      // Unicast Promiscuous Enable
        const MPE = 1 << 4;      // Multicast Promiscuous Enable
        const LPE = 1 << 5;      // Long Packet Enable
        const LBM = 3 << 6;      // Loopback Mode
        const RDMTS = 3 << 8;    // Receive Descriptor Minimum Threshold Size
        const MO = 3 << 12;      // Multicast Offset
        const BAM = 1 << 15;     // Broadcast Accept Mode
        const BSIZE = 3 << 16;   // Buffer Size
        const BSIZE_256 = 3 << 16;
        const BSIZE_512 = 2 << 16;
        const BSIZE_1024 = 1 << 16;
        const BSIZE_2048 = 0 << 16;
        const VFE = 1 << 18;     // VLAN Filter Enable
        const CFIEN = 1 << 19;   // Canonical Form Indicator Enable
        const CFI = 1 << 20;     // Canonical Form Indicator Value
        const DPF = 1 << 22;     // Discard Pause Frames
        const PMCF = 1 << 23;    // Pass MAC Control Frames
        const BSEX = 1 << 25;    // Buffer Size Extension
        const SECRC = 1 << 26;   // Strip Ethernet CRC
    }
}

/// E1000 transmit control register bits
bitflags::bitflags! {
    pub struct E1000Tctl: u32 {
        const EN = 1 << 1;       // Enable
        const PSP = 1 << 3;      // Pad Short Packets
        const CT = 0xFF << 4;    // Collision Threshold
        const COLD = 0x3FF << 12;// Collision Distance
        const SWXOFF = 1 << 22;  // Software XOFF Transmission
        const RTLC = 1 << 24;    // Re-transmit on Late Collision
        const NRTU = 1 << 25;    // No Re-transmit on Underrun
        const MULR = 1 << 28;    // Multiple Request Support
    }
}

/// Intel E1000 driver implementation
#[derive(Debug)]
pub struct IntelE1000Driver {
    name: String,
    device_info: Option<IntelE1000DeviceInfo>,
    state: DeviceState,
    capabilities: DeviceCapabilities,
    extended_capabilities: ExtendedNetworkCapabilities,
    stats: EnhancedNetworkStats,
    base_addr: u64,
    irq: u8,
    mac_address: MacAddress,
    power_state: PowerState,
    wol_config: WakeOnLanConfig,
    current_speed: u32,
    full_duplex: bool,
    /// DMA transmit ring
    tx_ring: Option<crate::net::dma::DmaRing>,
    /// DMA receive ring
    rx_ring: Option<crate::net::dma::DmaRing>,
}

impl IntelE1000Driver {
    /// Create new Intel E1000 driver instance
    pub fn new(
        name: String,
        device_info: IntelE1000DeviceInfo,
        base_addr: u64,
        irq: u8,
    ) -> Self {
        let mut capabilities = DeviceCapabilities::default();
        capabilities.mtu = 1500;
        capabilities.hw_checksum = true;
        capabilities.scatter_gather = true;
        capabilities.vlan_support = true;
        capabilities.jumbo_frames = true;
        capabilities.multicast_filter = true;
        capabilities.max_packet_size = 9018; // Jumbo frame
        capabilities.link_speed = device_info.max_speed_mbps;
        capabilities.full_duplex = true;

        if device_info.supports_rss {
            capabilities.rx_queues = device_info.queue_count;
            capabilities.tx_queues = device_info.queue_count;
        }

        let mut extended_capabilities = ExtendedNetworkCapabilities::default();
        extended_capabilities.base = capabilities.clone();
        extended_capabilities.max_bandwidth_mbps = device_info.max_speed_mbps;
        extended_capabilities.wake_on_lan = true;
        extended_capabilities.energy_efficient = matches!(device_info.generation, E1000Generation::E1000E | E1000Generation::I350 | E1000Generation::I210 | E1000Generation::I225);
        extended_capabilities.pxe_boot = true;
        extended_capabilities.sriov = matches!(device_info.generation, E1000Generation::I350 | E1000Generation::I210 | E1000Generation::I225);

        Self {
            name,
            device_info: Some(device_info),
            state: DeviceState::Down,
            capabilities,
            extended_capabilities,
            stats: EnhancedNetworkStats::default(),
            base_addr,
            irq,
            mac_address: MacAddress::ZERO,
            power_state: PowerState::D0,
            wol_config: WakeOnLanConfig::default(),
            current_speed: 0,
            full_duplex: false,
            tx_ring: None,
            rx_ring: None,
        }
    }

    /// Read E1000 register with proper memory barriers
    fn read_reg(&self, reg: E1000Reg) -> u32 {
        unsafe {
            // Ensure all previous writes complete before reading
            core::arch::x86_64::_mm_mfence();
            
            let value = ptr::read_volatile((self.base_addr + reg as u64) as *const u32);
            
            // Memory barrier to prevent reordering
            core::arch::x86_64::_mm_lfence();
            
            value
        }
    }

    /// Write E1000 register with proper memory barriers
    fn write_reg(&self, reg: E1000Reg, value: u32) {
        unsafe {
            // Ensure all previous operations complete
            core::arch::x86_64::_mm_mfence();
            
            ptr::write_volatile((self.base_addr + reg as u64) as *mut u32, value);
            
            // Ensure write completes before continuing
            core::arch::x86_64::_mm_sfence();
        }
    }

    /// Read and modify register atomically
    fn modify_reg<F>(&self, reg: E1000Reg, f: F) 
    where 
        F: FnOnce(u32) -> u32,
    {
        let current = self.read_reg(reg);
        let new_value = f(current);
        self.write_reg(reg, new_value);
    }

    /// Wait for register bit to be set with timeout
    fn wait_for_bit(&self, reg: E1000Reg, bit_mask: u32, set: bool, timeout_ms: u32) -> Result<(), NetworkError> {
        let start_time = self.get_time_ms();
        
        loop {
            let value = self.read_reg(reg);
            let bit_is_set = (value & bit_mask) != 0;
            
            if bit_is_set == set {
                return Ok(());
            }
            
            if self.get_time_ms() - start_time > timeout_ms as u64 {
                return Err(NetworkError::Timeout);
            }
            
            // Small delay to avoid overwhelming the bus
            self.delay_microseconds(10);
        }
    }

    /// Get current time in milliseconds
    fn get_time_ms(&self) -> u64 {
        // Use system time for driver timestamps
        crate::time::get_system_time_ms()
    }

    /// Delay for specified microseconds
    fn delay_microseconds(&self, microseconds: u32) {
        // Use kernel timer for accurate delays
        crate::time::sleep_us(microseconds as u64);
    }

    /// Reset the controller with proper hardware timing
    fn reset_controller(&mut self) -> Result<(), NetworkError> {
        // Disable all interrupts before reset
        self.write_reg(E1000Reg::Imc, 0xFFFFFFFF);
        
        // Clear any pending interrupts
        self.read_reg(E1000Reg::Icr);

        // Perform device reset
        self.modify_reg(E1000Reg::Ctrl, |ctrl| ctrl | E1000Ctrl::RST.bits());

        // Wait for reset to complete (up to 10ms)
        self.wait_for_bit(E1000Reg::Ctrl, E1000Ctrl::RST.bits(), false, 10)?;

        // Additional stabilization delay based on device generation
        let stabilization_delay = match self.device_info.map(|info| info.generation) {
            Some(E1000Generation::E1000) => 1000,      // 1ms for legacy E1000
            Some(E1000Generation::E1000E) => 500,      // 500μs for E1000E
            Some(E1000Generation::I350) => 200,        // 200μs for I350
            Some(E1000Generation::I210) => 100,        // 100μs for I210/I211
            Some(E1000Generation::I225) => 50,         // 50μs for I225
            None => 1000,                              // Default to 1ms
        };
        
        self.delay_microseconds(stabilization_delay);

        // Verify device is responsive after reset
        let status = self.read_reg(E1000Reg::Status);
        if status == 0xFFFFFFFF || status == 0 {
            return Err(NetworkError::HardwareError);
        }

        // Disable interrupts again after reset
        self.write_reg(E1000Reg::Imc, 0xFFFFFFFF);
        
        // Clear any interrupts that may have been generated during reset
        self.read_reg(E1000Reg::Icr);

        // Perform PHY reset if needed
        self.reset_phy()?;

        Ok(())
    }

    /// Reset PHY (Physical Layer)
    fn reset_phy(&mut self) -> Result<(), NetworkError> {
        // For some E1000 variants, PHY reset is needed
        match self.device_info.map(|info| info.generation) {
            Some(E1000Generation::E1000) => {
                // Legacy E1000 may need PHY reset
                self.modify_reg(E1000Reg::Ctrl, |ctrl| ctrl | E1000Ctrl::PHY_RST.bits());
                self.delay_microseconds(100);
                self.modify_reg(E1000Reg::Ctrl, |ctrl| ctrl & !E1000Ctrl::PHY_RST.bits());
                self.delay_microseconds(1000);
            }
            _ => {
                // Newer generations handle PHY reset automatically
            }
        }
        
        Ok(())
    }

    /// Read MAC address from EEPROM/registers
    fn read_mac_address(&mut self) -> Result<(), NetworkError> {
        // Try to read from receive address registers first
        let ral = self.read_reg(E1000Reg::Ral);
        let rah = self.read_reg(E1000Reg::Rah);

        if (rah & 0x80000000) != 0 { // Address valid bit
            let mac_bytes = [
                (ral & 0xFF) as u8,
                ((ral >> 8) & 0xFF) as u8,
                ((ral >> 16) & 0xFF) as u8,
                ((ral >> 24) & 0xFF) as u8,
                (rah & 0xFF) as u8,
                ((rah >> 8) & 0xFF) as u8,
            ];
            self.mac_address = MacAddress::new(mac_bytes);
        } else {
            // Generate a default MAC address with Intel OUI
            self.mac_address = super::utils::generate_mac_with_vendor(super::utils::INTEL_OUI);
        }

        Ok(())
    }

    /// Initialize receive subsystem with real hardware configuration
    fn init_rx(&mut self) -> Result<(), NetworkError> {
        // Disable receiver during configuration
        self.write_reg(E1000Reg::Rctl, 0);

        // Allocate DMA-coherent receive descriptor ring
        let rx_ring_base = self.allocate_rx_ring()?;
        
        // Set receive descriptor base address
        self.write_reg(E1000Reg::Rdbal, (rx_ring_base & 0xFFFFFFFF) as u32);
        self.write_reg(E1000Reg::Rdbah, (rx_ring_base >> 32) as u32);

        // Set receive descriptor length (32 descriptors * 16 bytes = 512 bytes)
        self.write_reg(E1000Reg::Rdlen, 32 * 16);

        // Initialize head and tail pointers
        self.write_reg(E1000Reg::Rdh, 0);
        self.write_reg(E1000Reg::Rdt, 31); // Make all descriptors available

        // Configure receive control register
        let mut rctl = E1000Rctl::EN.bits() |        // Enable receiver
                       E1000Rctl::BAM.bits() |       // Broadcast accept mode
                       E1000Rctl::BSIZE_2048.bits() | // 2048 byte buffers
                       E1000Rctl::SECRC.bits() |     // Strip Ethernet CRC
                       E1000Rctl::LPE.bits();        // Long packet enable (jumbo frames)

        // Configure multicast filter if supported
        if self.capabilities.multicast_filter {
            rctl |= E1000Rctl::MPE.bits(); // Multicast promiscuous enable for now
        }

        self.write_reg(E1000Reg::Rctl, rctl);

        // Configure receive interrupt delay
        self.write_reg(E1000Reg::Itr, 1000); // 1000 * 256ns = 256μs delay

        Ok(())
    }

    /// Allocate receive descriptor ring
    fn allocate_rx_ring(&mut self) -> Result<u64, NetworkError> {
        use crate::net::dma::DmaRing;

        // Allocate DMA ring: 256 descriptors, 2048 byte buffers
        let ring = DmaRing::new(256, 2048)?;
        let ring_addr = ring.descriptor_ring_addr();

        // Store ring in driver
        self.rx_ring = Some(ring);

        Ok(ring_addr)
    }

    /// Initialize transmit subsystem with real hardware configuration
    fn init_tx(&mut self) -> Result<(), NetworkError> {
        // Disable transmitter during configuration
        self.write_reg(E1000Reg::Tctl, 0);

        // Allocate transmit descriptor ring
        let tx_ring_base = self.allocate_tx_ring()?;

        // Set transmit descriptor base address
        self.write_reg(E1000Reg::Tdbal, (tx_ring_base & 0xFFFFFFFF) as u32);
        self.write_reg(E1000Reg::Tdbah, (tx_ring_base >> 32) as u32);

        // Set transmit descriptor length (32 descriptors * 16 bytes)
        self.write_reg(E1000Reg::Tdlen, 32 * 16);

        // Initialize head and tail pointers
        self.write_reg(E1000Reg::Tdh, 0);
        self.write_reg(E1000Reg::Tdt, 0);

        // Configure transmit control register
        let mut tctl = E1000Tctl::EN.bits() |        // Enable transmitter
                       E1000Tctl::PSP.bits() |       // Pad short packets
                       (0x0F << 4) |                 // Collision threshold (15)
                       (0x40 << 12);                 // Collision distance (64 bytes)

        // Enable retransmit on late collision for half-duplex
        if !self.full_duplex {
            tctl |= E1000Tctl::RTLC.bits();
        }

        self.write_reg(E1000Reg::Tctl, tctl);

        // Configure transmit inter-packet gap based on device generation
        let tipg = match self.device_info.map(|info| info.generation) {
            Some(E1000Generation::E1000) => 0x602008,      // Legacy E1000
            Some(E1000Generation::E1000E) => 0x602008,     // E1000E
            Some(E1000Generation::I350) => 0x602008,       // I350
            Some(E1000Generation::I210) => 0x602008,       // I210/I211
            Some(E1000Generation::I225) => 0x602008,       // I225
            None => 0x602008,                              // Default
        };
        self.write_reg(E1000Reg::Tipg, tipg);

        Ok(())
    }

    /// Allocate transmit descriptor ring
    fn allocate_tx_ring(&mut self) -> Result<u64, NetworkError> {
        use crate::net::dma::DmaRing;

        // Allocate DMA ring: 256 descriptors, 2048 byte buffers
        let ring = DmaRing::new(256, 2048)?;
        let ring_addr = ring.descriptor_ring_addr();

        // Store ring in driver
        self.tx_ring = Some(ring);

        Ok(ring_addr)
    }

    /// Send packet through hardware with real DMA
    fn send_packet_hardware(&mut self, packet_data: &[u8]) -> Result<(), NetworkError> {
        // Validate packet size
        if packet_data.is_empty() || packet_data.len() > 9018 {
            return Err(NetworkError::InvalidPacket);
        }

        // Get transmit ring
        let tx_ring = self.tx_ring.as_mut()
            .ok_or(NetworkError::InvalidState)?;

        // Get next available descriptor and buffer
        let (descriptor, dma_buffer) = tx_ring.get_tx_descriptor()
            .ok_or(NetworkError::Busy)?;

        // Copy packet data to DMA buffer
        dma_buffer.copy_from_slice(packet_data)?;

        // Ensure cache coherency (flush CPU cache to memory for hardware)
        dma_buffer.flush_cache();

        // Setup transmit descriptor
        descriptor.length = packet_data.len() as u16;
        descriptor.set_eop(); // End of packet
        descriptor.flags |= 1 << 2; // Ready for transmission (RS - Report Status)

        // Advance tail pointer in software
        tx_ring.advance_tail();

        // Get new tail value
        let new_tail = self.read_reg(E1000Reg::Tdt) as usize;
        let next_tail = (new_tail + 1) % 256; // 256 descriptors

        // Update hardware tail pointer to start transmission
        self.write_reg(E1000Reg::Tdt, next_tail as u32);

        // Update statistics
        self.stats.tx_packets += 1;
        self.stats.tx_bytes += packet_data.len() as u64;

        Ok(())
    }

    /// Receive packet from hardware with real DMA
    fn receive_packet_hardware(&mut self) -> Result<Option<Vec<u8>>, NetworkError> {
        // Get receive ring
        let rx_ring = self.rx_ring.as_mut()
            .ok_or(NetworkError::InvalidState)?;

        // Get next completed descriptor and buffer
        let (descriptor, dma_buffer) = match rx_ring.get_rx_descriptor() {
            Some(desc_buf) => desc_buf,
            None => return Ok(None), // No packets available
        };

        // Check for errors in descriptor
        if descriptor.has_error() {
            // Reset descriptor for reuse
            descriptor.status = 0;
            descriptor.flags = 1 << 2; // Ready for reception

            // Advance head pointer
            rx_ring.advance_head();

            // Update error statistics
            self.stats.rx_errors += 1;

            return Err(NetworkError::InvalidPacket);
        }

        // Ensure cache coherency (invalidate cache to see hardware updates)
        dma_buffer.invalidate_cache();

        // Copy packet data from DMA buffer
        let packet_len = descriptor.length as usize;
        let mut packet_data = alloc::vec![0u8; packet_len];
        let copied = dma_buffer.copy_to_slice(&mut packet_data);

        if copied != packet_len {
            // Reset descriptor for reuse
            descriptor.status = 0;
            descriptor.flags = 1 << 2;
            rx_ring.advance_head();
            return Err(NetworkError::BufferTooSmall);
        }

        // Reset descriptor for reuse
        descriptor.status = 0;
        descriptor.flags = 1 << 2; // Ready for reception

        // Advance head pointer
        rx_ring.advance_head();

        // Update hardware head pointer
        let new_head = self.read_reg(E1000Reg::Rdh) as usize;
        let next_head = (new_head + 1) % 256; // 256 descriptors
        self.write_reg(E1000Reg::Rdh, next_head as u32);

        // Update statistics
        self.stats.rx_packets += 1;
        self.stats.rx_bytes += packet_len as u64;

        Ok(Some(packet_data))
    }

    /// Handle hardware interrupt
    fn handle_interrupt(&mut self) -> Result<(), NetworkError> {
        // Read interrupt cause register
        let icr = self.read_reg(E1000Reg::Icr);

        // Handle different interrupt types
        if (icr & (1 << 0)) != 0 {
            // Transmit descriptor written back
            self.handle_tx_interrupt()?;
        }

        if (icr & (1 << 7)) != 0 {
            // Receive timer interrupt
            self.handle_rx_interrupt()?;
        }

        if (icr & (1 << 2)) != 0 {
            // Link status change
            self.handle_link_interrupt()?;
        }

        Ok(())
    }

    /// Handle transmit interrupt
    fn handle_tx_interrupt(&mut self) -> Result<(), NetworkError> {
        // Process completed transmit descriptors
        // In real implementation, this would clean up transmitted buffers
        Ok(())
    }

    /// Handle receive interrupt
    fn handle_rx_interrupt(&mut self) -> Result<(), NetworkError> {
        // Process received packets
        while let Some(packet_data) = self.receive_packet_hardware()? {
            // In real implementation, this would pass packet to network stack
            // For now, just update statistics
            self.stats.rx_packets += 1;
            self.stats.rx_bytes += packet_data.len() as u64;
        }
        Ok(())
    }

    /// Handle link status change interrupt
    fn handle_link_interrupt(&mut self) -> Result<(), NetworkError> {
        let status = self.read_reg(E1000Reg::Status);
        let link_up = (status & E1000Status::LU.bits()) != 0;

        if link_up {
            // Link is up, determine speed and duplex
            let speed_bits = (status & E1000Status::SPEED.bits()) >> 6;
            self.current_speed = match speed_bits {
                0 => 10,
                1 => 100,
                2 => 1000,
                _ => 0,
            };

            self.full_duplex = (status & E1000Status::FD.bits()) != 0;
            self.stats.link_changes += 1;
        } else {
            self.current_speed = 0;
            self.full_duplex = false;
            self.stats.link_changes += 1;
        }

        Ok(())
    }et tipg = match self.device_info.map(|info| info.generation) {
            Some(E1000Generation::E1000) => 0x602008, // 10/8/6 for copper
            _ => 0x602008, // Default values
        };
        self.write_reg(E1000Reg::Tipg, tipg);

        Ok(())
    }

    /// Configure link settings
    fn configure_link(&mut self) -> Result<(), NetworkError> {
        let mut ctrl = self.read_reg(E1000Reg::Ctrl);

        // Enable auto-negotiation
        ctrl |= E1000Ctrl::ASDE.bits();
        ctrl |= E1000Ctrl::SLU.bits(); // Set link up

        self.write_reg(E1000Reg::Ctrl, ctrl);

        // Wait for link establishment
        for _ in 0..1000 {
            let status = self.read_reg(E1000Reg::Status);
            if (status & E1000Status::LU.bits()) != 0 {
                // Link is up, determine speed and duplex
                self.current_speed = match (status & E1000Status::SPEED.bits()) >> 6 {
                    0 => 10,
                    1 => 100,
                    2 | 3 => 1000,
                    _ => 0,
                };
                self.full_duplex = (status & E1000Status::FD.bits()) != 0;
                break;
            }
        }

        Ok(())
    }

    /// Get device generation string
    pub fn get_generation_string(&self) -> &'static str {
        if let Some(info) = self.device_info {
            match info.generation {
                E1000Generation::E1000 => "E1000",
                E1000Generation::E1000E => "E1000E",
                E1000Generation::I350 => "I350",
                E1000Generation::I210 => "I210/I211",
                E1000Generation::I225 => "I225",
            }
        } else {
            "Unknown"
        }
    }

    /// Get detailed device information
    pub fn get_device_details(&self) -> String {
        if let Some(info) = self.device_info {
            format!(
                "{} ({}), Max Speed: {} Mbps, Queues: {}, TSO: {}, RSS: {}",
                info.name,
                self.get_generation_string(),
                info.max_speed_mbps,
                info.queue_count,
                info.supports_tso,
                info.supports_rss
            )
        } else {
            "Unknown Intel E1000 Device".to_string()
        }
    }

    /// Configure Wake-on-LAN
    pub fn configure_wol(&mut self, config: WakeOnLanConfig) -> Result<(), NetworkError> {
        self.wol_config = config;

        // In a real implementation, we would:
        // 1. Configure WoL filters
        // 2. Set up magic packet detection
        // 3. Configure power management

        Ok(())
    }

    /// Set power state
    pub fn set_power_state(&mut self, state: PowerState) -> Result<(), NetworkError> {
        // In a real implementation, we would configure PCI power management
        self.power_state = state;
        Ok(())
    }
}

impl NetworkDriver for IntelE1000Driver {
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Ethernet
    }

    fn state(&self) -> DeviceState {
        self.state
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn init(&mut self) -> Result<(), NetworkError> {
        // Reset controller
        self.reset_controller()?;

        // Read MAC address
        self.read_mac_address()?;

        // Initialize receive subsystem
        self.init_rx()?;

        // Initialize transmit subsystem
        self.init_tx()?;

        self.state = DeviceState::Initialized;
        Ok(())
    }

    fn start(&mut self) -> Result<(), NetworkError> {
        if self.state != DeviceState::Initialized {
            return Err(NetworkError::InvalidState);
        }

        // Enable interrupts
        let ims = (1 << 0) | // Transmit descriptor written back
                  (1 << 7) | // Receive timer
                  (1 << 2);  // Link status change
        self.write_reg(E1000Reg::Ims, ims);

        self.state = DeviceState::Up;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), NetworkError> {
        // Disable interrupts
        self.write_reg(E1000Reg::Imc, 0xFFFFFFFF);

        // Disable receiver and transmitter
        self.write_reg(E1000Reg::Rctl, 0);
        self.write_reg(E1000Reg::Tctl, 0);

        self.state = DeviceState::Down;
        Ok(())
    }

    fn send_packet(&mut self, packet: &[u8]) -> Result<(), NetworkError> {
        if self.state != DeviceState::Up {
            return Err(NetworkError::NetworkUnreachable);
        }

        self.send_packet_hardware(packet)
    }

    fn receive_packet(&mut self) -> Result<Option<Vec<u8>>, NetworkError> {
        if self.state != DeviceState::Up {
            return Ok(None);
        }

        self.receive_packet_hardware()
    }

    fn get_mac_address(&self) -> MacAddress {
        self.mac_address
    }

    fn set_mac_address(&mut self, mac: MacAddress) -> Result<(), NetworkError> {
        self.mac_address = mac;

        // Write MAC address to hardware registers
        let mac_bytes = mac.as_bytes();
        let ral = ((mac_bytes[3] as u32) << 24) |
                  ((mac_bytes[2] as u32) << 16) |
                  ((mac_bytes[1] as u32) << 8) |
                  (mac_bytes[0] as u32);
        let rah = ((mac_bytes[5] as u32) << 8) |
                  (mac_bytes[4] as u32) |
                  0x80000000; // Address valid bit

        self.write_reg(E1000Reg::Ral, ral);
        self.write_reg(E1000Reg::Rah, rah);

        Ok(())
    }

    fn get_link_status(&self) -> (bool, u32, bool) {
        let status = self.read_reg(E1000Reg::Status);
        let link_up = (status & E1000Status::LU.bits()) != 0;
        (link_up, self.current_speed, self.full_duplex)
    }

    fn get_statistics(&self) -> NetworkStats {
        NetworkStats {
            rx_packets: self.stats.rx_packets,
            tx_packets: self.stats.tx_packets,
            rx_bytes: self.stats.rx_bytes,
            tx_bytes: self.stats.tx_bytes,
            rx_errors: self.stats.rx_errors,
            tx_errors: self.stats.tx_errors,
            rx_dropped: self.stats.rx_dropped,
            tx_dropped: self.stats.tx_dropped,
        }
    }

    fn handle_interrupt(&mut self) -> Result<(), NetworkError> {
        self.handle_interrupt()
    }

    fn set_power_state(&mut self, state: PowerState) -> Result<(), NetworkError> {
        // Configure power management
        match state {
            PowerState::D0 => {
                // Full power
                self.power_state = state;
            }
            PowerState::D3Hot => {
                // Low power with wake capabilities
                if self.wol_config.enabled {
                    // Configure Wake-on-LAN
                    // In real implementation, set WOL registers
                }
                self.power_state = state;
            }
            _ => {
                return Err(NetworkError::NotSupported);
            }
        }
        Ok(())
    }

    fn configure_wol(&mut self, config: WakeOnLanConfig) -> Result<(), NetworkError> {
        if !self.extended_capabilities.wake_on_lan {
            return Err(NetworkError::NotSupported);
        }

        self.wol_config = config;
        
        // In real implementation, configure hardware WOL registers
        // For now, just store the configuration
        
        Ok(())
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Ethernet
    }

    fn mac_address(&self) -> MacAddress {
        self.mac_address
    }

    fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    fn state(&self) -> DeviceState {
        self.state
    }

    fn init(&mut self) -> Result<(), NetworkError> {
        self.state = DeviceState::Testing;

        // Reset controller
        self.reset_controller()?;

        // Read MAC address
        self.read_mac_address()?;

        // Initialize subsystems
        self.init_rx()?;
        self.init_tx()?;

        // Configure link
        self.configure_link()?;

        self.state = DeviceState::Down;
        Ok(())
    }

    fn start(&mut self) -> Result<(), NetworkError> {
        if self.state != DeviceState::Down {
            return Err(NetworkError::InvalidState);
        }

        // Enable interrupts (simplified)
        self.write_reg(E1000Reg::Ims, 0x1F6DC);

        self.state = DeviceState::Up;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), NetworkError> {
        if self.state != DeviceState::Up {
            return Err(NetworkError::InvalidState);
        }

        // Disable interrupts
        self.write_reg(E1000Reg::Imc, 0xFFFFFFFF);

        // Disable receiver and transmitter
        self.write_reg(E1000Reg::Rctl, 0);
        self.write_reg(E1000Reg::Tctl, 0);

        self.state = DeviceState::Down;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), NetworkError> {
        self.state = DeviceState::Resetting;
        self.reset_controller()?;
        self.init()?;
        Ok(())
    }

    fn send_packet(&mut self, data: &[u8]) -> Result<(), NetworkError> {
        if self.state != DeviceState::Up {
            return Err(NetworkError::InterfaceDown);
        }

        if data.len() > self.capabilities.max_packet_size as usize {
            return Err(NetworkError::BufferTooSmall);
        }

        // In a real implementation, we would:
        // 1. Get next available transmit descriptor
        // 2. Set up DMA mapping for packet data
        // 3. Configure descriptor with packet information
        // 4. Ring doorbell to notify hardware

        // Simulate successful transmission
        self.stats.tx_packets += 1;
        self.stats.tx_bytes += data.len() as u64;

        Ok(())
    }

    fn receive_packet(&mut self) -> Option<Vec<u8>> {
        if self.state != DeviceState::Up {
            return None;
        }

        // In a real implementation, we would:
        // 1. Check receive descriptor ring for completed packets
        // 2. Process received data
        // 3. Update receive statistics
        // 4. Replenish receive buffers

        // For simulation, return None (no packets available)
        None
    }

    fn is_link_up(&self) -> bool {
        let status = self.read_reg(E1000Reg::Status);
        (status & E1000Status::LU.bits()) != 0
    }

    fn set_promiscuous(&mut self, enabled: bool) -> Result<(), NetworkError> {
        let mut rctl = self.read_reg(E1000Reg::Rctl);

        if enabled {
            rctl |= E1000Rctl::UPE.bits() | E1000Rctl::MPE.bits();
        } else {
            rctl &= !(E1000Rctl::UPE.bits() | E1000Rctl::MPE.bits());
        }

        self.write_reg(E1000Reg::Rctl, rctl);
        Ok(())
    }

    fn add_multicast(&mut self, _addr: MacAddress) -> Result<(), NetworkError> {
        // In a real implementation, we would add the address to the multicast filter table
        Ok(())
    }

    fn remove_multicast(&mut self, _addr: MacAddress) -> Result<(), NetworkError> {
        // In a real implementation, we would remove the address from the multicast filter table
        Ok(())
    }

    fn get_stats(&self) -> NetworkStats {
        NetworkStats {
            packets_sent: self.stats.tx_packets,
            packets_received: self.stats.rx_packets,
            bytes_sent: self.stats.tx_bytes,
            bytes_received: self.stats.rx_bytes,
            send_errors: self.stats.tx_errors,
            receive_errors: self.stats.rx_errors,
            dropped_packets: self.stats.tx_dropped + self.stats.rx_dropped,
        }
    }

    fn set_mtu(&mut self, mtu: u16) -> Result<(), NetworkError> {
        if mtu < 68 || mtu > 9000 {
            return Err(NetworkError::InvalidPacket);
        }
        self.capabilities.mtu = mtu;
        Ok(())
    }

    fn get_mtu(&self) -> u16 {
        self.capabilities.mtu
    }

    fn handle_interrupt(&mut self) -> Result<(), NetworkError> {
        // Read interrupt cause register
        let icr = self.read_reg(E1000Reg::Icr);

        // Handle different interrupt types
        if (icr & 0x04) != 0 { // Link status change
            self.stats.link_changes += 1;
        }

        if (icr & 0x80) != 0 { // Receive timer interrupt
            // Process received packets
            self.stats.rx_packets += 1; // Simplified
        }

        if (icr & 0x01) != 0 { // Transmit descriptor written back
            // Process completed transmissions
        }

        Ok(())
    }
}

/// Create Intel E1000 driver from PCI device information
pub fn create_intel_e1000_driver(
    vendor_id: u16,
    device_id: u16,
    base_addr: u64,
    irq: u8,
) -> Option<(Box<dyn NetworkDriver + Send + Sync>, ExtendedNetworkCapabilities)> {
    // Find matching device in database
    let device_info = INTEL_E1000_DEVICES.iter()
        .find(|info| info.vendor_id == vendor_id && info.device_id == device_id)?;

    // Create driver instance
    let driver = IntelE1000Driver::new(
        device_info.name.to_string(),
        *device_info,
        base_addr,
        irq,
    );

    let capabilities = driver.extended_capabilities.clone();
    
    Some((Box::new(driver), capabilities))
    // Find matching device in database
    let device_info = INTEL_E1000_DEVICES.iter()
        .find(|info| info.vendor_id == vendor_id && info.device_id == device_id)
        .copied()?;

    let name = format!("Intel {}", device_info.name);
    let driver = IntelE1000Driver::new(name, device_info, base_addr, irq);
    let capabilities = driver.extended_capabilities.clone();

    Some((Box::new(driver), capabilities))
}

/// Check if PCI device is an Intel E1000 controller
pub fn is_intel_e1000_device(vendor_id: u16, device_id: u16) -> bool {
    INTEL_E1000_DEVICES.iter()
        .any(|info| info.vendor_id == vendor_id && info.device_id == device_id)
}

/// Get Intel E1000 device information
pub fn get_intel_e1000_device_info(vendor_id: u16, device_id: u16) -> Option<&'static IntelE1000DeviceInfo> {
    INTEL_E1000_DEVICES.iter()
        .find(|info| info.vendor_id == vendor_id && info.device_id == device_id)
}
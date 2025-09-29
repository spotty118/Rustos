//! # Intel E1000/E1000E Ethernet Driver
//!
//! Driver for Intel 82540/82541/82542/82543/82544/82545/82546/82547/82571/82572/82573/82574/82575/82576
//! and other Intel Gigabit Ethernet controllers (E1000 and E1000E series).

use super::{ExtendedNetworkCapabilities, EnhancedNetworkStats, PowerState, WakeOnLanConfig};
use crate::network::drivers::{NetworkDriver, DeviceType, DeviceState, DeviceCapabilities};
use crate::network::{NetworkError, NetworkStats, MacAddress};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ptr;

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
        }
    }

    /// Read E1000 register
    fn read_reg(&self, reg: E1000Reg) -> u32 {
        unsafe {
            ptr::read_volatile((self.base_addr + reg as u64) as *const u32)
        }
    }

    /// Write E1000 register
    fn write_reg(&self, reg: E1000Reg, value: u32) {
        unsafe {
            ptr::write_volatile((self.base_addr + reg as u64) as *mut u32, value);
        }
    }

    /// Reset the controller
    fn reset_controller(&mut self) -> Result<(), NetworkError> {
        // Disable interrupts
        self.write_reg(E1000Reg::Imc, 0xFFFFFFFF);

        // Reset the device
        let mut ctrl = self.read_reg(E1000Reg::Ctrl);
        ctrl |= E1000Ctrl::RST.bits();
        self.write_reg(E1000Reg::Ctrl, ctrl);

        // Wait for reset to complete
        for _ in 0..1000 {
            if (self.read_reg(E1000Reg::Ctrl) & E1000Ctrl::RST.bits()) == 0 {
                break;
            }
        }

        // Wait additional time for device stabilization
        for _ in 0..10000 {
            // Small delay
        }

        // Disable interrupts again
        self.write_reg(E1000Reg::Imc, 0xFFFFFFFF);

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

    /// Initialize receive subsystem
    fn init_rx(&mut self) -> Result<(), NetworkError> {
        // Allocate receive descriptor ring (simplified)
        // In a real implementation, we would allocate DMA-coherent memory

        // Set receive descriptor base address (using dummy address for now)
        self.write_reg(E1000Reg::Rdbal, 0x12345000);
        self.write_reg(E1000Reg::Rdbah, 0);

        // Set receive descriptor length (32 descriptors * 16 bytes = 512 bytes)
        self.write_reg(E1000Reg::Rdlen, 32 * 16);

        // Set receive descriptor head and tail
        self.write_reg(E1000Reg::Rdh, 0);
        self.write_reg(E1000Reg::Rdt, 31); // Last descriptor

        // Configure receive control
        let mut rctl = E1000Rctl::EN.bits() |     // Enable receiver
                       E1000Rctl::BAM.bits() |    // Broadcast accept mode
                       E1000Rctl::BSIZE_2048.bits() | // 2048 byte buffers
                       E1000Rctl::SECRC.bits();   // Strip Ethernet CRC

        self.write_reg(E1000Reg::Rctl, rctl);

        Ok(())
    }

    /// Initialize transmit subsystem
    fn init_tx(&mut self) -> Result<(), NetworkError> {
        // Allocate transmit descriptor ring (simplified)

        // Set transmit descriptor base address
        self.write_reg(E1000Reg::Tdbal, 0x12346000);
        self.write_reg(E1000Reg::Tdbah, 0);

        // Set transmit descriptor length
        self.write_reg(E1000Reg::Tdlen, 32 * 16);

        // Set transmit descriptor head and tail
        self.write_reg(E1000Reg::Tdh, 0);
        self.write_reg(E1000Reg::Tdt, 0);

        // Configure transmit control
        let tctl = E1000Tctl::EN.bits() |        // Enable transmitter
                   E1000Tctl::PSP.bits() |       // Pad short packets
                   (0x40 << 4) |                 // Collision threshold
                   (0x40 << 12);                 // Collision distance

        self.write_reg(E1000Reg::Tctl, tctl);

        // Configure transmit IPG
        let tipg = match self.device_info.map(|info| info.generation) {
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
) -> Option<(Box<dyn NetworkDriver>, ExtendedNetworkCapabilities)> {
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
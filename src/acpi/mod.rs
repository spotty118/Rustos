//! ACPI subsystem scaffolding for RustOS
//!
//! This module stores bootloader-provided ACPI discovery information so the
//! kernel can parse ACPI tables once physical memory mappings are established.

use alloc::{collections::BTreeSet, vec::Vec};
use core::{mem, slice};
use lazy_static::lazy_static;
use spin::RwLock;

/// ACPI discovery information captured during boot
#[derive(Debug, Clone)]
pub struct AcpiInfo {
    /// Physical address of the ACPI Root System Description Pointer (RSDP)
    pub rsdp_physical: u64,
    /// Optional virtual address where the RSDP can be accessed (requires physical offset)
    pub rsdp_virtual: Option<usize>,
    /// Physical memory offset supplied by the bootloader for identity mappings
    pub physical_memory_offset: Option<u64>,
    /// Whether the full ACPI tables have been parsed and cached
    pub tables_initialized: bool,
    /// Cached system description tables discovered during enumeration
    pub tables: Option<AcpiTables>,
    /// Cached MADT information
    pub madt: Option<MadtInfo>,
    /// Cached FADT information
    pub fadt: Option<FadtInfo>,
    /// Cached MCFG information
    pub mcfg: Option<McfgInfo>,
}

impl AcpiInfo {
    fn new(rsdp_physical: u64, physical_memory_offset: Option<u64>) -> Result<Self, &'static str> {
        let rsdp_virtual = if let Some(offset) = physical_memory_offset {
            match offset.checked_add(rsdp_physical) {
                Some(virt) => Some(virt as usize),
                None => return Err("Physical memory offset + RSDP address overflowed"),
            }
        } else {
            None
        };

        Ok(Self {
            rsdp_physical,
            rsdp_virtual,
            physical_memory_offset,
            tables_initialized: false,
            tables: None,
            madt: None,
            fadt: None,
            mcfg: None,
        })
    }
}

/// Parsed RSDP information that downstream subsystems can use
#[derive(Debug, Clone)]
pub struct RsdpInfo {
    pub oem_id: [u8; 6],
    pub revision: u8,
    pub rsdt_address: u32,
    pub xsdt_address: Option<u64>,
}

/// Parsed ACPI system description tables
#[derive(Debug, Clone, Default)]
pub struct AcpiTables {
    pub rsdt_entries: Vec<u64>,
    pub xsdt_entries: Vec<u64>,
    pub descriptors: Vec<AcpiTableDescriptor>,
}

/// Metadata for a discovered ACPI system description table
#[derive(Debug, Clone)]
pub struct AcpiTableDescriptor {
    pub signature: [u8; 4],
    pub phys_addr: u64,
    pub virt_addr: Option<usize>,
/// Multiprocessor APIC configuration extracted from the MADT
#[derive(Debug, Clone, Default)]
pub struct MadtInfo {
    pub local_apic_address: u32,
    pub flags: u32,
    pub io_apics: Vec<IoApic>,
    pub interrupt_overrides: Vec<InterruptOverride>,
    pub processors: Vec<ProcessorInfo>,
}

/// Processor information from MADT
#[derive(Debug, Clone)]
pub struct ProcessorInfo {
    pub processor_id: u8,
    pub apic_id: u8,
    pub flags: u32,
}

#[derive(Debug, Clone)]
pub struct IoApic {
    pub id: u8,
    pub address: u32,
    pub global_system_interrupt_base: u32,
}

#[derive(Debug, Clone)]
pub struct InterruptOverride {
    pub bus_source: u8,
    pub irq_source: u8,
    pub global_system_interrupt: u32,
    pub flags: u16,
}

/// Fixed ACPI Description Table (FADT / FACP) summary
#[derive(Debug, Clone, Default)]
pub struct FadtInfo {
    pub firmware_ctrl: Option<u32>,
    pub dsdt: Option<u32>,
    pub sci_interrupt: Option<u16>,
    pub smi_command: Option<u32>,
    pub acpi_enable: Option<u8>,
    pub acpi_disable: Option<u8>,
    pub pm1a_control_block: Option<u32>,
    pub pm_timer_block: Option<u32>,
    pub flags: Option<u32>,
    pub x_pm_timer_block: Option<u64>,
}

/// Memory Mapped Configuration (MCFG) table entry
#[derive(Debug, Clone)]
pub struct McfgEntry {
    pub base_address: u64,
    pub segment_group: u16,
    pub start_bus: u8,
    pub end_bus: u8,
}

/// MCFG table information for PCIe MMCONFIG
#[derive(Debug, Clone, Default)]
pub struct McfgInfo {
    pub entries: Vec<McfgEntry>,
}

lazy_static! {
    static ref ACPI_STATE: RwLock<Option<AcpiInfo>> = RwLock::new(None);
}

/// Initialize the ACPI subsystem with the provided RSDP pointer
pub fn init(rsdp_physical: u64, physical_memory_offset: Option<u64>) -> Result<(), &'static str> {
    let mut state = ACPI_STATE.write();

    if state.is_some() {
        return Ok(());
    }

    *state = Some(AcpiInfo::new(rsdp_physical, physical_memory_offset)?);
    Ok(())
}

/// Retrieve a snapshot of the ACPI discovery information
pub fn acpi_info() -> Option<AcpiInfo> {
    ACPI_STATE.read().clone()
}

/// Mark that ACPI tables have been fully parsed
pub fn mark_tables_initialized() {
    let mut state = ACPI_STATE.write();
    if let Some(info) = state.as_mut() {
        info.tables_initialized = true;
    }
}

/// Check if the ACPI subsystem has been initialized
pub fn is_initialized() -> bool {
    ACPI_STATE.read().is_some()
}

/// Attempt to parse the RSDP structure pointed to by the bootloader
pub fn parse_rsdp() -> Result<RsdpInfo, &'static str> {
    let state = ACPI_STATE.read();
    let state = state.as_ref().ok_or("ACPI subsystem not initialized")?;

    let rsdp_addr = state
        .rsdp_virtual
        .ok_or("Physical memory offset unavailable; cannot access ACPI tables")?;

    unsafe {
        let rsdp_v1 = &*(rsdp_addr as *const RsdpDescriptorV1);

        if &rsdp_v1.signature != b"RSD PTR " {
            return Err("Invalid RSDP signature");
        }

        if !checksum_bytes(slice::from_raw_parts(
            rsdp_addr as *const u8,
            mem::size_of::<RsdpDescriptorV1>(),
        )) {
            return Err("RSDP checksum validation failed");
        }

        let revision = rsdp_v1.revision;
        let mut result = RsdpInfo {
            oem_id: rsdp_v1.oem_id,
            revision,
            rsdt_address: rsdp_v1.rsdt_address,
            xsdt_address: None,
        };

        if revision >= 2 {
            let rsdp_v2 = &*(rsdp_addr as *const RsdpDescriptorV2);

            if rsdp_v2.length as usize >= mem::size_of::<RsdpDescriptorV2>() {
                if !checksum_bytes(slice::from_raw_parts(
                    rsdp_addr as *const u8,
                    rsdp_v2.length as usize,
                )) {
                    return Err("Extended RSDP checksum validation failed");
                }

                result.xsdt_address = Some(rsdp_v2.xsdt_address);
            }
        }

        Ok(result)
    }
}

/// Enumerate the ACPI system description tables referenced by the RSDP
pub fn enumerate_system_description_tables() -> Result<AcpiTables, &'static str> {
    let rsdp = parse_rsdp()?;

    let state_guard = ACPI_STATE.read();
    let state = state_guard
        .as_ref()
        .ok_or("ACPI subsystem not initialized")?;

    let physical_offset = state
        .physical_memory_offset
        .ok_or("Physical memory offset unavailable; cannot access ACPI tables")?;

    let mut tables = AcpiTables::default();

    if rsdp.rsdt_address != 0 {
        let rsdt_entries = unsafe {
            read_sdt_entries(
                phys_to_virt(rsdp.rsdt_address as u64, physical_offset)
                    .ok_or("Failed to translate RSDT physical address")?,
                4,
            )?
        };
        tables.rsdt_entries = rsdt_entries;
    }

    if let Some(xsdt_phys) = rsdp.xsdt_address {
        if xsdt_phys != 0 {
            let xsdt_entries = unsafe {
                read_sdt_entries(
                    phys_to_virt(xsdt_phys, physical_offset)
                        .ok_or("Failed to translate XSDT physical address")?,
                    8,
                )?
            };
            tables.xsdt_entries = xsdt_entries;
        }
    }

    let mut unique_entries: BTreeSet<u64> = BTreeSet::new();
    unique_entries.extend(tables.rsdt_entries.iter().copied());
    unique_entries.extend(tables.xsdt_entries.iter().copied());

    let mut descriptors = Vec::new();

    for phys in unique_entries {
        if phys == 0 {
            continue;
        }

        let virt = phys_to_virt(phys, physical_offset)
            .ok_or("Failed to translate ACPI SDT physical address")?;

        let header = unsafe { &*(virt as *const SdtHeader) };

        if header.length as usize >= mem::size_of::<SdtHeader>() {
            // Validate table checksum before accepting it
            let table_slice = unsafe {
                slice::from_raw_parts(virt as *const u8, header.length as usize)
            };

            if !checksum_bytes(table_slice) {
                continue;
            }

            descriptors.push(AcpiTableDescriptor {
                signature: header.signature,
                phys_addr: phys,
                virt_addr: Some(virt),
                length: header.length,
                revision: header.revision,
            });
        }
    }

    tables.descriptors = descriptors;

    drop(state_guard);

    {
        let mut state_write = ACPI_STATE.write();
        if let Some(info) = state_write.as_mut() {
            info.tables = Some(tables.clone());
            info.tables_initialized = true;
        }
    }

    Ok(tables)
}

/// ACPI RSDP descriptor for revision 1.0
#[repr(C, packed)]
struct RsdpDescriptorV1 {
    signature: [u8; 8],
    checksum: u8,
    oem_id: [u8; 6],
    revision: u8,
    rsdt_address: u32,
}

/// ACPI RSDP descriptor for revision 2.0+
#[repr(C, packed)]
struct RsdpDescriptorV2 {
    v1: RsdpDescriptorV1,
    length: u32,
    xsdt_address: u64,
    extended_checksum: u8,
    reserved: [u8; 3],
}

fn checksum_bytes(bytes: &[u8]) -> bool {
    bytes.iter().fold(0u8, |acc, b| acc.wrapping_add(*b)) == 0
}

#[repr(C, packed)]
struct SdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: u32,
    creator_revision: u32,
}

unsafe fn read_sdt_entries(virt_addr: usize, entry_size: usize) -> Result<Vec<u64>, &'static str> {
    let header = &*(virt_addr as *const SdtHeader);
    let total_length = header.length as usize;
    let header_size = mem::size_of::<SdtHeader>();

    if total_length < header_size {
        return Err("ACPI SDT length shorter than header");
    }

    let entries_length = total_length - header_size;

    if entries_length % entry_size != 0 {
        return Err("ACPI SDT entry area misaligned");
    }

    let entry_count = entries_length / entry_size;
    let entries_ptr = (virt_addr + header_size) as *const u8;
    let entries_slice = slice::from_raw_parts(entries_ptr, entries_length);

    let mut entries = Vec::with_capacity(entry_count);

    for chunk in entries_slice.chunks(entry_size) {
        let value = match entry_size {
            4 => {
                let mut array = [0u8; 4];
                array.copy_from_slice(chunk);
                u32::from_le_bytes(array) as u64
            }
            8 => {
                let mut array = [0u8; 8];
                array.copy_from_slice(chunk);
                u64::from_le_bytes(array)
            }
            _ => return Err("Unsupported SDT entry size"),
        };
        entries.push(value);
    }
    if !checksum_bytes(slice::from_raw_parts(virt_addr as *const u8, total_length)) {
        return Err("ACPI SDT checksum validation failed");
    }

    Ok(McfgInfo { entries })
}

const MADT_ENTRY_PROCESSOR: u8 = 0;
const MADT_ENTRY_IO_APIC: u8 = 1;
const MADT_ENTRY_INTERRUPT_OVERRIDE: u8 = 2;
const MADT_PROCESSOR_LEN: usize = 8;
const MADT_IO_APIC_LEN: usize = 12;
const MADT_INTERRUPT_OVERRIDE_LEN: usize = 10;

fn phys_to_virt(phys: u64, offset: u64) -> Option<usize> {
    offset.checked_add(phys).map(|addr| addr as usize)
}
/// Retrieve a clone of the cached ACPI tables (if enumeration has completed)
pub fn tables() -> Option<AcpiTables> {
    ACPI_STATE.read().as_ref()?.tables.clone()
}

/// Find a specific ACPI table by its four-character signature
pub fn find_table(signature: &[u8; 4]) -> Option<AcpiTableDescriptor> {
    let state = ACPI_STATE.read();
    let info = state.as_ref()?;
    let tables = info.tables.as_ref()?;
    tables
        .descriptors
        .iter()
        .find(|desc| &desc.signature == signature)
        .cloned()
}

/// Get cached MADT information if previously parsed
pub fn madt() -> Option<MadtInfo> {
    ACPI_STATE.read().as_ref()?.madt.clone()
}

/// Get cached FADT information if previously parsed
pub fn fadt() -> Option<FadtInfo> {
    ACPI_STATE.read().as_ref()?.fadt.clone()
}

/// Get cached MCFG information if previously parsed
pub fn mcfg() -> Option<McfgInfo> {
    ACPI_STATE.read().as_ref()?.mcfg.clone()
}

/// Parse the Multiple APIC Description Table (MADT) to extract interrupt controller topology
pub fn parse_madt() -> Result<MadtInfo, &'static str> {
    let descriptor = find_table(b"APIC").ok_or("MADT (APIC) table not found")?;

    let virt = descriptor
        .virt_addr
        .or_else(|| {
            acpi_info()
                .and_then(|info| info.physical_memory_offset)
                .and_then(|offset| phys_to_virt(descriptor.phys_addr, offset))
        })
        .ok_or("Failed to map MADT virtual address")?;

    let info = unsafe { parse_madt_from_address(virt, descriptor.length as usize) }?;

    {
        let mut state = ACPI_STATE.write();
        if let Some(acpi) = state.as_mut() {
            acpi.madt = Some(info.clone());
        }
    }

    Ok(info)
}

unsafe fn parse_madt_from_address(virt_addr: usize, table_length: usize) -> Result<MadtInfo, &'static str> {
    if table_length < mem::size_of::<MadtHeader>() {
        return Err("MADT shorter than expected header size");
{{ ... }}
    let virt = descriptor
        .virt_addr
        .or_else(|| {
            acpi_info()
                .and_then(|info| info.physical_memory_offset)
                .and_then(|offset| phys_to_virt(descriptor.phys_addr, offset))
        })
        .ok_or("Failed to map FADT virtual address")?;

    let info = unsafe { parse_fadt_from_address(virt, descriptor.length as usize) }?;

    {
        let mut state = ACPI_STATE.write();
        if let Some(acpi) = state.as_mut() {
            acpi.fadt = Some(info.clone());
        }
    }

    Ok(info)
}

unsafe fn parse_fadt_from_address(virt_addr: usize, table_length: usize) -> Result<FadtInfo, &'static str> {
    if table_length < mem::size_of::<SdtHeader>() {
        return Err("FADT shorter than SDT header");
{{ ... }}

    let table_slice = slice::from_raw_parts(virt_addr as *const u8, table_length);
    if !checksum_bytes(table_slice) {
        return Err("FADT checksum validation failed");
    }

    let mut info = FadtInfo::default();

    info.firmware_ctrl = read_u32(table_slice, FADT_FIRMWARE_CTRL_OFFSET);
    info.dsdt = read_u32(table_slice, FADT_DSDT_OFFSET);
    info.sci_interrupt = read_u16(table_slice, FADT_SCI_INTERRUPT_OFFSET);
    info.smi_command = read_u32(table_slice, FADT_SMI_CMD_OFFSET);
    info.acpi_enable = read_u8(table_slice, FADT_ACPI_ENABLE_OFFSET);
    info.acpi_disable = read_u8(table_slice, FADT_ACPI_DISABLE_OFFSET);
    info.pm1a_control_block = read_u32(table_slice, FADT_PM1A_CONTROL_OFFSET);
    info.pm_timer_block = read_u32(table_slice, FADT_PM_TIMER_BLOCK_OFFSET);
    info.flags = read_u32(table_slice, FADT_FLAGS_OFFSET);
    info.x_pm_timer_block = read_u64(table_slice, FADT_X_PM_TIMER_BLOCK_OFFSET);

    Ok(info)
}

const FADT_FIRMWARE_CTRL_OFFSET: usize = mem::size_of::<SdtHeader>();
const FADT_DSDT_OFFSET: usize = FADT_FIRMWARE_CTRL_OFFSET + 4;
const FADT_SCI_INTERRUPT_OFFSET: usize = FADT_FIRMWARE_CTRL_OFFSET + 8 + 1 + 1;
const FADT_SMI_CMD_OFFSET: usize = FADT_FIRMWARE_CTRL_OFFSET + 12;
const FADT_ACPI_ENABLE_OFFSET: usize = FADT_SMI_CMD_OFFSET + 4;
const FADT_ACPI_DISABLE_OFFSET: usize = FADT_ACPI_ENABLE_OFFSET + 1;
const FADT_PM1A_CONTROL_OFFSET: usize = FADT_FIRMWARE_CTRL_OFFSET + 24;
const FADT_PM_TIMER_BLOCK_OFFSET: usize = FADT_FIRMWARE_CTRL_OFFSET + 40;
const FADT_FLAGS_OFFSET: usize = FADT_FIRMWARE_CTRL_OFFSET + 44;
const FADT_X_PM_TIMER_BLOCK_OFFSET: usize = FADT_FIRMWARE_CTRL_OFFSET + 76;

fn read_u8(data: &[u8], offset: usize) -> Option<u8> {
    data.get(offset).copied()
}

fn read_u16(data: &[u8], offset: usize) -> Option<u16> {
    if offset + 2 <= data.len() {
        Some(u16::from_le_bytes([data[offset], data[offset + 1]]))
    } else {
        None
    }
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    if offset + 4 <= data.len() {
        Some(u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]))
    } else {
        None
    }
}

fn read_u64(data: &[u8], offset: usize) -> Option<u64> {
    if offset + 8 <= data.len() {
        Some(u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]))
    } else {
        None
    }
}

/// Parse the Memory Mapped Configuration (MCFG) table for PCIe MMCONFIG
pub fn parse_mcfg() -> Result<McfgInfo, &'static str> {
    let descriptor = find_table(b"MCFG").ok_or("MCFG table not found")?;

    let virt = descriptor
        .virt_addr
        .or_else(|| {
            acpi_info()
                .and_then(|info| info.physical_memory_offset)
                .and_then(|offset| phys_to_virt(descriptor.phys_addr, offset))
        })
        .ok_or("Failed to map MCFG virtual address")?;

    let info = unsafe { parse_mcfg_from_address(virt, descriptor.length as usize) }?;

    {
        let mut state = ACPI_STATE.write();
        if let Some(acpi) = state.as_mut() {
            acpi.mcfg = Some(info.clone());
        }
    }

    Ok(info)
}

unsafe fn parse_mcfg_from_address(virt_addr: usize, table_length: usize) -> Result<McfgInfo, &'static str> {
    if table_length < mem::size_of::<SdtHeader>() + 8 {
        return Err("MCFG table too short");
    }

    let table_slice = slice::from_raw_parts(virt_addr as *const u8, table_length);
    if !checksum_bytes(table_slice) {
        return Err("MCFG checksum validation failed");
    }

    let header_size = mem::size_of::<SdtHeader>();
    let reserved_size = 8; // 8 bytes reserved after header
    let entry_start = header_size + reserved_size;
    
    if table_length < entry_start {
        return Err("MCFG has no entries");
    }

    let entries_data = &table_slice[entry_start..];
    let entry_size = 16; // Each MCFG entry is 16 bytes
    let entry_count = entries_data.len() / entry_size;

    let mut entries = Vec::with_capacity(entry_count);

    for i in 0..entry_count {
        let entry_offset = i * entry_size;
        if entry_offset + entry_size > entries_data.len() {
            break;
        }

        let entry_type = entries_data[entry_offset];
        let entry_length = entries_data[entry_offset + 1] as usize;

        match entry_type {
            MADT_ENTRY_PROCESSOR if entry_length >= MADT_PROCESSOR_LEN => {
                let processor_id = entries_data[entry_offset + 2];
                let apic_id = entries_data[entry_offset + 3];
                let flags = u32::from_le_bytes([
                    entries_data[entry_offset + 4],
                    entries_data[entry_offset + 5],
                    entries_data[entry_offset + 6],
                    entries_data[entry_offset + 7],
                ]);

                info.processors.push(ProcessorInfo {
                    processor_id,
                    apic_id,
                    flags,
                });
            }
            MADT_ENTRY_IO_APIC if entry_length >= MADT_IO_APIC_LEN => {
                let id = entries_data[entry_offset + 2];
                let address = u32::from_le_bytes([
                    entries_data[entry_offset + 4],
                    entries_data[entry_offset + 5],
                    entries_data[entry_offset + 6],
                    entries_data[entry_offset + 7],
                ]);

                let si_base = u32::from_le_bytes([
                    entries_data[entry_offset + 8],
                    entries_data[entry_offset + 9],
                    entries_data[entry_offset + 10],
                    entries_data[entry_offset + 11],
                ]);

                info.io_apics.push(IoApic {
                    id,
                    address,
                    global_system_interrupt_base: si_base,
                });
            }
            MADT_ENTRY_INTERRUPT_OVERRIDE if entry_length >= MADT_INTERRUPT_OVERRIDE_LEN => {
                let bus_source = entries_data[entry_offset + 2];
                let irq_source = entries_data[entry_offset + 3];
                let si = u32::from_le_bytes([
                    entries_data[entry_offset + 4],
                    entries_data[entry_offset + 5],
                    entries_data[entry_offset + 6],
                    entries_data[entry_offset + 7],
                ]);

                let flags = u16::from_le_bytes([
                    entries_data[entry_offset + 8],
                    entries_data[entry_offset + 9],
                ]);

                info.interrupt_overrides.push(InterruptOverride {
                    bus_source,
                    irq_source,
                    global_system_interrupt: si,
                    flags,
                });
            }
            _ => {}
        }
    }

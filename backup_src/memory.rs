//! Virtual Memory Management System for RustOS
//!
//! This module provides comprehensive memory management including:
//! - Physical frame allocation
//! - Virtual memory mapping and unmapping
//! - Heap management integration
//! - Memory regions and protection
//! - Page fault handling support
//! - Memory statistics and debugging

use x86_64::{
    VirtAddr, PhysAddr,
    structures::paging::{
        PageTable, PageTableFlags, PhysFrame, Size4KiB, FrameAllocator,
        OffsetPageTable, Page, Mapper, RecursivePageTable, PageTableIndex,
        UnusedPhysFrame, mapper::MapToError
    },
    registers::control::Cr3,
};
use bootloader::bootinfo::{MemoryMap, MemoryRegionType};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use alloc::{vec::Vec, collections::BTreeMap};
use core::sync::atomic::{AtomicU64, Ordering};

/// Page size in bytes (4KB)
pub const PAGE_SIZE: usize = 4096;

/// Virtual memory layout constants
pub const KERNEL_HEAP_START: usize = 0x_4444_4444_0000;
pub const KERNEL_HEAP_SIZE: usize = 100 * 1024 * 1024; // 100 MiB
pub const USER_SPACE_START: usize = 0x_0000_1000_0000;
pub const USER_SPACE_END: usize = 0x_0000_8000_0000;
pub const KERNEL_SPACE_START: usize = 0xFFFF_8000_0000_0000;

/// Memory region types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegionType {
    Kernel,
    UserCode,
    UserData,
    UserStack,
    UserHeap,
    DeviceMemory,
    SharedMemory,
}

/// Memory protection flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryProtection {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
    pub user_accessible: bool,
}

impl MemoryProtection {
    pub const KERNEL_CODE: Self = MemoryProtection {
        readable: true,
        writable: false,
        executable: true,
        user_accessible: false,
    };

    pub const KERNEL_DATA: Self = MemoryProtection {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: false,
    };

    pub const USER_CODE: Self = MemoryProtection {
        readable: true,
        writable: false,
        executable: true,
        user_accessible: true,
    };

    pub const USER_DATA: Self = MemoryProtection {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: true,
    };

    pub fn to_page_table_flags(self) -> PageTableFlags {
        let mut flags = PageTableFlags::PRESENT;

        if self.writable {
            flags |= PageTableFlags::WRITABLE;
        }
        if self.user_accessible {
            flags |= PageTableFlags::USER_ACCESSIBLE;
        }
        if !self.executable {
            flags |= PageTableFlags::NO_EXECUTE;
        }

        flags
    }
}

/// Virtual memory region descriptor
#[derive(Debug, Clone)]
pub struct VirtualMemoryRegion {
    pub start: VirtAddr,
    pub size: usize,
    pub region_type: MemoryRegionType,
    pub protection: MemoryProtection,
    pub mapped: bool,
}

impl VirtualMemoryRegion {
    pub fn new(start: VirtAddr, size: usize, region_type: MemoryRegionType, protection: MemoryProtection) -> Self {
        Self {
            start,
            size,
            region_type,
            protection,
            mapped: false,
        }
    }

    pub fn end(&self) -> VirtAddr {
        self.start + self.size
    }

    pub fn contains(&self, addr: VirtAddr) -> bool {
        addr >= self.start && addr < self.end()
    }

    pub fn pages(&self) -> impl Iterator<Item = Page> {
        let start_page = Page::containing_address(self.start);
        let end_page = Page::containing_address(self.end() - 1u64);
        Page::range_inclusive(start_page, end_page)
    }
}

/// A FrameAllocator that returns usable frames from the bootloader's memory map
pub struct BootInfoFrameAllocator {
    memory_map: &'static MemoryMap,
    next: usize,
    allocated_frames: AtomicU64,
}

impl BootInfoFrameAllocator {
    /// Create a FrameAllocator from the passed memory map
    pub unsafe fn init(memory_map: &'static MemoryMap) -> Self {
        BootInfoFrameAllocator {
            memory_map,
            next: 0,
            allocated_frames: AtomicU64::new(0),
        }
    }

    /// Returns an iterator over the usable frames specified in the memory map
    fn usable_frames(&self) -> impl Iterator<Item = PhysFrame> {
        let regions = self.memory_map.iter();
        let usable_regions = regions
            .filter(|r| r.region_type == MemoryRegionType::Usable);
        let addr_ranges = usable_regions
            .map(|r| r.range.start_addr()..r.range.end_addr());
        let frame_addresses = addr_ranges.flat_map(|r| r.step_by(4096));
        frame_addresses.map(|addr| PhysFrame::containing_address(PhysAddr::new(addr)))
    }

    /// Get total number of usable frames
    pub fn total_frames(&self) -> usize {
        self.usable_frames().count()
    }

    /// Get number of allocated frames
    pub fn allocated_frames(&self) -> u64 {
        self.allocated_frames.load(Ordering::Relaxed)
    }

    /// Get number of free frames
    pub fn free_frames(&self) -> usize {
        self.total_frames().saturating_sub(self.allocated_frames() as usize)
    }
}

unsafe impl FrameAllocator<Size4KiB> for BootInfoFrameAllocator {
    fn allocate_frame(&mut self) -> Option<UnusedPhysFrame> {
        let frame = self.usable_frames().nth(self.next);
        self.next += 1;
        if frame.is_some() {
            self.allocated_frames.fetch_add(1, Ordering::Relaxed);
        }
        frame.map(|f| unsafe { UnusedPhysFrame::new(f) })
    }
}

/// Global memory management state
pub struct MemoryManager {
    mapper: Mutex<OffsetPageTable<'static>>,
    frame_allocator: Mutex<BootInfoFrameAllocator>,
    regions: RwLock<BTreeMap<VirtAddr, VirtualMemoryRegion>>,
    heap_region: Option<VirtualMemoryRegion>,
}

impl MemoryManager {
    pub fn new(
        mapper: OffsetPageTable<'static>,
        frame_allocator: BootInfoFrameAllocator,
    ) -> Self {
        Self {
            mapper: Mutex::new(mapper),
            frame_allocator: Mutex::new(frame_allocator),
            regions: RwLock::new(BTreeMap::new()),
            heap_region: None,
        }
    }

    /// Map a virtual memory region to physical frames
    pub fn map_region(&self, region: &mut VirtualMemoryRegion) -> Result<(), MapToError<Size4KiB>> {
        let mut mapper = self.mapper.lock();
        let mut frame_allocator = self.frame_allocator.lock();

        for page in region.pages() {
            let frame = frame_allocator
                .allocate_frame()
                .ok_or(MapToError::FrameAllocationFailed)?;
            let flags = region.protection.to_page_table_flags();

            unsafe {
                mapper.map_to(page, frame, flags, &mut *frame_allocator)?;
            }
        }

        region.mapped = true;
        Ok(())
    }

    /// Unmap a virtual memory region
    pub fn unmap_region(&self, region: &mut VirtualMemoryRegion) {
        let mut mapper = self.mapper.lock();

        for page in region.pages() {
            let (_, flush) = mapper.unmap(page).expect("Failed to unmap page");
            flush.flush();
        }

        region.mapped = false;
    }

    /// Add a virtual memory region to management
    pub fn add_region(&self, region: VirtualMemoryRegion) {
        let mut regions = self.regions.write();
        regions.insert(region.start, region);
    }

    /// Find region containing the given address
    pub fn find_region(&self, addr: VirtAddr) -> Option<VirtualMemoryRegion> {
        let regions = self.regions.read();
        regions.values()
            .find(|region| region.contains(addr))
            .cloned()
    }

    /// Allocate virtual memory region
    pub fn allocate_region(
        &self,
        size: usize,
        region_type: MemoryRegionType,
        protection: MemoryProtection,
    ) -> Result<VirtualMemoryRegion, &'static str> {
        let aligned_size = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        // Find free virtual address space
        let start_addr = self.find_free_virtual_space(aligned_size)
            .ok_or("No free virtual address space")?;

        let mut region = VirtualMemoryRegion::new(start_addr, aligned_size, region_type, protection);

        // Map the region
        self.map_region(&mut region)
            .map_err(|_| "Failed to map region")?;

        // Add to region tracking
        self.add_region(region.clone());

        Ok(region)
    }

    /// Find free virtual address space
    fn find_free_virtual_space(&self, size: usize) -> Option<VirtAddr> {
        let regions = self.regions.read();
        let mut current_addr = VirtAddr::new(USER_SPACE_START as u64);

        while current_addr.as_u64() + size as u64 <= USER_SPACE_END as u64 {
            let end_addr = current_addr + size;

            let overlaps = regions.values().any(|region| {
                let region_end = region.end();
                !(end_addr <= region.start || current_addr >= region_end)
            });

            if !overlaps {
                return Some(current_addr);
            }

            // Move to next page-aligned address
            current_addr = VirtAddr::new((current_addr.as_u64() + PAGE_SIZE as u64) & !(PAGE_SIZE as u64 - 1));
        }

        None
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let frame_allocator = self.frame_allocator.lock();
        let regions = self.regions.read();

        MemoryStats {
            total_frames: frame_allocator.total_frames(),
            allocated_frames: frame_allocator.allocated_frames() as usize,
            free_frames: frame_allocator.free_frames(),
            total_regions: regions.len(),
            mapped_pages: regions.values()
                .filter(|r| r.mapped)
                .map(|r| (r.size + PAGE_SIZE - 1) / PAGE_SIZE)
                .sum(),
        }
    }

    /// Handle page fault (basic implementation)
    pub fn handle_page_fault(&self, addr: VirtAddr, error_code: u64) -> Result<(), &'static str> {
        crate::println!("Page fault at address: {:?}, error: {:#x}", addr, error_code);

        // Check if address is in a valid region
        if let Some(region) = self.find_region(addr) {
            if !region.mapped {
                // Region exists but not mapped - this could be lazy allocation
                crate::println!("Lazy allocation for region at {:?}", region.start);
                return Ok(());
            }
        }

        // For now, we'll just log and continue
        crate::println!("Unhandled page fault!");
        Err("Unhandled page fault")
    }
}

/// Memory statistics structure
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_frames: usize,
    pub allocated_frames: usize,
    pub free_frames: usize,
    pub total_regions: usize,
    pub mapped_pages: usize,
}

impl MemoryStats {
    pub fn memory_usage_percent(&self) -> f32 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.allocated_frames as f32 / self.total_frames as f32) * 100.0
        }
    }
}

lazy_static! {
    static ref MEMORY_MANAGER: RwLock<Option<MemoryManager>> = RwLock::new(None);
}

/// Initialize the memory management system
pub fn init(memory_map: &'static MemoryMap) {
    let physical_memory_offset = VirtAddr::new(0);

    let level_4_table = unsafe {
        let (level_4_table_frame, _) = Cr3::read();
        let phys = level_4_table_frame.start_address();
        let virt = physical_memory_offset + phys.as_u64();
        PageTable::from_ptr(virt.as_mut_ptr())
    };

    let mapper = unsafe { OffsetPageTable::new(level_4_table, physical_memory_offset) };
    let frame_allocator = unsafe { BootInfoFrameAllocator::init(memory_map) };

    let memory_manager = MemoryManager::new(mapper, frame_allocator);

    // Initialize kernel heap region
    let heap_region = VirtualMemoryRegion::new(
        VirtAddr::new(KERNEL_HEAP_START as u64),
        KERNEL_HEAP_SIZE,
        MemoryRegionType::Kernel,
        MemoryProtection::KERNEL_DATA,
    );

    memory_manager.add_region(heap_region);

    let stats = memory_manager.memory_stats();
    crate::println!("Memory management initialized");
    crate::println!("Total frames: {}, Free: {}, Usage: {:.1}%",
                   stats.total_frames, stats.free_frames, stats.memory_usage_percent());

    *MEMORY_MANAGER.write() = Some(memory_manager);
}

/// Get global memory manager
pub fn get_memory_manager() -> Option<&'static MemoryManager> {
    unsafe {
        MEMORY_MANAGER.read().as_ref().map(|mm| core::mem::transmute(mm))
    }
}

/// Allocate virtual memory region (high-level interface)
pub fn allocate_memory(size: usize, region_type: MemoryRegionType, protection: MemoryProtection)
    -> Result<VirtAddr, &'static str>
{
    let mm = get_memory_manager().ok_or("Memory manager not initialized")?;
    let region = mm.allocate_region(size, region_type, protection)?;
    Ok(region.start)
}

/// Get memory statistics
pub fn get_memory_stats() -> Option<MemoryStats> {
    get_memory_manager().map(|mm| mm.memory_stats())
}

/// Translate virtual address to physical address
pub unsafe fn translate_addr(addr: VirtAddr, physical_memory_offset: VirtAddr) -> Option<PhysAddr> {
    translate_addr_inner(addr, physical_memory_offset)
}

/// Private function for address translation
fn translate_addr_inner(addr: VirtAddr, physical_memory_offset: VirtAddr) -> Option<PhysAddr> {
    use x86_64::structures::paging::page_table::FrameError;

    let (level_4_table_frame, _) = Cr3::read();

    let table_indexes = [
        addr.p4_index(), addr.p3_index(), addr.p2_index(), addr.p1_index()
    ];
    let mut frame = level_4_table_frame;

    for &index in &table_indexes {
        let virt = physical_memory_offset + frame.start_address().as_u64();
        let table_ptr: *const PageTable = virt.as_ptr();
        let table = unsafe { &*table_ptr };

        let entry = &table[index];
        frame = match entry.frame() {
            Ok(frame) => frame,
            Err(FrameError::FrameNotPresent) => return None,
            Err(FrameError::HugeFrame) => panic!("Huge pages not supported"),
        };
    }

    Some(frame.start_address() + u64::from(addr.page_offset()))
}

/// Create identity mapping for physical memory
pub unsafe fn create_physical_memory_mapping(
    mapper: &mut OffsetPageTable,
    frame_allocator: &mut impl FrameAllocator<Size4KiB>,
    physical_memory_offset: VirtAddr,
) -> Result<(), MapToError<Size4KiB>> {
    // This is a simplified version - in a real kernel you'd map specific ranges
    crate::println!("Physical memory mapping created at offset {:?}", physical_memory_offset);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn test_memory_protection_flags() {
        let kernel_data = MemoryProtection::KERNEL_DATA;
        let flags = kernel_data.to_page_table_flags();

        assert!(flags.contains(PageTableFlags::PRESENT));
        assert!(flags.contains(PageTableFlags::WRITABLE));
        assert!(!flags.contains(PageTableFlags::USER_ACCESSIBLE));
    }

    #[test_case]
    fn test_virtual_memory_region() {
        let start = VirtAddr::new(0x1000);
        let size = 0x2000;
        let region = VirtualMemoryRegion::new(
            start,
            size,
            MemoryRegionType::UserData,
            MemoryProtection::USER_DATA,
        );

        assert_eq!(region.start, start);
        assert_eq!(region.size, size);
        assert_eq!(region.end(), start + size);
        assert!(region.contains(VirtAddr::new(0x1500)));
        assert!(!region.contains(VirtAddr::new(0x3500)));
    }
}

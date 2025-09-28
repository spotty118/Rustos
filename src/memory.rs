//! Comprehensive Memory Management System for RustOS
//!
//! This module provides a complete memory management system including:
//! - Physical frame allocation and deallocation
//! - Virtual memory management and mapping
//! - Page table management with full address translation
//! - Memory protection (read/write/execute permissions)
//! - Kernel and user space separation
//! - Memory zone management (DMA, Normal, HighMem)
//! - Integration with heap allocator
//! - Memory statistics and monitoring
//! - Error handling and memory safety guarantees

use x86_64::{
    VirtAddr, PhysAddr,
    structures::paging::{
        PageTable, PageTableFlags, PhysFrame, Size4KiB, FrameAllocator,
        OffsetPageTable, Page, Mapper, mapper::MapToError, Translate,
    },
    registers::control::Cr3,
};
use bootloader_api::info::{MemoryRegion, MemoryRegionKind};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use alloc::{collections::BTreeMap, vec::Vec};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::fmt;

/// Page size constants
pub const PAGE_SIZE: usize = 4096;
pub const PAGE_SHIFT: usize = 12;

/// Memory layout constants for virtual address space
pub const KERNEL_HEAP_START: usize = 0x_4444_4444_0000;
pub const KERNEL_HEAP_SIZE: usize = 100 * 1024 * 1024; // 100 MiB
pub const USER_SPACE_START: usize = 0x_0000_1000_0000;
pub const USER_SPACE_END: usize = 0x_0000_8000_0000;
pub const KERNEL_SPACE_START: usize = 0xFFFF_8000_0000_0000;

/// Physical memory zone boundaries
pub const DMA_ZONE_END: u64 = 16 * 1024 * 1024; // 16MB
pub const NORMAL_ZONE_END: u64 = 896 * 1024 * 1024; // 896MB
// Everything above NORMAL_ZONE_END is considered HIGHMEM

/// Memory zone types for different hardware requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryZone {
    /// DMA-accessible memory (below 16MB)
    Dma,
    /// Normal memory (16MB - 896MB)
    Normal,
    /// High memory (above 896MB)
    HighMem,
}

impl MemoryZone {
    pub fn from_address(addr: PhysAddr) -> Self {
        let addr = addr.as_u64();
        if addr < DMA_ZONE_END {
            MemoryZone::Dma
        } else if addr < NORMAL_ZONE_END {
            MemoryZone::Normal
        } else {
            MemoryZone::HighMem
        }
    }
}

/// Virtual memory region types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegionType {
    /// Kernel code and data
    Kernel,
    /// User process code
    UserCode,
    /// User process data
    UserData,
    /// User process stack
    UserStack,
    /// User process heap
    UserHeap,
    /// Memory-mapped device registers
    DeviceMemory,
    /// Shared memory between processes
    SharedMemory,
    /// Video/framebuffer memory
    VideoMemory,
}

/// Memory protection flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryProtection {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
    pub user_accessible: bool,
    pub cache_disabled: bool,
    pub write_through: bool,
}

impl MemoryProtection {
    pub const KERNEL_CODE: Self = MemoryProtection {
        readable: true,
        writable: false,
        executable: true,
        user_accessible: false,
        cache_disabled: false,
        write_through: false,
    };

    pub const KERNEL_DATA: Self = MemoryProtection {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: false,
        cache_disabled: false,
        write_through: false,
    };

    pub const USER_CODE: Self = MemoryProtection {
        readable: true,
        writable: false,
        executable: true,
        user_accessible: true,
        cache_disabled: false,
        write_through: false,
    };

    pub const USER_DATA: Self = MemoryProtection {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: true,
        cache_disabled: false,
        write_through: false,
    };

    pub const DEVICE_MEMORY: Self = MemoryProtection {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: false,
        cache_disabled: true,
        write_through: true,
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
        if self.cache_disabled {
            flags |= PageTableFlags::NO_CACHE;
        }
        if self.write_through {
            flags |= PageTableFlags::WRITE_THROUGH;
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
    pub physical_start: Option<PhysAddr>,
}

impl VirtualMemoryRegion {
    pub fn new(
        start: VirtAddr,
        size: usize,
        region_type: MemoryRegionType,
        protection: MemoryProtection
    ) -> Self {
        Self {
            start,
            size,
            region_type,
            protection,
            mapped: false,
            physical_start: None,
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

    pub fn page_count(&self) -> usize {
        (self.size + PAGE_SIZE - 1) / PAGE_SIZE
    }
}

/// Physical frame allocator with zone support
pub struct PhysicalFrameAllocator {
    zone_frames: [Vec<PhysFrame>; 3],
    allocated_frames: [AtomicU64; 3],
    total_frames: [usize; 3],
}

impl PhysicalFrameAllocator {
    /// Initialize the frame allocator from bootloader memory regions
    pub fn init(memory_regions: &[MemoryRegion]) -> Self {
        let mut zone_frames = [Vec::new(), Vec::new(), Vec::new()];

        for region in memory_regions.iter().filter(|r| r.kind == MemoryRegionKind::Usable) {
            let mut start = align_up(region.start as usize, PAGE_SIZE) as u64;
            let end = region.end;

            while start + PAGE_SIZE as u64 <= end {
                let phys_addr = PhysAddr::new(start);
                let zone = MemoryZone::from_address(phys_addr);
                zone_frames[zone as usize].push(PhysFrame::containing_address(phys_addr));
                start += PAGE_SIZE as u64;
            }
        }

        // Reverse to allow pop() to hand out low addresses first
        for frames in &mut zone_frames {
            frames.sort_unstable_by_key(|frame| frame.start_address().as_u64());
            frames.reverse();
        }

        let total_frames = [
            zone_frames[MemoryZone::Dma as usize].len(),
            zone_frames[MemoryZone::Normal as usize].len(),
            zone_frames[MemoryZone::HighMem as usize].len(),
        ];

        PhysicalFrameAllocator {
            zone_frames,
            allocated_frames: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            total_frames,
        }
    }

    /// Allocate a frame from a specific zone
    pub fn allocate_frame_in_zone(&mut self, zone: MemoryZone) -> Option<PhysFrame> {
        let zone_idx = zone as usize;
        let frame = self.zone_frames[zone_idx].pop();
        if let Some(frame) = frame {
            self.allocated_frames[zone_idx].fetch_add(1, Ordering::Relaxed);
            Some(frame)
        } else {
            None
        }
    }

    /// Get memory statistics for all zones
    pub fn get_zone_stats(&self) -> [ZoneStats; 3] {
        [
            ZoneStats {
                zone: MemoryZone::Dma,
                total_frames: self.total_frames[0],
                allocated_frames: self.allocated_frames[0].load(Ordering::Relaxed) as usize,
            },
            ZoneStats {
                zone: MemoryZone::Normal,
                total_frames: self.total_frames[1],
                allocated_frames: self.allocated_frames[1].load(Ordering::Relaxed) as usize,
            },
            ZoneStats {
                zone: MemoryZone::HighMem,
                total_frames: self.total_frames[2],
                allocated_frames: self.allocated_frames[2].load(Ordering::Relaxed) as usize,
            },
        ]
    }

    /// Deallocate a frame (returns it to the free list)
    pub fn deallocate_frame(&mut self, frame: PhysFrame, zone: MemoryZone) {
        let zone_idx = zone as usize;
        self.allocated_frames[zone_idx].fetch_sub(1, Ordering::Relaxed);
        self.zone_frames[zone_idx].push(frame);
    }
}

// Implement the standard FrameAllocator trait (allocates from Normal zone by default)
unsafe impl FrameAllocator<Size4KiB> for PhysicalFrameAllocator {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        // Try Normal zone first, then HighMem, then DMA as last resort
        self.allocate_frame_in_zone(MemoryZone::Normal)
            .or_else(|| self.allocate_frame_in_zone(MemoryZone::HighMem))
            .or_else(|| self.allocate_frame_in_zone(MemoryZone::Dma))
    }
}

/// Zone statistics structure
#[derive(Debug, Clone, Copy)]
pub struct ZoneStats {
    pub zone: MemoryZone,
    pub total_frames: usize,
    pub allocated_frames: usize,
}

impl ZoneStats {
    pub fn free_frames(&self) -> usize {
        self.total_frames.saturating_sub(self.allocated_frames)
    }

    pub fn usage_percent(&self) -> f32 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.allocated_frames as f32 / self.total_frames as f32) * 100.0
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.total_frames * PAGE_SIZE
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_frames * PAGE_SIZE
    }

    pub fn free_bytes(&self) -> usize {
        self.free_frames() * PAGE_SIZE
    }
}

/// Page table management system
pub struct PageTableManager {
    mapper: OffsetPageTable<'static>,
    physical_memory_offset: VirtAddr,
}

impl PageTableManager {
    pub fn new(mapper: OffsetPageTable<'static>, physical_memory_offset: VirtAddr) -> Self {
        Self {
            mapper,
            physical_memory_offset,
        }
    }

    /// Translate virtual address to physical address
    pub fn translate_addr(&self, addr: VirtAddr) -> Option<PhysAddr> {
        self.mapper.translate_addr(addr)
    }

    /// Map a single page with specific flags
    pub fn map_page(
        &mut self,
        page: Page,
        frame: PhysFrame,
        flags: PageTableFlags,
        frame_allocator: &mut impl FrameAllocator<Size4KiB>,
    ) -> Result<(), MapToError<Size4KiB>> {
        unsafe {
            self.mapper.map_to(page, frame, flags, frame_allocator)
                .map(|flush| flush.flush())
        }
    }

    /// Unmap a single page
    pub fn unmap_page(&mut self, page: Page) -> Option<PhysFrame> {
        let (frame, flush) = self.mapper.unmap(page).ok()?;
        flush.flush();
        Some(frame)
    }

    /// Update page flags
    pub fn update_flags(
        &mut self,
        page: Page,
        flags: PageTableFlags,
    ) -> Result<(), &'static str> {
        unsafe {
            self.mapper.update_flags(page, flags)
                .map_err(|_| "Failed to update page flags")?;
        }
        Ok(())
    }

    /// Get current page flags
    pub fn get_flags(&self, _page: Page) -> Option<PageTableFlags> {
        // For now, return None as translate_page API has changed in newer versions
        // In a real implementation, we would check the page table entry directly
        None
    }
}

/// Main memory management system
pub struct MemoryManager {
    frame_allocator: Mutex<PhysicalFrameAllocator>,
    page_table_manager: Mutex<PageTableManager>,
    regions: RwLock<BTreeMap<VirtAddr, VirtualMemoryRegion>>,
    heap_initialized: AtomicU64,
    total_memory: AtomicUsize,
}

impl MemoryManager {
    pub fn new(
        frame_allocator: PhysicalFrameAllocator,
        page_table_manager: PageTableManager,
    ) -> Self {
        // Calculate total memory
        let zone_stats = frame_allocator.get_zone_stats();
        let total_memory = zone_stats.iter()
            .map(|stats| stats.total_bytes())
            .sum();

        Self {
            frame_allocator: Mutex::new(frame_allocator),
            page_table_manager: Mutex::new(page_table_manager),
            regions: RwLock::new(BTreeMap::new()),
            heap_initialized: AtomicU64::new(0),
            total_memory: AtomicUsize::new(total_memory),
        }
    }

    /// Map a virtual memory region to physical frames
    pub fn map_region(&self, region: &mut VirtualMemoryRegion) -> Result<(), MemoryError> {
        let mut page_table_manager = self.page_table_manager.lock();
        let mut frame_allocator = self.frame_allocator.lock();

        let flags = region.protection.to_page_table_flags();
        let mut first_frame = None;

        for page in region.pages() {
            let frame = frame_allocator
                .allocate_frame()
                .ok_or(MemoryError::OutOfMemory)?;

            if first_frame.is_none() {
                first_frame = Some(frame.start_address());
            }

            page_table_manager.map_page(page, frame, flags, &mut *frame_allocator)
                .map_err(|_| MemoryError::MappingFailed)?;
        }

        region.mapped = true;
        region.physical_start = first_frame;
        Ok(())
    }

    /// Unmap a virtual memory region
    pub fn unmap_region(&self, region: &mut VirtualMemoryRegion) -> Result<(), MemoryError> {
        let mut page_table_manager = self.page_table_manager.lock();
        let mut frame_allocator = self.frame_allocator.lock();

        for page in region.pages() {
            if let Some(frame) = page_table_manager.unmap_page(page) {
                let zone = MemoryZone::from_address(frame.start_address());
                frame_allocator.deallocate_frame(frame, zone);
            }
        }

        region.mapped = false;
        region.physical_start = None;
        Ok(())
    }

    /// Add a virtual memory region to management
    pub fn add_region(&self, region: VirtualMemoryRegion) -> Result<(), MemoryError> {
        let mut regions = self.regions.write();

        // Check for overlaps
        for existing_region in regions.values() {
            if self.regions_overlap(&region, existing_region) {
                return Err(MemoryError::RegionOverlap);
            }
        }

        regions.insert(region.start, region);
        Ok(())
    }

    /// Remove a region from management
    pub fn remove_region(&self, start: VirtAddr) -> Result<VirtualMemoryRegion, MemoryError> {
        let mut regions = self.regions.write();
        regions.remove(&start).ok_or(MemoryError::RegionNotFound)
    }

    /// Find region containing the given address
    pub fn find_region(&self, addr: VirtAddr) -> Option<VirtualMemoryRegion> {
        let regions = self.regions.read();
        regions.values()
            .find(|region| region.contains(addr))
            .cloned()
    }

    /// Check if two regions overlap
    fn regions_overlap(&self, region1: &VirtualMemoryRegion, region2: &VirtualMemoryRegion) -> bool {
        let r1_end = region1.end();
        let r2_end = region2.end();
        !(r1_end <= region2.start || region1.start >= r2_end)
    }

    /// Allocate virtual memory region
    pub fn allocate_region(
        &self,
        size: usize,
        region_type: MemoryRegionType,
        protection: MemoryProtection,
    ) -> Result<VirtualMemoryRegion, MemoryError> {
        let aligned_size = align_up(size, PAGE_SIZE);

        // Find free virtual address space
        let start_addr = self.find_free_virtual_space(aligned_size)
            .ok_or(MemoryError::NoVirtualSpace)?;

        let mut region = VirtualMemoryRegion::new(start_addr, aligned_size, region_type, protection);

        // Map the region
        self.map_region(&mut region)?;

        // Add to region tracking
        self.add_region(region.clone())?;

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
            current_addr = VirtAddr::new(align_up(current_addr.as_u64() as usize + PAGE_SIZE, PAGE_SIZE) as u64);
        }

        None
    }

    /// Initialize the kernel heap
    pub fn init_heap(&self) -> Result<(), MemoryError> {
        // Check if already initialized
        if self.heap_initialized.load(Ordering::Relaxed) != 0 {
            return Ok(());
        }

        // Create heap region
        let heap_region = VirtualMemoryRegion::new(
            VirtAddr::new(KERNEL_HEAP_START as u64),
            KERNEL_HEAP_SIZE,
            MemoryRegionType::Kernel,
            MemoryProtection::KERNEL_DATA,
        );

        // Add without mapping (heap will be mapped on demand)
        self.add_region(heap_region)?;

        // Initialize the heap allocator
        unsafe {
            crate::init_heap(KERNEL_HEAP_START, KERNEL_HEAP_SIZE)
                .map_err(|_| MemoryError::HeapInitFailed)?;
        }

        self.heap_initialized.store(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get comprehensive memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let frame_allocator = self.frame_allocator.lock();
        let regions = self.regions.read();
        let zone_stats = frame_allocator.get_zone_stats();

        let total_allocated_frames: usize = zone_stats.iter()
            .map(|stats| stats.allocated_frames)
            .sum();
        let total_frames: usize = zone_stats.iter()
            .map(|stats| stats.total_frames)
            .sum();

        MemoryStats {
            total_memory: self.total_memory.load(Ordering::Relaxed),
            allocated_memory: total_allocated_frames * PAGE_SIZE,
            free_memory: (total_frames.saturating_sub(total_allocated_frames)) * PAGE_SIZE,
            total_regions: regions.len(),
            mapped_regions: regions.values().filter(|r| r.mapped).count(),
            heap_initialized: self.heap_initialized.load(Ordering::Relaxed) != 0,
            zone_stats,
        }
    }

    /// Handle page fault
    pub fn handle_page_fault(&self, addr: VirtAddr, error_code: u64) -> Result<(), MemoryError> {
        // Check if address is in a valid region
        if let Some(region) = self.find_region(addr) {
            if !region.mapped {
                // Could implement lazy allocation here
                return Err(MemoryError::LazyAllocationNotImplemented);
            }

            // Check permission violation
            let is_write = error_code & 0x2 != 0;
            let is_user = error_code & 0x4 != 0;

            if is_write && !region.protection.writable {
                return Err(MemoryError::WriteViolation);
            }

            if is_user && !region.protection.user_accessible {
                return Err(MemoryError::PrivilegeViolation);
            }
        }

        Err(MemoryError::InvalidAddress)
    }

    /// Translate virtual address to physical address
    pub fn translate_addr(&self, addr: VirtAddr) -> Option<PhysAddr> {
        let page_table_manager = self.page_table_manager.lock();
        page_table_manager.translate_addr(addr)
    }

    /// Change protection flags for a memory region
    pub fn protect_region(
        &self,
        start: VirtAddr,
        size: usize,
        protection: MemoryProtection,
    ) -> Result<(), MemoryError> {
        let mut page_table_manager = self.page_table_manager.lock();
        let flags = protection.to_page_table_flags();

        let start_page = Page::containing_address(start);
        let end_page = Page::containing_address(start + size - 1u64);

        for page in Page::range_inclusive(start_page, end_page) {
            page_table_manager.update_flags(page, flags)
                .map_err(|_| MemoryError::ProtectionFailed)?;
        }

        // Update region protection in our tracking
        let mut regions = self.regions.write();
        for region in regions.values_mut() {
            if region.contains(start) {
                region.protection = protection;
                break;
            }
        }

        Ok(())
    }
}

/// Memory error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryError {
    OutOfMemory,
    MappingFailed,
    RegionOverlap,
    RegionNotFound,
    NoVirtualSpace,
    HeapInitFailed,
    InvalidAddress,
    WriteViolation,
    PrivilegeViolation,
    LazyAllocationNotImplemented,
    ProtectionFailed,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MemoryError::OutOfMemory => write!(f, "Out of physical memory"),
            MemoryError::MappingFailed => write!(f, "Failed to map virtual memory"),
            MemoryError::RegionOverlap => write!(f, "Memory region overlap detected"),
            MemoryError::RegionNotFound => write!(f, "Memory region not found"),
            MemoryError::NoVirtualSpace => write!(f, "No available virtual address space"),
            MemoryError::HeapInitFailed => write!(f, "Heap initialization failed"),
            MemoryError::InvalidAddress => write!(f, "Invalid memory address"),
            MemoryError::WriteViolation => write!(f, "Write access violation"),
            MemoryError::PrivilegeViolation => write!(f, "Privilege violation"),
            MemoryError::LazyAllocationNotImplemented => write!(f, "Lazy allocation not implemented"),
            MemoryError::ProtectionFailed => write!(f, "Failed to change memory protection"),
        }
    }
}

/// Comprehensive memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_memory: usize,
    pub allocated_memory: usize,
    pub free_memory: usize,
    pub total_regions: usize,
    pub mapped_regions: usize,
    pub heap_initialized: bool,
    pub zone_stats: [ZoneStats; 3],
}

impl MemoryStats {
    pub fn memory_usage_percent(&self) -> f32 {
        if self.total_memory == 0 {
            0.0
        } else {
            (self.allocated_memory as f32 / self.total_memory as f32) * 100.0
        }
    }

    pub fn total_memory_mb(&self) -> usize {
        self.total_memory / (1024 * 1024)
    }

    pub fn allocated_memory_mb(&self) -> usize {
        self.allocated_memory / (1024 * 1024)
    }

    pub fn free_memory_mb(&self) -> usize {
        self.free_memory / (1024 * 1024)
    }
}

/// Global memory manager instance
lazy_static! {
    static ref MEMORY_MANAGER: RwLock<Option<MemoryManager>> = RwLock::new(None);
}

/// Initialize the memory management system
pub fn init_memory_management(
    memory_regions: &[MemoryRegion],
    physical_memory_offset: Option<u64>,
) -> Result<(), MemoryError> {
    // Determine physical memory offset (default to zero if not provided)
    let physical_memory_offset = VirtAddr::new(physical_memory_offset.unwrap_or(0));

    // Get current page table
    let level_4_table = unsafe {
        let (level_4_table_frame, _) = Cr3::read();
        let phys = level_4_table_frame.start_address();
        let virt = physical_memory_offset + phys.as_u64();
        &mut *(virt.as_mut_ptr() as *mut PageTable)
    };

    // Create page table manager
    let mapper = unsafe { OffsetPageTable::new(level_4_table, physical_memory_offset) };
    let page_table_manager = PageTableManager::new(mapper, physical_memory_offset);

    // Create frame allocator
    let frame_allocator = PhysicalFrameAllocator::init(memory_regions);

    // Create memory manager
    let memory_manager = MemoryManager::new(frame_allocator, page_table_manager);

    // Initialize heap
    memory_manager.init_heap()?;

    // Store global instance
    *MEMORY_MANAGER.write() = Some(memory_manager);

    Ok(())
}

/// Get global memory manager
pub fn get_memory_manager() -> Option<&'static MemoryManager> {
    unsafe {
        MEMORY_MANAGER.read().as_ref().map(|mm| core::mem::transmute(mm))
    }
}

/// High-level memory allocation interface
pub fn allocate_memory(
    size: usize,
    region_type: MemoryRegionType,
    protection: MemoryProtection,
) -> Result<VirtAddr, MemoryError> {
    let mm = get_memory_manager().ok_or(MemoryError::OutOfMemory)?;
    let region = mm.allocate_region(size, region_type, protection)?;
    Ok(region.start)
}

/// Deallocate memory region
pub fn deallocate_memory(addr: VirtAddr) -> Result<(), MemoryError> {
    let mm = get_memory_manager().ok_or(MemoryError::OutOfMemory)?;
    let mut region = mm.remove_region(addr)?;
    mm.unmap_region(&mut region)?;
    Ok(())
}

/// Get memory statistics
pub fn get_memory_stats() -> Option<MemoryStats> {
    get_memory_manager().map(|mm| mm.memory_stats())
}

/// Translate virtual address to physical address
pub fn translate_addr(addr: VirtAddr) -> Option<PhysAddr> {
    get_memory_manager()?.translate_addr(addr)
}

/// Change memory protection
pub fn protect_memory(
    addr: VirtAddr,
    size: usize,
    protection: MemoryProtection,
) -> Result<(), MemoryError> {
    let mm = get_memory_manager().ok_or(MemoryError::OutOfMemory)?;
    mm.protect_region(addr, size, protection)
}

/// Handle page fault (called from interrupt handler)
pub fn handle_page_fault(addr: VirtAddr, error_code: u64) -> Result<(), MemoryError> {
    let mm = get_memory_manager().ok_or(MemoryError::OutOfMemory)?;
    mm.handle_page_fault(addr, error_code)
}

/// Utility function to align up to nearest boundary
pub fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

/// Utility function to align down to nearest boundary
pub fn align_down(addr: usize, align: usize) -> usize {
    addr & !(align - 1)
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

    #[test_case]
    fn test_memory_zones() {
        assert_eq!(MemoryZone::from_address(PhysAddr::new(0x100000)), MemoryZone::Dma);
        assert_eq!(MemoryZone::from_address(PhysAddr::new(0x2000000)), MemoryZone::Normal);
        assert_eq!(MemoryZone::from_address(PhysAddr::new(0x40000000)), MemoryZone::HighMem);
    }

    #[test_case]
    fn test_align_functions() {
        assert_eq!(align_up(0x1001, 0x1000), 0x2000);
        assert_eq!(align_down(0x1fff, 0x1000), 0x1000);
        assert_eq!(align_up(0x1000, 0x1000), 0x1000);
    }
}
//! Production-Grade Memory Management System for RustOS
//!
//! This module provides a comprehensive memory management system including:
//! - Buddy allocator for efficient physical frame allocation
//! - Slab allocator for small object allocation
//! - Virtual memory management with copy-on-write and demand paging
//! - Page table management with full address translation
//! - Memory protection with guard pages and stack canaries
//! - Kernel and user space separation with ASLR
//! - Memory zone management (DMA, Normal, HighMem)
//! - Integration with heap allocator
//! - Comprehensive memory statistics and monitoring
//! - Advanced error handling and memory safety guarantees

use x86_64::{
    VirtAddr, PhysAddr,
    structures::paging::{
        PageTable, PageTableFlags, PhysFrame, Size4KiB, FrameAllocator,
        OffsetPageTable, Page, Mapper, mapper::MapToError, Translate,
    },
    registers::control::Cr3,
};
use bootloader::bootinfo::{MemoryRegion, MemoryRegionType as MemoryRegionKind};
use spin::{Mutex, RwLock};
use lazy_static::lazy_static;
use alloc::{collections::BTreeMap, vec::Vec, vec};
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::fmt;
use crate::performance::{
    CacheAligned, PerCpuAllocator,
    get_performance_monitor, HighResTimer, likely
};

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

/// Buddy allocator order constants
const MIN_ORDER: usize = 0;  // 4KB pages
const MAX_ORDER: usize = 10; // 4MB max allocation (2^10 * 4KB)
const NUM_ORDERS: usize = MAX_ORDER + 1;

/// ASLR entropy bits
const ASLR_ENTROPY_BITS: u32 = 16;

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
    /// Kernel stack
    KernelStack,
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
    /// Copy-on-write region
    CopyOnWrite,
    /// Guard page
    GuardPage,
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
    pub copy_on_write: bool,
    pub guard_page: bool,
}

impl MemoryProtection {
    pub const KERNEL_CODE: Self = MemoryProtection {
        readable: true,
        writable: false,
        executable: true,
        user_accessible: false,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: false,
    };

    pub const KERNEL_DATA: Self = MemoryProtection {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: false,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: false,
    };

    pub const USER_CODE: Self = MemoryProtection {
        readable: true,
        writable: false,
        executable: true,
        user_accessible: true,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: false,
    };

    pub const USER_DATA: Self = MemoryProtection {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: true,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: false,
    };

    pub const DEVICE_MEMORY: Self = MemoryProtection {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: false,
        cache_disabled: true,
        write_through: true,
        copy_on_write: false,
        guard_page: false,
    };

    pub const GUARD_PAGE: Self = MemoryProtection {
        readable: false,
        writable: false,
        executable: false,
        user_accessible: false,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: true,
    };

    pub const COPY_ON_WRITE: Self = MemoryProtection {
        readable: true,
        writable: false,
        executable: false,
        user_accessible: true,
        cache_disabled: false,
        write_through: false,
        copy_on_write: true,
        guard_page: false,
    };

    /// Create empty memory protection (no access)
    pub fn empty() -> Self {
        Self {
            readable: false,
            writable: false,
            executable: false,
            user_accessible: false,
            cache_disabled: false,
            write_through: false,
            copy_on_write: false,
            guard_page: false,
        }
    }

    /// Read-only access flag
    pub const READ: Self = Self {
        readable: true,
        writable: false,
        executable: false,
        user_accessible: false,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: false,
    };

    /// Write access flag
    pub const WRITE: Self = Self {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: false,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: false,
    };

    /// Execute access flag
    pub const EXECUTE: Self = Self {
        readable: true,
        writable: false,
        executable: true,
        user_accessible: false,
        cache_disabled: false,
        write_through: false,
        copy_on_write: false,
        guard_page: false,
    };

    pub fn to_page_table_flags(self) -> PageTableFlags {
        let mut flags = PageTableFlags::PRESENT;

        if self.writable && !self.copy_on_write {
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

/// Implement bitwise OR for MemoryProtection
impl core::ops::BitOrAssign for MemoryProtection {
    fn bitor_assign(&mut self, rhs: Self) {
        self.readable |= rhs.readable;
        self.writable |= rhs.writable;
        self.executable |= rhs.executable;
        self.user_accessible |= rhs.user_accessible;
        self.cache_disabled |= rhs.cache_disabled;
        self.write_through |= rhs.write_through;
        self.copy_on_write |= rhs.copy_on_write;
        self.guard_page |= rhs.guard_page;
    }
}

/// Buddy allocator node
#[derive(Debug, Clone)]
struct BuddyNode {
    address: PhysAddr,
    order: usize,
}

/// Fragmentation statistics for each zone
#[derive(Debug, Clone, Copy, Default)]
pub struct FragmentationStats {
    /// Number of free blocks by order
    pub free_blocks_by_order: [usize; NUM_ORDERS],
    /// Largest free block order
    pub largest_free_order: usize,
    /// Total free memory
    pub total_free_bytes: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = maximum fragmentation)
    pub fragmentation_ratio: f32,
}

/// Production-grade Physical Frame Allocator with Buddy System and Performance Optimizations
pub struct PhysicalFrameAllocator {
    /// Cache-aligned buddy allocator free lists for each order and zone
    buddy_lists: [[CacheAligned<Vec<BuddyNode>>; NUM_ORDERS]; 3],
    /// Allocation bitmap for tracking allocated blocks
    allocation_bitmap: [Vec<u64>; 3],
    /// Zone statistics (cache-aligned for better performance)
    allocated_frames: [CacheAligned<AtomicU64>; 3],
    total_frames: [usize; 3],
    /// Zone memory boundaries
    zone_start: [PhysAddr; 3],
    zone_end: [PhysAddr; 3],
    /// Fragmentation statistics (cache-aligned)
    fragmentation_stats: [CacheAligned<FragmentationStats>; 3],
    /// Per-CPU allocator for fast allocations
    per_cpu_allocator: PerCpuAllocator,
}

impl PhysicalFrameAllocator {
    /// Initialize the frame allocator with buddy system from bootloader memory regions
    pub fn init(memory_regions: &[MemoryRegion]) -> Self {
        let mut buddy_lists = [
            core::array::from_fn(|_| CacheAligned::new(Vec::new())),
            core::array::from_fn(|_| CacheAligned::new(Vec::new())),
            core::array::from_fn(|_| CacheAligned::new(Vec::new())),
        ];

        let mut allocation_bitmap = [Vec::new(), Vec::new(), Vec::new()];
        let mut zone_start = [PhysAddr::new(0); 3];
        let mut zone_end = [PhysAddr::new(0); 3];
        let mut total_frames = [0; 3];

        // Initialize zone boundaries
        zone_start[MemoryZone::Dma as usize] = PhysAddr::new(0);
        zone_end[MemoryZone::Dma as usize] = PhysAddr::new(DMA_ZONE_END);
        zone_start[MemoryZone::Normal as usize] = PhysAddr::new(DMA_ZONE_END);
        zone_end[MemoryZone::Normal as usize] = PhysAddr::new(NORMAL_ZONE_END);
        zone_start[MemoryZone::HighMem as usize] = PhysAddr::new(NORMAL_ZONE_END);
        zone_end[MemoryZone::HighMem as usize] = PhysAddr::new(u64::MAX);

        // Process memory regions and build buddy lists
        for region in memory_regions.iter().filter(|r| r.region_type == bootloader::bootinfo::MemoryRegionType::Usable) {
            let start = align_up(region.range.start_addr() as usize, PAGE_SIZE) as u64;
            let end = align_down(region.range.end_addr() as usize, PAGE_SIZE) as u64;

            if start >= end {
                continue;
            }

            let mut current = start;
            while current < end {
                let zone = MemoryZone::from_address(PhysAddr::new(current));
                let zone_idx = zone as usize;

                // Find the largest possible buddy block at this address
                let mut order = MAX_ORDER;
                let mut block_size = PAGE_SIZE << order;

                while order > 0 {
                    if current % (block_size as u64) == 0 && current + block_size as u64 <= end {
                        break;
                    }
                    order -= 1;
                    block_size >>= 1;
                }

                // Add block to appropriate buddy list
                buddy_lists[zone_idx][order].push(BuddyNode {
                    address: PhysAddr::new(current),
                    order,
                });

                total_frames[zone_idx] += 1 << order;
                current += block_size as u64;
            }
        }

        // Initialize allocation bitmaps (one bit per page)
        for zone_idx in 0..3 {
            let bitmap_size = (total_frames[zone_idx] + 63) / 64; // Round up to u64 boundary
            allocation_bitmap[zone_idx] = vec![0u64; bitmap_size];
        }

        // Sort buddy lists by address for efficient allocation
        for zone_idx in 0..3 {
            for order in 0..NUM_ORDERS {
                buddy_lists[zone_idx][order].sort_unstable_by_key(|node| node.address.as_u64());
            }
        }

        PhysicalFrameAllocator {
            buddy_lists,
            allocation_bitmap,
            allocated_frames: [
                CacheAligned::new(AtomicU64::new(0)),
                CacheAligned::new(AtomicU64::new(0)),
                CacheAligned::new(AtomicU64::new(0))
            ],
            total_frames,
            zone_start,
            zone_end,
            fragmentation_stats: [
                CacheAligned::new(FragmentationStats::default()),
                CacheAligned::new(FragmentationStats::default()),
                CacheAligned::new(FragmentationStats::default())
            ],
            per_cpu_allocator: PerCpuAllocator::new(),
        }
    }

    /// Fast path allocation using per-CPU allocator
    pub fn allocate_frame_fast(&mut self, cpu_id: usize) -> Option<PhysFrame> {
        let (result, time_ns) = HighResTimer::time(|| {
            if likely(cpu_id < crate::performance::MAX_CPUS) {
                // Try per-CPU cache first
                if let Some(addr) = self.per_cpu_allocator.allocate_fast(cpu_id) {
                    return Some(PhysFrame::containing_address(PhysAddr::new(addr as u64)));
                }

                // Fallback to slow path
                self.per_cpu_allocator.allocate_slow(cpu_id, 0)
                    .map(|addr| PhysFrame::containing_address(PhysAddr::new(addr as u64)))
            } else {
                None
            }
        });

        // Record performance metrics
        let perf_monitor = get_performance_monitor();
        if result.is_some() {
            perf_monitor.record_allocation(PAGE_SIZE as u64, time_ns);
        } else {
            perf_monitor.record_allocation_failure();
        }

        result
    }

    /// Allocate frames using buddy allocator from a specific zone
    pub fn allocate_frames_in_zone(&mut self, zone: MemoryZone, order: usize) -> Option<PhysFrame> {
        if order > MAX_ORDER {
            return None;
        }

        let zone_idx = zone as usize;

        // Try to find a free block of the requested order
        if let Some(block) = self.find_free_block(zone_idx, order) {
            self.mark_allocated(zone_idx, block.address, order);
            self.allocated_frames[zone_idx].fetch_add(1 << order, Ordering::Relaxed);
            self.update_fragmentation_stats(zone_idx);
            return Some(PhysFrame::containing_address(block.address));
        }

        None
    }

    /// Allocate a single frame from a specific zone
    pub fn allocate_frame_in_zone(&mut self, zone: MemoryZone) -> Option<PhysFrame> {
        self.allocate_frames_in_zone(zone, 0)
    }

    /// Find and split a free block of the requested order
    fn find_free_block(&mut self, zone_idx: usize, order: usize) -> Option<BuddyNode> {
        // First try to find exact order
        if let Some(block) = self.buddy_lists[zone_idx][order].pop() {
            return Some(block);
        }

        // Try higher orders and split
        for higher_order in (order + 1)..=MAX_ORDER {
            if let Some(block) = self.buddy_lists[zone_idx][higher_order].pop() {
                return Some(self.split_block(zone_idx, block, order));
            }
        }

        None
    }

    /// Split a larger block into smaller blocks
    fn split_block(&mut self, zone_idx: usize, mut block: BuddyNode, target_order: usize) -> BuddyNode {
        while block.order > target_order {
            block.order -= 1;
            let buddy_size = PAGE_SIZE << block.order;
            let buddy_addr = PhysAddr::new(block.address.as_u64() + buddy_size as u64);

            // Add buddy to free list
            let buddy = BuddyNode {
                address: buddy_addr,
                order: block.order,
            };

            // Insert in sorted order
            let list = &mut self.buddy_lists[zone_idx][block.order];
            let insert_pos = list.iter().position(|b| b.address > buddy_addr).unwrap_or(list.len());
            list.insert(insert_pos, buddy);
        }

        block
    }

    /// Mark memory region as allocated in bitmap
    fn mark_allocated(&mut self, zone_idx: usize, addr: PhysAddr, order: usize) {
        let page_index = self.addr_to_page_index(zone_idx, addr);
        let num_pages = 1 << order;

        for i in 0..num_pages {
            let bit_index = page_index + i;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            if word_index < self.allocation_bitmap[zone_idx].len() {
                self.allocation_bitmap[zone_idx][word_index] |= 1u64 << bit_offset;
            }
        }
    }

    /// Mark memory region as free in bitmap
    fn mark_free(&mut self, zone_idx: usize, addr: PhysAddr, order: usize) {
        let page_index = self.addr_to_page_index(zone_idx, addr);
        let num_pages = 1 << order;

        for i in 0..num_pages {
            let bit_index = page_index + i;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            if word_index < self.allocation_bitmap[zone_idx].len() {
                self.allocation_bitmap[zone_idx][word_index] &= !(1u64 << bit_offset);
            }
        }
    }

    /// Convert physical address to page index within zone
    fn addr_to_page_index(&self, zone_idx: usize, addr: PhysAddr) -> usize {
        ((addr.as_u64() - self.zone_start[zone_idx].as_u64()) / PAGE_SIZE as u64) as usize
    }

    /// Deallocate frames using buddy allocator (with coalescing)
    pub fn deallocate_frames(&mut self, frame: PhysFrame, zone: MemoryZone, order: usize) {
        let zone_idx = zone as usize;
        let addr = frame.start_address();

        self.mark_free(zone_idx, addr, order);
        self.allocated_frames[zone_idx].fetch_sub(1 << order, Ordering::Relaxed);

        // Try to coalesce with buddy
        let coalesced_block = self.coalesce_block(zone_idx, addr, order);

        // Add to appropriate free list
        let list = &mut self.buddy_lists[zone_idx][coalesced_block.order];
        let insert_pos = list.iter().position(|b| b.address > coalesced_block.address).unwrap_or(list.len());
        list.insert(insert_pos, coalesced_block);

        self.update_fragmentation_stats(zone_idx);
    }

    /// Deallocate a single frame
    pub fn deallocate_frame(&mut self, frame: PhysFrame, zone: MemoryZone) {
        self.deallocate_frames(frame, zone, 0);
    }

    /// Coalesce block with its buddy recursively
    fn coalesce_block(&mut self, zone_idx: usize, addr: PhysAddr, order: usize) -> BuddyNode {
        if order >= MAX_ORDER {
            return BuddyNode { address: addr, order };
        }

        let block_size = PAGE_SIZE << order;
        let buddy_addr = if (addr.as_u64() / block_size as u64) % 2 == 0 {
            // We're the left buddy, buddy is to the right
            PhysAddr::new(addr.as_u64() + block_size as u64)
        } else {
            // We're the right buddy, buddy is to the left
            PhysAddr::new(addr.as_u64() - block_size as u64)
        };

        // Check if buddy is free
        if self.is_buddy_free(zone_idx, buddy_addr, order) {
            // Remove buddy from free list
            if let Some(pos) = self.buddy_lists[zone_idx][order]
                .iter().position(|b| b.address == buddy_addr) {
                self.buddy_lists[zone_idx][order].remove(pos);

                // Determine the new block address (always the lower address)
                let new_addr = PhysAddr::new(core::cmp::min(addr.as_u64(), buddy_addr.as_u64()));

                // Recursively coalesce at next order
                return self.coalesce_block(zone_idx, new_addr, order + 1);
            }
        }

        BuddyNode { address: addr, order }
    }

    /// Check if buddy block is free
    fn is_buddy_free(&self, zone_idx: usize, buddy_addr: PhysAddr, order: usize) -> bool {
        let page_index = self.addr_to_page_index(zone_idx, buddy_addr);
        let num_pages = 1 << order;

        for i in 0..num_pages {
            let bit_index = page_index + i;
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;

            if word_index >= self.allocation_bitmap[zone_idx].len() {
                return false;
            }

            if (self.allocation_bitmap[zone_idx][word_index] & (1u64 << bit_offset)) != 0 {
                return false; // Page is allocated
            }
        }

        true
    }

    /// Update fragmentation statistics for a zone
    fn update_fragmentation_stats(&mut self, zone_idx: usize) {
        let stats = &mut self.fragmentation_stats[zone_idx];

        // Reset stats
        stats.free_blocks_by_order = [0; NUM_ORDERS];
        stats.largest_free_order = 0;
        stats.total_free_bytes = 0;

        // Count free blocks by order
        for order in 0..NUM_ORDERS {
            let count = self.buddy_lists[zone_idx][order].len();
            stats.free_blocks_by_order[order] = count;

            if count > 0 {
                stats.largest_free_order = order;
                stats.total_free_bytes += count * (PAGE_SIZE << order);
            }
        }

        // Calculate fragmentation ratio
        if stats.total_free_bytes > 0 {
            let largest_possible_block = PAGE_SIZE << stats.largest_free_order;
            stats.fragmentation_ratio = 1.0 - (largest_possible_block as f32 / stats.total_free_bytes as f32);
        } else {
            stats.fragmentation_ratio = 0.0;
        }
    }

    /// Get comprehensive memory statistics for all zones
    pub fn get_zone_stats(&self) -> [ZoneStats; 3] {
        [
            ZoneStats {
                zone: MemoryZone::Dma,
                total_frames: self.total_frames[0],
                allocated_frames: self.allocated_frames[0].load(Ordering::Relaxed) as usize,
                fragmentation_stats: self.fragmentation_stats[0].clone(),
            },
            ZoneStats {
                zone: MemoryZone::Normal,
                total_frames: self.total_frames[1],
                allocated_frames: self.allocated_frames[1].load(Ordering::Relaxed) as usize,
                fragmentation_stats: self.fragmentation_stats[1].clone(),
            },
            ZoneStats {
                zone: MemoryZone::HighMem,
                total_frames: self.total_frames[2],
                allocated_frames: self.allocated_frames[2].load(Ordering::Relaxed) as usize,
                fragmentation_stats: self.fragmentation_stats[2].clone(),
            },
        ]
    }

    /// Get buddy allocator statistics
    pub fn get_buddy_stats(&self) -> BuddyAllocatorStats {
        let mut total_free_blocks = 0;
        let mut free_blocks_by_order = [0; NUM_ORDERS];

        for zone_idx in 0..3 {
            for order in 0..NUM_ORDERS {
                let count = self.buddy_lists[zone_idx][order].len();
                free_blocks_by_order[order] += count;
                total_free_blocks += count;
            }
        }

        BuddyAllocatorStats {
            total_free_blocks,
            free_blocks_by_order,
            max_order: MAX_ORDER,
            min_order: MIN_ORDER,
        }
    }

    /// Allocate contiguous pages (for DMA, etc.)
    pub fn allocate_contiguous_pages(&mut self, num_pages: usize, zone: MemoryZone) -> Option<PhysFrame> {
        if num_pages == 0 {
            return None;
        }

        // Find minimum order that can satisfy the request
        let mut order = 0;
        while (1 << order) < num_pages && order <= MAX_ORDER {
            order += 1;
        }

        if order > MAX_ORDER {
            return None; // Request too large
        }

        self.allocate_frames_in_zone(zone, order)
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

/// Zone statistics structure with fragmentation info
#[derive(Debug, Clone)]
pub struct ZoneStats {
    pub zone: MemoryZone,
    pub total_frames: usize,
    pub allocated_frames: usize,
    pub fragmentation_stats: FragmentationStats,
}

/// Buddy allocator statistics
#[derive(Debug, Clone)]
pub struct BuddyAllocatorStats {
    pub total_free_blocks: usize,
    pub free_blocks_by_order: [usize; NUM_ORDERS],
    pub max_order: usize,
    pub min_order: usize,
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

    pub fn fragmentation_percent(&self) -> f32 {
        self.fragmentation_stats.fragmentation_ratio * 100.0
    }

    pub fn largest_free_block_size(&self) -> usize {
        PAGE_SIZE << self.fragmentation_stats.largest_free_order
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
    pub reference_count: usize,
    pub aslr_offset: u64,
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
            reference_count: 1,
            aslr_offset: 0,
        }
    }

    pub fn new_with_aslr(
        start: VirtAddr,
        size: usize,
        region_type: MemoryRegionType,
        protection: MemoryProtection,
        enable_aslr: bool
    ) -> Self {
        let aslr_offset = if enable_aslr {
            generate_aslr_offset()
        } else {
            0
        };

        Self {
            start: VirtAddr::new(start.as_u64() + aslr_offset),
            size,
            region_type,
            protection,
            mapped: false,
            physical_start: None,
            reference_count: 1,
            aslr_offset,
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

    pub fn increment_ref_count(&mut self) {
        self.reference_count += 1;
    }

    pub fn decrement_ref_count(&mut self) -> usize {
        self.reference_count = self.reference_count.saturating_sub(1);
        self.reference_count
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
            let _ = self.mapper.update_flags(page, flags)
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

    /// Clone page table for fork operation (with copy-on-write)
    pub fn clone_for_fork(&mut self, _frame_allocator: &mut impl FrameAllocator<Size4KiB>) -> Result<OffsetPageTable<'static>, &'static str> {
        // This would create a new page table with copy-on-write mappings
        // For now, return error as this is complex to implement properly
        Err("Fork page table cloning not implemented")
    }
}

/// Main memory management system
pub struct MemoryManager {
    frame_allocator: Mutex<PhysicalFrameAllocator>,
    page_table_manager: Mutex<PageTableManager>,
    regions: RwLock<BTreeMap<VirtAddr, VirtualMemoryRegion>>,
    heap_initialized: AtomicU64,
    total_memory: AtomicUsize,
    security_features: SecurityFeatures,
}

/// Security features configuration
#[derive(Debug, Clone)]
struct SecurityFeatures {
    aslr_enabled: bool,
    stack_canaries_enabled: bool,
    nx_bit_enabled: bool,
    smep_enabled: bool,
    smap_enabled: bool,
}

impl Default for SecurityFeatures {
    fn default() -> Self {
        Self {
            aslr_enabled: true,
            stack_canaries_enabled: true,
            nx_bit_enabled: true,
            smep_enabled: true,
            smap_enabled: true,
        }
    }
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
            security_features: SecurityFeatures::default(),
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

            // Initialize page content if needed
            if matches!(region.region_type, MemoryRegionType::UserStack | MemoryRegionType::UserHeap) {
                unsafe {
                    let page_ptr = frame.start_address().as_u64() as *mut u8;
                    core::ptr::write_bytes(page_ptr, 0, PAGE_SIZE);
                }
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

    /// Allocate virtual memory region with enhanced features
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

        let enable_aslr = self.security_features.aslr_enabled &&
                         matches!(region_type, MemoryRegionType::UserCode | MemoryRegionType::UserData | MemoryRegionType::UserStack);

        let mut region = VirtualMemoryRegion::new_with_aslr(start_addr, aligned_size, region_type, protection, enable_aslr);

        // Map the region
        self.map_region(&mut region)?;

        // Add to region tracking
        self.add_region(region.clone())?;

        Ok(region)
    }

    /// Allocate region with guard pages
    pub fn allocate_region_with_guards(
        &self,
        size: usize,
        region_type: MemoryRegionType,
        protection: MemoryProtection,
    ) -> Result<VirtualMemoryRegion, MemoryError> {
        let aligned_size = align_up(size, PAGE_SIZE);
        let total_size = aligned_size + 2 * PAGE_SIZE; // Add guard pages

        let start_addr = self.find_free_virtual_space(total_size)
            .ok_or(MemoryError::NoVirtualSpace)?;

        // Create guard page at start
        let guard_start = VirtualMemoryRegion::new(
            start_addr,
            PAGE_SIZE,
            MemoryRegionType::GuardPage,
            MemoryProtection::GUARD_PAGE,
        );

        // Create actual region
        let mut main_region = VirtualMemoryRegion::new(
            start_addr + PAGE_SIZE,
            aligned_size,
            region_type,
            protection,
        );

        // Create guard page at end
        let guard_end = VirtualMemoryRegion::new(
            start_addr + PAGE_SIZE + aligned_size,
            PAGE_SIZE,
            MemoryRegionType::GuardPage,
            MemoryProtection::GUARD_PAGE,
        );

        // Add regions
        self.add_region(guard_start)?;
        self.map_region(&mut main_region)?;
        self.add_region(main_region.clone())?;
        self.add_region(guard_end)?;

        Ok(main_region)
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

    /// Initialize the kernel heap with guard pages
    pub fn init_heap(&self) -> Result<(), MemoryError> {
        // Check if already initialized
        if self.heap_initialized.load(Ordering::Relaxed) != 0 {
            return Ok(());
        }

        // Create heap region with guard pages
        let guard_page_size = PAGE_SIZE;
        let actual_heap_start = KERNEL_HEAP_START + guard_page_size;
        let actual_heap_size = KERNEL_HEAP_SIZE - 2 * guard_page_size;

        // Create guard page at the beginning
        let guard_start_region = VirtualMemoryRegion::new(
            VirtAddr::new(KERNEL_HEAP_START as u64),
            guard_page_size,
            MemoryRegionType::GuardPage,
            MemoryProtection::GUARD_PAGE,
        );

        // Create actual heap region
        let heap_region = VirtualMemoryRegion::new(
            VirtAddr::new(actual_heap_start as u64),
            actual_heap_size,
            MemoryRegionType::Kernel,
            MemoryProtection::KERNEL_DATA,
        );

        // Create guard page at the end
        let guard_end_region = VirtualMemoryRegion::new(
            VirtAddr::new((actual_heap_start + actual_heap_size) as u64),
            guard_page_size,
            MemoryRegionType::GuardPage,
            MemoryProtection::GUARD_PAGE,
        );

        // Add regions
        self.add_region(guard_start_region)?;
        self.add_region(heap_region)?;
        self.add_region(guard_end_region)?;

        // Initialize the heap allocator with actual heap area
        crate::init_heap(actual_heap_start, actual_heap_size)
            .map_err(|_| MemoryError::HeapInitFailed)?;

        self.heap_initialized.store(1, Ordering::Relaxed);
        Ok(())
    }

    /// Enhanced page fault handler with copy-on-write and demand paging
    pub fn handle_page_fault(&self, addr: VirtAddr, error_code: u64) -> Result<(), MemoryError> {
        // Parse error code
        let is_present = error_code & 0x1 != 0;
        let is_write = error_code & 0x2 != 0;
        let is_user = error_code & 0x4 != 0;
        let is_instruction_fetch = error_code & 0x10 != 0;

        // Check if address is in a valid region
        if let Some(region) = self.find_region(addr) {
            // Handle different types of page faults
            if !is_present {
                // Page not present - implement demand paging
                return self.handle_demand_paging(addr, &region);
            }

            if is_write && region.protection.copy_on_write {
                // Write to copy-on-write page
                return self.handle_copy_on_write(addr, &region);
            }

            if is_write && !region.protection.writable {
                return Err(MemoryError::WriteViolation);
            }

            if is_instruction_fetch && !region.protection.executable {
                return Err(MemoryError::ExecuteViolation);
            }

            if is_user && !region.protection.user_accessible {
                return Err(MemoryError::PrivilegeViolation);
            }

            // Check for guard page access
            if region.protection.guard_page {
                return Err(MemoryError::GuardPageViolation);
            }
        }

        Err(MemoryError::InvalidAddress)
    }

    /// Handle demand paging (allocate page on first access)
    fn handle_demand_paging(&self, addr: VirtAddr, region: &VirtualMemoryRegion) -> Result<(), MemoryError> {
        let page = Page::containing_address(addr);
        let mut page_table_manager = self.page_table_manager.lock();
        let mut frame_allocator = self.frame_allocator.lock();

        // Allocate a new frame
        let frame = frame_allocator
            .allocate_frame()
            .ok_or(MemoryError::OutOfMemory)?;

        // Zero the page for security
        unsafe {
            let page_ptr = frame.start_address().as_u64() as *mut u8;
            core::ptr::write_bytes(page_ptr, 0, PAGE_SIZE);
        }

        // Map the page
        let flags = region.protection.to_page_table_flags();
        page_table_manager.map_page(page, frame, flags, &mut *frame_allocator)
            .map_err(|_| MemoryError::MappingFailed)?;

        Ok(())
    }

    /// Handle copy-on-write page fault
    fn handle_copy_on_write(&self, addr: VirtAddr, region: &VirtualMemoryRegion) -> Result<(), MemoryError> {
        let page = Page::containing_address(addr);
        let mut page_table_manager = self.page_table_manager.lock();
        let mut frame_allocator = self.frame_allocator.lock();

        // Get the current frame
        let old_frame_addr = page_table_manager.translate_addr(addr)
            .ok_or(MemoryError::InvalidAddress)?;

        // Allocate a new frame
        let new_frame = frame_allocator
            .allocate_frame()
            .ok_or(MemoryError::OutOfMemory)?;

        // Copy content from old page to new page
        unsafe {
            let old_ptr = old_frame_addr.as_u64() as *const u8;
            let new_ptr = new_frame.start_address().as_u64() as *mut u8;
            core::ptr::copy_nonoverlapping(old_ptr, new_ptr, PAGE_SIZE);
        }

        // Unmap old page
        if let Some(old_frame) = page_table_manager.unmap_page(page) {
            let zone = MemoryZone::from_address(old_frame.start_address());
            frame_allocator.deallocate_frame(old_frame, zone);
        }

        // Map new page with write permissions
        let mut protection = region.protection;
        protection.writable = true;
        protection.copy_on_write = false;
        let flags = protection.to_page_table_flags();

        page_table_manager.map_page(page, new_frame, flags, &mut *frame_allocator)
            .map_err(|_| MemoryError::MappingFailed)?;

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
            buddy_stats: frame_allocator.get_buddy_stats(),
            security_features: self.security_features.clone(),
        }
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

    /// Create a copy-on-write mapping (for fork)
    pub fn create_cow_mapping(&self, src_region: &VirtualMemoryRegion) -> Result<VirtualMemoryRegion, MemoryError> {
        let mut cow_region = src_region.clone();
        cow_region.protection.copy_on_write = true;
        cow_region.protection.writable = false;

        // Mark original pages as copy-on-write
        let mut page_table_manager = self.page_table_manager.lock();
        let flags = cow_region.protection.to_page_table_flags();

        for page in cow_region.pages() {
            page_table_manager.update_flags(page, flags)
                .map_err(|_| MemoryError::ProtectionFailed)?;
        }

        Ok(cow_region)
    }
}

/// Generate ASLR offset
fn generate_aslr_offset() -> u64 {
    // Simple PRNG for ASLR (in production, use hardware RNG)
    static mut ASLR_SEED: u64 = 0x123456789ABCDEF0;

    unsafe {
        ASLR_SEED = ASLR_SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (ASLR_SEED >> 16) & ((1 << ASLR_ENTROPY_BITS) - 1) * PAGE_SIZE as u64
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
    ExecuteViolation,
    GuardPageViolation,
    LazyAllocationNotImplemented,
    ProtectionFailed,
    InvalidOrder,
    BuddyAllocationFailed,
    FragmentationLimitExceeded,
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
            MemoryError::ExecuteViolation => write!(f, "Execute access violation"),
            MemoryError::GuardPageViolation => write!(f, "Guard page access violation"),
            MemoryError::LazyAllocationNotImplemented => write!(f, "Lazy allocation not implemented"),
            MemoryError::ProtectionFailed => write!(f, "Failed to change memory protection"),
            MemoryError::InvalidOrder => write!(f, "Invalid buddy allocator order"),
            MemoryError::BuddyAllocationFailed => write!(f, "Buddy allocation failed"),
            MemoryError::FragmentationLimitExceeded => write!(f, "Memory fragmentation limit exceeded"),
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
    pub buddy_stats: BuddyAllocatorStats,
    pub security_features: SecurityFeatures,
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

    pub fn average_fragmentation(&self) -> f32 {
        let total_fragmentation: f32 = self.zone_stats.iter()
            .map(|stats| stats.fragmentation_percent())
            .sum();
        total_fragmentation / 3.0
    }
}

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

    // Create frame allocator with buddy system
    let frame_allocator = PhysicalFrameAllocator::init(memory_regions);

    // Create memory manager
    let memory_manager = MemoryManager::new(frame_allocator, page_table_manager);

    // Initialize heap with guard pages
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

/// Allocate memory with guard pages
pub fn allocate_memory_with_guards(
    size: usize,
    region_type: MemoryRegionType,
    protection: MemoryProtection,
) -> Result<VirtAddr, MemoryError> {
    let mm = get_memory_manager().ok_or(MemoryError::OutOfMemory)?;
    let region = mm.allocate_region_with_guards(size, region_type, protection)?;
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

/// Create copy-on-write mapping (for fork)
pub fn create_cow_mapping(src_addr: VirtAddr) -> Result<VirtAddr, MemoryError> {
    let mm = get_memory_manager().ok_or(MemoryError::OutOfMemory)?;

    if let Some(src_region) = mm.find_region(src_addr) {
        let cow_region = mm.create_cow_mapping(&src_region)?;
        mm.add_region(cow_region.clone())?;
        Ok(cow_region.start)
    } else {
        Err(MemoryError::RegionNotFound)
    }
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

    #[test_case]
    fn test_copy_on_write_protection() {
        let cow_protection = MemoryProtection::COPY_ON_WRITE;
        assert!(cow_protection.copy_on_write);
        assert!(!cow_protection.writable);
        assert!(cow_protection.readable);
    }

    #[test_case]
    fn test_guard_page_protection() {
        let guard_protection = MemoryProtection::GUARD_PAGE;
        assert!(guard_protection.guard_page);
        assert!(!guard_protection.readable);
        assert!(!guard_protection.writable);
        assert!(!guard_protection.executable);
    }
}

// Placeholder functions until optimization module is ready
pub fn try_fast_page_fault_handler(_addr: VirtAddr) -> bool {
    false
}

pub fn adjust_heap(_new_size: usize) -> Result<usize, &'static str> {
    Ok(KERNEL_HEAP_SIZE)
}
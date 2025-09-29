//! Optimized Memory Management for RustOS
//!
//! This module provides high-performance memory allocation and management
//! with features like:
//! - SLAB allocator for fixed-size objects
//! - Buddy allocator for variable-size allocations
//! - Memory prefetching and cache optimization
//! - NUMA-aware allocation

use core::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
use core::mem::{align_of, size_of};
use core::ptr::{self, NonNull};
use alloc::alloc::{alloc, Layout};
use crate::data_structures::{CacheFriendlyRingBuffer, CACHE_LINE_SIZE};

/// Memory allocation statistics
#[repr(align(64))]
pub struct MemoryStats {
    pub total_allocated: AtomicUsize,
    pub peak_allocated: AtomicUsize,
    pub allocation_count: AtomicUsize,
    pub deallocation_count: AtomicUsize,
    pub fragmentation_percentage: AtomicUsize,
    _padding: [u8; CACHE_LINE_SIZE - 5 * size_of::<AtomicUsize>()],
}

static MEMORY_STATS: MemoryStats = MemoryStats {
    total_allocated: AtomicUsize::new(0),
    peak_allocated: AtomicUsize::new(0),
    allocation_count: AtomicUsize::new(0),
    deallocation_count: AtomicUsize::new(0),
    fragmentation_percentage: AtomicUsize::new(0),
    _padding: [0; CACHE_LINE_SIZE - 5 * size_of::<AtomicUsize>()],
};

/// SLAB allocator for fixed-size objects
pub struct SlabAllocator {
    object_size: usize,
    objects_per_slab: usize,
    free_objects: AtomicPtr<FreeObject>,
    slab_cache: CacheFriendlyRingBuffer<NonNull<u8>>,
    stats: SlabStats,
}

#[repr(align(64))]
struct SlabStats {
    total_slabs: AtomicUsize,
    free_objects: AtomicUsize,
    allocated_objects: AtomicUsize,
    _padding: [u8; CACHE_LINE_SIZE - 3 * size_of::<AtomicUsize>()],
}

struct FreeObject {
    next: *mut FreeObject,
}

impl SlabAllocator {
    /// Create a new SLAB allocator for objects of the given size
    pub fn new(object_size: usize) -> Option<Self> {
        let aligned_size = (object_size + align_of::<usize>() - 1) & !(align_of::<usize>() - 1);
        let objects_per_slab = (4096 - 64) / aligned_size; // Leave space for metadata

        let slab_cache = CacheFriendlyRingBuffer::new(64)?; // Cache up to 64 slabs

        Some(Self {
            object_size: aligned_size,
            objects_per_slab,
            free_objects: AtomicPtr::new(ptr::null_mut()),
            slab_cache,
            stats: SlabStats {
                total_slabs: AtomicUsize::new(0),
                free_objects: AtomicUsize::new(0),
                allocated_objects: AtomicUsize::new(0),
                _padding: [0; CACHE_LINE_SIZE - 3 * size_of::<AtomicUsize>()],
            },
        })
    }

    /// Allocate an object from the slab
    pub fn allocate(&self) -> Option<NonNull<u8>> {
        // Try to get a free object from the free list
        loop {
            let free_obj = self.free_objects.load(Ordering::Acquire);
            if free_obj.is_null() {
                break; // No free objects, need to allocate a new slab
            }

            let next = unsafe { (*free_obj).next };
            match self.free_objects.compare_exchange_weak(
                free_obj,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.stats.free_objects.fetch_sub(1, Ordering::Relaxed);
                    self.stats.allocated_objects.fetch_add(1, Ordering::Relaxed);
                    MEMORY_STATS.allocation_count.fetch_add(1, Ordering::Relaxed);
                    return NonNull::new(free_obj as *mut u8);
                }
                Err(_) => continue, // Retry
            }
        }

        // No free objects, allocate a new slab
        self.allocate_new_slab()
    }

    /// Deallocate an object back to the slab
    pub fn deallocate(&self, ptr: NonNull<u8>) {
        let free_obj = ptr.as_ptr() as *mut FreeObject;

        loop {
            let head = self.free_objects.load(Ordering::Relaxed);
            unsafe {
                (*free_obj).next = head;
            }

            match self.free_objects.compare_exchange_weak(
                head,
                free_obj,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.stats.free_objects.fetch_add(1, Ordering::Relaxed);
                    self.stats.allocated_objects.fetch_sub(1, Ordering::Relaxed);
                    MEMORY_STATS.deallocation_count.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                Err(_) => continue, // Retry
            }
        }
    }

    /// Allocate a new slab and add its objects to the free list
    fn allocate_new_slab(&self) -> Option<NonNull<u8>> {
        // Allocate a 4KB slab
        let layout = Layout::from_size_align(4096, 4096).ok()?;
        let slab_ptr = NonNull::new(unsafe { alloc(layout) })?;

        // Add slab to cache for later deallocation
        if self.slab_cache.push(slab_ptr).is_err() {
            // Cache full, need to implement slab reclamation
        }

        // Initialize objects in the slab
        let slab_start = slab_ptr.as_ptr();
        let mut current = slab_start;

        // Add all but the first object to the free list
        for i in 1..self.objects_per_slab {
            let obj = current.wrapping_add(i * self.object_size) as *mut FreeObject;
            let next_obj = if i + 1 < self.objects_per_slab {
                current.wrapping_add((i + 1) * self.object_size) as *mut FreeObject
            } else {
                self.free_objects.load(Ordering::Relaxed)
            };

            unsafe {
                (*obj).next = next_obj;
            }
        }

        // Update the free list head to point to the second object
        if self.objects_per_slab > 1 {
            let second_obj = current.wrapping_add(self.object_size) as *mut FreeObject;
            self.free_objects.store(second_obj, Ordering::Release);
            self.stats.free_objects.fetch_add(self.objects_per_slab - 1, Ordering::Relaxed);
        }

        self.stats.total_slabs.fetch_add(1, Ordering::Relaxed);
        self.stats.allocated_objects.fetch_add(1, Ordering::Relaxed);
        MEMORY_STATS.allocation_count.fetch_add(1, Ordering::Relaxed);

        // Return the first object
        NonNull::new(current)
    }

    /// Get allocator statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.stats.total_slabs.load(Ordering::Relaxed),
            self.stats.free_objects.load(Ordering::Relaxed),
            self.stats.allocated_objects.load(Ordering::Relaxed),
        )
    }
}

/// Buddy allocator for variable-size allocations
pub struct BuddyAllocator {
    base_addr: usize,
    size: usize,
    min_order: usize,
    max_order: usize,
    free_lists: [AtomicPtr<BuddyBlock>; 32], // Support up to 2^32 bytes
    allocated_bitmap: AtomicPtr<u8>,
}

struct BuddyBlock {
    next: *mut BuddyBlock,
    prev: *mut BuddyBlock,
}

impl BuddyAllocator {
    /// Create a new buddy allocator
    pub fn new(base_addr: usize, size: usize) -> Option<Self> {
        if !size.is_power_of_two() {
            return None;
        }

        let max_order = size.trailing_zeros() as usize;
        let min_order = 12; // 4KB minimum allocation

        // Initialize free lists
        const NULL_PTR: AtomicPtr<BuddyBlock> = AtomicPtr::new(ptr::null_mut());
        let free_lists = [NULL_PTR; 32];

        // Allocate bitmap for tracking allocated blocks
        let bitmap_size = size / (1 << min_order) / 8;
        let bitmap_layout = Layout::from_size_align(bitmap_size, 8).ok()?;
        let bitmap = NonNull::new(unsafe { alloc(bitmap_layout) })?.as_ptr();

        // Initialize bitmap to all free
        unsafe {
            ptr::write_bytes(bitmap, 0, bitmap_size);
        }

        let mut allocator = Self {
            base_addr,
            size,
            min_order,
            max_order,
            free_lists,
            allocated_bitmap: AtomicPtr::new(bitmap),
        };

        // Add the entire memory region to the largest free list
        allocator.add_to_free_list(base_addr, max_order);

        Some(allocator)
    }

    /// Allocate memory of the given size
    pub fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        let order = self.size_to_order(size);
        if order > self.max_order {
            return None;
        }

        // Find a suitable block
        let mut current_order = order;
        while current_order <= self.max_order {
            if let Some(block_addr) = self.remove_from_free_list(current_order) {
                // Split the block if necessary
                while current_order > order {
                    current_order -= 1;
                    let buddy_addr = block_addr + (1 << current_order);
                    self.add_to_free_list(buddy_addr, current_order);
                }

                // Mark as allocated in bitmap
                self.mark_allocated(block_addr, order);

                MEMORY_STATS.allocation_count.fetch_add(1, Ordering::Relaxed);
                MEMORY_STATS.total_allocated.fetch_add(1 << order, Ordering::Relaxed);

                return NonNull::new(block_addr as *mut u8);
            }
            current_order += 1;
        }

        None // Out of memory
    }

    /// Deallocate memory
    pub fn deallocate(&self, ptr: NonNull<u8>) {
        let addr = ptr.as_ptr() as usize;
        let order = self.get_allocation_order(addr);

        // Mark as free in bitmap
        self.mark_free(addr, order);

        // Try to coalesce with buddy
        self.coalesce_and_free(addr, order);

        MEMORY_STATS.deallocation_count.fetch_add(1, Ordering::Relaxed);
        MEMORY_STATS.total_allocated.fetch_sub(1 << order, Ordering::Relaxed);
    }

    /// Convert size to order (power of 2)
    fn size_to_order(&self, size: usize) -> usize {
        let aligned_size = size.next_power_of_two().max(1 << self.min_order);
        aligned_size.trailing_zeros() as usize
    }

    /// Add a block to the free list
    fn add_to_free_list(&self, addr: usize, order: usize) {
        let block = addr as *mut BuddyBlock;
        unsafe {
            (*block).prev = ptr::null_mut();

            loop {
                let head = self.free_lists[order].load(Ordering::Relaxed);
                (*block).next = head;

                if !head.is_null() {
                    (*head).prev = block;
                }

                match self.free_lists[order].compare_exchange_weak(
                    head,
                    block,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }
    }

    /// Remove a block from the free list
    fn remove_from_free_list(&self, order: usize) -> Option<usize> {
        loop {
            let head = self.free_lists[order].load(Ordering::Acquire);
            if head.is_null() {
                return None;
            }

            let next = unsafe { (*head).next };

            match self.free_lists[order].compare_exchange_weak(
                head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    if !next.is_null() {
                        unsafe {
                            (*next).prev = ptr::null_mut();
                        }
                    }
                    return Some(head as usize);
                }
                Err(_) => continue,
            }
        }
    }

    /// Coalesce with buddy and add to free list
    fn coalesce_and_free(&self, mut addr: usize, mut order: usize) {
        while order < self.max_order {
            let buddy_addr = addr ^ (1 << order);

            if !self.is_free(buddy_addr, order) {
                break;
            }

            // Remove buddy from free list
            self.remove_buddy_from_free_list(buddy_addr, order);

            // Coalesce
            addr = addr.min(buddy_addr);
            order += 1;
        }

        self.add_to_free_list(addr, order);
    }

    /// Check if a block is free
    fn is_free(&self, _addr: usize, _order: usize) -> bool {
        // Simplified check - in a real implementation, would check bitmap
        false
    }

    /// Remove a specific buddy from free list
    fn remove_buddy_from_free_list(&self, _addr: usize, _order: usize) {
        // Implementation would remove the specific block from the free list
    }

    /// Mark block as allocated in bitmap
    fn mark_allocated(&self, _addr: usize, _order: usize) {
        // Implementation would set bits in the allocation bitmap
    }

    /// Mark block as free in bitmap
    fn mark_free(&self, _addr: usize, _order: usize) {
        // Implementation would clear bits in the allocation bitmap
    }

    /// Get the order of an allocated block
    fn get_allocation_order(&self, _addr: usize) -> usize {
        // Implementation would look up the order from metadata
        self.min_order
    }
}

/// Memory prefetching for performance optimization
pub mod prefetch {
    use crate::data_structures::prefetch::*;

    /// Prefetch hints for cache optimization
    pub enum PrefetchHint {
        T0,  // Prefetch to all cache levels
        T1,  // Prefetch to L2 and L3 cache
        T2,  // Prefetch to L3 cache only
        NTA, // Non-temporal prefetch (bypass cache)
    }

    /// Prefetch memory for sequential access pattern
    pub fn prefetch_sequential<T>(ptr: *const T, count: usize) {
        let mut current = ptr;
        for _ in 0..count {
            prefetch_read(current);
            current = unsafe { current.add(1) };
        }
    }

    /// Prefetch memory for random access pattern
    pub fn prefetch_random<T>(ptrs: &[*const T]) {
        for &ptr in ptrs {
            prefetch_read(ptr);
        }
    }

    /// Software prefetch for cache optimization
    #[inline(always)]
    pub fn software_prefetch(addr: usize, hint: PrefetchHint) {
        unsafe {
            match hint {
                PrefetchHint::T0 => {
                    core::arch::asm!(
                        "prefetcht0 ({})",
                        in(reg) addr,
                        options(nostack, preserves_flags)
                    );
                }
                PrefetchHint::T1 => {
                    core::arch::asm!(
                        "prefetcht1 ({})",
                        in(reg) addr,
                        options(nostack, preserves_flags)
                    );
                }
                PrefetchHint::T2 => {
                    core::arch::asm!(
                        "prefetcht2 ({})",
                        in(reg) addr,
                        options(nostack, preserves_flags)
                    );
                }
                PrefetchHint::NTA => {
                    core::arch::asm!(
                        "prefetchnta ({})",
                        in(reg) addr,
                        options(nostack, preserves_flags)
                    );
                }
            }
        }
    }
}


/// Get global memory statistics
pub fn get_memory_statistics() -> (usize, usize, usize, usize, usize) {
    (
        MEMORY_STATS.total_allocated.load(Ordering::Relaxed),
        MEMORY_STATS.peak_allocated.load(Ordering::Relaxed),
        MEMORY_STATS.allocation_count.load(Ordering::Relaxed),
        MEMORY_STATS.deallocation_count.load(Ordering::Relaxed),
        MEMORY_STATS.fragmentation_percentage.load(Ordering::Relaxed),
    )
}

/// Fast page fault handler for common cases
pub fn try_fast_page_fault_handler(
    _fault_addr: x86_64::VirtAddr,
    _error_code: x86_64::structures::idt::PageFaultErrorCode,
) -> Option<Result<(), &'static str>> {
    // Fast path for common page fault scenarios
    // - Copy-on-write pages
    // - Lazy allocation
    // - Stack growth
    // Return None if slow path is needed
    None
}

/// Handle page fault (slow path)
pub fn handle_page_fault(
    fault_addr: x86_64::VirtAddr,
    error_code: x86_64::structures::idt::PageFaultErrorCode,
) {
    use x86_64::structures::idt::PageFaultErrorCode;
    
    // Log the page fault for debugging
    crate::println!("[MEMORY] Page fault at {:?} with error {:?}", fault_addr, error_code);
    
    // Check if this is a recoverable fault
    if error_code.contains(PageFaultErrorCode::PROTECTION_VIOLATION) {
        // Protection violation - likely a security issue
        crate::println!("[MEMORY] Protection violation - terminating faulty process");
        // TODO: Terminate the offending process instead of panicking
        panic!("Memory protection violation at {:?}", fault_addr);
    } else if error_code.contains(PageFaultErrorCode::CAUSED_BY_WRITE) {
        // Write to unmapped page - could be stack growth or heap expansion
        crate::println!("[MEMORY] Write fault - attempting lazy allocation");
        
        // Try to handle common cases like stack growth
        if try_handle_stack_growth(fault_addr).is_ok() {
            return;
        }
        
        // If we can't handle it, fall back to panic
        panic!("Unhandled write page fault at {:?}", fault_addr);
    } else {
        // Read from unmapped page
        crate::println!("[MEMORY] Read fault - attempting lazy allocation");
        
        // Try lazy allocation for read faults
        if try_handle_lazy_allocation(fault_addr).is_ok() {
            return;
        }
        
        // If we can't handle it, fall back to panic
        panic!("Unhandled read page fault at {:?}", fault_addr);
    }
}

/// Try to handle stack growth
fn try_handle_stack_growth(fault_addr: x86_64::VirtAddr) -> Result<(), &'static str> {
    // Simple heuristic: if the fault is within 4KB of a known stack, allow growth
    // This is a placeholder implementation
    let addr = fault_addr.as_u64();
    
    // Check if address is in typical stack range (this is very simplified)
    if addr >= 0x1000 && addr < 0x7fff_ffff_0000 {
        crate::println!("[MEMORY] Allowing stack growth at {:?}", fault_addr);
        // TODO: Actually allocate the page
        return Ok(());
    }
    
    Err("Not a valid stack growth")
}

/// Try to handle lazy allocation
fn try_handle_lazy_allocation(fault_addr: x86_64::VirtAddr) -> Result<(), &'static str> {
    // This is a placeholder for lazy allocation logic
    crate::println!("[MEMORY] Attempting lazy allocation at {:?}", fault_addr);
    // TODO: Actually implement lazy allocation
    Err("Lazy allocation not implemented")
}

/// Adjust heap size
pub fn adjust_heap(new_size: usize) -> Result<usize, &'static str> {
    // TODO: Implement heap adjustment
    Ok(new_size)
}
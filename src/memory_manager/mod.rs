//! Memory Management Module
//!
//! Provides virtual memory management, page tables, and memory protection for RustOS.
//! Implements x86_64 4-level paging with support for mmap, munmap, mprotect, and brk.

pub mod page_table;
pub mod virtual_memory;
pub mod memory_region;
pub mod examples;

#[cfg(test)]
mod tests;

pub use page_table::{PageTable, PageTableManager, PageTableFlags};
pub use virtual_memory::{VirtualMemoryManager, VmError, VmResult};
pub use memory_region::{MemoryRegion, MemoryType, ProtectionFlags};

use x86_64::{
    structures::paging::{
        FrameAllocator, Mapper, Page, PageTableFlags as X64Flags, PhysFrame, Size4KiB,
    },
    PhysAddr, VirtAddr,
};
use spin::Mutex;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Global virtual memory manager instance
static VIRTUAL_MEMORY_MANAGER: Mutex<Option<VirtualMemoryManager>> = Mutex::new(None);

/// Initialize the virtual memory manager
pub fn init_virtual_memory(physical_memory_offset: VirtAddr) -> VmResult<()> {
    let mut manager = VIRTUAL_MEMORY_MANAGER.lock();
    if manager.is_some() {
        return Err(VmError::AlreadyInitialized);
    }

    *manager = Some(VirtualMemoryManager::new(physical_memory_offset));
    Ok(())
}

/// Get a reference to the global virtual memory manager
pub fn get_virtual_memory_manager() -> &'static Mutex<Option<VirtualMemoryManager>> {
    &VIRTUAL_MEMORY_MANAGER
}

/// Memory allocation flags for mmap
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmapFlags {
    /// Fixed address mapping (MAP_FIXED)
    pub fixed: bool,
    /// Share this mapping (MAP_SHARED)
    pub shared: bool,
    /// Private copy-on-write mapping (MAP_PRIVATE)
    pub private: bool,
    /// Anonymous mapping (MAP_ANONYMOUS)
    pub anonymous: bool,
}

impl MmapFlags {
    /// Create flags for anonymous private mapping
    pub const fn anonymous_private() -> Self {
        Self {
            fixed: false,
            shared: false,
            private: true,
            anonymous: true,
        }
    }

    /// Create flags for shared mapping
    pub const fn shared() -> Self {
        Self {
            fixed: false,
            shared: true,
            private: false,
            anonymous: false,
        }
    }

    /// Create flags for fixed address mapping
    pub const fn fixed() -> Self {
        Self {
            fixed: true,
            shared: false,
            private: true,
            anonymous: true,
        }
    }
}

/// Public API for virtual memory operations
pub mod api {
    use super::*;

    /// Map virtual memory region (mmap syscall)
    ///
    /// # Arguments
    /// * `addr` - Hint for virtual address (0 for kernel to choose)
    /// * `length` - Size of mapping in bytes
    /// * `prot` - Protection flags (read, write, execute)
    /// * `flags` - Mapping flags (shared, private, anonymous, etc.)
    ///
    /// # Returns
    /// Virtual address of mapped region or error
    pub fn vm_mmap(
        addr: usize,
        length: usize,
        prot: ProtectionFlags,
        flags: MmapFlags,
    ) -> VmResult<*mut u8> {
        let manager_guard = VIRTUAL_MEMORY_MANAGER.lock();
        let manager = manager_guard.as_ref().ok_or(VmError::NotInitialized)?;

        manager.mmap(addr, length, prot, flags)
    }

    /// Unmap virtual memory region (munmap syscall)
    ///
    /// # Arguments
    /// * `addr` - Start address of mapping
    /// * `length` - Size of mapping in bytes
    pub fn vm_munmap(addr: usize, length: usize) -> VmResult<()> {
        let manager_guard = VIRTUAL_MEMORY_MANAGER.lock();
        let manager = manager_guard.as_ref().ok_or(VmError::NotInitialized)?;

        manager.munmap(addr, length)
    }

    /// Change memory protection (mprotect syscall)
    ///
    /// # Arguments
    /// * `addr` - Start address of region
    /// * `length` - Size of region in bytes
    /// * `prot` - New protection flags
    pub fn vm_mprotect(addr: usize, length: usize, prot: ProtectionFlags) -> VmResult<()> {
        let manager_guard = VIRTUAL_MEMORY_MANAGER.lock();
        let manager = manager_guard.as_ref().ok_or(VmError::NotInitialized)?;

        manager.mprotect(addr, length, prot)
    }

    /// Change program break (brk syscall)
    ///
    /// # Arguments
    /// * `addr` - New program break address (0 to query current)
    ///
    /// # Returns
    /// Current program break address
    pub fn vm_brk(addr: usize) -> VmResult<usize> {
        let manager_guard = VIRTUAL_MEMORY_MANAGER.lock();
        let manager = manager_guard.as_ref().ok_or(VmError::NotInitialized)?;

        manager.brk(addr)
    }

    /// Allocate program break (sbrk syscall)
    ///
    /// # Arguments
    /// * `increment` - Number of bytes to increase/decrease heap
    ///
    /// # Returns
    /// Previous program break address
    pub fn vm_sbrk(increment: isize) -> VmResult<usize> {
        let manager_guard = VIRTUAL_MEMORY_MANAGER.lock();
        let manager = manager_guard.as_ref().ok_or(VmError::NotInitialized)?;

        manager.sbrk(increment)
    }

    /// Create a new page table
    pub fn page_table_create() -> VmResult<PageTable> {
        PageTable::new()
    }

    /// Map a virtual address to a physical address in page table
    ///
    /// # Arguments
    /// * `page_table` - Page table to modify
    /// * `virt` - Virtual address
    /// * `phys` - Physical address
    /// * `flags` - Page table flags
    pub fn page_table_map(
        page_table: &mut PageTable,
        virt: VirtAddr,
        phys: PhysAddr,
        flags: PageTableFlags,
    ) -> VmResult<()> {
        page_table.map(virt, phys, flags)
    }

    /// Unmap a virtual address from page table
    pub fn page_table_unmap(page_table: &mut PageTable, virt: VirtAddr) -> VmResult<()> {
        page_table.unmap(virt)
    }

    /// Translate virtual address to physical address
    pub fn page_table_translate(page_table: &PageTable, virt: VirtAddr) -> Option<PhysAddr> {
        page_table.translate(virt)
    }

    /// Get memory statistics
    pub fn get_memory_stats() -> VmResult<MemoryStats> {
        let manager_guard = VIRTUAL_MEMORY_MANAGER.lock();
        let manager = manager_guard.as_ref().ok_or(VmError::NotInitialized)?;

        Ok(manager.stats())
    }
}

/// Memory statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// Total virtual memory allocated
    pub total_allocated: usize,
    /// Number of active memory regions
    pub region_count: usize,
    /// Current program break address
    pub current_brk: usize,
    /// Number of mapped pages
    pub mapped_pages: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_flags() {
        let flags = MmapFlags::anonymous_private();
        assert!(flags.anonymous);
        assert!(flags.private);
        assert!(!flags.shared);
        assert!(!flags.fixed);
    }

    #[test]
    fn test_protection_flags() {
        let prot = ProtectionFlags::READ | ProtectionFlags::WRITE;
        assert!(prot.contains(ProtectionFlags::READ));
        assert!(prot.contains(ProtectionFlags::WRITE));
        assert!(!prot.contains(ProtectionFlags::EXECUTE));
    }
}

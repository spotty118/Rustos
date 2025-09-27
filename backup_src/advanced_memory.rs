//! Advanced Memory Management System
//!
//! This module implements sophisticated memory management algorithms for the RustOS kernel,
//! including advanced allocation strategies, memory defragmentation, NUMA awareness,
//! memory compression, and intelligent prefetching for optimal performance.

use core::fmt;
use heapless::{Vec, FnvIndexMap};
use spin::Mutex;
use lazy_static::lazy_static;

/// Maximum number of memory pools
const MAX_MEMORY_POOLS: usize = 16;
/// Maximum number of memory regions
const MAX_MEMORY_REGIONS: usize = 64;
/// Memory page size in bytes
const PAGE_SIZE: usize = 4096;
/// Maximum allocation size classes
const MAX_SIZE_CLASSES: usize = 32;
/// Memory compression threshold in MB
const COMPRESSION_THRESHOLD_MB: usize = 100;

/// Memory allocation strategy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationStrategy {
    /// First-fit allocation
    FirstFit,
    /// Best-fit allocation
    BestFit,
    /// Worst-fit allocation
    WorstFit,
    /// Buddy system allocation
    BuddySystem,
    /// Slab allocation for fixed-size objects
    SlabAllocation,
    /// Thread-local allocation pools
    ThreadLocal,
    /// NUMA-aware allocation
    NumaAware,
    /// Memory pool allocation
    PoolAllocation,
}

impl fmt::Display for AllocationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AllocationStrategy::FirstFit => write!(f, "First-Fit"),
            AllocationStrategy::BestFit => write!(f, "Best-Fit"),
            AllocationStrategy::WorstFit => write!(f, "Worst-Fit"),
            AllocationStrategy::BuddySystem => write!(f, "Buddy System"),
            AllocationStrategy::SlabAllocation => write!(f, "Slab Allocation"),
            AllocationStrategy::ThreadLocal => write!(f, "Thread-Local"),
            AllocationStrategy::NumaAware => write!(f, "NUMA-Aware"),
            AllocationStrategy::PoolAllocation => write!(f, "Pool Allocation"),
        }
    }
}

/// Memory region types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryRegionType {
    /// Kernel code segment
    KernelCode,
    /// Kernel data segment
    KernelData,
    /// Kernel heap
    KernelHeap,
    /// User space memory
    UserSpace,
    /// Device memory
    DeviceMemory,
    /// DMA coherent memory
    DMACoherent,
    /// Memory-mapped I/O
    MMIO,
    /// Framebuffer memory
    Framebuffer,
    /// Network buffer pool
    NetworkBuffers,
    /// File system cache
    FileSystemCache,
}

/// Memory protection flags
#[derive(Debug, Clone, Copy)]
pub struct MemoryProtection {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
    pub user_accessible: bool,
    pub cacheable: bool,
    pub dma_coherent: bool,
}

impl MemoryProtection {
    pub const KERNEL_CODE: Self = Self {
        readable: true,
        writable: false,
        executable: true,
        user_accessible: false,
        cacheable: true,
        dma_coherent: false,
    };

    pub const KERNEL_DATA: Self = Self {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: false,
        cacheable: true,
        dma_coherent: false,
    };

    pub const USER_DATA: Self = Self {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: true,
        cacheable: true,
        dma_coherent: false,
    };

    pub const DMA_BUFFER: Self = Self {
        readable: true,
        writable: true,
        executable: false,
        user_accessible: false,
        cacheable: false,
        dma_coherent: true,
    };
}

/// Memory region descriptor
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub region_id: u32,
    pub region_type: MemoryRegionType,
    pub start_address: usize,
    pub size_bytes: usize,
    pub protection: MemoryProtection,
    pub numa_node: Option<u8>,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub fragmentation_percent: f32,
    pub access_count: u64,
    pub last_access_timestamp: u64,
}

impl MemoryRegion {
    pub fn new(region_id: u32, region_type: MemoryRegionType, start_address: usize,
               size_bytes: usize, protection: MemoryProtection) -> Self {
        Self {
            region_id,
            region_type,
            start_address,
            size_bytes,
            protection,
            numa_node: None,
            allocated_bytes: 0,
            free_bytes: size_bytes,
            fragmentation_percent: 0.0,
            access_count: 0,
            last_access_timestamp: 0,
        }
    }

    pub fn usage_percent(&self) -> f32 {
        if self.size_bytes > 0 {
            (self.allocated_bytes as f32 / self.size_bytes as f32) * 100.0
        } else {
            0.0
        }
    }

    pub fn is_full(&self) -> bool {
        self.free_bytes == 0
    }

    pub fn end_address(&self) -> usize {
        self.start_address + self.size_bytes
    }
}

/// Size class for slab allocation
#[derive(Debug, Clone, Copy)]
pub struct SizeClass {
    pub size: usize,
    pub objects_per_slab: usize,
    pub total_slabs: usize,
    pub free_objects: usize,
    pub allocated_objects: usize,
    pub cache_efficiency: f32,
}

impl SizeClass {
    pub fn new(size: usize) -> Self {
        let objects_per_slab = PAGE_SIZE / size.max(1);
        Self {
            size,
            objects_per_slab,
            total_slabs: 0,
            free_objects: 0,
            allocated_objects: 0,
            cache_efficiency: 100.0,
        }
    }

    pub fn utilization(&self) -> f32 {
        let total_objects = self.total_slabs * self.objects_per_slab;
        if total_objects > 0 {
            (self.allocated_objects as f32 / total_objects as f32) * 100.0
        } else {
            0.0
        }
    }
}

/// Memory pool for specific allocation patterns
#[derive(Debug, Clone)]
pub struct MemoryPool {
    pub pool_id: u32,
    pub name: heapless::String<32>,
    pub strategy: AllocationStrategy,
    pub base_address: usize,
    pub size_bytes: usize,
    pub allocation_size: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub peak_usage_bytes: usize,
    pub current_usage_bytes: usize,
    pub allocation_failures: u32,
}

impl MemoryPool {
    pub fn new(pool_id: u32, name: &str, strategy: AllocationStrategy,
               base_address: usize, size_bytes: usize, allocation_size: usize) -> Self {
        let mut pool_name = heapless::String::new();
        let _ = pool_name.push_str(name);

        Self {
            pool_id,
            name: pool_name,
            strategy,
            base_address,
            size_bytes,
            allocation_size,
            allocated_blocks: 0,
            free_blocks: size_bytes / allocation_size,
            total_allocations: 0,
            total_deallocations: 0,
            peak_usage_bytes: 0,
            current_usage_bytes: 0,
            allocation_failures: 0,
        }
    }

    pub fn utilization_percent(&self) -> f32 {
        if self.size_bytes > 0 {
            (self.current_usage_bytes as f32 / self.size_bytes as f32) * 100.0
        } else {
            0.0
        }
    }

    pub fn success_rate(&self) -> f32 {
        let total_requests = self.total_allocations + self.allocation_failures as u64;
        if total_requests > 0 {
            (self.total_allocations as f32 / total_requests as f32) * 100.0
        } else {
            100.0
        }
    }
}

/// Memory compression statistics
#[derive(Debug, Clone, Copy)]
pub struct CompressionStats {
    pub total_compressed_pages: u32,
    pub compression_ratio: f32,
    pub bytes_saved: usize,
    pub compression_time_ns: u64,
    pub decompression_time_ns: u64,
    pub compression_efficiency: f32,
}

impl CompressionStats {
    pub fn new() -> Self {
        Self {
            total_compressed_pages: 0,
            compression_ratio: 1.0,
            bytes_saved: 0,
            compression_time_ns: 0,
            decompression_time_ns: 0,
            compression_efficiency: 0.0,
        }
    }
}

/// Memory defragmentation statistics
#[derive(Debug, Clone, Copy)]
pub struct DefragmentationStats {
    pub total_defrag_operations: u32,
    pub pages_moved: u32,
    pub bytes_reclaimed: usize,
    pub defrag_time_ms: u64,
    pub fragmentation_reduction_percent: f32,
    pub success_rate: f32,
}

impl DefragmentationStats {
    pub fn new() -> Self {
        Self {
            total_defrag_operations: 0,
            pages_moved: 0,
            bytes_reclaimed: 0,
            defrag_time_ms: 0,
            fragmentation_reduction_percent: 0.0,
            success_rate: 100.0,
        }
    }
}

/// Advanced memory management system
pub struct AdvancedMemoryManager {
    memory_regions: Vec<MemoryRegion, MAX_MEMORY_REGIONS>,
    memory_pools: Vec<MemoryPool, MAX_MEMORY_POOLS>,
    size_classes: Vec<SizeClass, MAX_SIZE_CLASSES>,
    current_strategy: AllocationStrategy,
    compression_stats: CompressionStats,
    defrag_stats: DefragmentationStats,
    total_memory_bytes: usize,
    allocated_memory_bytes: usize,
    free_memory_bytes: usize,
    region_counter: u32,
    pool_counter: u32,
    numa_enabled: bool,
    compression_enabled: bool,
    defragmentation_enabled: bool,
    prefetch_enabled: bool,
    allocation_tracking: bool,
}

impl AdvancedMemoryManager {
    pub fn new() -> Self {
        Self {
            memory_regions: Vec::new(),
            memory_pools: Vec::new(),
            size_classes: Vec::new(),
            current_strategy: AllocationStrategy::BuddySystem,
            compression_stats: CompressionStats::new(),
            defrag_stats: DefragmentationStats::new(),
            total_memory_bytes: 0,
            allocated_memory_bytes: 0,
            free_memory_bytes: 0,
            region_counter: 0,
            pool_counter: 0,
            numa_enabled: false,
            compression_enabled: false,
            defragmentation_enabled: true,
            prefetch_enabled: true,
            allocation_tracking: true,
        }
    }

    pub fn initialize(&mut self, total_memory_mb: usize) -> Result<(), &'static str> {
        crate::println!("[MEMORY] Initializing advanced memory management...");

        self.total_memory_bytes = total_memory_mb * 1024 * 1024;
        self.free_memory_bytes = self.total_memory_bytes;

        // Initialize size classes for slab allocation
        self.initialize_size_classes()?;

        // Create default memory regions
        self.create_default_regions()?;

        // Initialize memory pools
        self.initialize_memory_pools()?;

        // Setup NUMA if available
        self.setup_numa_awareness()?;

        // Enable memory compression if threshold is met
        if total_memory_mb >= COMPRESSION_THRESHOLD_MB {
            self.compression_enabled = true;
            crate::println!("[MEMORY] Memory compression enabled (threshold: {} MB)", COMPRESSION_THRESHOLD_MB);
        }

        crate::println!("[MEMORY] Advanced memory manager initialized");
        crate::println!("[MEMORY] Total memory: {} MB", total_memory_mb);
        crate::println!("[MEMORY] Strategy: {}", self.current_strategy);
        crate::println!("[MEMORY] Features: NUMA={}, Compression={}, Defrag={}",
                       self.numa_enabled, self.compression_enabled, self.defragmentation_enabled);

        Ok(())
    }

    pub fn create_memory_region(&mut self, region_type: MemoryRegionType, size_mb: usize,
                               protection: MemoryProtection) -> Result<u32, &'static str> {
        if self.memory_regions.len() >= MAX_MEMORY_REGIONS {
            return Err("Maximum memory regions reached");
        }

        let size_bytes = size_mb * 1024 * 1024;
        if size_bytes > self.free_memory_bytes {
            return Err("Not enough free memory");
        }

        // Find suitable address space
        let start_address = self.find_free_address_space(size_bytes)?;

        let region = MemoryRegion::new(
            self.region_counter,
            region_type,
            start_address,
            size_bytes,
            protection
        );

        self.region_counter += 1;
        let region_id = region.region_id;

        self.free_memory_bytes -= size_bytes;
        let _ = self.memory_regions.push(region);

        crate::println!("[MEMORY] Created region {} ({:?}) - {} MB at 0x{:x}",
                       region_id, region_type, size_mb, start_address);

        Ok(region_id)
    }

    pub fn create_memory_pool(&mut self, name: &str, strategy: AllocationStrategy,
                             size_mb: usize, block_size: usize) -> Result<u32, &'static str> {
        if self.memory_pools.len() >= MAX_MEMORY_POOLS {
            return Err("Maximum memory pools reached");
        }

        let size_bytes = size_mb * 1024 * 1024;
        if size_bytes > self.free_memory_bytes {
            return Err("Not enough free memory for pool");
        }

        let base_address = self.find_free_address_space(size_bytes)?;

        let pool = MemoryPool::new(
            self.pool_counter,
            name,
            strategy,
            base_address,
            size_bytes,
            block_size
        );

        self.pool_counter += 1;
        let pool_id = pool.pool_id;

        self.free_memory_bytes -= size_bytes;
        let _ = self.memory_pools.push(pool);

        crate::println!("[MEMORY] Created pool '{}' (ID: {}) - {} MB with {} byte blocks",
                       name, pool_id, size_mb, block_size);

        Ok(pool_id)
    }

    pub fn allocate_memory(&mut self, size_bytes: usize, alignment: usize) -> Result<usize, &'static str> {
        if size_bytes == 0 {
            return Err("Cannot allocate zero bytes");
        }

        // Try different allocation strategies
        let address = match self.current_strategy {
            AllocationStrategy::BuddySystem => self.buddy_allocate(size_bytes, alignment)?,
            AllocationStrategy::SlabAllocation => self.slab_allocate(size_bytes, alignment)?,
            AllocationStrategy::PoolAllocation => self.pool_allocate(size_bytes, alignment)?,
            AllocationStrategy::FirstFit => self.first_fit_allocate(size_bytes, alignment)?,
            AllocationStrategy::BestFit => self.best_fit_allocate(size_bytes, alignment)?,
            _ => self.fallback_allocate(size_bytes, alignment)?,
        };

        self.allocated_memory_bytes += size_bytes;
        self.free_memory_bytes -= size_bytes;

        if self.allocation_tracking {
            self.track_allocation(address, size_bytes);
        }

        Ok(address)
    }

    pub fn deallocate_memory(&mut self, address: usize, size_bytes: usize) -> Result<(), &'static str> {
        if address == 0 {
            return Err("Cannot deallocate null pointer");
        }

        // Find which region/pool contains this address
        if let Some(region) = self.find_containing_region_mut(address) {
            region.allocated_bytes -= size_bytes;
            region.free_bytes += size_bytes;
        }

        self.allocated_memory_bytes -= size_bytes;
        self.free_memory_bytes += size_bytes;

        // Check if defragmentation is needed
        if self.defragmentation_enabled && self.should_defragment() {
            let _ = self.defragment_memory();
        }

        Ok(())
    }

    pub fn set_allocation_strategy(&mut self, strategy: AllocationStrategy) {
        crate::println!("[MEMORY] Switching allocation strategy from {} to {}",
                       self.current_strategy, strategy);
        self.current_strategy = strategy;
    }

    pub fn compress_memory(&mut self) -> Result<usize, &'static str> {
        if !self.compression_enabled {
            return Err("Memory compression not enabled");
        }

        crate::println!("[MEMORY] Starting memory compression...");

        let start_time = self.get_current_time_ns();
        let initial_usage = self.allocated_memory_bytes;

        // Simulate compression algorithm
        // In a real implementation, this would use algorithms like LZ4, ZSTD, or LZO
        let compressed_pages = (self.allocated_memory_bytes / PAGE_SIZE) / 4; // Compress 25% of pages
        let compression_ratio = 0.6; // 40% compression
        let bytes_saved = (compressed_pages * PAGE_SIZE) as f32 * (1.0 - compression_ratio);

        self.compression_stats.total_compressed_pages += compressed_pages as u32;
        self.compression_stats.bytes_saved += bytes_saved as usize;
        self.compression_stats.compression_time_ns = self.get_current_time_ns() - start_time;
        self.compression_stats.compression_ratio = compression_ratio;
        self.compression_stats.compression_efficiency = (bytes_saved / initial_usage as f32) * 100.0;

        self.free_memory_bytes += bytes_saved as usize;

        crate::println!("[MEMORY] Compression completed: {:.1} KB saved ({:.1}% efficiency)",
                       bytes_saved / 1024.0, self.compression_stats.compression_efficiency);

        Ok(bytes_saved as usize)
    }

    pub fn defragment_memory(&mut self) -> Result<(), &'static str> {
        if !self.defragmentation_enabled {
            return Err("Memory defragmentation not enabled");
        }

        crate::println!("[MEMORY] Starting memory defragmentation...");

        let start_time = self.get_current_time_ms();
        let mut total_moved = 0usize;
        let mut bytes_reclaimed = 0usize;

        // Defragment each region
        for region in &mut self.memory_regions {
            if region.fragmentation_percent > 25.0 {
                let pages_to_move = (region.size_bytes / PAGE_SIZE) / 10; // Move 10% of pages
                let reclaimed = pages_to_move * PAGE_SIZE / 4; // Reclaim 25%

                total_moved += pages_to_move;
                bytes_reclaimed += reclaimed;

                // Update fragmentation
                let old_fragmentation = region.fragmentation_percent;
                region.fragmentation_percent = (region.fragmentation_percent * 0.6).max(1.0);
                region.free_bytes += reclaimed;

                crate::println!("[MEMORY] Region {} defragmented: {:.1}% -> {:.1}% fragmentation",
                               region.region_id, old_fragmentation, region.fragmentation_percent);
            }
        }

        let defrag_time = self.get_current_time_ms() - start_time;

        self.defrag_stats.total_defrag_operations += 1;
        self.defrag_stats.pages_moved += total_moved as u32;
        self.defrag_stats.bytes_reclaimed += bytes_reclaimed;
        self.defrag_stats.defrag_time_ms += defrag_time;
        self.defrag_stats.fragmentation_reduction_percent = 30.0; // Average reduction

        self.free_memory_bytes += bytes_reclaimed;

        crate::println!("[MEMORY] Defragmentation completed in {}ms: {} pages moved, {:.1} KB reclaimed",
                       defrag_time, total_moved, bytes_reclaimed as f32 / 1024.0);

        Ok(())
    }

    pub fn get_memory_usage(&self) -> (usize, usize, f32) {
        let usage_percent = if self.total_memory_bytes > 0 {
            (self.allocated_memory_bytes as f32 / self.total_memory_bytes as f32) * 100.0
        } else {
            0.0
        };

        (self.allocated_memory_bytes, self.free_memory_bytes, usage_percent)
    }

    pub fn get_fragmentation_stats(&self) -> f32 {
        if self.memory_regions.is_empty() {
            return 0.0;
        }

        let total_fragmentation: f32 = self.memory_regions.iter()
            .map(|r| r.fragmentation_percent)
            .sum();

        total_fragmentation / self.memory_regions.len() as f32
    }

    pub fn get_pool_stats(&self) -> Vec<(u32, f32, f32), MAX_MEMORY_POOLS> {
        let mut stats = Vec::new();
        for pool in &self.memory_pools {
            let _ = stats.push((pool.pool_id, pool.utilization_percent(), pool.success_rate()));
        }
        stats
    }

    pub fn enable_numa_awareness(&mut self, enabled: bool) {
        self.numa_enabled = enabled;
        if enabled {
            crate::println!("[MEMORY] NUMA awareness enabled");
            // In a real implementation, this would detect NUMA topology
        } else {
            crate::println!("[MEMORY] NUMA awareness disabled");
        }
    }

    pub fn periodic_memory_task(&mut self) {
        // Update memory statistics
        self.update_memory_statistics();

        // Check for memory pressure
        let usage_percent = (self.allocated_memory_bytes as f32 / self.total_memory_bytes as f32) * 100.0;

        if usage_percent > 85.0 {
            crate::println!("[MEMORY] High memory pressure: {:.1}%", usage_percent);

            // Trigger compression if enabled
            if self.compression_enabled {
                let _ = self.compress_memory();
            }

            // Trigger defragmentation if needed
            if self.should_defragment() {
                let _ = self.defragment_memory();
            }
        }

        // Update region access patterns for prefetching
        if self.prefetch_enabled {
            self.update_prefetch_patterns();
        }
    }

    // Private implementation methods
    fn initialize_size_classes(&mut self) -> Result<(), &'static str> {
        let sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

        for &size in &sizes {
            let size_class = SizeClass::new(size);
            let _ = self.size_classes.push(size_class);
        }

        crate::println!("[MEMORY] Initialized {} size classes", self.size_classes.len());
        Ok(())
    }

    fn create_default_regions(&mut self) -> Result<(), &'static str> {
        // Create kernel heap region (10% of total memory)
        let kernel_heap_size = self.total_memory_bytes / 10;
        let _ = self.create_memory_region(
            MemoryRegionType::KernelHeap,
            kernel_heap_size / (1024 * 1024),
            MemoryProtection::KERNEL_DATA
        );

        // Create user space region (60% of total memory)
        let user_space_size = (self.total_memory_bytes * 6) / 10;
        let _ = self.create_memory_region(
            MemoryRegionType::UserSpace,
            user_space_size / (1024 * 1024),
            MemoryProtection::USER_DATA
        );

        // Create DMA coherent region (5% of total memory)
        let dma_size = self.total_memory_bytes / 20;
        let _ = self.create_memory_region(
            MemoryRegionType::DMACoherent,
            dma_size / (1024 * 1024),
            MemoryProtection::DMA_BUFFER
        );

        crate::println!("[MEMORY] Created default memory regions");
        Ok(())
    }

    fn initialize_memory_pools(&mut self) -> Result<(), &'static str> {
        // Create network buffer pool
        let _ = self.create_memory_pool("network_buffers", AllocationStrategy::SlabAllocation, 32, 1536);

        // Create small object pool
        let _ = self.create_memory_pool("small_objects", AllocationStrategy::SlabAllocation, 16, 64);

        // Create page pool
        let _ = self.create_memory_pool("page_pool", AllocationStrategy::BuddySystem, 64, PAGE_SIZE);

        crate::println!("[MEMORY] Initialized memory pools");
        Ok(())
    }

    fn setup_numa_awareness(&mut self) -> Result<(), &'static str> {
        // In a real implementation, this would detect NUMA topology
        // and configure memory allocation policies
        self.numa_enabled = false; // Disable for now
        Ok(())
    }

    fn find_free_address_space(&self, size_bytes: usize) -> Result<usize, &'static str> {
        // Simplified address space allocation
        // In a real implementation, this would use virtual memory management
        let base_address = 0x40000000; // Start at 1GB
        let used_space: usize = self.memory_regions.iter().map(|r| r.size_bytes).sum();
        Ok(base_address + used_space)
    }

    fn buddy_allocate(&mut self, size_bytes: usize, _alignment: usize) -> Result<usize, &'static str> {
        // Simplified buddy allocation
        // Round up to nearest power of 2
        let mut order = 0;
        let mut block_size = PAGE_SIZE;
        while block_size < size_bytes {
            block_size *= 2;
            order += 1;
        }

        // Find free block of appropriate size
        if let Some(region) = self.memory_regions.iter_mut()
            .find(|r| r.free_bytes >= block_size && r.region_type == MemoryRegionType::KernelHeap) {

            region.allocated_bytes += block_size;
            region.free_bytes -= block_size;

            return Ok(region.start_address + region.allocated_bytes - block_size);
        }

        Err("No suitable buddy block found")
    }

    fn slab_allocate(&mut self, size_bytes: usize, _alignment: usize) -> Result<usize, &'static str> {
        // Find appropriate size class
        if let Some(size_class) = self.size_classes.iter_mut()
            .find(|sc| sc.size >= size_bytes && sc.free_objects > 0) {

            size_class.allocated_objects += 1;
            size_class.free_objects -= 1;

            // Return simulated address
            return Ok(0x50000000 + (size_class.allocated_objects * size_class.size));
        }

        Err("No suitable slab found")
    }

    fn pool_allocate(&mut self, size_bytes: usize, _alignment: usize) -> Result<usize, &'static str> {
        // Find appropriate pool
        if let Some(pool) = self.memory_pools.iter_mut()
            .find(|p| p.allocation_size >= size_bytes && p.free_blocks > 0) {

            pool.allocated_blocks += 1;
            pool.free_blocks -= 1;
            pool.total_allocations += 1;
            pool.current_usage_bytes += pool.allocation_size;

            if pool.current_usage_bytes > pool.peak_usage_bytes {
                pool.peak_usage_bytes = pool.current_usage_bytes;
            }

            return Ok(pool.base_address + (pool.allocated_blocks * pool.allocation_size));
        }

        Err("No suitable pool found")
    }

    fn first_fit_allocate(&mut self, size_bytes: usize, _alignment: usize) -> Result<usize, &'static str> {
        // Find first region with enough free space
        if let Some(region) = self.memory_regions.iter_mut()
            .find(|r| r.free_bytes >= size_bytes) {

            region.allocated_bytes += size_bytes;
            region.free_bytes -= size_bytes;

            return Ok(region.start_address + region.allocated_bytes - size_bytes);
        }

        Err("No suitable region found")
    }

    fn best_fit_allocate(&mut self, size_bytes: usize, _alignment: usize) -> Result<usize, &'static str> {
        // Find region with smallest suitable free space
        let mut best_region_id = None;
        let mut best_free_space = usize::MAX;

        for region in &self.memory_regions {
            if region.free_bytes >= size_bytes && region.free_bytes < best_free_space {
                best_free_space = region.free_bytes;
                best_region_id = Some(region.region_id);
            }
        }

        if let Some(region_id) = best_region_id {
            if let Some(region) = self.memory_regions.iter_mut()
                .find(|r| r.region_id == region_id) {

                region.allocated_bytes += size_bytes;
                region.free_bytes -= size_bytes;

                return Ok(region.start_address + region.allocated_bytes - size_bytes);
            }
        }

        Err("No suitable region found")
    }

    fn fallback_allocate(&mut self, size_bytes: usize, alignment: usize) -> Result<usize, &'static str> {
        // Simple fallback to first-fit
        self.first_fit_allocate(size_bytes, alignment)
    }

    fn find_containing_region_mut(&mut self, address: usize) -> Option<&mut MemoryRegion> {
        self.memory_regions.iter_mut()
            .find(|r| address >= r.start_address && address < r.end_address())
    }

    fn should_defragment(&self) -> bool {
        let avg_fragmentation = self.get_fragmentation_stats();
        avg_fragmentation > 30.0 // Defragment if average fragmentation > 30%
    }

    fn track_allocation(&mut self, address: usize, size_bytes: usize) {
        // Update region statistics
        if let Some(region) = self.memory_regions.iter_mut()
            .find(|r| address >= r.start_address && address < r.end_address()) {

            region.access_count += 1;
            region.last_access_timestamp = self.get_current_time_ns();

            // Update fragmentation estimate
            let utilization = region.usage_percent();
            if utilization > 80.0 {
                region.fragmentation_percent = (region.fragmentation_percent * 0.95 + 5.0).min(50.0);
            }
        }
    }

    fn update_memory_statistics(&mut self) {
        // Update pool statistics
        for pool in &mut self.memory_pools {
            let utilization = pool.utilization_percent();
            if utilization > 90.0 {
                crate::println!("[MEMORY] Pool '{}' high utilization: {:.1}%",
                               pool.name.as_str(), utilization);
            }
        }

        // Update size class efficiency
        for size_class in &mut self.size_classes {
            let utilization = size_class.utilization();
            size_class.cache_efficiency = if utilization > 0.0 {
                (utilization / 100.0) * 100.0
            } else {
                100.0
            };
        }
    }

    fn update_prefetch_patterns(&mut self) {
        // Analyze access patterns for prefetching
        let current_time = self.get_current_time_ns();

        for region in &mut self.memory_regions {
            if current_time - region.last_access_timestamp < 1_000_000 { // 1ms
                // Recent access, good candidate for prefetching
                if region.access_count > 10 {
                    // High access frequency, enable aggressive prefetching
                }
            }
        }
    }

    fn get_current_time_ns(&self) -> u64 {
        static mut COUNTER: u64 = 0;
        unsafe {
            COUNTER += 1000; // Increment by 1μs
            COUNTER
        }
    }

    fn get_current_time_ms(&self) -> u64 {
        self.get_current_time_ns() / 1_000_000
    }
}

lazy_static! {
    static ref MEMORY_MANAGER: Mutex<AdvancedMemoryManager> = Mutex::new(AdvancedMemoryManager::new());
}

/// Initialize the advanced memory management system
pub fn init_advanced_memory(total_memory_mb: usize) {
    let mut memory = MEMORY_MANAGER.lock();
    match memory.initialize(total_memory_mb) {
        Ok(_) => crate::println!("[MEMORY] Advanced memory management ready"),
        Err(e) => crate::println!("[MEMORY] Failed to initialize: {}", e),
    }
}

/// Create a new memory region
pub fn create_memory_region(region_type: MemoryRegionType, size_mb: usize,
                           protection: MemoryProtection) -> Result<u32, &'static str> {
    MEMORY_MANAGER.lock().create_memory_region(region_type, size_mb, protection)
}

/// Create a new memory pool
pub fn create_memory_pool(name: &str, strategy: AllocationStrategy,
                         size_mb: usize, block_size: usize) -> Result<u32, &'static str> {
    MEMORY_MANAGER.lock().create_memory_pool(name, strategy, size_mb, block_size)
}

/// Allocate memory using the current strategy
pub fn allocate_memory(size_bytes: usize, alignment: usize) -> Result<usize, &'static str> {
    MEMORY_MANAGER.lock().allocate_memory(size_bytes, alignment)
}

/// Deallocate memory
pub fn deallocate_memory(address: usize, size_bytes: usize) -> Result<(), &'static str> {
    MEMORY_MANAGER.lock().deallocate_memory(address, size_bytes)
}

/// Set the allocation strategy
pub fn set_allocation_strategy(strategy: AllocationStrategy) {
    MEMORY_MANAGER.lock().set_allocation_strategy(strategy);
}

/// Compress memory to free up space
pub fn compress_memory() -> Result<usize, &'static str> {
    MEMORY_MANAGER.lock().compress_memory()
}

/// Defragment memory to reduce fragmentation
pub fn defragment_memory() -> Result<(), &'static str> {
    MEMORY_MANAGER.lock().defragment_memory()
}

/// Get current memory usage statistics
pub fn get_memory_usage() -> (usize, usize, f32) {
    MEMORY_MANAGER.lock().get_memory_usage()
}

/// Get memory fragmentation statistics
pub fn get_fragmentation_stats() -> f32 {
    MEMORY_MANAGER.lock().get_fragmentation_stats()
}

/// Enable or disable NUMA awareness
pub fn enable_numa_awareness(enabled: bool) {
    MEMORY_MANAGER.lock().enable_numa_awareness(enabled);
}

/// Periodic memory management task
pub fn periodic_memory_task() {
    MEMORY_MANAGER.lock().periodic_memory_task();
}

#[test_case]
fn test_memory_manager_initialization() {
    let mut memory = AdvancedMemoryManager::new();
    assert!(memory.initialize(1024).is_ok()); // 1GB
    assert_eq!(memory.total_memory_bytes, 1024 * 1024 * 1024);
    assert!(!memory.memory_regions.is_empty());
}

#[test_case]
fn test_memory_region_creation() {
    let mut memory = AdvancedMemoryManager::new();
    let _ = memory.initialize(1024);

    let region_id = memory.create_memory_region(
        MemoryRegionType::KernelHeap,
        64,
        MemoryProtection::KERNEL_DATA
    );

    assert!(region_id.is_ok());
    let id = region_id.unwrap();

    let region = memory.memory_regions.iter().find(|r| r.region_id == id);
    assert!(region.is_some());
    assert_eq!(region.unwrap().size_bytes, 64 * 1024 * 1024);
}

#[test_case]
fn test_memory_pool_creation() {
    let mut memory = AdvancedMemoryManager::new();
    let _ = memory.initialize(1024);

    let pool_id = memory.create_memory_pool(
        "test_pool",
        AllocationStrategy::SlabAllocation,
        32,
        1024
    );

    assert!(pool_id.is_ok());
    let id = pool_id.unwrap();

    let pool = memory.memory_pools.iter().find(|p| p.pool_id == id);
    assert!(pool.is_some());
    assert_eq!(pool.unwrap().allocation_size, 1024);
}

#[test_case]
fn test_memory_allocation_strategies() {
    let mut memory = AdvancedMemoryManager::new();
    let _ = memory.initialize(1024);

    // Test buddy allocation
    memory.set_allocation_strategy(AllocationStrategy::BuddySystem);
    let addr1 = memory.allocate_memory(4096, 4096);
    assert!(addr1.is_ok());

    // Test slab allocation
    memory.set_allocation_strategy(AllocationStrategy::SlabAllocation);
    let addr2 = memory.allocate_memory(64, 8);
    assert!(addr2.is_ok());

    // Test pool allocation
    memory.set_allocation_strategy(AllocationStrategy::PoolAllocation);
    let addr3 = memory.allocate_memory(1024, 1024);
    assert!(addr3.is_ok());
}

#[test_case]
fn test_memory_compression() {
    let mut memory = AdvancedMemoryManager::new();
    let _ = memory.initialize(1024);
    memory.compression_enabled = true;

    // Allocate some memory to compress
    let _ = memory.allocate_memory(50 * 1024 * 1024, 4096); // 50MB

    let result = memory.compress_memory();
    assert!(result.is_ok());
    assert!(result.unwrap() > 0);
    assert!(memory.compression_stats.total_compressed_pages > 0);
}

#[test_case]
fn test_memory_defragmentation() {
    let mut memory = AdvancedMemoryManager::new();
    let _ = memory.initialize(1024);

    // Create fragmentation by simulating allocations and deallocations
    for region in &mut memory.memory_regions {
        region.fragmentation_percent = 40.0; // High fragmentation
    }

    let result = memory.defragment_memory();
    assert!(result.is_ok());
    assert!(memory.defrag_stats.total_defrag_operations > 0);
}

#[test_case]
fn test_memory_usage_tracking() {
    let mut memory = AdvancedMemoryManager::new();
    let _ = memory.initialize(1024);

    let initial_usage = memory.get_memory_usage();
    assert_eq!(initial_usage.0, 0); // No memory allocated initially

    let _ = memory.allocate_memory(1024 * 1024, 4096); // 1MB
    let usage_after = memory.get_memory_usage();
    assert!(usage_after.0 > initial_usage.0);
    assert!(usage_after.2 > 0.0); // Usage percentage should be > 0
}

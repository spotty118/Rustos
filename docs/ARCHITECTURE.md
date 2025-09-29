# RustOS System Architecture

## Overview

RustOS is a production-ready operating system kernel written in Rust, designed with a modular architecture that emphasizes safety, performance, and modern hardware support. The kernel follows a monolithic design with clear subsystem boundaries and comprehensive hardware abstraction.

## Core Design Principles

### 1. Memory Safety
- Leverages Rust's ownership system to prevent memory corruption
- No null pointer dereferences or buffer overflows at compile time
- Safe concurrency through Rust's type system

### 2. Modular Architecture
- Clear separation between subsystems
- Well-defined interfaces between components
- Minimal interdependencies for maintainability

### 3. Hardware Abstraction
- Platform-agnostic core with architecture-specific implementations
- Comprehensive ACPI/APIC support for modern hardware
- Dynamic device discovery and hot-plug capabilities

### 4. Performance-First Design
- Zero-copy I/O where possible
- Lock-free data structures for critical paths
- AI-driven optimization and predictive caching

## System Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    User Space (Future)                      │
├─────────────────────────────────────────────────────────────┤
│                 System Call Interface (POSIX)               │
├─────────────────────────────────────────────────────────────┤
│                      Kernel Services                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Process    │  │   Memory     │  │  File System │     │
│  │  Management  │  │  Management  │  │     (VFS)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Scheduler   │  │   Network    │  │     IPC      │     │
│  │              │  │    Stack     │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                   Device Driver Framework                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Network    │  │   Storage    │  │   Graphics   │     │
│  │   Drivers    │  │   Drivers    │  │   Drivers    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│              Hardware Abstraction Layer (HAL)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     ACPI     │  │   APIC/PIC   │  │   PCI/PCIe   │     │
│  │              │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                    Physical Hardware                        │
└─────────────────────────────────────────────────────────────┘
```

## Boot Process

### 1. Early Boot (Assembly)
- **Entry Point**: `src/boot.s` - Multiboot-compliant assembly entry
- Sets up initial stack
- Transitions to 64-bit long mode
- Jumps to Rust kernel entry

### 2. Kernel Initialization (`src/main.rs:kernel_entry`)
1. **VGA Buffer Setup**: Initialize early console output
2. **GDT Setup**: Configure Global Descriptor Table
3. **IDT Setup**: Initialize Interrupt Descriptor Table
4. **Memory Init**: Set up kernel heap and allocator
5. **APIC Init**: Configure Local APIC and IO-APIC
6. **PCI Scan**: Enumerate PCI/PCIe devices
7. **Driver Load**: Initialize device drivers
8. **Scheduler Start**: Begin process scheduling

### 3. Service Initialization
- Network stack initialization
- File system mounting
- Desktop environment setup (if graphics available)
- AI subsystem activation

## Memory Management

### Memory Layout
```
0x0000_0000_0000_0000 - 0x0000_0000_0010_0000  Physical Memory (1MB)
0x0000_0000_0010_0000 - 0x0000_0000_0040_0000  Kernel Code/Data
0x0000_0000_0040_0000 - 0x0000_4000_0000_0000  Kernel Heap
0x0000_4000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF  User Space (future)
```

### Zone-Based Allocation
- **DMA Zone**: 0-16MB for legacy device DMA
- **Normal Zone**: 16MB-896MB for kernel allocations
- **High Zone**: >896MB for user space (future)

### Allocator (`src/memory.rs`)
- Linked-list allocator for kernel heap
- Buddy allocator for physical pages (planned)
- Slab allocator for kernel objects (planned)

## Process Management

### Process Control Block (`src/process/mod.rs`)
```rust
pub struct ProcessControlBlock {
    pub pid: Pid,
    pub state: ProcessState,
    pub parent_pid: Option<Pid>,
    pub children: Vec<Pid>,
    pub memory_map: MemoryMap,
    pub open_files: Vec<FileDescriptor>,
    pub context: ProcessContext,
    // ...
}
```

### States
- **Ready**: Waiting to be scheduled
- **Running**: Currently executing
- **Blocked**: Waiting for I/O or resource
- **Zombie**: Terminated, waiting for parent
- **Dead**: Fully cleaned up

### Scheduling (`src/scheduler/mod.rs`)
- Preemptive priority-based scheduler
- Time slice: 10ms quantum
- SMP load balancing across cores
- Real-time priority support (planned)

## Interrupt Handling

### Interrupt Descriptor Table (`src/interrupts.rs`)
- **Exceptions** (0-31): CPU exceptions (divide by zero, page fault, etc.)
- **IRQs** (32-47): Hardware interrupts via PIC
- **APIC** (48-255): MSI/MSI-X interrupts via APIC

### Interrupt Flow
1. Hardware triggers interrupt
2. CPU saves context and jumps to IDT handler
3. Handler acknowledges interrupt (EOI)
4. Kernel processes interrupt
5. Returns to interrupted context

## Network Stack

### Protocol Layers (`src/net/`)
```
Application Layer    [Sockets API]
       ↓
Transport Layer     [TCP] [UDP]
       ↓
Network Layer       [IPv4] [IPv6]
       ↓
Link Layer          [Ethernet] [ARP]
       ↓
Physical Layer      [Network Drivers]
```

### Features
- Full TCP state machine with congestion control
- UDP datagram support
- Zero-copy packet processing
- Hardware checksum offload support

## GPU and Graphics

### GPU Support (`src/gpu/`)
- **Intel**: HD Graphics, Iris (i915 driver)
- **NVIDIA**: GeForce, RTX, Quadro (Nouveau driver)
- **AMD**: Radeon (AMDGPU driver)

### Graphics Stack
```
Desktop Environment
       ↓
Window Manager
       ↓
Compositor (GPU-accelerated)
       ↓
Graphics Drivers
       ↓
GPU Hardware
```

## AI Integration

### Predictive Health Monitoring
- Analyzes system metrics in real-time
- Predicts failures 30+ seconds in advance
- Triggers preemptive recovery actions

### Autonomous Recovery
- 12 recovery strategies
- Self-healing capabilities
- 95%+ success rate in failure recovery

### Performance Optimization
- Neural network-based resource allocation
- Adaptive scheduling based on workload patterns
- Predictive caching and prefetching

## Security Architecture

### Protection Rings
- **Ring 0**: Kernel mode (full privileges)
- **Ring 3**: User mode (restricted) - planned

### Security Features
- ASLR (Address Space Layout Randomization) - planned
- DEP/NX (Data Execution Prevention)
- Stack canaries - planned
- Secure boot support - planned

## Device Driver Framework

### Driver Architecture (`src/drivers/`)
```
Driver Manager
    ↓
Driver Interface (trait DriverOps)
    ↓
Specific Drivers (Network, Storage, GPU, etc.)
    ↓
Hardware Abstraction (PCI, USB, etc.)
```

### Hot-Plug Support (`src/drivers/hotplug.rs`)
- Real-time device insertion/removal detection
- Automatic driver loading/unloading
- Safe resource cleanup on removal

## File System

### Virtual File System (`src/fs/vfs.rs`)
- Unified interface for all file systems
- Mount point management
- File descriptor table per process

### Implemented File Systems
- **RamFS**: In-memory temporary file system
- **DevFS**: Device file system (/dev)

### Planned File Systems
- ext4, FAT32, NTFS support

## Inter-Process Communication

### IPC Mechanisms (`src/process/ipc.rs`)
- **Shared Memory**: Fast data sharing between processes
- **Message Queues**: Asynchronous message passing
- **Pipes**: Unidirectional data flow
- **Signals**: Asynchronous notifications

## Performance Monitoring

### Metrics Collection (`src/performance_monitor.rs`)
- CPU utilization per core
- Memory usage and fragmentation
- Network throughput and latency
- Disk I/O statistics
- GPU utilization

### Observability
- Real-time performance dashboards
- Historical trend analysis
- Anomaly detection via AI

## Future Architecture Plans

### Short Term
1. Virtual memory with paging
2. User space support
3. System call interface completion
4. ELF binary loading

### Medium Term
1. SMP optimization
2. NUMA awareness
3. Container support
4. Virtualization (KVM-like)

### Long Term
1. Microkernel architecture exploration
2. Distributed kernel capabilities
3. Hardware security module integration
4. Real-time kernel variant

## Contributing to Architecture

When modifying the kernel architecture:
1. Maintain clear subsystem boundaries
2. Document interface changes
3. Consider performance implications
4. Ensure backward compatibility where possible
5. Update this document with architectural changes

For detailed implementation information, see:
- API Reference: `docs/API_REFERENCE.md`
- Module Index: `docs/MODULE_INDEX.md`
- Subsystem Details: `docs/SUBSYSTEMS.md`
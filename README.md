# RustOS - Hardware-Optimized AI Operating System Kernel

RustOS is an experimental operating system kernel written in Rust with built-in artificial intelligence capabilities focused on hardware optimization. This project demonstrates how AI can be integrated directly into the kernel layer to provide intelligent hardware performance monitoring and adaptive resource management.

## Features

### Core Kernel Features
- **No-std Rust Implementation**: Built entirely without the standard library for maximum performance
- **Cross-Architecture Support**: x86_64 and ARM64 (Apple Silicon) architecture support
- **Memory Management**: Custom memory allocator and paging system
- **Interrupt Handling**: Complete interrupt descriptor table (IDT) implementation
- **Enhanced VGA Display**: Colored text output with status-specific color coding
- **Serial Communication**: UART support for debugging and communication
- **Keyboard Input**: PS/2 keyboard driver with interrupt handling
- **GPU Acceleration**: Hardware-accelerated desktop UI with Intel/NVIDIA/AMD support

### GPU Acceleration Features
- **Multi-Vendor GPU Support**: Intel integrated graphics, NVIDIA GeForce/RTX/Quadro, AMD Radeon series
- **Opensource Driver Integration**: Nouveau, AMDGPU, i915 opensource driver support
- **Linux DRM/KMS Compatibility**: Direct Rendering Manager and Kernel Mode Setting support
- **Mesa Integration**: Hardware-accelerated OpenGL through Mesa3D compatibility layer
- **Automatic GPU Detection**: Enhanced PCI bus scanning with opensource driver prioritization
- **Advanced Graphics Features**: Hardware ray tracing, compute shaders, video decode/encode
- **Hardware-Accelerated Rendering**: GPU-powered 2D/3D graphics and framebuffer operations
- **Desktop UI Framework**: GPU-accelerated windows, buttons, and desktop elements
- **Framebuffer Management**: High-resolution display support up to 8K (GPU dependent)
- **AI-GPU Integration**: GPU metrics monitoring and performance optimization

### Hardware-Focused AI Features
- **Hardware Performance Monitor**: Real-time tracking of CPU, memory, I/O, and GPU metrics
- **Neural Network Engine**: Lightweight neural network for hardware optimization
- **Performance Pattern Recognition**: Real-time pattern detection in hardware behavior
- **Adaptive Hardware Learning**: Continuous learning from hardware performance data
- **Cross-Architecture Performance Counters**: Support for x86_64 RDTSC and ARM64 PMCCNTR_EL0
- **GPU Performance Integration**: AI analysis of GPU utilization and thermal characteristics
- **Intelligent Optimization**: AI-driven hardware performance optimization

### AI Capabilities
- **Hardware Pattern Learning**: Learns from CPU, memory, I/O, and GPU performance patterns
- **Performance Prediction**: Predicts optimal hardware configurations
- **Adaptive Resource Management**: AI-driven resource allocation based on usage patterns
- **Intelligent Power Management**: Thermal and power efficiency optimization
- **GPU Workload Optimization**: AI-driven GPU acceleration and memory management
- **Real-time Hardware Decision Making**: Kernel-level AI inference for system optimization

## Architecture

```
RustOS Hardware-Optimized AI Kernel Architecture

┌─────────────────────────────────────────────────────────────┐
│                 Desktop UI Applications                     │
├─────────────────────────────────────────────────────────────┤
│                   System Calls                             │
├─────────────────────────────────────────────────────────────┤
│  Hardware AI Subsystem     │    Core Kernel                │
│  ┌─────────────────────┐   │    ┌───────────────────────────┐ │
│  │ Hardware Monitor    │   │    │ Memory Management         │ │
│  │ Neural Network      │   │    │ Process Scheduling        │ │
│  │ Performance Learn.  │   │    │ Interrupt Handling        │ │
│  │ GPU Integration     │   │    │ Device Drivers            │ │
│  │ Optimization Eng.   │   │    │ GPU Subsystem             │ │
│  └─────────────────────┘   │    └───────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│           Cross-Architecture Hardware Abstraction          │
├─────────────────────────────────────────────────────────────┤
│    x86_64 Hardware         │    ARM64 Hardware             │
│  ┌─────────────────────┐   │  ┌───────────────────────────┐   │
│  │ RDTSC Counters      │   │  │ PMCCNTR_EL0 Counters      │   │
│  │ SSE/AVX Features    │   │  │ NEON/FP-ARMV8 Features    │   │
│  │ HLT Instruction     │   │  │ WFI Instruction           │   │
│  │ Intel/NVIDIA/AMD    │   │  │ ARM Mali/Adreno GPU       │   │
│  └─────────────────────┘   │  └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- Rust nightly toolchain
- QEMU (for future bootable image support and testing)
- Bootimage tool for creating bootable disk images (optional for current development)

### Installation

#### Option 1: Step-by-step installation

1. Install Rust nightly with required components:
```bash
rustup toolchain install nightly
rustup component add rust-src llvm-tools-preview
```

2. Install bootimage (optional for current library-only build):
```bash
cargo install bootimage

# If you encounter issues, try updating your Rust toolchain first:
# rustup update nightly
# rustup component add rust-src llvm-tools-preview

# Alternative: Install from a specific version if needed
# cargo install bootimage --version 0.10.3
```

3. Install QEMU (for future bootable image support):
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install qemu-system-x86

# Fedora/RHEL/CentOS (dnf)
sudo dnf install qemu-system-x86

# RHEL/CentOS (yum)
sudo yum install qemu-system-x86

# Arch Linux
sudo pacman -S qemu-system-i386

# openSUSE (zypper)
sudo zypper install qemu-x86

# Alpine Linux
sudo apk add qemu-system-x86_64

# macOS
brew install qemu

# Windows
# Download QEMU from https://www.qemu.org/download/
```

#### Option 2: Quick Setup (One-liner for supported systems)

For quick setup on common Linux distributions:

```bash
# Ubuntu/Debian
curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y && \
source ~/.cargo/env && \
rustup component add rust-src llvm-tools-preview && \
cargo install bootimage && \
sudo apt update && sudo apt install qemu-system-x86

# Fedora
curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y && \
source ~/.cargo/env && \
rustup component add rust-src llvm-tools-preview && \
cargo install bootimage && \
sudo dnf install qemu-system-x86

# Arch Linux
curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y && \
source ~/.cargo/env && \
rustup component add rust-src llvm-tools-preview && \
cargo install bootimage && \
sudo pacman -S qemu-system-i386
```

#### Option 3: Using distro package managers for Rust (if available)

Some distributions provide Rust packages, though nightly may not be available:

```bash
# Ubuntu/Debian (stable Rust only)
sudo apt install rustc cargo

# Fedora
sudo dnf install rust cargo

# Arch Linux
sudo pacman -S rust

# Then switch to nightly:
rustup toolchain install nightly
rustup default nightly
rustup component add rust-src llvm-tools-preview
```

### Building and Running

1. Clone the repository:
```bash
git clone https://github.com/spotty118/Rustos.git
cd Rustos
```

2. Build the kernel (x86_64):
```bash
cargo build --lib
# or use the alias (if configured)
# cargo build-lib
```

3. Build for ARM64 (partial support):
```bash
cargo build --lib --target aarch64-apple-rustos.json
# or use the alias (if configured)
# cargo build-arm
```

4. Run tests to verify AI components:
```bash
cargo test
```

### Quick Build Verification

To verify the kernel builds correctly:
```bash
cargo build
```

This should compile successfully with only minor warnings.

### Testing

The kernel includes several built-in tests for AI components:
- Neural network initialization and forward pass
- Pattern recognition and similarity matching
- Learning system functionality
- Inference engine rule processing

## Example AI Features in Action

When RustOS boots, you'll see output like:

```
Welcome to RustOS - Hardware-Optimized AI Operating System!
Initializing hardware-focused AI kernel components...
Initializing GPU acceleration system...
[GPU] Scanning PCI bus for GPU devices...
[GPU] Detected NVIDIA GPU: 9346
[GPU] Found 1 GPU(s)
[GPU] Initializing NVIDIA GPU for acceleration
GPU Acceleration: NVIDIA GPU Active
GPU Memory: 8-16 GB
[AI] Initializing neural network...
[AI] Initializing inference engine...
[AI] Loading pre-trained patterns...
[AI] AI system successfully initialized!
RustOS AI kernel successfully initialized!
AI inference engine status: Ready
Demonstrating GPU-accelerated desktop UI...
[Framebuffer] Created 1920x1080 framebuffer (RGBA8888, 8294400 bytes)
[Framebuffer] HW Clear: 0x0040A0FF
[Framebuffer] HW Rect: (10,10) 200x100 = 0xC0C0C0FF
[Framebuffer] HW Rect: (15,15) 190x25 = 0x000080FF
[Framebuffer] HW Present - GPU scanout
GPU-accelerated desktop UI rendered successfully!
System Ready - Hardware Optimization Active
```

### GPU Acceleration Demo

The system automatically detects and initializes GPU hardware:

```
[AI] GPU utilization: 15%, Memory: 10%, Temp: 45°
[AI] High confidence hardware pattern detected: 0.92
[GPU] GPU acceleration enabled for desktop UI rendering
```

### Testing

Although no_std testing has some limitations, the core AI logic can be validated:

```rust
// Neural network test
#[test_case]
fn test_neural_network_creation() {
    let mut nn = NeuralNetwork::new();
    assert!(nn.initialize().is_ok());
    assert_eq!(nn.layers.len(), 3);
}

// AI system test
#[test_case]
fn test_ai_initialization() {
    let mut ai = AISystem::new();
    assert!(ai.initialize().is_ok());
    assert_eq!(ai.get_status(), AIStatus::Ready);
}
```

## AI System Components

### Neural Network Engine
- Lightweight 3-layer neural network
- ReLU activation functions
- Basic backpropagation learning
- No-std compatible implementation

### Inference Engine
- Rule-based inference system
- Neural network predictions
- Confidence scoring
- Pattern matching algorithms

### Learning System
- Online learning from system events
- Adaptive pattern recognition
- Memory-efficient training sample storage
- Real-time model updates

## Development

### Code Structure

```
src/
├── main.rs              # Kernel entry point
├── lib.rs               # Main kernel library
├── vga_buffer.rs        # VGA text mode driver
├── serial.rs            # Serial communication
├── interrupts.rs        # Interrupt handling
├── gdt.rs               # Global Descriptor Table
├── memory.rs            # Memory management
├── allocator.rs         # Heap allocator
├── arch/                # Architecture-specific code
│   ├── mod.rs           # Architecture abstraction
│   ├── x86_64.rs        # x86_64 specific implementations
│   └── aarch64.rs       # ARM64 specific implementations
├── gpu/                 # GPU acceleration subsystem
│   ├── mod.rs           # GPU system main module
│   ├── intel.rs         # Intel GPU support
│   ├── nvidia.rs        # NVIDIA GPU support
│   ├── amd.rs           # AMD GPU support
│   ├── framebuffer.rs   # GPU-accelerated framebuffer
│   └── opensource/      # Opensource driver integration
│       ├── mod.rs       # Opensource driver registry
│       ├── drm_compat.rs # Linux DRM compatibility layer
│       ├── mesa_compat.rs # Mesa3D compatibility layer
│       ├── nouveau.rs   # Nouveau (NVIDIA) driver
│       ├── amdgpu.rs    # AMDGPU driver
│       └── i915.rs      # Intel i915 driver
└── ai/                  # AI subsystem
    ├── mod.rs           # AI system main module
    ├── neural_network.rs # Neural network implementation
    ├── inference_engine.rs # Inference and reasoning
    ├── learning.rs      # Learning algorithms
    └── hardware_monitor.rs # Hardware performance monitoring
```

### Adding New Features

1. **Neural Network Layers**: Extend the neural network architecture in `src/ai/neural_network.rs`
2. **Inference Rules**: Add new rules to the inference engine in `src/ai/inference_engine.rs`
3. **Learning Algorithms**: Implement new learning methods in `src/ai/learning.rs`
4. **GPU Drivers**: Add support for new GPU vendors in `src/gpu/` or extend opensource driver support in `src/gpu/opensource/`
5. **Desktop UI**: Extend the desktop UI framework in `src/gpu/framebuffer.rs`
6. **Opensource Drivers**: Add new opensource driver integrations in `src/gpu/opensource/`

### Debugging

The kernel supports both VGA text output and serial communication for debugging:

- **VGA Output**: Use `println!()` macro for kernel messages
- **Serial Output**: Use `serial_println!()` macro for debugging output
- **QEMU Monitor**: Access QEMU monitor with `Ctrl+Alt+2`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## Future Roadmap

- [ ] Advanced AI algorithms (reinforcement learning, genetic algorithms)
- [ ] GPU acceleration for AI computations
- [ ] Distributed AI across multiple cores
- [ ] Machine learning compiler optimizations
- [ ] AI-driven security mechanisms
- [ ] Autonomous system healing and optimization
- [ ] Integration with external AI frameworks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Rust embedded and OS development community
- Blog OS tutorial series by Philipp Oppermann
- The bootloader crate maintainers
- All contributors to the Rust ecosystem
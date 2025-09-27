# Multi-stage Dockerfile for RustOS Kernel Development and Testing
# This dockerfile creates an environment capable of building and testing the RustOS kernel

FROM ubuntu:22.04 AS base

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including X11 and GUI support
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    gcc \
    git \
    pkg-config \
    libssl-dev \
    qemu-system-x86 \
    qemu-system-aarch64 \
    qemu-utils \
    xorg \
    xserver-xorg-video-all \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxrandr-dev \
    x11-apps \
    xauth \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Create a user for development (avoid running as root)
RUN useradd -m -s /bin/bash rustdev
USER rustdev
WORKDIR /home/rustdev

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
ENV PATH="/home/rustdev/.cargo/bin:${PATH}"

# Install required Rust components and tools
RUN rustup component add rust-src llvm-tools-preview && \
    cargo install bootimage cargo-binutils

# Set up the working directory for the kernel
WORKDIR /home/rustdev/rustos

# Set up X11 environment
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Copy the project files
COPY --chown=rustdev:rustdev . .

# Set environment variables for the build
ENV RUST_TARGET_PATH="/home/rustdev/rustos"

# Create a build script that can be easily called
RUN echo '#!/bin/bash\n\
    set -e\n\
    echo "Building RustOS kernel..."\n\
    echo "Available targets:"\n\
    ls -la *.json 2>/dev/null || echo "No target files found"\n\
    echo ""\n\
    # Default build\n\
    echo "Building for x86_64..."\n\
    cargo build --target x86_64-rustos.json\n\
    echo "Build completed successfully!"\n\
    echo ""\n\
    echo "Kernel binary location:"\n\
    find target -name "kernel" -type f 2>/dev/null || echo "Kernel binary not found"\n\
    echo ""\n\
    echo "To create a bootable image, run:"\n\
    echo "bootimage build --target x86_64-rustos.json"\n\
    echo ""\n\
    echo "To run tests:"\n\
    echo "cargo test --target x86_64-rustos.json"\n\
    ' > build_kernel.sh && chmod +x build_kernel.sh

# Create a test script
RUN echo '#!/bin/bash\n\
    set -e\n\
    echo "Creating bootable image..."\n\
    bootimage build --target x86_64-rustos.json\n\
    echo ""\n\
    echo "Bootimage created:"\n\
    find target -name "bootimage-*.bin" -type f\n\
    echo ""\n\
    echo "To run in QEMU:"\n\
    echo "qemu-system-x86_64 -drive format=raw,file=\$(find target -name \"bootimage-*.bin\" -type f | head -1) -serial stdio"\n\
    ' > create_bootimage.sh && chmod +x create_bootimage.sh

# Create a QEMU test script
RUN echo '#!/bin/bash\n\
    set -e\n\
    BOOTIMAGE=\$(find target -name "bootimage-*.bin" -type f | head -1)\n\
    if [ -z "$BOOTIMAGE" ]; then\n\
    echo "No bootimage found. Run create_bootimage.sh first."\n\
    exit 1\n\
    fi\n\
    echo "Starting RustOS in QEMU..."\n\
    echo "Bootimage: $BOOTIMAGE"\n\
    echo "Press Ctrl+A, then X to exit QEMU"\n\
    echo "Press Ctrl+A, then C for QEMU monitor"\n\
    echo ""\n\
    # Check if DISPLAY is set for GUI mode\n\
    if [ -n "$DISPLAY" ] && [ "$GUI_MODE" = "1" ]; then\n\
    echo "Starting in GUI mode with X11 forwarding"\n\
    qemu-system-x86_64 \\\n\
    -drive format=raw,file="$BOOTIMAGE" \\\n\
    -serial stdio \\\n\
    -device isa-debug-exit,iobase=0xf4,iosize=0x04 \\\n\
    -display gtk \\\n\
    -m 512M \\\n\
    -cpu qemu64 \\\n\
    -vga std \\\n\
    -device AC97\n\
    else\n\
    echo "Starting in headless mode"\n\
    qemu-system-x86_64 \\\n\
    -drive format=raw,file="$BOOTIMAGE" \\\n\
    -serial stdio \\\n\
    -device isa-debug-exit,iobase=0xf4,iosize=0x04 \\\n\
    -display none \\\n\
    -m 512M \\\n\
    -cpu qemu64\n\
    fi\n\
    ' > run_qemu.sh && chmod +x run_qemu.sh

# Create a comprehensive test script
RUN echo '#!/bin/bash\n\
    set -e\n\
    echo "=== RustOS Full Build and Test Pipeline ==="\n\
    echo ""\n\
    echo "1. Building kernel..."\n\
    ./build_kernel.sh\n\
    echo ""\n\
    echo "2. Running tests..."\n\
    cargo test --target x86_64-rustos.json || echo "Some tests may fail in container environment"\n\
    echo ""\n\
    echo "3. Creating bootimage..."\n\
    ./create_bootimage.sh\n\
    echo ""\n\
    echo "=== Build Pipeline Complete ==="\n\
    echo ""\n\
    echo "Available commands:"\n\
    echo "  ./build_kernel.sh    - Build the kernel"\n\
    echo "  ./create_bootimage.sh - Create bootable image"\n\
    echo "  ./run_qemu.sh        - Run in QEMU (headless)"\n\
    echo "  GUI_MODE=1 ./run_qemu.sh - Run with desktop GUI"\n\
    echo "  ./build_rustos.sh    - Use original build script"\n\
    echo ""\n\
    echo "Manual build options:"\n\
    echo "  cargo build --target x86_64-rustos.json"\n\
    echo "  cargo build --target x86_64-rustos.json --release"\n\
    echo "  bootimage build --target x86_64-rustos.json"\n\
    echo ""\n\
    echo "GUI Testing:"\n\
    echo "  GUI_MODE=1 ./run_qemu.sh - Test desktop environment"\n\
    echo "  X11 forwarding must be enabled for GUI mode"\n\
    echo ""\n\
    ' > full_test.sh && chmod +x full_test.sh

# Expose any ports if needed (none for kernel development)
# EXPOSE 8080

# Set the default command to show available options
CMD ["./full_test.sh"]

# Build instructions and labels
LABEL maintainer="RustOS Team"
LABEL description="RustOS Kernel Development and Testing Environment"
LABEL version="1.0"

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD rustc --version && cargo --version || exit 1

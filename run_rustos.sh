#!/bin/bash

# RustOS QEMU Runner Script  
# This script runs RustOS in QEMU with proper configuration

echo "🚀 Starting RustOS in QEMU..."

# Build the kernel first
echo "🔨 Building RustOS kernel..."
cargo build --bin rustos

if [ $? -ne 0 ]; then
    echo "❌ Kernel build failed"
    exit 1
fi

echo "✅ Kernel built successfully"

# Create GRUB boot structure
echo "📦 Setting up multiboot environment..."
mkdir -p isodir/boot/grub

# Copy kernel
cp target/x86_64-rustos/debug/rustos isodir/boot/rustos

# Create GRUB config
cat > isodir/boot/grub/grub.cfg << EOF
menuentry "RustOS" {
    multiboot2 /boot/rustos
}
EOF

echo "🖥️  Launching QEMU with GRUB multiboot..."

# Try direct kernel boot first (simpler)
qemu-system-x86_64 \
    -kernel target/x86_64-rustos/debug/rustos \
    -m 256M \
    -display gtk \
    -serial stdio \
    -no-reboot

echo "🏁 RustOS session completed"

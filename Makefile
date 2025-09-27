# RustOS Kernel Makefile
# Provides convenient build targets for the RustOS kernel

# Default configuration
KERNEL_NAME = rustos
TARGET_X86 = x86_64-rustos.json
TARGET_ARM = aarch64-apple-rustos.json
DEFAULT_TARGET = $(TARGET_X86)
BUILD_SCRIPT = ./build_rustos.sh

# Build modes
DEBUG_DIR = target/$(DEFAULT_TARGET)/debug
RELEASE_DIR = target/$(DEFAULT_TARGET)/release
BOOTIMAGE_DEBUG = $(DEBUG_DIR)/bootimage-kernel.bin
BOOTIMAGE_RELEASE = $(RELEASE_DIR)/bootimage-kernel.bin

# QEMU configuration
QEMU_ARGS = -m 256M -serial stdio -device isa-debug-exit,iobase=0xf4,iosize=0x04
QEMU_X86 = qemu-system-x86_64
QEMU_ARM = qemu-system-aarch64

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
PURPLE = \033[0;35m
CYAN = \033[0;36m
NC = \033[0m # No Color

.PHONY: help all build build-release clean test run run-release install-deps check bootimage bootimage-release

# Default target
all: build

# Help target - shows available commands
help:
	@echo "$(PURPLE)RustOS Kernel Build System$(NC)"
	@echo "$(PURPLE)=========================$(NC)"
	@echo ""
	@echo "$(CYAN)Available targets:$(NC)"
	@echo "  $(GREEN)build$(NC)           - Build debug kernel"
	@echo "  $(GREEN)build-release$(NC)   - Build release kernel"
	@echo "  $(GREEN)bootimage$(NC)       - Create bootable debug image"
	@echo "  $(GREEN)bootimage-release$(NC) - Create bootable release image"
	@echo "  $(GREEN)run$(NC)             - Build and run debug kernel in QEMU"
	@echo "  $(GREEN)run-release$(NC)     - Build and run release kernel in QEMU"
	@echo "  $(GREEN)run-vnc$(NC)         - Run with VNC display"
	@echo "  $(GREEN)test$(NC)            - Run kernel tests"
	@echo "  $(GREEN)check$(NC)           - Check compilation without building"
	@echo "  $(GREEN)clean$(NC)           - Clean build artifacts"
	@echo "  $(GREEN)install-deps$(NC)    - Install build dependencies"
	@echo ""
	@echo "$(CYAN)Architecture targets:$(NC)"
	@echo "  $(GREEN)build-x86$(NC)       - Build for x86_64"
	@echo "  $(GREEN)build-arm$(NC)       - Build for AArch64"
	@echo "  $(GREEN)run-x86$(NC)         - Run x86_64 kernel in QEMU"
	@echo "  $(GREEN)run-arm$(NC)         - Run AArch64 kernel in QEMU"
	@echo ""
	@echo "$(CYAN)Utility targets:$(NC)"
	@echo "  $(GREEN)info$(NC)            - Show build information"
	@echo "  $(GREEN)size$(NC)            - Show kernel binary size"
	@echo "  $(GREEN)objdump$(NC)         - Show kernel disassembly"
	@echo "  $(GREEN)nm$(NC)              - Show kernel symbols"
	@echo ""
	@echo "$(CYAN)Examples:$(NC)"
	@echo "  make build           # Build debug kernel"
	@echo "  make run-release     # Build release kernel and run in QEMU"
	@echo "  make clean build     # Clean and build"
	@echo ""

# Install build dependencies
install-deps:
	@echo "$(BLUE)[INFO]$(NC) Installing build dependencies..."
	@$(BUILD_SCRIPT) --install-deps

# Check compilation without building
check:
	@echo "$(BLUE)[INFO]$(NC) Checking kernel compilation..."
	@$(BUILD_SCRIPT) --check-only

# Build debug kernel
build:
	@echo "$(BLUE)[INFO]$(NC) Building RustOS kernel (debug)..."
	@$(BUILD_SCRIPT)

# Build release kernel
build-release:
	@echo "$(BLUE)[INFO]$(NC) Building RustOS kernel (release)..."
	@$(BUILD_SCRIPT) --release

# Build for x86_64
build-x86:
	@echo "$(BLUE)[INFO]$(NC) Building RustOS kernel for x86_64..."
	@$(BUILD_SCRIPT) --target $(TARGET_X86)

# Build for AArch64
build-arm:
	@echo "$(BLUE)[INFO]$(NC) Building RustOS kernel for AArch64..."
	@$(BUILD_SCRIPT) --target $(TARGET_ARM)

# Create bootable debug image
bootimage: build
	@echo "$(BLUE)[INFO]$(NC) Creating bootable image (debug)..."
	@$(BUILD_SCRIPT) --bootimage

# Create bootable release image
bootimage-release: build-release
	@echo "$(BLUE)[INFO]$(NC) Creating bootable image (release)..."
	@$(BUILD_SCRIPT) --release --bootimage

# Run debug kernel in QEMU
run: bootimage
	@echo "$(BLUE)[INFO]$(NC) Running RustOS in QEMU (debug)..."
	@$(BUILD_SCRIPT) --qemu

# Run release kernel in QEMU
run-release: bootimage-release
	@echo "$(BLUE)[INFO]$(NC) Running RustOS in QEMU (release)..."
	@$(BUILD_SCRIPT) --release --qemu

# Run with VNC display (for headless systems)
run-vnc: bootimage
	@echo "$(BLUE)[INFO]$(NC) Running RustOS in QEMU with VNC..."
	@$(QEMU_X86) $(QEMU_ARGS) -drive format=raw,file=$(BOOTIMAGE_DEBUG) -vnc :1

# Run x86_64 kernel
run-x86:
	@echo "$(BLUE)[INFO]$(NC) Running x86_64 kernel in QEMU..."
	@$(BUILD_SCRIPT) --target $(TARGET_X86) --qemu --bootimage

# Run AArch64 kernel
run-arm:
	@echo "$(BLUE)[INFO]$(NC) Running AArch64 kernel in QEMU..."
	@$(BUILD_SCRIPT) --target $(TARGET_ARM) --qemu --bootimage

# Run kernel tests
test:
	@echo "$(BLUE)[INFO]$(NC) Running kernel tests..."
	@$(BUILD_SCRIPT) --test

# Clean build artifacts
clean:
	@echo "$(BLUE)[INFO]$(NC) Cleaning build artifacts..."
	@$(BUILD_SCRIPT) --clean

# Full clean and rebuild
rebuild: clean build

# Full clean and rebuild release
rebuild-release: clean build-release

# Show build information
info:
	@echo "$(PURPLE)RustOS Build Information$(NC)"
	@echo "$(PURPLE)========================$(NC)"
	@echo "$(CYAN)Kernel Name:$(NC) $(KERNEL_NAME)"
	@echo "$(CYAN)Default Target:$(NC) $(DEFAULT_TARGET)"
	@echo "$(CYAN)Build Script:$(NC) $(BUILD_SCRIPT)"
	@echo "$(CYAN)Debug Directory:$(NC) $(DEBUG_DIR)"
	@echo "$(CYAN)Release Directory:$(NC) $(RELEASE_DIR)"
	@if command -v rustc >/dev/null 2>&1; then \
		echo "$(CYAN)Rust Version:$(NC) $$(rustc --version)"; \
	else \
		echo "$(YELLOW)Rust not installed$(NC)"; \
	fi
	@if command -v qemu-system-x86_64 >/dev/null 2>&1; then \
		echo "$(CYAN)QEMU Version:$(NC) $$(qemu-system-x86_64 --version | head -n1)"; \
	else \
		echo "$(YELLOW)QEMU not available$(NC)"; \
	fi

# Show kernel binary size
size: build
	@if [ -f "$(DEBUG_DIR)/kernel" ]; then \
		echo "$(CYAN)Kernel Binary Size:$(NC)"; \
		ls -lh "$(DEBUG_DIR)/kernel" | awk '{print "  Debug: " $$5}'; \
	fi
	@if [ -f "$(RELEASE_DIR)/kernel" ]; then \
		ls -lh "$(RELEASE_DIR)/kernel" | awk '{print "  Release: " $$5}'; \
	fi
	@if [ -f "$(BOOTIMAGE_DEBUG)" ]; then \
		ls -lh "$(BOOTIMAGE_DEBUG)" | awk '{print "  Boot Image (Debug): " $$5}'; \
	fi
	@if [ -f "$(BOOTIMAGE_RELEASE)" ]; then \
		ls -lh "$(BOOTIMAGE_RELEASE)" | awk '{print "  Boot Image (Release): " $$5}'; \
	fi

# Show kernel disassembly
objdump: build
	@if command -v cargo-objdump >/dev/null 2>&1 && [ -f "$(DEBUG_DIR)/kernel" ]; then \
		echo "$(CYAN)Kernel Disassembly (first 50 lines):$(NC)"; \
		cargo objdump --target $(DEFAULT_TARGET) --bin kernel -- --disassemble --demangle | head -50; \
	else \
		echo "$(YELLOW)cargo-objdump not available or kernel not built$(NC)"; \
	fi

# Show kernel symbols
nm: build
	@if command -v cargo-nm >/dev/null 2>&1 && [ -f "$(DEBUG_DIR)/kernel" ]; then \
		echo "$(CYAN)Kernel Symbols:$(NC)"; \
		cargo nm --target $(DEFAULT_TARGET) --bin kernel | head -20; \
	else \
		echo "$(YELLOW)cargo-nm not available or kernel not built$(NC)"; \
	fi

# Debug targets for development
debug: build
	@echo "$(GREEN)Debug build completed$(NC)"
	@make size

release: build-release
	@echo "$(GREEN)Release build completed$(NC)"
	@make size

# Quick development cycle
dev: clean build run

# CI/CD friendly targets
ci-build: install-deps check build test
	@echo "$(GREEN)CI build pipeline completed$(NC)"

ci-test: install-deps test
	@echo "$(GREEN)CI test pipeline completed$(NC)"

# Create distribution package
dist: clean build-release bootimage-release
	@echo "$(BLUE)[INFO]$(NC) Creating distribution package..."
	@mkdir -p dist
	@cp $(BOOTIMAGE_RELEASE) dist/rustos-kernel.img
	@cp $(RELEASE_DIR)/kernel dist/rustos-kernel.elf
	@echo "$(GREEN)Distribution package created in dist/$(NC)"

# Benchmark build times
benchmark:
	@echo "$(BLUE)[INFO]$(NC) Benchmarking build times..."
	@echo "Clean build (debug):"
	@time make clean build >/dev/null
	@echo "Clean build (release):"
	@time make clean build-release >/dev/null
	@echo "Incremental build:"
	@time make build >/dev/null

# Watch for file changes and rebuild (requires inotify-tools)
watch:
	@if command -v inotifywait >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(NC) Watching for changes... Press Ctrl+C to stop"; \
		while inotifywait -r -e modify,create,delete src/ >/dev/null 2>&1; do \
			make build; \
		done; \
	else \
		echo "$(YELLOW)inotifywait not available. Install inotify-tools for file watching.$(NC)"; \
	fi

# Format code (requires rustfmt)
format:
	@if command -v rustfmt >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(NC) Formatting code..."; \
		cargo fmt; \
		echo "$(GREEN)Code formatted$(NC)"; \
	else \
		echo "$(YELLOW)rustfmt not available$(NC)"; \
	fi

# Lint code (requires clippy)
lint:
	@if command -v cargo-clippy >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(NC) Linting code..."; \
		cargo clippy --target $(DEFAULT_TARGET); \
	else \
		echo "$(YELLOW)clippy not available$(NC)"; \
	fi

# Generate documentation
docs:
	@echo "$(BLUE)[INFO]$(NC) Generating documentation..."
	@cargo doc --target $(DEFAULT_TARGET) --document-private-items
	@echo "$(GREEN)Documentation generated$(NC)"

# Show disk usage
disk-usage:
	@echo "$(CYAN)Disk Usage:$(NC)"
	@du -sh target/ 2>/dev/null || echo "No build artifacts"
	@du -sh ~/.cargo/ 2>/dev/null || echo "No Cargo cache"

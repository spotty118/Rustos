# RustOS Package Management System

## Quick Start

The RustOS package management system provides experimental adapters for working with Linux packages (.deb, .rpm, .apk), package repository APIs, and app stores.

### What's Implemented

✅ **Package Format Validation**
- .deb (Debian/Ubuntu) - Full validation
- .rpm (Fedora/RHEL) - Magic number validation
- .apk (Alpine Linux) - Format detection
- .rustos (Native) - Custom format specification

✅ **Archive Support**
- AR archive parsing (used by .deb packages)
- Member extraction and header parsing

✅ **Package Database**
- Track installed packages
- Search and query functionality
- Package status management

✅ **Adapter Framework**
- Extensible adapter pattern
- Repository API adapters (stubs)
- App store integration framework

### What's Not Yet Implemented

❌ Full archive extraction (TAR, GZIP, XZ)
❌ Network stack integration for downloads
❌ Dependency resolution
❌ File system installation
❌ Script execution
❌ Dynamic linking support

## Usage

### Basic Package Operations

```rust
use rustos::package::{PackageManager, PackageManagerType, PackageOperation};

// Create package manager for Debian packages
let mut pm = PackageManager::new(PackageManagerType::Apt);

// List installed packages
pm.execute_operation(PackageOperation::List, "")?;

// Search for packages
pm.execute_operation(PackageOperation::Search, "htop")?;

// Get package info
pm.execute_operation(PackageOperation::Info, "htop")?;
```

### Validate Package Format

```rust
use rustos::package::adapters::{PackageAdapter, DebAdapter};

let adapter = DebAdapter::new();

// Check if file is a valid .deb package
if adapter.validate(&package_data)? {
    println!("Valid .deb package");
    
    // Extract metadata
    let metadata = adapter.parse_metadata(&package_data)?;
    println!("Package: {} v{}", metadata.name, metadata.version);
}
```

### Parse AR Archives

```rust
use rustos::package::archive::ArArchive;

// Parse .deb package (which uses AR format)
let ar = ArArchive::parse(&deb_data)?;

// Find control archive
if let Some(control_data) = ar.find_member("control.tar.gz") {
    println!("Found control.tar.gz");
}

// List all archive members
for member in ar.members() {
    println!("{}: {} bytes", member.name, member.size);
}
```

### Manage Package Database

```rust
use rustos::package::{PackageDatabase, PackageInfo, PackageMetadata};

let mut db = PackageDatabase::new();

// Add package
let metadata = PackageMetadata::new(
    "htop".into(),
    "3.0.5".into(),
    "amd64".into()
);
let info = PackageInfo { metadata, /* ... */ };
db.add_package(info)?;

// Search packages
let results = db.search("http");

// Check if installed
if db.is_installed("htop") {
    println!("htop is installed");
}
```

## Documentation

- **[EXPERIMENTAL_PACKAGE_ADAPTERS.md](../docs/EXPERIMENTAL_PACKAGE_ADAPTERS.md)** - Complete adapter documentation
- **[LINUX_APP_SUPPORT.md](../docs/LINUX_APP_SUPPORT.md)** - Implementation roadmap and requirements
- **[package_manager_integration.md](../docs/package_manager_integration.md)** - Future vision

## Examples

See `examples/package_manager_demo.rs` for comprehensive usage examples.

## Architecture

```
src/package/
├── mod.rs              - Core types, errors, operations
├── types.rs            - Package metadata structures
├── adapters/           - Format-specific adapters
│   ├── deb.rs         - Debian .deb packages
│   ├── rpm.rs         - RPM packages
│   ├── apk.rs         - Alpine APK packages
│   └── native.rs      - Native RustOS packages
├── archive/            - Archive format support
│   └── ar.rs          - AR archive parser
├── database.rs         - Package database
├── api.rs              - Repository/app store APIs
└── manager.rs          - Package manager orchestrator
```

## Roadmap

To achieve full Linux package support, the following must be implemented:

**Phase 1: Dynamic Linker (3-4 months)**
- Parse PT_DYNAMIC segment
- Load shared libraries
- Symbol resolution
- Relocation processing

**Phase 2: POSIX Support (3-4 months)**
- Extended syscalls
- Filesystem support (ext4)
- POSIX threads
- IPC mechanisms

**Phase 3: Userspace Tools (4-5 months)**
- Shell (bash/sh)
- Core utilities
- Archive tools (tar, gzip)

**Phase 4: Package Management (2-3 months)**
- Complete archive extraction
- Dependency resolution
- Script execution
- Package operations

**Total: 15-20 months**

## Contributing

To contribute to package management:

1. **Start with the dynamic linker** - Highest impact
2. **Implement archive extraction** - TAR, GZIP support
3. **Add filesystem operations** - Install files to disk
4. **Build dependency resolver** - Package dependencies
5. **Test with real packages** - Validate against .deb files

## Current Status

**Maturity**: Experimental (Foundation Layer)
**Functionality**: ~15% complete
**Production Ready**: No - requires infrastructure

**What works today**:
- Package format detection
- AR archive parsing
- Database management
- Adapter architecture

**What's needed**:
- Dynamic linker
- C library (libc)
- Extended syscalls
- Filesystem support
- Network integration

## Testing

```bash
# Check compilation
cargo check --lib

# Run examples (when environment supports it)
cargo run --example package_manager_demo

# Run tests
cargo test --lib package
```

## License

Part of RustOS - See main repository license.

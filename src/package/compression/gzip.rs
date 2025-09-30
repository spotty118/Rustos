//! Gzip decompression support
//!
//! Basic gzip/DEFLATE decompression for package archives.
//! This is a minimal implementation - for production use, consider
//! integrating a full implementation like flate2 or miniz_oxide.

use alloc::vec::Vec;
use crate::package::{PackageResult, PackageError};

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const GZIP_HEADER_SIZE: usize = 10;

/// Gzip file flags
mod flags {
    pub const FTEXT: u8 = 0x01;
    pub const FHCRC: u8 = 0x02;
    pub const FEXTRA: u8 = 0x04;
    pub const FNAME: u8 = 0x08;
    pub const FCOMMENT: u8 = 0x10;
}

/// Gzip decoder
pub struct GzipDecoder;

impl GzipDecoder {
    /// Decode gzip-compressed data
    pub fn decode(data: &[u8]) -> PackageResult<Vec<u8>> {
        // Validate gzip header
        if data.len() < GZIP_HEADER_SIZE {
            return Err(PackageError::InvalidFormat(
                "File too small to be gzip".into()
            ));
        }

        if data[0] != GZIP_MAGIC[0] || data[1] != GZIP_MAGIC[1] {
            return Err(PackageError::InvalidFormat(
                "Invalid gzip magic number".into()
            ));
        }

        // Check compression method (should be 8 for DEFLATE)
        if data[2] != 8 {
            return Err(PackageError::InvalidFormat(
                format!("Unsupported compression method: {}", data[2])
            ));
        }

        let flags = data[3];
        let mut offset = GZIP_HEADER_SIZE;

        // Skip extra fields if present
        if flags & flags::FEXTRA != 0 {
            if data.len() < offset + 2 {
                return Err(PackageError::InvalidFormat("Truncated extra field".into()));
            }
            let xlen = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2 + xlen;
        }

        // Skip filename if present
        if flags & flags::FNAME != 0 {
            while offset < data.len() && data[offset] != 0 {
                offset += 1;
            }
            offset += 1; // skip null terminator
        }

        // Skip comment if present
        if flags & flags::FCOMMENT != 0 {
            while offset < data.len() && data[offset] != 0 {
                offset += 1;
            }
            offset += 1; // skip null terminator
        }

        // Skip header CRC if present
        if flags & flags::FHCRC != 0 {
            offset += 2;
        }

        if offset >= data.len() {
            return Err(PackageError::InvalidFormat(
                "Gzip header extends beyond data".into()
            ));
        }

        // The compressed data is everything except the last 8 bytes (CRC32 + size)
        if data.len() < offset + 8 {
            return Err(PackageError::InvalidFormat(
                "Gzip file too short for footer".into()
            ));
        }

        let compressed_data = &data[offset..data.len() - 8];

        // Decompress using miniz_oxide
        Self::decompress_deflate(compressed_data)
    }

    /// Decompress DEFLATE-compressed data using miniz_oxide
    fn decompress_deflate(compressed: &[u8]) -> PackageResult<Vec<u8>> {
        use miniz_oxide::inflate::decompress_to_vec;

        decompress_to_vec(compressed)
            .map_err(|e| PackageError::ExtractionError(
                format!("DEFLATE decompression failed: {:?}", e)
            ))
    }

    /// Validate gzip format without decompression
    pub fn validate(data: &[u8]) -> bool {
        data.len() >= 2 && data[0] == GZIP_MAGIC[0] && data[1] == GZIP_MAGIC[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gzip_validation() {
        let valid_gzip = [0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(GzipDecoder::validate(&valid_gzip));

        let invalid = [0x00, 0x00, 0x00, 0x00];
        assert!(!GzipDecoder::validate(&invalid));
    }
}

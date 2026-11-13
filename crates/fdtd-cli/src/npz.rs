//! Minimal NPZ writer: each array is a `.npy` v1.0 file (little-endian f64,
//! C order) stored uncompressed in a zip, which is exactly what `numpy.load` reads.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

/// One named array: `shape` is C order (last index fastest), `data.len()` must
/// equal the product of `shape` (an empty shape is a scalar).
pub struct Array<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub data: &'a [f64],
}

/// Serializes one array as `.npy` v1.0 bytes.
pub fn npy_bytes(shape: &[usize], data: &[f64]) -> Vec<u8> {
    let dims = match shape {
        [] => String::new(),
        [n] => format!("{n},"),
        _ => shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    };
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({dims}), }}");
    // Magic (6) + version (2) + length (2) + header + '\n' must be a multiple of 64.
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');

    let mut out = Vec::with_capacity(10 + header.len() + 8 * data.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Writes `arrays` to an NPZ file at `path`, creating or overwriting it.
// ponytail: whole array buffered in memory and no zip64, fine below 4 GB per array.
pub fn write_npz(path: &Path, arrays: &[Array]) -> io::Result<()> {
    for a in arrays {
        let expected: usize = a.shape.iter().product();
        if expected != a.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "array {} has {} values, shape needs {expected}",
                    a.name,
                    a.data.len()
                ),
            ));
        }
    }
    let mut zip = ZipWriter::new(File::create(path)?);
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    for a in arrays {
        zip.start_file(format!("{}.npy", a.name), opts)
            .map_err(io::Error::other)?;
        zip.write_all(&npy_bytes(a.shape, a.data))?;
    }
    zip.finish().map_err(io::Error::other)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn npz_roundtrip() {
        let path = std::env::temp_dir().join(format!("fdtd_npz_test_{}.npz", std::process::id()));
        let data: Vec<f64> = (0..6).map(|i| i as f64 * 0.5 - 1.0).collect();
        write_npz(
            &path,
            &[
                Array {
                    name: "ez",
                    shape: &[2, 3],
                    data: &data,
                },
                Array {
                    name: "dt",
                    shape: &[],
                    data: &[0.5],
                },
            ],
        )
        .unwrap();

        let mut archive = zip::ZipArchive::new(File::open(&path).unwrap()).unwrap();
        let mut bytes = Vec::new();
        archive
            .by_name("ez.npy")
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0, "data must start 64-byte aligned");
        let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
        assert!(header.contains("'descr': '<f8'"));
        assert!(header.contains("'shape': (2, 3)"));
        assert!(header.ends_with('\n'));
        let values: Vec<f64> = bytes[10 + hlen..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        assert_eq!(values, data);

        let mut scalar = Vec::new();
        archive
            .by_name("dt.npy")
            .unwrap()
            .read_to_end(&mut scalar)
            .unwrap();
        let hlen = u16::from_le_bytes([scalar[8], scalar[9]]) as usize;
        assert!(
            std::str::from_utf8(&scalar[10..10 + hlen])
                .unwrap()
                .contains("'shape': ()")
        );
        assert_eq!(
            f64::from_le_bytes(scalar[10 + hlen..].try_into().unwrap()),
            0.5
        );
        let _ = std::fs::remove_file(&path);

        let bad = [Array {
            name: "bad",
            shape: &[2],
            data: &[1.0],
        }];
        assert!(write_npz(&path, &bad).is_err());
        assert!(
            !path.exists(),
            "a rejected write must not leave a file behind"
        );
    }
}

//! Loading float32 tensors from a `.safetensors` checkpoint.
//!
//! Only the header is parsed up front; each tensor is read and converted
//! when it is asked for, and the caller packs it into its own layout before
//! asking for the next. So a model's load holds the checkpoint's bytes at
//! most once (and not at all when they stay in a file or outside the
//! process) plus one tensor at a time, whatever the model's size. Building
//! just the encoder or just the decoder never reads the other half.
//!
//! The bytes come from a file ([`Weights::load`]), a slice
//! ([`Weights::from_bytes`]) or any reader of byte ranges
//! ([`Weights::from_reader`]): the WebAssembly bindings read straight from
//! the browser's buffer, so the file is never copied into wasm memory.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;

/// Largest header the loader accepts, as in the reference implementation.
const MAX_HEADER_BYTES: u64 = 100_000_000;
/// Bytes read and converted at a time: tensors are converted to `f32`
/// without a byte copy of the whole tensor.
const READ_PIECE: usize = 64 * 1024;

/// One tensor's entry in the safetensors header.
#[derive(Debug, Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

/// Reads the bytes at a file offset into the whole buffer.
type ReadAt<'a> = Box<dyn Fn(u64, &mut [u8]) -> Result<()> + 'a>;

/// Where the tensors' bytes come from.
enum Source<'a> {
    /// The whole file in memory.
    Bytes(&'a [u8]),
    /// Byte ranges read on demand (a file, or a buffer outside the process).
    Reader(ReadAt<'a>),
}

impl fmt::Debug for Source<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(bytes) => write!(f, "Bytes({} bytes)", bytes.len()),
            Self::Reader(_) => f.write_str("Reader"),
        }
    }
}

/// The float32 tensors of a checkpoint, read on demand.
#[derive(Debug)]
pub struct Weights<'a> {
    index: HashMap<String, Entry>,
    /// File offset of the data section.
    data_start: u64,
    source: Source<'a>,
}

impl Weights<'static> {
    /// Opens a `.safetensors` file and reads its header.
    ///
    /// # Errors
    /// Fails when the file cannot be read or its header is invalid.
    pub fn load(path: &Path) -> Result<Self> {
        let open = || -> Result<Self> {
            let file = File::open(path)?;
            let len = file.metadata()?.len();
            Self::from_reader(
                len,
                Box::new(move |offset, buf| {
                    let mut reader = &file;
                    let _: u64 = reader.seek(SeekFrom::Start(offset))?;
                    reader.read_exact(buf)?;
                    Ok(())
                }),
            )
        };
        open().with_context(|| format!("reading {}", path.display()))
    }
}

impl<'a> Weights<'a> {
    /// Parses the header of a whole `.safetensors` file held in memory; the
    /// tensors are converted from `bytes` as they are asked for.
    ///
    /// # Errors
    /// Fails when the bytes are not a valid safetensors file.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let prefix: [u8; 8] = bytes
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .context("safetensors data is shorter than its header prefix")?;
        let header_len = usize::try_from(u64::from_le_bytes(prefix))?;
        let data_start = header_len
            .checked_add(8)
            .filter(|&start| start <= bytes.len())
            .context("invalid safetensors header length")?;
        let data_len = u64::try_from(bytes.len() - data_start)?;
        Ok(Self {
            index: parse_header(&bytes[8..data_start], data_len)?,
            data_start: u64::try_from(data_start)?,
            source: Source::Bytes(bytes),
        })
    }

    /// Reads a `.safetensors` file of `len` bytes through `read`, which
    /// fills a buffer with the bytes at a file offset: the header now, each
    /// tensor when it is asked for, a piece at a time.
    ///
    /// # Errors
    /// Fails when `read` fails or the header is invalid.
    pub fn from_reader(len: u64, read: ReadAt<'a>) -> Result<Self> {
        let mut prefix = [0_u8; 8];
        ensure!(
            len >= 8,
            "safetensors data is shorter than its header prefix"
        );
        read(0, &mut prefix)?;
        let header_len = u64::from_le_bytes(prefix);
        ensure!(
            header_len <= MAX_HEADER_BYTES && header_len <= len - 8,
            "invalid safetensors header length {header_len}"
        );
        let mut header = vec![0_u8; usize::try_from(header_len)?];
        read(8, &mut header)?;
        let data_start = 8 + header_len;
        Ok(Self {
            index: parse_header(&header, len - data_start)?,
            data_start,
            source: Source::Reader(read),
        })
    }

    /// Whether the checkpoint holds `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// The shape of tensor `name`, if the checkpoint holds it.
    #[must_use]
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|entry| entry.shape.as_slice())
    }

    /// The values of tensor `name`, checking its shape.
    ///
    /// # Errors
    /// Fails when the tensor is missing, is not float32 or has a different
    /// shape, or its bytes cannot be read.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let entry = self
            .index
            .get(name)
            .with_context(|| format!("checkpoint is missing tensor {name}"))?;
        if entry.dtype != "F32" {
            bail!("tensor {name} is not float32");
        }
        if entry.shape != shape {
            bail!(
                "tensor {name} has shape {}, expected {}",
                fmt_shape(&entry.shape),
                fmt_shape(shape)
            );
        }
        let (start, end) = entry.data_offsets;
        let len = usize::try_from(end - start)?;
        match &self.source {
            Source::Bytes(bytes) => {
                let at = usize::try_from(self.data_start + start)?;
                Ok(to_f32(&bytes[at..at + len]))
            }
            Source::Reader(read) => {
                let mut values = Vec::with_capacity(len / 4);
                let mut piece = vec![0_u8; READ_PIECE.min(len)];
                let mut done = 0;
                while done < len {
                    let buf = &mut piece[..READ_PIECE.min(len - done)];
                    read(self.data_start + start + u64::try_from(done)?, buf)
                        .with_context(|| format!("reading tensor {name}"))?;
                    values.extend(
                        buf.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|&b| f32::from_le_bytes(b)),
                    );
                    done += buf.len();
                }
                Ok(values)
            }
        }
    }
}

/// Parses the JSON header and checks every entry against the data size.
fn parse_header(header: &[u8], data_len: u64) -> Result<HashMap<String, Entry>> {
    let mut raw: HashMap<String, serde_json::Value> =
        serde_json::from_slice(header).context("parsing the safetensors header")?;
    drop(raw.remove("__metadata__"));
    let mut index = HashMap::with_capacity(raw.len());
    for (name, value) in raw {
        let entry: Entry = serde_json::from_value(value)
            .with_context(|| format!("parsing the header entry of tensor {name}"))?;
        let (start, end) = entry.data_offsets;
        ensure!(
            start <= end && end <= data_len,
            "tensor {name} lies outside the file"
        );
        if entry.dtype == "F32" {
            let values = entry
                .shape
                .iter()
                .try_fold(1_u64, |n, &dim| n.checked_mul(u64::try_from(dim).ok()?));
            ensure!(
                values.and_then(|n| n.checked_mul(4)) == Some(end - start),
                "tensor {name} has {} bytes, which does not match shape {}",
                end - start,
                fmt_shape(&entry.shape)
            );
        }
        drop(index.insert(name, entry));
    }
    Ok(index)
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

fn fmt_shape(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("[{}]", dims.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<u8> {
        let header = br#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,1],"data_offsets":[0,8]},"n":{"dtype":"I64","shape":[1],"data_offsets":[8,16]}}"#;
        let mut bytes = u64::try_from(header.len())
            .unwrap_or(0)
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.5_f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.0_f32).to_le_bytes());
        bytes.extend_from_slice(&7_i64.to_le_bytes());
        bytes
    }

    #[test]
    fn reads_tensors_from_bytes() -> Result<()> {
        let bytes = sample();
        let weights = Weights::from_bytes(&bytes)?;
        assert_eq!(weights.get("a", &[2, 1])?, [1.5, -2.0]);
        assert!(weights.contains("n"));
        assert!(weights.get("n", &[1]).is_err(), "not float32");
        assert!(weights.get("a", &[2]).is_err(), "wrong shape");
        assert!(weights.get("missing", &[1]).is_err());
        Ok(())
    }

    #[test]
    fn file_and_bytes_agree() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "encodec-rs-weights-test-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, sample())?;
        let from_file = Weights::load(&path);
        let values = from_file.and_then(|w| w.get("a", &[2, 1]));
        std::fs::remove_file(&path)?;
        assert_eq!(values?, vec![1.5, -2.0]);
        Ok(())
    }

    /// A reader sees every tensor in pieces; the values equal the slice's.
    #[test]
    fn reader_reads_large_tensors_in_pieces() -> Result<()> {
        let values: Vec<f32> = (0..40_000_u16).map(f32::from).collect();
        let header = format!(
            r#"{{"t":{{"dtype":"F32","shape":[200,200],"data_offsets":[0,{}]}}}}"#,
            values.len() * 4
        );
        let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        let calls = std::cell::Cell::new(0);
        let reader = Weights::from_reader(
            u64::try_from(bytes.len())?,
            Box::new(|offset, buf| {
                calls.set(calls.get() + 1);
                let at = usize::try_from(offset)?;
                buf.copy_from_slice(bytes.get(at..at + buf.len()).context("past the end")?);
                Ok(())
            }),
        )?;
        assert_eq!(reader.get("t", &[200, 200])?, values);
        assert_eq!(Weights::from_bytes(&bytes)?.get("t", &[200, 200])?, values);
        // Prefix, header, then 160 000 bytes in pieces of 64 KiB.
        assert_eq!(calls.get(), 2 + 3);
        Ok(())
    }

    #[test]
    fn rejects_truncated_data() {
        let mut bytes = sample();
        bytes.truncate(bytes.len() - 12);
        assert!(Weights::from_bytes(&bytes).is_err());
        assert!(Weights::from_bytes(&[1, 2, 3]).is_err());
    }
}

//! Loading float32 tensors from a `.safetensors` checkpoint.
//!
//! A checkpoint on disk is read lazily: only the header is parsed up front and
//! each tensor is read when it is asked for, so building just the encoder or
//! just the decoder never pulls the other half of the file into memory.
//! [`Weights::from_bytes`] serves targets without a filesystem.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::Path;

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;

/// Largest header the loader accepts, as in the reference implementation.
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// One tensor's entry in the safetensors header.
#[derive(Debug, Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

#[derive(Debug)]
enum Source {
    /// Tensors are read from the file on demand; offsets are relative to `data_start`.
    File { file: File, data_start: u64 },
    /// Tensors already converted to `f32`, keyed by name.
    Memory(HashMap<String, Vec<f32>>),
}

/// The float32 tensors of a checkpoint.
#[derive(Debug)]
pub struct Weights {
    index: HashMap<String, Entry>,
    source: Source,
}

impl Weights {
    /// Opens a `.safetensors` file and reads its header.
    ///
    /// # Errors
    /// Fails when the file cannot be read or its header is invalid.
    pub fn load(path: &Path) -> Result<Self> {
        let open = || -> Result<Self> {
            let mut file = File::open(path)?;
            let file_len = file.metadata()?.len();
            let mut prefix = [0_u8; 8];
            file.read_exact(&mut prefix)?;
            let header_len = u64::from_le_bytes(prefix);
            ensure!(
                header_len <= MAX_HEADER_BYTES && header_len <= file_len.saturating_sub(8),
                "invalid safetensors header length {header_len}"
            );
            let mut header = vec![0_u8; usize::try_from(header_len)?];
            file.read_exact(&mut header)?;
            let data_start = 8 + header_len;
            let index = parse_header(&header, file_len - data_start)?;
            Ok(Self {
                index,
                source: Source::File { file, data_start },
            })
        };
        open().with_context(|| format!("reading {}", path.display()))
    }

    /// Parses a whole `.safetensors` file held in memory.
    ///
    /// # Errors
    /// Fails when the bytes are not a valid safetensors file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let prefix: [u8; 8] = bytes
            .get(..8)
            .and_then(|b| b.try_into().ok())
            .context("safetensors data is shorter than its header prefix")?;
        let header_len = usize::try_from(u64::from_le_bytes(prefix))?;
        let data_start = header_len
            .checked_add(8)
            .filter(|&start| start <= bytes.len())
            .context("invalid safetensors header length")?;
        let data = &bytes[data_start..];
        let index = parse_header(&bytes[8..data_start], u64::try_from(data.len())?)?;
        let mut tensors = HashMap::with_capacity(index.len());
        for (name, entry) in &index {
            if entry.dtype != "F32" {
                continue;
            }
            let start = usize::try_from(entry.data_offsets.0)?;
            let end = usize::try_from(entry.data_offsets.1)?;
            drop(tensors.insert(name.clone(), to_f32(&data[start..end])));
        }
        Ok(Self {
            index,
            source: Source::Memory(tensors),
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
    /// Fails when the tensor is missing, is not float32 or has a different shape.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Cow<'_, [f32]>> {
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
        match &self.source {
            Source::Memory(tensors) => tensors
                .get(name)
                .map(|values| Cow::Borrowed(values.as_slice()))
                .with_context(|| format!("checkpoint is missing tensor {name}")),
            Source::File { file, data_start } => {
                let (start, end) = entry.data_offsets;
                let mut bytes = vec![0_u8; usize::try_from(end - start)?];
                let mut reader = file;
                let _: u64 = reader
                    .seek(SeekFrom::Start(data_start + start))
                    .with_context(|| format!("seeking to tensor {name}"))?;
                reader
                    .read_exact(&mut bytes)
                    .with_context(|| format!("reading tensor {name}"))?;
                Ok(Cow::Owned(to_f32(&bytes)))
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
        let weights = Weights::from_bytes(&sample())?;
        assert_eq!(&*weights.get("a", &[2, 1])?, &[1.5, -2.0]);
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
        let values = from_file.and_then(|w| w.get("a", &[2, 1]).map(Cow::into_owned));
        std::fs::remove_file(&path)?;
        assert_eq!(values?, vec![1.5, -2.0]);
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

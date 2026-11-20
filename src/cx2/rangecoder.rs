//! Carry-less 64-bit range coder (Subbotin style) over integer frequencies.
//!
//! All arithmetic is on `u64` with explicit wrapping, so the encoder and
//! decoder agree bit for bit on every platform. Symbol totals may be any
//! value up to [`MAX_TOTAL`].

use anyhow::{Result, ensure};

const TOP: u64 = 1 << 56;
const BOTTOM: u64 = 1 << 48;

/// Largest frequency total a symbol may be coded with.
pub const MAX_TOTAL: u64 = BOTTOM;

/// Range encoder writing to an owned byte buffer.
#[derive(Debug, Default)]
pub struct Encoder {
    low: u64,
    range: u64,
    out: Vec<u8>,
}

impl Encoder {
    /// Creates an empty encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            low: 0,
            range: u64::MAX,
            out: Vec::new(),
        }
    }

    /// Codes a symbol occupying `[cum, cum + freq)` out of `total`.
    ///
    /// # Errors
    /// Fails when the interval is empty or outside `total`, or `total` exceeds [`MAX_TOTAL`].
    pub fn encode(&mut self, cum: u64, freq: u64, total: u64) -> Result<()> {
        ensure!(
            freq > 0 && total <= MAX_TOTAL && cum.checked_add(freq).is_some_and(|e| e <= total),
            "invalid coding interval {cum}+{freq}/{total}"
        );
        let step = self.range / total;
        self.low = self.low.wrapping_add(step * cum);
        self.range = step * freq;
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) >= TOP {
                if self.range >= BOTTOM {
                    break;
                }
                self.range = self.low.wrapping_neg() & (BOTTOM - 1);
            }
            self.out.push(self.low.to_be_bytes()[0]);
            self.low <<= 8;
            self.range <<= 8;
        }
        Ok(())
    }

    /// Flushes the coder state and returns the bytes.
    ///
    /// The decoder reads zeros past the end, so it is enough to write the shortest
    /// byte prefix whose zero-padded value lies inside the final interval.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        let mut tail = self.low.to_be_bytes().to_vec();
        for keep in 1..8_u32 {
            let mask = (1_u64 << (64 - 8 * keep)) - 1;
            let fits = self
                .low
                .checked_add(mask)
                .map(|v| v & !mask)
                .filter(|v| v - self.low < self.range);
            if let Some(value) = fits {
                tail = value.to_be_bytes()[..usize::try_from(keep).unwrap_or(8)].to_vec();
                break;
            }
        }
        self.out.extend_from_slice(&tail);
        self.out
    }
}

/// Range decoder over a byte slice; reads past the end yield zero bytes.
#[derive(Debug)]
pub struct Decoder<'a> {
    low: u64,
    range: u64,
    code: u64,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    /// Starts decoding `bytes`.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        let mut decoder = Self {
            low: 0,
            range: u64::MAX,
            code: 0,
            bytes,
            pos: 0,
        };
        for _ in 0..8 {
            decoder.code = (decoder.code << 8) | decoder.next_byte();
        }
        decoder
    }

    fn next_byte(&mut self) -> u64 {
        let byte = self.bytes.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        u64::from(byte)
    }

    /// Whether the decoder has read further past the end of its input than any stream
    /// from [`Encoder::finish`] makes it: the input is truncated or not a coded stream.
    ///
    /// The encoder writes one byte per normalisation step plus at least one at the end,
    /// and the decoder is eight bytes ahead of those steps, so an intact stream never
    /// takes it more than seven bytes past the end.
    #[must_use]
    pub const fn overrun(&self) -> bool {
        self.pos > self.bytes.len() + 8
    }

    /// Returns the cumulative-frequency target of the next symbol, in `0..total`.
    ///
    /// # Errors
    /// Fails when `total` is zero or exceeds [`MAX_TOTAL`].
    pub fn target(&mut self, total: u64) -> Result<u64> {
        ensure!(total > 0 && total <= MAX_TOTAL, "invalid total {total}");
        self.range /= total;
        Ok((self.code.wrapping_sub(self.low) / self.range).min(total - 1))
    }

    /// Consumes the symbol occupying `[cum, cum + freq)`; call after [`Self::target`].
    pub fn consume(&mut self, cum: u64, freq: u64) {
        self.low = self.low.wrapping_add(self.range * cum);
        self.range *= freq;
        loop {
            if (self.low ^ self.low.wrapping_add(self.range)) >= TOP {
                if self.range >= BOTTOM {
                    break;
                }
                self.range = self.low.wrapping_neg() & (BOTTOM - 1);
            }
            self.code = (self.code << 8) | self.next_byte();
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    /// Number of input bytes consumed so far (including the 8-byte preload).
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random stream for tests.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state >> 33
    }

    #[test]
    fn round_trips_skewed_symbols() -> Result<()> {
        let freqs: Vec<u64> = (0..1024_u64).map(|i| 1 + (i * i) % 977 * 4096).collect();
        let total: u64 = freqs.iter().sum();
        let mut state = 7;
        let symbols: Vec<usize> = (0..20_000)
            .map(|_| {
                let r = lcg(&mut state);
                usize::try_from(if r.is_multiple_of(3) {
                    r % 1024
                } else {
                    5 + r % 3
                })
                .unwrap_or(0)
            })
            .collect();
        let mut encoder = Encoder::new();
        for &s in &symbols {
            let cum: u64 = freqs[..s].iter().sum();
            encoder.encode(cum, freqs[s], total)?;
        }
        let bytes = encoder.finish();
        let mut decoder = Decoder::new(&bytes);
        for &s in &symbols {
            let target = decoder.target(total)?;
            let mut cum = 0;
            let mut found = 0;
            for (i, &f) in freqs.iter().enumerate() {
                if target < cum + f {
                    found = i;
                    break;
                }
                cum += f;
            }
            assert_eq!(found, s);
            decoder.consume(cum, freqs[found]);
        }
        Ok(())
    }

    #[test]
    fn rejects_empty_interval() {
        let mut encoder = Encoder::new();
        assert!(encoder.encode(0, 0, 10).is_err());
        assert!(encoder.encode(8, 4, 10).is_err());
    }
}

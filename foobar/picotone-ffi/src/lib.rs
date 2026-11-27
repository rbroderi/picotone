//! A C interface to picotone's [`Player`], for the foobar2000 input component
//! (`foo_input_pico`), which links it as a static library.
//!
//! - [`pt_init`] names the folders holding the weights and the cx2 tables;
//!   the tables load at once, each network the first time a file needs it.
//! - [`pt_header`] and [`pt_tags`] read a file's headers without decoding.
//! - [`pt_open`] opens a file (with its `.picol` layer, when given) at a
//!   sample, [`pt_decode`] returns the next block of final samples, and
//!   [`pt_close`] frees it. A seek is a new [`pt_open`] at the new sample.
//!
//! Every function catches panics, and reports failures through its return
//! value; [`pt_last_error`] then gives the message (per thread). All strings
//! are UTF-8.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context as _, Result, anyhow, bail};
use picotone::ModelKind;
use picotone::model::LoadOptions;
use picotone::pico::{self, ChannelLayout, Codec, DacStream};
use picotone::player::{Net, Player, Tables, no_clock};
use picotone::vocos::VocosDecoder;

/// The codec of a file, in [`PtHeader::codec`].
const CODEC_HQ: u32 = 0;
const CODEC_ENCODEC_48K: u32 = 1;
const CODEC_ENCODEC_24K: u32 = 2;

/// What [`pt_header`] reports about a file.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PtHeader {
    /// 0 = HQ, 1 = EnCodec 48 kHz, 2 = EnCodec 24 kHz.
    pub codec: u32,
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Output channels.
    pub channels: u32,
    /// Codebooks per frame (HQ: of the first coded channel, with the layer
    /// when it fits).
    pub codebooks: u32,
    /// HQ mid/side: the side's codebooks; otherwise 0.
    pub side_codebooks: u32,
    /// HQ stereo layout: 0 = mono, 1 = left/right, 2 = mid/side,
    /// 3 = parametric; EnCodec: 255.
    pub layout: u32,
    /// 1 when the file is a seamless loop.
    pub looped: u32,
    /// 1 when the layer given was this file's and is used.
    pub layered: u32,
    /// Playable length in samples per channel.
    pub length: u64,
}

/// An open file being decoded.
#[derive(Debug)]
pub struct PtStream {
    player: Player,
    /// The last block [`pt_decode`] returned, interleaved.
    block: Vec<f32>,
}

/// The loaded tables and networks.
#[derive(Debug)]
struct Engine {
    weights: PathBuf,
    tables: Tables,
    hq: OnceLock<Result<Net, String>>,
    encodec48: OnceLock<Result<Net, String>>,
    encodec24: OnceLock<Result<Net, String>>,
}

static ENGINE: OnceLock<Engine> = OnceLock::new();

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: &str) {
    let text = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = text);
}

/// Runs `f`, turning an error or a panic into `failed` and the thread's last
/// error.
fn guard<T>(failed: T, f: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            set_error(&format!("{error:#}"));
            failed
        }
        Err(_) => {
            set_error("picotone panicked");
            failed
        }
    }
}

/// A UTF-8 path from C.
///
/// # Safety
/// `text` must be null or a NUL-terminated string.
unsafe fn path_arg(text: *const c_char) -> Result<PathBuf> {
    if text.is_null() {
        bail!("a folder path is null");
    }
    // SAFETY: the caller passes a NUL-terminated string.
    let text = unsafe { CStr::from_ptr(text) };
    Ok(PathBuf::from(
        text.to_str().context("a folder path is not UTF-8")?,
    ))
}

/// A byte buffer from C; an empty slice when `len` is 0.
///
/// # Safety
/// `data` must point to `len` readable bytes (or be anything when `len` is 0).
unsafe fn bytes_arg<'a>(data: *const u8, len: usize) -> Result<&'a [u8]> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        bail!("a buffer is null");
    }
    // SAFETY: the caller passes `len` readable bytes at `data`.
    Ok(unsafe { std::slice::from_raw_parts(data, len) })
}

fn engine() -> Result<&'static Engine> {
    ENGINE.get().context("pt_init has not been called")
}

impl Engine {
    fn load(weights: PathBuf, tables_dir: &Path) -> Result<Self> {
        let mut tables = Tables::new();
        let entries = std::fs::read_dir(tables_dir)
            .with_context(|| format!("reading the tables folder {}", tables_dir.display()))?;
        for entry in entries {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("ecxm"))
            {
                let bytes = std::fs::read(&path)?;
                // Tables of other kinds or versions are skipped: a file that
                // needs one reports it missing.
                drop(tables.add(&bytes));
            }
        }
        Ok(Self {
            weights,
            tables,
            hq: OnceLock::new(),
            encodec48: OnceLock::new(),
            encodec24: OnceLock::new(),
        })
    }

    /// The network a file of `codec` needs, loaded on first use.
    fn net(&self, codec: Codec) -> Result<Net> {
        let dir = &self.weights;
        let decoder_only = LoadOptions {
            encoder: false,
            decoder: true,
            codebooks: None,
        };
        let load_encodec = |kind: ModelKind| -> Result<Net, String> {
            kind.load_with(dir, decoder_only)
                .map(|model| Net::Encodec(Box::leak(Box::new(model))))
                .map_err(|e| format!("{e:#}"))
        };
        let slot = match codec {
            Codec::Dac44 => self.hq.get_or_init(|| {
                VocosDecoder::load(&dir.join("dac44_vocos.safetensors"))
                    .map(|decoder| Net::Hq(Box::leak(Box::new(decoder))))
                    .map_err(|e| format!("{e:#}"))
            }),
            Codec::Encodec(ModelKind::Stereo48k) => self
                .encodec48
                .get_or_init(|| load_encodec(ModelKind::Stereo48k)),
            Codec::Encodec(ModelKind::Mono24k) => self
                .encodec24
                .get_or_init(|| load_encodec(ModelKind::Mono24k)),
        };
        slot.clone().map_err(|e| anyhow!(e))
    }
}

/// Loads the cx2 tables (`*.ecxm`) from `tables_dir` and remembers
/// `weights_dir` for the networks. Only the first successful call counts.
/// Returns 0, or -1 on failure.
///
/// # Safety
/// Both arguments must be NUL-terminated UTF-8 strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_init(weights_dir: *const c_char, tables_dir: *const c_char) -> i32 {
    guard(-1, || {
        if ENGINE.get().is_some() {
            return Ok(0);
        }
        // SAFETY: the caller passes NUL-terminated strings.
        let (weights, tables) = unsafe { (path_arg(weights_dir)?, path_arg(tables_dir)?) };
        let engine = Engine::load(weights, &tables)?;
        // A racing first call may have won; its engine is as good.
        drop(ENGINE.set(engine));
        Ok(0)
    })
}

/// The message of the last failure on this thread (never null; valid until
/// the next call on this thread).
#[unsafe(no_mangle)]
pub extern "C" fn pt_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

fn header(bytes: &[u8], layer: &[u8]) -> Result<PtHeader> {
    Ok(match pico::codec_of(bytes)? {
        Codec::Dac44 => {
            let mut stream = DacStream::parse(bytes)?;
            let layered = !layer.is_empty() && stream.attach_layer(layer).is_ok();
            let info = &stream.info;
            let books = |i: usize| info.codebooks.get(i).copied().unwrap_or(0);
            let layout = match info.layout {
                ChannelLayout::Mono => 0,
                ChannelLayout::LeftRight => 1,
                ChannelLayout::MidSide => 2,
                ChannelLayout::Parametric(_) => 3,
            };
            PtHeader {
                codec: CODEC_HQ,
                sample_rate: picotone::dac::SAMPLE_RATE,
                channels: u32::try_from(info.layout.output_channels())?,
                codebooks: u32::try_from(books(0))?,
                side_codebooks: if layout == 2 {
                    u32::try_from(books(1))?
                } else {
                    0
                },
                layout,
                looped: u32::from(info.looped),
                layered: u32::from(layered),
                length: u64::try_from(info.playable_samples())?,
            }
        }
        Codec::Encodec(kind) => {
            let info = pico::Stream::parse(bytes)?.info;
            let (codec, sample_rate, channels) = match kind {
                ModelKind::Stereo48k => (CODEC_ENCODEC_48K, 48_000, 2),
                ModelKind::Mono24k => (CODEC_ENCODEC_24K, 24_000, 1),
            };
            PtHeader {
                codec,
                sample_rate,
                channels,
                codebooks: u32::try_from(info.num_codebooks)?,
                side_codebooks: 0,
                layout: 255,
                looped: u32::from(info.looped),
                layered: 0,
                length: u64::try_from(info.playable_samples())?,
            }
        }
    })
}

/// Reads a file's headers into `out` without decoding audio or loading a
/// network. `layer` (may be null with `layer_len` 0) is the file's `.picol`;
/// `out.layered` says whether it fits. Returns 0, or -1 on failure.
///
/// # Safety
/// `pico` must hold `pico_len` bytes, `layer` `layer_len` bytes, and `out`
/// must point to a writable [`PtHeader`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_header(
    pico: *const u8,
    pico_len: usize,
    layer: *const u8,
    layer_len: usize,
    out: *mut PtHeader,
) -> i32 {
    guard(-1, || {
        // SAFETY: the caller passes readable buffers of these lengths.
        let (bytes, layer) = unsafe { (bytes_arg(pico, pico_len)?, bytes_arg(layer, layer_len)?) };
        let value = header(bytes, layer)?;
        if out.is_null() {
            bail!("the header output is null");
        }
        // SAFETY: the caller passes a writable PtHeader.
        unsafe { out.write(value) };
        Ok(0)
    })
}

/// Writes a file's tags (`KEY=value`, each NUL-terminated, then one more NUL)
/// into `buf` when they fit in `cap` bytes, and returns the bytes they need;
/// -1 on failure.
///
/// # Safety
/// `pico` must hold `pico_len` bytes and `buf` must have `cap` writable bytes
/// (or be anything when `cap` is 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_tags(
    pico: *const u8,
    pico_len: usize,
    buf: *mut c_char,
    cap: usize,
) -> isize {
    guard(-1, || {
        // SAFETY: the caller passes a readable buffer of this length.
        let bytes = unsafe { bytes_arg(pico, pico_len)? };
        let tags = match pico::codec_of(bytes)? {
            Codec::Dac44 => DacStream::parse(bytes)?.info.tags,
            Codec::Encodec(_) => pico::Stream::parse(bytes)?.info.tags,
        };
        let mut packed = Vec::new();
        for tag in &tags {
            packed.extend(tag.bytes().filter(|&b| b != 0));
            packed.push(0);
        }
        packed.push(0);
        if packed.len() <= cap && !buf.is_null() {
            // SAFETY: `buf` has `cap` writable bytes, and `packed` fits.
            unsafe {
                std::ptr::copy_nonoverlapping(packed.as_ptr(), buf.cast::<u8>(), packed.len());
            }
        }
        Ok(isize::try_from(packed.len())?)
    })
}

/// Opens a file for decoding from playable sample `start_sample`; returns the
/// stream, or null on failure.
///
/// The file's network loads on first use. `layer` (may be null with
/// `layer_len` 0) is its `.picol`; one that does not fit fails the call, so
/// the caller opens again without it. The buffers can be freed once this
/// returns.
///
/// # Safety
/// `pico` must hold `pico_len` bytes and `layer` `layer_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_open(
    pico: *const u8,
    pico_len: usize,
    layer: *const u8,
    layer_len: usize,
    start_sample: u64,
) -> *mut PtStream {
    guard(std::ptr::null_mut(), || {
        // SAFETY: the caller passes readable buffers of these lengths.
        let (bytes, layer) = unsafe { (bytes_arg(pico, pico_len)?, bytes_arg(layer, layer_len)?) };
        let engine = engine()?;
        let net = engine.net(pico::codec_of(bytes)?)?;
        let layer = (!layer.is_empty()).then_some(layer);
        let start = usize::try_from(start_sample)?;
        let player = Player::open(net, bytes, layer, &engine.tables, start, no_clock)?;
        Ok(Box::into_raw(Box::new(PtStream {
            player,
            block: Vec::new(),
        })))
    })
}

/// Decodes up to `max_units` more units (about a second each); returns 1 with
/// a block, 0 at the end, -1 on failure.
///
/// `data` then points at the final samples, interleaved, `frames` per
/// channel; the block stays valid until the next call on this stream.
///
/// # Safety
/// `stream` must come from [`pt_open`] and not be closed; `data` and `frames`
/// must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_decode(
    stream: *mut PtStream,
    max_units: u32,
    data: *mut *const f32,
    frames: *mut usize,
) -> i32 {
    guard(-1, || {
        if stream.is_null() || data.is_null() || frames.is_null() {
            bail!("a pt_decode argument is null");
        }
        // SAFETY: the caller passes a live stream from pt_open, used by one
        // thread at a time.
        let stream = unsafe { &mut *stream };
        let units = usize::try_from(max_units)?.max(1);
        let Some(chunk) = stream.player.decode_units(units)? else {
            return Ok(0);
        };
        stream.block = chunk.audio.data().to_vec();
        // SAFETY: the caller passes writable outputs.
        unsafe {
            data.write(stream.block.as_ptr());
            frames.write(chunk.audio.len());
        }
        Ok(1)
    })
}

/// Frees a stream from [`pt_open`] (null is ignored).
///
/// # Safety
/// `stream` must come from [`pt_open`] and not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_close(stream: *mut PtStream) {
    if !stream.is_null() {
        // SAFETY: the caller passes a stream from pt_open, not used after this.
        drop(unsafe { Box::from_raw(stream) });
    }
}

//! A C interface to picotone's [`Player`] and encoder, for the foobar2000
//! input component (`foo_input_pico`) and the `FFmpeg` wrapper, which link it
//! as a static library. The header is `include/picotone.h`.
//!
//! - [`pt_init`] names the folders holding the weights and the cx2 tables
//!   (or finds them as the command line does); the tables load at once, each
//!   network the first time a file needs it.
//! - [`pt_header`] and [`pt_tags`] read a file's headers without decoding.
//! - [`pt_open`] opens a file (with its `.picol` layer, when given) at a
//!   sample, [`pt_decode`] returns the next block of final samples, and
//!   [`pt_close`] frees it. A seek is a new [`pt_open`] at the new sample.
//! - [`pt_encoder_new`] starts an encode of interleaved float audio,
//!   [`pt_encoder_set`] and [`pt_encoder_tag`] take `picotone encode`'s
//!   options, [`pt_encoder_push`] adds audio, [`pt_encoder_finish`] codes it
//!   to the `.pico` bytes (and a `.picol` layer's) exactly as the command
//!   line does, and [`pt_encoder_free`] frees it.
//!
//! Every function catches panics, and reports failures through its return
//! value; [`pt_last_error`] then gives the message (per thread). All strings
//! are UTF-8.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use picotone::ModelKind;
use picotone::audio::Audio;
use picotone::encode::{self, Encoded, Models, Options};
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

/// An encode in progress: the audio pushed so far and the options, then the
/// encoded file.
#[derive(Debug)]
pub struct PtEncoder {
    sample_rate: u32,
    /// The audio, one vector per channel.
    channels: Vec<Vec<f32>>,
    options: Options,
    /// Whether [`pt_encoder_finish`] has been called.
    finished: bool,
    /// Its result.
    encoded: Option<Encoded>,
}

/// The loaded tables and networks.
#[derive(Debug)]
struct Engine {
    weights: PathBuf,
    tables: Tables,
    /// The HQ decoder, shared with the encoder's stereo pick.
    hq: OnceLock<Result<&'static Arc<VocosDecoder>, String>>,
    encodec48: OnceLock<Result<Net, String>>,
    encodec24: OnceLock<Result<Net, String>>,
    /// The encoder's networks and tables; one encode at a time.
    encoder: Mutex<Models>,
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

/// A UTF-8 path from C; `None` for null.
///
/// # Safety
/// `text` must be null or a NUL-terminated string.
unsafe fn path_arg(text: *const c_char) -> Result<Option<PathBuf>> {
    if text.is_null() {
        return Ok(None);
    }
    // SAFETY: the caller passes a NUL-terminated string.
    let text = unsafe { str_arg(text)? };
    Ok(Some(PathBuf::from(text)))
}

/// A UTF-8 string from C.
///
/// # Safety
/// `text` must be null or a NUL-terminated string that outlives `'a`.
unsafe fn str_arg<'a>(text: *const c_char) -> Result<&'a str> {
    if text.is_null() {
        bail!("a string argument is null");
    }
    // SAFETY: the caller passes a NUL-terminated string.
    let text = unsafe { CStr::from_ptr(text) };
    text.to_str().context("a string argument is not UTF-8")
}

/// The encoder behind a pointer from C, before [`pt_encoder_finish`].
///
/// # Safety
/// `enc` must be null or come from [`pt_encoder_new`], not freed, used by one
/// thread at a time.
unsafe fn open_encoder<'a>(enc: *mut PtEncoder) -> Result<&'a mut PtEncoder> {
    if enc.is_null() {
        bail!("the encoder is null");
    }
    // SAFETY: the caller passes a live encoder from pt_encoder_new.
    let enc = unsafe { &mut *enc };
    ensure!(!enc.finished, "the encoder has already finished");
    Ok(enc)
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
            encoder: Mutex::new(Models::new(weights.clone(), tables_dir.to_path_buf())),
            weights,
            tables,
            hq: OnceLock::new(),
            encodec48: OnceLock::new(),
            encodec24: OnceLock::new(),
        })
    }

    /// The HQ decoder, loaded on first use (by a decode or an encode's
    /// stereo pick).
    fn hq_decoder(&self) -> Result<&'static Arc<VocosDecoder>> {
        let slot = self.hq.get_or_init(|| {
            VocosDecoder::load(&self.weights.join(encode::hq::DECODER_FILE))
                .map(|decoder| &*Box::leak(Box::new(Arc::new(decoder))))
                .map_err(|e| format!("{e:#}"))
        });
        slot.clone().map_err(|e| anyhow!(e))
    }

    /// Encodes `audio` with the shared networks.
    fn encode(&self, audio: Audio, options: &Options) -> Result<Encoded> {
        let mut models = self.encoder.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(Ok(decoder)) = self.hq.get() {
            models.set_hq_decoder(Arc::clone(decoder));
        }
        let encoded = encode::encode(&mut models, audio, options, &mut |_| {});
        let decoder = models.loaded_hq_decoder();
        drop(models);
        // A decoder the pick loaded serves later decodes too.
        if let Some(decoder) = decoder {
            let _ = self.hq.get_or_init(|| Ok(&*Box::leak(Box::new(decoder))));
        }
        encoded
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
            Codec::Dac44 => return Ok(Net::Hq(&**self.hq_decoder()?)),
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
/// `weights_dir` for the networks (decoders and encoders).
///
/// Either may be null for the command line's default: `$PICOTONE_WEIGHTS` (or
/// `$PICOTONE_TABLES`), then `weights/` (or `models/`) next to the
/// executable or its parents, then the picotone project's. Only the first
/// successful call counts. Returns 0, or -1 on failure.
///
/// # Safety
/// Each argument must be null or a NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_init(weights_dir: *const c_char, tables_dir: *const c_char) -> i32 {
    guard(-1, || {
        if ENGINE.get().is_some() {
            return Ok(0);
        }
        // SAFETY: the caller passes null or NUL-terminated strings.
        let (weights, tables) = unsafe { (path_arg(weights_dir)?, path_arg(tables_dir)?) };
        let weights = encode::weights_dir(weights.as_deref());
        let tables = encode::tables_dir(tables.as_deref());
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

/// Starts an encode of interleaved 32-bit float audio at `sample_rate` Hz
/// with `channels` channels.
///
/// The options start as `picotone encode`'s defaults. Any rate and channel
/// count the command line reads works: the audio is
/// converted as it converts a file's. Returns null on failure.
#[unsafe(no_mangle)]
pub extern "C" fn pt_encoder_new(sample_rate: u32, channels: u32) -> *mut PtEncoder {
    guard(std::ptr::null_mut(), || {
        ensure!(sample_rate > 0, "the sample rate is 0");
        ensure!(channels > 0, "the channel count is 0");
        Ok(Box::into_raw(Box::new(PtEncoder {
            sample_rate,
            channels: vec![Vec::new(); usize::try_from(channels)?],
            options: Options::default(),
            finished: false,
            encoded: None,
        })))
    })
}

/// Sets an encode option from its text.
///
/// The key is `picotone encode`'s long flag without the dashes
/// (`model`, `fast`, `channel-format`, `stereo`, `bandwidth`, `preset`,
/// `codebooks`, `side-codebooks`, `layer-codebooks`, `layer-side-codebooks`,
/// `ps-bands`, `ps-frames`, `stride`, `crossfade`, `beam`, `loop`), its value
/// as the flag's text; `fast` and `loop` take `1` or `0`. Returns 0, or -1 for
/// an unknown key or a value that does not parse; settings that do not go
/// together fail [`pt_encoder_finish`].
///
/// # Safety
/// `enc` must come from [`pt_encoder_new`] and not be freed; `key` and
/// `value` must be NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_encoder_set(
    enc: *mut PtEncoder,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    guard(-1, || {
        // SAFETY: the caller passes a live encoder and NUL-terminated strings.
        let (enc, key, value) = unsafe { (open_encoder(enc)?, str_arg(key)?, str_arg(value)?) };
        enc.options.set(key, value)?;
        Ok(0)
    })
}

/// Adds a `KEY=value` comment (as `--tag`). Returns 0, or -1 when it is not
/// `KEY=value`.
///
/// # Safety
/// `enc` must come from [`pt_encoder_new`] and not be freed; `tag` must be a
/// NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_encoder_tag(enc: *mut PtEncoder, tag: *const c_char) -> i32 {
    guard(-1, || {
        // SAFETY: the caller passes a live encoder and a NUL-terminated string.
        let (enc, tag) = unsafe { (open_encoder(enc)?, str_arg(tag)?) };
        enc.options.add_tag(tag)?;
        Ok(0)
    })
}

/// Appends `frames` interleaved frames (`frames` x channels floats at
/// `data`). Returns 0, or -1 on failure.
///
/// # Safety
/// `enc` must come from [`pt_encoder_new`] and not be freed; `data` must hold
/// `frames` x channels readable floats (or be anything when `frames` is 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_encoder_push(
    enc: *mut PtEncoder,
    data: *const f32,
    frames: usize,
) -> i32 {
    guard(-1, || {
        // SAFETY: the caller passes a live encoder.
        let enc = unsafe { open_encoder(enc)? };
        if frames == 0 {
            return Ok(0);
        }
        ensure!(!data.is_null(), "the audio is null");
        let width = enc.channels.len();
        let count = frames.checked_mul(width).context("too many frames")?;
        // SAFETY: the caller passes `frames` x channels readable floats.
        let samples = unsafe { std::slice::from_raw_parts(data, count) };
        for (c, channel) in enc.channels.iter_mut().enumerate() {
            channel.extend(samples.iter().skip(c).step_by(width));
        }
        Ok(0)
    })
}

/// Encodes everything pushed, as `picotone encode` codes the same audio with
/// the same options.
///
/// `*pico`/`*pico_len` get the `.pico` file, and
/// `*layer`/`*layer_len` the `.picol` enhancement layer (null and 0 when the
/// options ask for none; `layer` and `layer_len` may be null). The bytes stay
/// valid until [`pt_encoder_free`]. The networks load on first use and are
/// kept; encodes run one at a time. Only once per encoder; [`pt_init`] must
/// have succeeded. Returns 0, or -1 on failure.
///
/// # Safety
/// `enc` must come from [`pt_encoder_new`] and not be freed; `pico` and
/// `pico_len` must be writable, and `layer` and `layer_len` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_encoder_finish(
    enc: *mut PtEncoder,
    pico: *mut *const u8,
    pico_len: *mut usize,
    layer: *mut *const u8,
    layer_len: *mut usize,
) -> i32 {
    guard(-1, || {
        // SAFETY: the caller passes a live encoder.
        let enc = unsafe { open_encoder(enc)? };
        if pico.is_null() || pico_len.is_null() {
            bail!("a pt_encoder_finish output is null");
        }
        let engine = engine()?;
        enc.finished = true;
        let audio = Audio {
            sample_rate: enc.sample_rate,
            channels: std::mem::take(&mut enc.channels),
        };
        let encoded = enc.encoded.insert(engine.encode(audio, &enc.options)?);
        let (layer_ptr, layer_size) = encoded
            .layer
            .as_ref()
            .map_or((std::ptr::null(), 0), |l| (l.as_ptr(), l.len()));
        // SAFETY: the caller passes writable outputs (the layer's may be null).
        unsafe {
            pico.write(encoded.bytes.as_ptr());
            pico_len.write(encoded.bytes.len());
            if !layer.is_null() {
                layer.write(layer_ptr);
            }
            if !layer_len.is_null() {
                layer_len.write(layer_size);
            }
        }
        Ok(0)
    })
}

/// Frees an encoder from [`pt_encoder_new`] and its output (null is
/// ignored).
///
/// # Safety
/// `enc` must come from [`pt_encoder_new`] and not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pt_encoder_free(enc: *mut PtEncoder) {
    if !enc.is_null() {
        // SAFETY: the caller passes an encoder from pt_encoder_new, not used
        // after this.
        drop(unsafe { Box::from_raw(enc) });
    }
}

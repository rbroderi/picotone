//! WebAssembly bindings for picotone's decoder.
//!
//! A [`Tables`] holds the cx2 entropy tables, found by the hash a `.pico`
//! stream stores. A [`Decoder`] holds one decoder network: a decoder-only
//! EnCodec model, or an HQ decoder of DAC 44 kHz codes (`dac44_student` or
//! `dac44_vocos`; any of them plays any HQ file).
//! [`Decoder::open`] parses a `.pico` file into a [`Stream`], and
//! [`Stream::decode_next`] decodes the next part of it, returning only samples
//! that later input can no longer change, so playback can start before the
//! whole file is decoded. The cx2 decode runs page by page in step with the
//! network, in the same integer Rust code as the native decoder, so the codes
//! are bit-identical on every machine.
//!
//! [`readCodes`](read_codes) decodes only the codes of an EnCodec file (for
//! the WebGPU decoder, which runs the network on the GPU).
//!
//! Segmented models (`encodec_48khz`) decode each segment with
//! [`Encodec::decode_segment`] and stitch them with [`OverlapAdd`]; causal
//! models (`encodec_24khz`) run a [`StreamDecoder`] over the codes. Either
//! way the output equals [`Encodec::decode`] on the whole file, and memory
//! depends on the chunk size, not on the file length. HQ files run through
//! [`HqStream`], the native decoder's own streaming path, whose output equals
//! a whole-file decode bit for bit.

use std::collections::VecDeque;

use anyhow::{Context as _, Result, bail, ensure};
use picotone::cx2;
use picotone::cx2::hq::HqModel;
use picotone::dac::{self, DacDecoder};
use picotone::hqstream::{HqDecoder, HqStream};
use picotone::layout::Trimmer;
use picotone::model::{Encodec, EncodedFrame, OverlapAdd, StreamDecoder};
use picotone::nn::Seq;
use picotone::pico::{self, ChannelLayout, Codec, DacInfo, DacStream, Info};
use picotone::vocos::{self, VocosDecoder};
use picotone::weights::Weights;
use wasm_bindgen::prelude::*;

/// Name of the HQ decoder network, and of its weight files.
const HQ_DECODER: &str = "dac44_student";
/// Name of the HQ code space in headers (`model`), whatever decodes it.
const HQ_MODEL: &str = "dac44";

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

/// Milliseconds from an arbitrary origin.
#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    performance_now()
}

/// Milliseconds from an arbitrary origin.
#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// Bytes of WebAssembly memory: its peak so far, since it never shrinks.
#[cfg(target_arch = "wasm32")]
fn memory_bytes() -> usize {
    core::arch::wasm32::memory_size::<0>() * 65_536
}

/// Bytes of WebAssembly memory (0 off wasm).
#[cfg(not(target_arch = "wasm32"))]
const fn memory_bytes() -> usize {
    0
}

/// The memory size at each phase of a load, in order.
#[derive(Debug, Default)]
struct Phases(Vec<(&'static str, usize)>);

impl Phases {
    fn mark(&mut self, phase: &'static str) {
        self.0.push((phase, memory_bytes()));
    }

    fn json(&self) -> String {
        let map: serde_json::Map<String, serde_json::Value> = self
            .0
            .iter()
            .map(|&(phase, bytes)| (phase.to_owned(), serde_json::json!(bytes)))
            .collect();
        serde_json::Value::Object(map).to_string()
    }
}

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

/// Sends panic messages to `console.error`; a panic still aborts (traps).
#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| {
        console_error(&format!("picotone panicked: {info}"));
    }));
}

/// Whether this build was compiled with the Web Worker thread pool.
#[wasm_bindgen(js_name = isThreadedBuild)]
#[must_use]
#[expect(
    clippy::missing_const_for_fn,
    reason = "wasm_bindgen cannot export const fns"
)]
pub fn is_threaded_build() -> bool {
    cfg!(feature = "threads")
}

/// Number of threads rayon runs on: 1 in the single-threaded build.
#[wasm_bindgen(js_name = threadCount)]
#[must_use]
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

/// Parses a `.pico` file's headers and page index and returns them as JSON.
///
/// The fields: `{"model", "sampleRate", "channels", "audioLength",
/// "codedLength", "numCodebooks", "stride", "crossfade", "preSkip",
/// "endTrim", "loop", "tableHash", "pageLength", "lengthInHeader",
/// "outputGain", "tags"}`. `audioLength` is the playable length,
/// `codedLength` the coded one; `stride` is null for the causal model.
///
/// HQ files (`"model": "dac44"`) have `stride` and `crossfade` null,
/// `numCodebooks` the largest count, and also `"codebooks"` (per coded
/// channel), `"layout"` (`mono`, `lr`, `ms` or `parametric`) and
/// `"codebookHash"`.
///
/// # Errors
/// Fails on a malformed or damaged file.
#[wasm_bindgen(js_name = readHeader)]
pub fn read_header(pico_bytes: &[u8]) -> Result<String, JsError> {
    header_json(pico_bytes)
        .map(|v| v.to_string())
        .map_err(|e| to_js(&e))
}

fn header_json(bytes: &[u8]) -> Result<serde_json::Value> {
    Ok(match pico::codec_of(bytes)? {
        Codec::Encodec(_) => info_json(&pico::Stream::parse(bytes)?.info),
        Codec::Dac44 => dac_info_json(&DacStream::parse(bytes)?.info),
    })
}

fn info_json(info: &Info) -> serde_json::Value {
    let (rate, channels) = match info.model {
        picotone::ModelKind::Mono24k => (24_000, 1),
        picotone::ModelKind::Stereo48k => (48_000, 2),
    };
    serde_json::json!({
        "model": info.model.name(),
        "sampleRate": rate,
        "channels": channels,
        "audioLength": info.playable_samples(),
        "codedLength": info.audio_length,
        "numCodebooks": info.num_codebooks,
        "stride": info.stride,
        "crossfade": info.crossfade.code(),
        "preSkip": info.pre_skip,
        "endTrim": info.end_trim,
        "loop": info.looped,
        "tableHash": format!("{:016x}", info.table_hash),
        "pageLength": info.page_length,
        "lengthInHeader": info.length_in_header,
        "outputGain": info.output_gain,
        "tags": info.tags,
    })
}

fn dac_info_json(info: &DacInfo) -> serde_json::Value {
    let layout = match info.layout {
        ChannelLayout::Mono => "mono",
        ChannelLayout::LeftRight => "lr",
        ChannelLayout::MidSide => "ms",
        ChannelLayout::Parametric(_) => "parametric",
    };
    serde_json::json!({
        "model": HQ_MODEL,
        "sampleRate": dac::SAMPLE_RATE,
        "channels": info.layout.output_channels(),
        "layout": layout,
        "codebooks": info.codebooks,
        "codebookHash": format!("{:016x}", info.codebook_hash),
        "audioLength": info.playable_samples(),
        "codedLength": info.audio_length,
        "numCodebooks": info.codebooks.iter().copied().max().unwrap_or(0),
        "stride": null,
        "crossfade": null,
        "preSkip": info.pre_skip,
        "endTrim": info.end_trim,
        "loop": info.looped,
        "tableHash": format!("{:016x}", info.table_hash),
        "pageLength": info.page_length,
        "lengthInHeader": info.length_in_header,
        "outputGain": info.output_gain,
        "tags": info.tags,
    })
}

/// Refuses HQ files on the codes-only (WebGPU) paths.
fn ensure_encodec(bytes: &[u8]) -> Result<()> {
    if pico::codec_of(bytes)? == Codec::Dac44 {
        bail!(
            "HQ files decode only through Decoder.open (the WebGPU decoder has no HQ network yet)"
        );
    }
    Ok(())
}

/// The cx2 entropy tables a page can decode with, keyed by their hash: EnCodec
/// tables (`ECXM` version 5) and HQ tables (`ECXH`).
///
/// Tables live as long as the module instance (streams borrow them across
/// calls), so `free()` releases only the handle.
#[wasm_bindgen]
#[derive(Debug, Default)]
pub struct Tables {
    list: Vec<&'static cx2::Model>,
    hq: Vec<&'static HqModel>,
}

#[wasm_bindgen]
impl Tables {
    /// An empty set.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a self-contained (version 5) `.ecxm` table or an HQ table and
    /// returns its hash as 16 hex digits. Adding a table twice keeps one copy.
    ///
    /// # Errors
    /// Fails when the bytes are not a valid table of either kind.
    pub fn add(&mut self, table: &[u8]) -> Result<String, JsError> {
        self.add_table(table).map_err(|e| to_js(&e))
    }

    /// Whether the table with this hash (16 hex digits) is loaded.
    #[must_use]
    pub fn has(&self, hash: &str) -> bool {
        u64::from_str_radix(hash, 16)
            .is_ok_and(|h| self.find(h).is_some() || self.find_hq(h).is_some())
    }
}

impl Tables {
    fn add_table(&mut self, table: &[u8]) -> Result<String> {
        let hash = if let Some(hash) = cx2::hq::table_hash(table) {
            if self.find_hq(hash).is_none() {
                let model = HqModel::from_table(table)?;
                self.hq.push(Box::leak(Box::new(model)));
            }
            hash
        } else {
            let hash = cx2::model::self_contained_hash(table)
                .context("not a self-contained (version 5) cx2 table or an HQ table")?;
            if self.find(hash).is_none() {
                let model = cx2::Model::from_table(table)?;
                self.list.push(Box::leak(Box::new(model)));
            }
            hash
        };
        Ok(format!("{hash:016x}"))
    }

    fn find(&self, hash: u64) -> Option<&'static cx2::Model> {
        self.list.iter().copied().find(|t| t.hash() == hash)
    }

    fn find_hq(&self, hash: u64) -> Option<&'static HqModel> {
        self.hq.iter().copied().find(|t| t.hash() == hash)
    }

    fn for_stream(&self, info: &Info) -> Result<&'static cx2::Model> {
        self.find(info.table_hash).with_context(|| {
            format!(
                "the cx2 table {:016x} this file was coded with is not loaded",
                info.table_hash
            )
        })
    }

    fn for_hq_stream(&self, info: &DacInfo) -> Result<&'static HqModel> {
        self.find_hq(info.table_hash).with_context(|| {
            format!(
                "the HQ cx2 table {:016x} this file was coded with is not loaded",
                info.table_hash
            )
        })
    }
}

/// A file's codes, decoded without the network (for the WebGPU decoder).
#[wasm_bindgen]
#[derive(Debug)]
pub struct Codes {
    info: String,
    values: Vec<u16>,
    scales: Vec<f32>,
}

#[wasm_bindgen]
impl Codes {
    /// The header as [`readHeader`](read_header) gives it, plus
    /// `"segmentFrames"`: latent frames per segment (one entry for the causal model).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn info(&self) -> String {
        self.info.clone()
    }

    /// All codes, frame-major (`[frame][codebook]`), segments one after another.
    #[must_use]
    pub fn codes(&self) -> Vec<u16> {
        self.values.clone()
    }

    /// One scale per segment (empty for the causal model).
    #[must_use]
    pub fn scales(&self) -> Vec<f32> {
        self.scales.clone()
    }
}

/// Decodes every page of a `.pico` file into codes and scales.
///
/// # Errors
/// Fails when the file is malformed or damaged, or its table is not in `tables`.
#[wasm_bindgen(js_name = readCodes)]
pub fn read_codes(pico_bytes: &[u8], tables: &Tables) -> Result<Codes, JsError> {
    codes_of(pico_bytes, tables).map_err(|e| to_js(&e))
}

/// A file's codes, range-decoded one Ogg page at a time (for the WebGPU
/// decoder, so its first audio does not wait for the whole file).
#[wasm_bindgen]
#[derive(Debug)]
pub struct CodePages {
    info: String,
    pages: Pages,
}

#[wasm_bindgen]
impl CodePages {
    /// Parses a `.pico` file and finds its table; decodes nothing yet.
    ///
    /// # Errors
    /// Fails when the file is malformed or its table is not in `tables`.
    #[wasm_bindgen(constructor)]
    pub fn new(pico_bytes: &[u8], tables: &Tables) -> Result<Self, JsError> {
        Self::open(pico_bytes, tables).map_err(|e| to_js(&e))
    }

    fn open(bytes: &[u8], tables: &Tables) -> Result<Self> {
        ensure_encodec(bytes)?;
        let stream = pico::Stream::parse(bytes)?;
        let table = tables.for_stream(&stream.info)?;
        let mut info = info_json(&stream.info);
        info["segmentFrames"] = serde_json::json!(stream.segment_frames);
        Ok(Self {
            info: info.to_string(),
            pages: Pages {
                stream,
                table,
                next: 0,
                cx2_ms: 0.0,
            },
        })
    }

    fn next_page(&mut self) -> Result<Option<Codes>> {
        Ok(self.pages.next()?.map(|frames| Codes {
            info: String::new(),
            scales: frames.iter().filter_map(|f| f.scale).collect(),
            values: frames.into_iter().flat_map(|f| f.codes).collect(),
        }))
    }

    /// As [`Codes::info`].
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn info(&self) -> String {
        self.info.clone()
    }

    /// Ogg pages holding codes.
    #[wasm_bindgen(getter, js_name = pageCount)]
    #[must_use]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "wasm_bindgen cannot export const fns"
    )]
    pub fn page_count(&self) -> usize {
        self.pages.stream.pages.len()
    }

    /// The next page's codes (its segments, or for the causal model its
    /// latent frames), or `undefined` after the last page.
    ///
    /// # Errors
    /// Fails when the page is damaged or missing.
    #[wasm_bindgen(js_name = nextPage)]
    pub fn decode_next(&mut self) -> Result<Option<Codes>, JsError> {
        self.next_page().map_err(|e| to_js(&e))
    }
}

fn codes_of(bytes: &[u8], tables: &Tables) -> Result<Codes> {
    ensure_encodec(bytes)?;
    let stream = pico::Stream::parse(bytes)?;
    let table = tables.for_stream(&stream.info)?;
    let frames = stream.decode_all(table)?;
    let mut info = info_json(&stream.info);
    info["segmentFrames"] = serde_json::json!(stream.segment_frames);
    Ok(Codes {
        info: info.to_string(),
        scales: frames.iter().filter_map(|f| f.scale).collect(),
        values: frames.into_iter().flat_map(|f| f.codes).collect(),
    })
}

/// The network a [`Decoder`] holds.
#[derive(Debug, Clone, Copy)]
enum Net {
    /// A decoder-only EnCodec model.
    Encodec(&'static Encodec),
    /// A decoder of DAC 44 kHz codes, for HQ files.
    Hq(&'static dyn HqDecoder),
}

/// A decoder network: a decoder-only EnCodec model (`encodec_24khz`,
/// `encodec_48khz`) or the HQ decoder (`dac44_student`).
///
/// The network lives as long as the module instance: streams borrow it
/// across calls, so `free()` releases only the handle, not the weights. Load
/// each network once per worker (the worker does) and terminate the worker
/// to release everything.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Decoder {
    net: Net,
    /// WebAssembly memory after each phase of the build, as JSON.
    load_memory: String,
}

#[wasm_bindgen]
#[expect(
    clippy::missing_const_for_fn,
    reason = "wasm_bindgen cannot export const fns"
)]
impl Decoder {
    /// Builds network `name` from its `.json` config text and `.safetensors`
    /// bytes. For EnCodec those need to hold only the decoder and quantizer
    /// tensors; `dac44_student` takes its weight file as it is.
    ///
    /// The bytes stay in JavaScript: each tensor is read from them when it is
    /// packed, a piece at a time, so WebAssembly memory never holds the file.
    /// The caller can drop its buffer once this returns.
    ///
    /// # Errors
    /// Fails when the inputs do not describe a valid decoder.
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: &str,
        config_json: &str,
        safetensors: &js_sys::Uint8Array,
    ) -> Result<Self, JsError> {
        let len = safetensors.length();
        let read = move |offset: u64, buf: &mut [u8]| -> Result<()> {
            let start = u32::try_from(offset)?;
            let end = u32::try_from(buf.len())
                .ok()
                .and_then(|n| start.checked_add(n))
                .filter(|&end| end <= len)
                .context("read past the end of the weights")?;
            safetensors.subarray(start, end).copy_to(buf);
            Ok(())
        };
        Weights::from_reader(u64::from(len), Box::new(read))
            .and_then(|weights| Self::build(name, config_json, &weights))
            .map_err(|e| to_js(&e))
    }

    /// WebAssembly memory in bytes after each phase of building the network,
    /// as a JSON object from phase name to bytes, in order. The memory never
    /// shrinks, so each value is the peak up to that phase.
    #[wasm_bindgen(getter, js_name = loadMemory)]
    #[must_use]
    pub fn load_memory(&self) -> String {
        self.load_memory.clone()
    }

    /// Network name, e.g. `encodec_48khz` or `dac44_student`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn name(&self) -> String {
        match self.net {
            Net::Encodec(model) => model.name().to_owned(),
            Net::Hq(decoder) => decoder.name().to_owned(),
        }
    }

    /// The code space of the files it decodes, as `readHeader` names it
    /// (`model`): `encodec_24khz`, `encodec_48khz` or `dac44`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn model(&self) -> String {
        match self.net {
            Net::Encodec(model) => model.name().to_owned(),
            Net::Hq(_) => HQ_MODEL.to_owned(),
        }
    }

    /// Output sample rate in Hz.
    #[wasm_bindgen(getter, js_name = sampleRate)]
    #[must_use]
    pub fn sample_rate(&self) -> usize {
        match self.net {
            Net::Encodec(model) => model.config().sampling_rate,
            Net::Hq(_) => 44_100,
        }
    }

    /// Most output channels (an HQ file has 1 or 2, as its header says).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn channels(&self) -> usize {
        match self.net {
            Net::Encodec(model) => model.config().audio_channels,
            Net::Hq(_) => 2,
        }
    }

    /// Parses a `.pico` file for this network, to be decoded from its start.
    ///
    /// # Errors
    /// Fails when the file is malformed, truncated, made for another model, or
    /// coded with a table `tables` lacks.
    pub fn open(&self, pico_bytes: &[u8], tables: &Tables) -> Result<Stream, JsError> {
        Stream::new(self.net, pico_bytes, tables, 0).map_err(|e| to_js(&e))
    }

    /// Like [`open`](Self::open), but the first returned sample is playable
    /// sample `start_sample`. The 48 kHz model starts at the segment that
    /// sample needs (found from the page granules) and HQ files a few frames
    /// before it; the causal 24 kHz model and loops (whose end blends into
    /// their start) decode from the start and drop what comes before. The
    /// output equals the matching part of a whole-file decode, bit for bit.
    ///
    /// # Errors
    /// As [`open`](Self::open); also when `start_sample` is past the end.
    #[wasm_bindgen(js_name = openAt)]
    pub fn open_at(
        &self,
        pico_bytes: &[u8],
        tables: &Tables,
        start_sample: usize,
    ) -> Result<Stream, JsError> {
        Stream::new(self.net, pico_bytes, tables, start_sample).map_err(|e| to_js(&e))
    }
}

impl Decoder {
    /// Builds the network from its tensors; `parsed` is the memory once the
    /// header is read, `packed` once every tensor is packed.
    fn build(name: &str, config_json: &str, w: &Weights<'_>) -> Result<Self> {
        let mut phases = Phases::default();
        phases.mark("parsed");
        let net = if name == HQ_DECODER {
            let config: serde_json::Value =
                serde_json::from_str(config_json).context("parsing the decoder's config")?;
            let width = config.get("width").and_then(serde_json::Value::as_u64);
            ensure!(
                width.is_none_or(|w| w == 512),
                "this build decodes the width-512 HQ student only"
            );
            let decoder = DacDecoder::from_weights(w, 512).context("building the HQ decoder")?;
            let decoder: &'static DacDecoder = Box::leak(Box::new(decoder));
            Net::Hq(decoder)
        } else if name == vocos::NAME {
            let decoder = VocosDecoder::from_weights(w).context("building the Vocos HQ decoder")?;
            let decoder: &'static VocosDecoder = Box::leak(Box::new(decoder));
            Net::Hq(decoder)
        } else {
            Net::Encodec(Box::leak(Box::new(Encodec::decoder_from_weights(
                name,
                config_json,
                w,
            )?)))
        };
        phases.mark("packed");
        Ok(Self {
            net,
            load_memory: phases.json(),
        })
    }
}

/// Decodes a stream's pages on demand.
#[derive(Debug)]
struct Pages {
    stream: pico::Stream,
    table: &'static cx2::Model,
    next: usize,
    /// Milliseconds spent in the cx2 decode.
    cx2_ms: f64,
}

impl Pages {
    /// The next page's segments (causal model: one frame), or `None` at the end.
    fn next(&mut self) -> Result<Option<Vec<EncodedFrame>>> {
        if self.next == self.stream.pages.len() {
            return Ok(None);
        }
        let start = now_ms();
        let frames = self.stream.decode_page(self.table, self.next)?;
        self.cx2_ms += now_ms() - start;
        self.next += 1;
        Ok(Some(frames))
    }
}

/// How a stream turns codes into audio.
#[derive(Debug)]
enum Source {
    /// Normalised segments, decoded one by one and overlap-added.
    Segmented {
        /// Decoded segments not yet run through the network.
        queue: VecDeque<EncodedFrame>,
        /// Segments to drop from the first page (a seek into the page).
        skip: usize,
        /// Segments run through the network so far, counted from `first`.
        done: usize,
        /// Segments from the first decoded one to the end.
        count: usize,
        overlap: Option<OverlapAdd>,
    },
    /// One causal stream of latent frames.
    Causal {
        /// Codes decoded but not yet pushed, frame-major.
        codes: Vec<u16>,
        num_codebooks: usize,
        /// Latent frames per unit (one second).
        unit_frames: usize,
        /// Latent frames in the file.
        total: usize,
        /// Latent frames pushed so far.
        next: usize,
        decoder: Option<StreamDecoder<'static>>,
    },
}

/// A `.pico` file being decoded a few units at a time.
///
/// A unit is one coded segment (about 1 s) for segmented models, one second
/// of latent frames for causal models, and one page of frames (80 by
/// default, 0.93 s) for HQ files.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Stream {
    inner: Inner,
    /// The file's `model`, as [`readHeader`](read_header) names it.
    model: String,
    sample_rate: usize,
    channels: usize,
    audio_length: usize,
    looped: bool,
    num_codebooks: usize,
    /// Milliseconds spent in the cx2 decode, and in everything else.
    cx2_ms: f64,
    network_ms: f64,
}

/// The decode behind a [`Stream`].
#[derive(Debug)]
enum Inner {
    Encodec(Box<EncodecStream>),
    Hq(Box<HqStream<'static>>),
}

impl Stream {
    fn new(net: Net, bytes: &[u8], tables: &Tables, start_sample: usize) -> Result<Self> {
        let codec = pico::codec_of(bytes)?;
        let file_model = match codec {
            Codec::Encodec(kind) => kind.name(),
            Codec::Dac44 => HQ_MODEL,
        };
        let loaded = match net {
            Net::Encodec(model) => model.name(),
            Net::Hq(_) => HQ_MODEL,
        };
        ensure!(
            file_model == loaded,
            "this file needs model {file_model}, but {loaded} is loaded"
        );
        let header = header_json(bytes)?;
        let field = |name: &str| {
            header
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(0)
        };
        let (sample_rate, channels, num_codebooks) = (
            field("sampleRate"),
            field("channels"),
            field("numCodebooks"),
        );
        let inner = match net {
            Net::Encodec(model) => Inner::Encodec(Box::new(EncodecStream::new(
                model,
                bytes,
                tables,
                start_sample,
            )?)),
            Net::Hq(decoder) => {
                let stream = DacStream::parse(bytes)?;
                let table = tables.for_hq_stream(&stream.info)?;
                Inner::Hq(Box::new(HqStream::new(
                    stream,
                    table,
                    decoder,
                    start_sample,
                )?))
            }
        };
        let (audio_length, looped) = match &inner {
            Inner::Encodec(s) => (s.info.playable_samples(), s.info.looped),
            Inner::Hq(s) => (s.info().playable_samples(), s.info().looped),
        };
        Ok(Self {
            inner,
            model: file_model.to_owned(),
            sample_rate,
            channels,
            audio_length,
            looped,
            num_codebooks,
            cx2_ms: 0.0,
            network_ms: 0.0,
        })
    }

    /// Decodes up to `max_units` more units, timing the cx2 decode apart
    /// from the rest.
    fn decode_units(&mut self, max_units: usize) -> Result<Option<Chunk>> {
        let start = now_ms();
        match &mut self.inner {
            Inner::Encodec(s) => {
                let before = s.pages.cx2_ms;
                let chunk = s.decode_units(max_units)?;
                let cx2 = s.pages.cx2_ms - before;
                self.cx2_ms += cx2;
                self.network_ms += now_ms() - start - cx2;
                Ok(chunk)
            }
            Inner::Hq(s) => {
                s.fetch(max_units)?;
                let fetched = now_ms();
                let part = s.decode_next(max_units)?;
                self.cx2_ms += fetched - start;
                self.network_ms += now_ms() - fetched;
                Ok(part.map(|(offset, audio)| planar_chunk(offset, &audio)))
            }
        }
    }

    fn unit_count(&self) -> usize {
        match &self.inner {
            Inner::Encodec(s) => s.unit_count(),
            Inner::Hq(s) => s.unit_count(),
        }
    }

    fn units_done(&self) -> usize {
        match &self.inner {
            Inner::Encodec(s) => s.units_done(),
            Inner::Hq(s) => s.units_done(),
        }
    }
}

/// An EnCodec `.pico` file being decoded a few units at a time.
#[derive(Debug)]
struct EncodecStream {
    model: &'static Encodec,
    info: Info,
    pages: Pages,
    source: Source,
    /// Removes the pre-skip and end trim and joins loops.
    trimmer: Option<Trimmer>,
    /// Coded samples the raw output stops at, counted from its first sample.
    raw_length: usize,
    /// Trimmed samples still to drop before the first returned one.
    drop: usize,
    /// Playable sample the next returned chunk starts at.
    emitted: usize,
}

/// Final samples `[sample][channel]` from playable sample `offset`, as a
/// planar chunk.
fn planar_chunk(offset: usize, audio: &Seq) -> Chunk {
    let length = audio.len();
    let channels = audio.channels();
    let mut planar = vec![0.0_f32; length * channels];
    for (t, row) in audio.data().chunks_exact(channels.max(1)).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            planar[c * length + t] = v;
        }
    }
    Chunk {
        offset,
        length,
        channels,
        planar,
    }
}

impl EncodecStream {
    fn new(
        model: &'static Encodec,
        bytes: &[u8],
        tables: &Tables,
        start_sample: usize,
    ) -> Result<Self> {
        let stream = pico::Stream::parse(bytes)?;
        let info = stream.info.clone();
        ensure!(
            info.model.name() == model.name(),
            "this file needs model {}, but {} is loaded",
            info.model.name(),
            model.name()
        );
        let playable = info.playable_samples();
        ensure!(playable > 0, "the file holds no audio");
        ensure!(
            start_sample < playable,
            "start sample {start_sample} is past the end ({playable} samples)"
        );
        let table = tables.for_stream(&info)?;
        let cfg = model.config();
        let rate = cfg.sampling_rate;
        let channels = cfg.audio_channels;
        // A seek starts at a later segment only where nothing before it matters:
        // segmented, and not a loop (a loop's end blends into its start).
        let first_segment = match (info.stride, info.looped) {
            (Some(stride), false) if start_sample > 0 => {
                first_needed_segment(&info, stride, start_sample + info.pre_skip)?
            }
            _ => 0,
        };
        let raw_start = first_segment * info.stride.unwrap_or(0);
        let (trimmer, drop) = if first_segment == 0 {
            let trimmer = Trimmer::new(info.pre_skip, playable, info.looped, rate, channels)?;
            (trimmer, start_sample)
        } else {
            let pre_skip = start_sample + info.pre_skip - raw_start;
            let trimmer = Trimmer::new(pre_skip, playable - start_sample, false, rate, channels)?;
            (trimmer, 0)
        };
        let first_page = stream
            .spans
            .iter()
            .position(|span| span.segments.contains(&first_segment))
            .context("no page holds the start segment")?;
        let skip = first_segment - stream.spans[first_page].segments.start;
        let segment_count = stream.segment_frames.len();
        let total_frames: usize = stream.segment_frames.iter().sum();
        let source = if info.stride.is_some() {
            Source::Segmented {
                queue: VecDeque::new(),
                skip,
                done: 0,
                count: segment_count - first_segment,
                overlap: Some(model.overlap_add_with(info.segmenting())?),
            }
        } else {
            Source::Causal {
                codes: Vec::new(),
                num_codebooks: info.num_codebooks,
                unit_frames: cfg.frame_rate().max(1),
                total: total_frames,
                next: 0,
                decoder: Some(model.stream_decoder(info.num_codebooks)?),
            }
        };
        Ok(Self {
            model,
            raw_length: info.audio_length - raw_start,
            info,
            pages: Pages {
                stream,
                table,
                next: first_page,
                cx2_ms: 0.0,
            },
            source,
            trimmer: Some(trimmer),
            drop,
            emitted: start_sample,
        })
    }

    /// Decodes up to `max_units` more units and trims them; `None` once
    /// everything was returned. Keeps decoding while the trim swallows
    /// everything, so a returned chunk is never empty.
    fn decode_units(&mut self, max_units: usize) -> Result<Option<Chunk>> {
        loop {
            let Some((raw, finished)) = self.decode_raw(max_units)? else {
                return Ok(None);
            };
            let trimmer = self.trimmer.as_mut().context("stream already finished")?;
            let mut audio = trimmer.push(&raw)?;
            drop(raw);
            if finished {
                self.trimmer
                    .take()
                    .context("stream already finished")?
                    .finish()?;
            }
            if self.drop > 0 {
                let cut = self.drop.min(audio.len());
                self.drop -= cut;
                audio = audio.slice_rows(cut, audio.len());
            }
            if audio.is_empty() {
                if finished {
                    return Ok(None);
                }
                continue;
            }
            return Ok(Some(self.chunk(&audio)));
        }
    }

    /// Converts final samples to a planar chunk at the current offset.
    fn chunk(&mut self, audio: &Seq) -> Chunk {
        let chunk = planar_chunk(self.emitted, audio);
        self.emitted += chunk.length;
        chunk
    }

    /// Units this stream decodes.
    const fn unit_count(&self) -> usize {
        match &self.source {
            Source::Segmented { count, .. } => *count,
            Source::Causal {
                total, unit_frames, ..
            } => total.div_ceil(*unit_frames),
        }
    }

    /// Units decoded so far.
    const fn units_done(&self) -> usize {
        match &self.source {
            Source::Segmented { done, .. } => *done,
            Source::Causal {
                next, unit_frames, ..
            } => next.div_ceil(*unit_frames),
        }
    }

    /// Decodes up to `max_units` more units of coded samples, untrimmed, and
    /// whether they are the last; `None` once everything was returned.
    fn decode_raw(&mut self, max_units: usize) -> Result<Option<(Seq, bool)>> {
        let max_units = max_units.max(1);
        let raw_length = self.raw_length;
        let model = self.model;
        let pages = &mut self.pages;
        let audio = match &mut self.source {
            Source::Segmented {
                queue,
                skip,
                done,
                count,
                overlap,
            } => {
                let Some(acc) = overlap.as_mut() else {
                    return Ok(None);
                };
                let want = max_units.min(*count - *done);
                while queue.len() < want {
                    let frames = pages.next()?.context("the stream ended early")?;
                    queue.extend(frames.into_iter().skip(*skip));
                    *skip = 0;
                }
                let batch: Vec<EncodedFrame> = queue.drain(..want).collect();
                let mut parts = Vec::with_capacity(batch.len() + 1);
                // Together: the LSTM advances all segments of the batch in
                // one pass over its weights per time step, and the pool
                // shares the convolutions of a few segments at a time.
                for segment in &model.decode_segments(&batch)? {
                    parts.push(acc.push(segment)?);
                }
                *done += want;
                if *done == *count {
                    let acc = overlap.take().context("overlap-add already finished")?;
                    parts.push(acc.finish(raw_length)?);
                }
                Seq::concat(parts)?
            }
            Source::Causal {
                codes,
                num_codebooks,
                unit_frames,
                total,
                next,
                decoder,
            } => {
                let Some(stream) = decoder.as_mut() else {
                    return Ok(None);
                };
                let nc = (*num_codebooks).max(1);
                let step = max_units.saturating_mul(*unit_frames);
                let mut parts = Vec::new();
                // The first frames are held back until the reflection padding
                // is known, so keep pushing until some audio comes out.
                loop {
                    let end = next.saturating_add(step).min(*total);
                    while codes.len() < (end - *next) * nc {
                        let frames = pages.next()?.context("the stream ended early")?;
                        codes.extend(frames.into_iter().flat_map(|f| f.codes));
                    }
                    let take = (end - *next) * nc;
                    let audio = stream.push(&codes[..take])?;
                    drop(codes.drain(..take));
                    *next = end;
                    let got = audio.len();
                    parts.push(audio);
                    if end == *total {
                        let stream = decoder.take().context("stream already finished")?;
                        parts.push(stream.finish(raw_length)?);
                        break;
                    }
                    if got > 0 {
                        break;
                    }
                }
                Seq::concat(parts)?
            }
        };
        let finished = match &self.source {
            Source::Segmented { overlap, .. } => overlap.is_none(),
            Source::Causal { decoder, .. } => decoder.is_none(),
        };
        Ok(Some((audio, finished)))
    }
}

/// The first segment whose output reaches coded sample `coded_start`
/// unchanged by the segments before it: the last one that starts after the
/// previous segment ends, at or before `coded_start`.
fn first_needed_segment(info: &Info, stride: usize, coded_start: usize) -> Result<usize> {
    let layout = info.layout()?;
    let length = layout
        .segment_length()
        .context("a strided model has a segment length")?;
    let count = layout.segment_count(info.audio_length);
    // Segment j > 0 is alone from (j - 1) * stride + length on.
    let mut first = 0;
    for j in 1..count {
        if (j - 1) * stride + length <= coded_start {
            first = j;
        } else {
            break;
        }
    }
    Ok(first)
}

#[wasm_bindgen]
#[expect(
    clippy::missing_const_for_fn,
    reason = "wasm_bindgen cannot export const fns"
)]
impl Stream {
    /// Model the file was encoded with.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn model(&self) -> String {
        self.model.clone()
    }

    /// Output sample rate in Hz.
    #[wasm_bindgen(getter, js_name = sampleRate)]
    #[must_use]
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Output channel count.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Audio length in samples per channel, after the pre-skip and end trim.
    #[wasm_bindgen(getter, js_name = audioLength)]
    #[must_use]
    pub fn audio_length(&self) -> usize {
        self.audio_length
    }

    /// Whether the file is a seamless loop.
    #[wasm_bindgen(getter, js_name = isLoop)]
    #[must_use]
    pub fn is_loop(&self) -> bool {
        self.looped
    }

    /// Codebooks per latent frame (bandwidth); for HQ files the largest
    /// count of any coded channel.
    #[wasm_bindgen(getter, js_name = numCodebooks)]
    #[must_use]
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Milliseconds the cx2 (entropy) decode has taken so far.
    #[wasm_bindgen(getter, js_name = cx2Ms)]
    #[must_use]
    pub fn cx2_ms(&self) -> f64 {
        self.cx2_ms
    }

    /// Milliseconds everything else (the network, stereo, trim) has taken.
    #[wasm_bindgen(getter, js_name = networkMs)]
    #[must_use]
    pub fn network_ms(&self) -> f64 {
        self.network_ms
    }

    /// Number of units this stream decodes: coded segments, seconds of
    /// latent frames for causal models, or pages of frames for HQ files.
    #[wasm_bindgen(getter, js_name = segmentCount)]
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.unit_count()
    }

    /// Units decoded so far.
    #[wasm_bindgen(getter, js_name = segmentsDone)]
    #[must_use]
    pub fn segments_done(&self) -> usize {
        self.units_done()
    }

    /// Decodes up to `max_segments` more units and returns the samples that
    /// became final, or `undefined` once the whole file has been returned.
    ///
    /// # Errors
    /// Fails when `decoder` is not the model the stream was opened with or
    /// the codes do not fit it.
    #[wasm_bindgen(js_name = decodeNext)]
    pub fn decode_next(
        &mut self,
        decoder: &Decoder,
        max_segments: usize,
    ) -> Result<Option<Chunk>, JsError> {
        if decoder.model() != self.model {
            return Err(JsError::new("stream was opened with another model"));
        }
        self.decode_units(max_segments).map_err(|e| to_js(&e))
    }
}

/// Final decoded samples `[offset, offset + length)`, planar.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Chunk {
    offset: usize,
    length: usize,
    channels: usize,
    planar: Vec<f32>,
}

#[wasm_bindgen]
#[expect(
    clippy::missing_const_for_fn,
    reason = "wasm_bindgen cannot export const fns"
)]
impl Chunk {
    /// First sample, per channel, from the start of the playable audio.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Samples per channel.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Channel count.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// A copy of channel `index` (a fresh, transferable `Float32Array`).
    ///
    /// # Errors
    /// Fails when `index` is not a channel.
    pub fn channel(&self, index: usize) -> Result<Vec<f32>, JsError> {
        let start = index
            .checked_mul(self.length)
            .filter(|_| index < self.channels)
            .ok_or_else(|| JsError::new("channel index out of range"))?;
        self.planar
            .get(start..start + self.length)
            .map(<[f32]>::to_vec)
            .ok_or_else(|| JsError::new("channel index out of range"))
    }
}

fn to_js(error: &anyhow::Error) -> JsError {
    JsError::new(&format!("{error:#}"))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use picotone::ModelKind;
    use picotone::layout::{self, Crossfade};
    use picotone::model::EncodeParams;
    use picotone::pico::Layout;

    use super::*;

    /// The shipped default table of `kind`, from the repository's `models/`.
    fn table(kind: ModelKind) -> Result<Vec<u8>> {
        let name = match kind {
            ModelKind::Mono24k => "cx2_encodec_24khz_v5.ecxm",
            ModelKind::Stereo48k => "cx2_encodec_48khz_net2_v5.ecxm",
        };
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../models")
            .join(name);
        std::fs::read(&path).with_context(|| format!("reading {}", path.display()))
    }

    struct Case {
        kind: ModelKind,
        seconds: f64,
        stride: Option<usize>,
        crossfade: Crossfade,
        looped: bool,
        page_length: usize,
    }

    const STOCK48: Case = Case {
        kind: ModelKind::Stereo48k,
        seconds: 3.5,
        stride: Some(47_520),
        crossfade: Crossfade::Triangle,
        looped: false,
        page_length: 1,
    };

    /// Encodes a few seconds of synthetic audio into `.pico` and checks that
    /// streaming it in chunks of every size, from the start and from several
    /// seek points, reproduces the whole-file decode bit for bit.
    fn streams_like_whole_decode(case: &Case) -> Result<()> {
        let weights = Path::new(env!("CARGO_MANIFEST_DIR")).join("../weights");
        let model: &'static Encodec = Box::leak(Box::new(case.kind.load(&weights)?));
        let table_bytes = table(case.kind)?;
        let mut tables = Tables::new();
        let _hash = tables
            .add(&table_bytes)
            .map_err(|_| anyhow::anyhow!("bad table"))?;
        let cx = cx2::Model::from_table(&table_bytes)?;
        let cfg = model.config();
        let rate = cfg.sampling_rate;
        let channels = cfg.audio_channels;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "test signal synthesis"
        )]
        let (len, data) = {
            let len = (case.seconds * rate as f64) as usize;
            let data = (0..len * channels)
                .map(|i| {
                    let t = (i / channels) as f32 / rate as f32;
                    let c = (i % channels) as f32;
                    let tone = (t * 110.0f32.mul_add(c, 220.0) * std::f32::consts::TAU).sin();
                    0.3f32.mul_add(tone, 0.1 * (t * 3.0 * std::f32::consts::TAU).sin())
                })
                .collect();
            (len, data)
        };
        let mut audio = Seq::from_vec(len, channels, data)?;
        let num_codebooks = cfg.codebooks_for_bandwidth(6.0)?;
        let mut info = Info {
            model: case.kind,
            table_hash: cx.hash(),
            num_codebooks,
            stride: case.stride,
            crossfade: case.crossfade,
            looped: case.looped,
            pre_skip: 0,
            page_length: case.page_length,
            audio_length: 0,
            end_trim: 0,
            length_in_header: true,
            output_gain: 0,
            tags: Vec::new(),
        };
        if case.looped {
            let (pre, post) = layout::loop_padding(rate);
            audio = layout::wrap_loop(&audio, pre, post)?;
            info.pre_skip = pre;
            info.end_trim = post;
        }
        info.audio_length = audio.len();
        let _layout: Layout = info.layout()?;
        let params = EncodeParams {
            segmenting: info.segmenting(),
            ..EncodeParams::stock(num_codebooks)
        };
        let frames = model.encode_with(&audio, &params)?;
        let bytes = pico::write(&cx, &info, &frames)?;
        let raw = model.decode_with(&frames, audio.len(), info.segmenting())?;
        let mut trimmer = Trimmer::new(info.pre_skip, len, case.looped, rate, channels)?;
        let whole = trimmer.push(&raw)?;
        trimmer.finish()?;
        assert_eq!(whole.len(), len);
        let starts = [0, 1, len / 3, len / 2 + 7, len - 1];
        for (start, max_units) in starts
            .iter()
            .flat_map(|&s| [1, 2, 3, 100].map(move |u| (s, u)))
        {
            let mut stream = Stream::new(Net::Encodec(model), &bytes, &tables, start)?;
            let mut covered = start;
            let mut chunks = 0;
            let mut mismatches = 0;
            while let Some(chunk) = stream.decode_units(max_units)? {
                assert_eq!(chunk.offset, covered, "chunks must be contiguous");
                for c in 0..channels {
                    for t in 0..chunk.length {
                        let want = whole.data()[(chunk.offset + t) * channels + c];
                        if chunk.planar[c * chunk.length + t].to_bits() != want.to_bits() {
                            mismatches += 1;
                        }
                    }
                }
                covered += chunk.length;
                chunks += 1;
            }
            assert_eq!(covered, len, "start {start} max_units {max_units}");
            assert_eq!(stream.units_done(), stream.unit_count());
            assert!(
                start > 0 || max_units > 2 || chunks > 1,
                "max_units {max_units} should give several chunks"
            );
            assert_eq!(mismatches, 0, "start {start} max_units {max_units}");
        }
        check_codes_only(&bytes, &tables, &frames)
    }

    /// The codes-only paths (whole file and page at a time) give the encoder's codes.
    fn check_codes_only(bytes: &[u8], tables: &Tables, frames: &[EncodedFrame]) -> Result<()> {
        let all: Vec<u16> = frames
            .iter()
            .flat_map(|f| f.codes.iter().copied())
            .collect();
        assert_eq!(codes_of(bytes, tables)?.values, all);
        let mut reader = CodePages::open(bytes, tables)?;
        let mut paged = Vec::new();
        let mut count = 0;
        while let Some(page) = reader.next_page()? {
            paged.extend(page.values);
            count += 1;
        }
        assert_eq!(count, reader.page_count());
        assert_eq!(paged, all);
        Ok(())
    }

    #[test]
    fn stereo_stream_is_bit_exact() -> Result<()> {
        streams_like_whole_decode(&STOCK48)
    }

    #[test]
    fn stereo_stream_with_long_pages_is_bit_exact() -> Result<()> {
        streams_like_whole_decode(&Case {
            page_length: 2,
            seconds: 5.2,
            ..STOCK48
        })
    }

    #[test]
    fn mono_stream_is_bit_exact() -> Result<()> {
        streams_like_whole_decode(&Case {
            kind: ModelKind::Mono24k,
            stride: None,
            page_length: 75,
            ..STOCK48
        })
    }

    #[test]
    fn strided_edge_loop_stream_is_bit_exact() -> Result<()> {
        streams_like_whole_decode(&Case {
            seconds: 3.2,
            stride: Some(44_800),
            crossfade: Crossfade::EdgeAvoiding,
            looped: true,
            ..STOCK48
        })
    }

    #[test]
    fn strided_edge_stream_seeks_bit_exact() -> Result<()> {
        streams_like_whole_decode(&Case {
            seconds: 4.4,
            stride: Some(44_800),
            crossfade: Crossfade::EdgeAvoiding,
            page_length: 2,
            ..STOCK48
        })
    }

    #[test]
    fn mono_loop_stream_is_bit_exact() -> Result<()> {
        streams_like_whole_decode(&Case {
            kind: ModelKind::Mono24k,
            seconds: 2.5,
            stride: None,
            looped: true,
            page_length: 75,
            ..STOCK48
        })
    }

    /// The Vocos decoder builds by its name and plays HQ files: they name
    /// their code space, not a decoder.
    #[test]
    fn vocos_builds_as_an_hq_decoder() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let weights = std::fs::read(root.join("tests/data/vocos_tiny.safetensors"))?;
        let config = std::fs::read_to_string(root.join("tests/data/vocos_tiny.json"))?;
        let decoder = Decoder::build(vocos::NAME, &config, &Weights::from_bytes(&weights)?)?;
        assert_eq!(decoder.name(), "dac44_vocos");
        assert_eq!(decoder.model(), HQ_MODEL);
        let Net::Hq(hq) = decoder.net else {
            bail!("dac44_vocos built an EnCodec decoder");
        };
        let student = DacDecoder::from_bytes(&std::fs::read(
            root.join("weights/dac44_student.safetensors"),
        )?)?;
        assert_eq!(hq.codebook_hash(), student.codebook_hash());
        Ok(())
    }

    /// An HQ parametric stereo file through the bindings: the header, the
    /// tables, the streamed decode (equal to the library's whole-stream
    /// decode) and the refusals.
    #[test]
    fn hq_stream_decodes_through_the_bindings() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let weights = std::fs::read(root.join("weights/dac44_student.safetensors"))?;
        let config = std::fs::read_to_string(root.join("weights/dac44_student.json"))?;
        let table_bytes = std::fs::read(root.join("models/cx2_dac44.ecxm"))?;
        let decoder = Decoder::build(HQ_DECODER, &config, &Weights::from_bytes(&weights)?)?;
        let mut tables = Tables::new();
        let hash = tables.add_table(&table_bytes)?;
        let table = HqModel::from_table(&table_bytes)?;
        assert_eq!(hash, format!("{:016x}", table.hash()));
        assert!(tables.has(&hash));
        let Net::Hq(hq) = decoder.net else {
            bail!("dac44_student built an EnCodec decoder");
        };
        let cfg = picotone::pstereo::Config::DEFAULT;
        let frames = 170;
        let info = DacInfo {
            codebook_hash: hq.codebook_hash(),
            table_hash: table.hash(),
            layout: ChannelLayout::Parametric(cfg),
            codebooks: vec![9],
            looped: false,
            pre_skip: 700,
            page_length: pico::DEFAULT_PAGE_FRAMES,
            audio_length: frames * dac::HOP - 33,
            end_trim: 900,
            length_in_header: false,
            output_gain: 0,
            tags: Vec::new(),
        };
        let mut state = 5_u32;
        let codes: Vec<u16> = (0..info.frames() * 9)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                u16::try_from(state >> 22).unwrap_or(0)
            })
            .collect();
        let n = info.frames() * dac::HOP;
        #[expect(clippy::cast_precision_loss, reason = "test signal")]
        let left: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
        let right: Vec<f32> = left.iter().map(|v| v * 0.25).collect();
        let spans = info.pages();
        let params = picotone::pstereo::analyse(&left, &right, &spans, cfg);
        let pages = picotone::pstereo::write_pages(&params, &spans, cfg);
        let bytes = pico::write_dac_parametric(&info, &table, &[codes], &pages)?;

        let header = header_json(&bytes)?;
        assert_eq!(header["model"], HQ_MODEL);
        assert_eq!(header["channels"], 2);
        assert_eq!(header["sampleRate"], 44_100);
        assert_eq!(header["layout"], "parametric");
        assert_eq!(header["audioLength"], info.playable_samples());
        assert!(codes_of(&bytes, &tables).is_err());
        assert!(CodePages::open(&bytes, &tables).is_err());

        let whole = HqStream::new(DacStream::parse(&bytes)?, &table, hq, 0)?.decode_rest()?;
        let total = info.playable_samples();
        assert_eq!(whole.len(), total);
        for start in [0, total / 3] {
            let mut stream = Stream::new(decoder.net, &bytes, &tables, start)?;
            assert_eq!((stream.channels, stream.sample_rate), (2, 44_100));
            let mut covered = start;
            while let Some(chunk) = stream.decode_units(1)? {
                assert_eq!(chunk.offset, covered);
                for c in 0..2 {
                    for t in 0..chunk.length {
                        let want = whole.data()[(chunk.offset + t) * 2 + c];
                        assert_eq!(chunk.planar[c * chunk.length + t].to_bits(), want.to_bits());
                    }
                }
                covered += chunk.length;
            }
            assert_eq!(covered, total);
            assert_eq!(stream.units_done(), stream.unit_count());
            assert!(stream.cx2_ms > 0.0 && stream.network_ms > stream.cx2_ms);
        }
        // An EnCodec decoder refuses the file.
        let model: &'static Encodec =
            Box::leak(Box::new(ModelKind::Stereo48k.load(&root.join("weights"))?));
        assert!(Stream::new(Net::Encodec(model), &bytes, &tables, 0).is_err());
        Ok(())
    }

    #[test]
    fn a_missing_table_is_a_clear_error() -> Result<()> {
        let tables = Tables::new();
        let info = Info {
            model: ModelKind::Stereo48k,
            table_hash: 7,
            num_codebooks: 2,
            stride: Some(47_520),
            crossfade: Crossfade::Triangle,
            looped: false,
            pre_skip: 0,
            page_length: 1,
            audio_length: 10,
            end_trim: 0,
            length_in_header: true,
            output_gain: 0,
            tags: Vec::new(),
        };
        match tables.for_stream(&info) {
            Ok(_) => anyhow::bail!("a table was found in an empty set"),
            Err(error) => assert!(format!("{error}").contains("not loaded")),
        }
        Ok(())
    }
}

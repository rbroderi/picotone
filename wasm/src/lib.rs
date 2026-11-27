//! WebAssembly bindings for picotone's decoder.
//!
//! A [`Tables`] holds the cx2 entropy tables, found by the hash a `.pico`
//! stream stores. A [`Decoder`] holds one decoder network: a decoder-only
//! EnCodec model, or the HQ decoder of DAC 44 kHz codes (`dac44_vocos`).
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
//! The decode is the library's [`Player`], shared with the native player
//! plugin: its output equals a whole-file decode bit for bit, and memory
//! depends on the chunk size, not on the file length.

use anyhow::{Context as _, Result, bail};
use picotone::dac;
use picotone::model::Encodec;
use picotone::nn::Seq;
use picotone::pico::{self, ChannelLayout, Codec, DacInfo, DacStream, Info};
use picotone::player::{self, HQ_MODEL, Net, Pages, Player};
use picotone::vocos::{self, VocosDecoder};
use picotone::weights::Weights;
use wasm_bindgen::prelude::*;

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
    inner: player::Tables,
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
        u64::from_str_radix(hash, 16).is_ok_and(|h| self.inner.has(h))
    }
}

impl Tables {
    fn add_table(&mut self, table: &[u8]) -> Result<String> {
        Ok(format!("{:016x}", self.inner.add(table)?))
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
        let table = tables.inner.for_stream(&stream.info)?;
        let mut info = info_json(&stream.info);
        info["segmentFrames"] = serde_json::json!(stream.segment_frames);
        Ok(Self {
            info: info.to_string(),
            pages: Pages::new(stream, table, now_ms),
        })
    }

    fn next_page(&mut self) -> Result<Option<Codes>> {
        Ok(self.pages.next_page()?.map(|frames| Codes {
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
        self.pages.stream().pages.len()
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
    let table = tables.inner.for_stream(&stream.info)?;
    let frames = stream.decode_all(table)?;
    let mut info = info_json(&stream.info);
    info["segmentFrames"] = serde_json::json!(stream.segment_frames);
    Ok(Codes {
        info: info.to_string(),
        scales: frames.iter().filter_map(|f| f.scale).collect(),
        values: frames.into_iter().flat_map(|f| f.codes).collect(),
    })
}

/// A decoder network: a decoder-only EnCodec model (`encodec_24khz`,
/// `encodec_48khz`) or the HQ decoder (`dac44_vocos`).
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
    /// tensors; `dac44_vocos` takes its weight file as it is.
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

    /// Network name, e.g. `encodec_48khz` or `dac44_vocos`.
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
        Stream::new(self.net, pico_bytes, None, tables, 0).map_err(|e| to_js(&e))
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
        Stream::new(self.net, pico_bytes, None, tables, start_sample).map_err(|e| to_js(&e))
    }

    /// Like [`open_at`](Self::open_at), for an HQ file with its enhancement
    /// layer (`.picol`): decodes as the combined stream. The output equals the
    /// one-file encode of the combined codebooks.
    ///
    /// # Errors
    /// As [`open_at`](Self::open_at); also when the layer is not this file's,
    /// or any of its pages is damaged or missing. The caller then opens the
    /// file alone.
    #[wasm_bindgen(js_name = openLayered)]
    pub fn open_layered(
        &self,
        pico_bytes: &[u8],
        layer_bytes: &[u8],
        tables: &Tables,
        start_sample: usize,
    ) -> Result<Stream, JsError> {
        Stream::new(
            self.net,
            pico_bytes,
            Some(layer_bytes),
            tables,
            start_sample,
        )
        .map_err(|e| to_js(&e))
    }
}

impl Decoder {
    /// Builds the network from its tensors; `parsed` is the memory once the
    /// header is read, `packed` once every tensor is packed.
    fn build(name: &str, config_json: &str, w: &Weights<'_>) -> Result<Self> {
        let mut phases = Phases::default();
        phases.mark("parsed");
        let net = if name == vocos::NAME {
            let decoder = VocosDecoder::from_weights(w).context("building the HQ decoder")?;
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

/// A `.pico` file being decoded a few units at a time.
///
/// A unit is one coded segment (about 1 s) for segmented models, one second
/// of latent frames for causal models, and one page of frames (80 by
/// default, 0.93 s) for HQ files.
#[wasm_bindgen]
#[derive(Debug)]
pub struct Stream {
    player: Player,
}

impl Stream {
    fn new(
        net: Net,
        bytes: &[u8],
        layer: Option<&[u8]>,
        tables: &Tables,
        start_sample: usize,
    ) -> Result<Self> {
        let player = Player::open(net, bytes, layer, &tables.inner, start_sample, now_ms)?;
        Ok(Self { player })
    }

    /// Decodes up to `max_units` more units, as a planar chunk.
    fn decode_units(&mut self, max_units: usize) -> Result<Option<Chunk>> {
        Ok(self
            .player
            .decode_units(max_units)?
            .map(|chunk| planar_chunk(chunk.offset, &chunk.audio)))
    }
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
        self.player.model().to_owned()
    }

    /// Output sample rate in Hz.
    #[wasm_bindgen(getter, js_name = sampleRate)]
    #[must_use]
    pub fn sample_rate(&self) -> usize {
        self.player.sample_rate()
    }

    /// Output channel count.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.player.channels()
    }

    /// Audio length in samples per channel, after the pre-skip and end trim.
    #[wasm_bindgen(getter, js_name = audioLength)]
    #[must_use]
    pub fn audio_length(&self) -> usize {
        self.player.audio_length()
    }

    /// Whether the file is a seamless loop.
    #[wasm_bindgen(getter, js_name = isLoop)]
    #[must_use]
    pub fn is_loop(&self) -> bool {
        self.player.is_loop()
    }

    /// Codebooks per latent frame (bandwidth); for HQ files the largest
    /// count of any coded channel.
    #[wasm_bindgen(getter, js_name = numCodebooks)]
    #[must_use]
    pub fn num_codebooks(&self) -> usize {
        self.player.num_codebooks()
    }

    /// Milliseconds the cx2 (entropy) decode has taken so far.
    #[wasm_bindgen(getter, js_name = cx2Ms)]
    #[must_use]
    pub fn cx2_ms(&self) -> f64 {
        self.player.cx2_ms()
    }

    /// Milliseconds everything else (the network, stereo, trim) has taken.
    #[wasm_bindgen(getter, js_name = networkMs)]
    #[must_use]
    pub fn network_ms(&self) -> f64 {
        self.player.network_ms()
    }

    /// Number of units this stream decodes: coded segments, seconds of
    /// latent frames for causal models, or pages of frames for HQ files.
    #[wasm_bindgen(getter, js_name = segmentCount)]
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.player.unit_count()
    }

    /// Units decoded so far.
    #[wasm_bindgen(getter, js_name = segmentsDone)]
    #[must_use]
    pub fn segments_done(&self) -> usize {
        self.player.units_done()
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
        if decoder.model() != self.player.model() {
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

    use picotone::cx2;
    use picotone::cx2::hq::HqModel;
    use picotone::hqstream::HqStream;
    use picotone::layout::Trimmer;
    use picotone::model::EncodedFrame;

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
            let mut stream = Stream::new(Net::Encodec(model), &bytes, None, &tables, start)?;
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
            assert_eq!(stream.player.units_done(), stream.player.unit_count());
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

    /// The HQ decoder builds by its name and reads DAC's own codebooks, the
    /// quantiser's: HQ files name their code space, not a decoder.
    #[test]
    fn hq_decoder_builds_by_its_name() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let weights = std::fs::read(root.join("tests/data/vocos_tiny.safetensors"))?;
        let config = std::fs::read_to_string(root.join("tests/data/vocos_tiny.json"))?;
        let decoder = Decoder::build(vocos::NAME, &config, &Weights::from_bytes(&weights)?)?;
        assert_eq!(decoder.name(), "dac44_vocos");
        assert_eq!(decoder.model(), HQ_MODEL);
        let Net::Hq(hq) = decoder.net else {
            bail!("dac44_vocos built an EnCodec decoder");
        };
        let quantizer = dac::DacQuantizer::load(&root.join("weights/dac44_quantizer.safetensors"))?;
        assert_eq!(hq.codebook_hash(), quantizer.codebook_hash());
        Ok(())
    }

    /// An HQ parametric stereo file through the bindings: the header, the
    /// tables, the streamed decode (equal to the library's whole-stream
    /// decode) and the refusals. The decoder is the small Vocos fixture.
    #[test]
    fn hq_stream_decodes_through_the_bindings() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let weights = std::fs::read(root.join("tests/data/vocos_tiny.safetensors"))?;
        let config = std::fs::read_to_string(root.join("tests/data/vocos_tiny.json"))?;
        let table_bytes = std::fs::read(root.join("models/cx2_dac44.ecxm"))?;
        let decoder = Decoder::build(vocos::NAME, &config, &Weights::from_bytes(&weights)?)?;
        let mut tables = Tables::new();
        let hash = tables.add_table(&table_bytes)?;
        let table = HqModel::from_table(&table_bytes)?;
        assert_eq!(hash, format!("{:016x}", table.hash()));
        assert!(tables.has(&hash));
        let Net::Hq(hq) = decoder.net else {
            bail!("dac44_vocos built an EnCodec decoder");
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
            let mut stream = Stream::new(decoder.net, &bytes, None, &tables, start)?;
            assert_eq!((stream.channels(), stream.sample_rate()), (2, 44_100));
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
            assert_eq!(stream.player.units_done(), stream.player.unit_count());
            assert!(stream.cx2_ms() > 0.0 && stream.network_ms() > 0.0);
        }
        // An EnCodec decoder refuses the file.
        let model: &'static Encodec =
            Box::leak(Box::new(ModelKind::Stereo48k.load(&root.join("weights"))?));
        assert!(Stream::new(Net::Encodec(model), &bytes, None, &tables, 0).is_err());
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
        match tables.inner.for_stream(&info) {
            Ok(_) => anyhow::bail!("a table was found in an empty set"),
            Err(error) => assert!(format!("{error}").contains("not loaded")),
        }
        Ok(())
    }
}

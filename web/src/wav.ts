/** Minimal WAV reading (PCM 16/24/32, float 32) and writing (PCM 16 or float 32). */

export interface PlanarAudio {
  readonly sampleRate: number;
  readonly channels: readonly Float32Array[];
}

/** Encodes planar audio as a RIFF WAVE file. */
export function encodeWav(audio: PlanarAudio, format: 'pcm16' | 'float32' = 'pcm16'): Uint8Array {
  const C = audio.channels.length;
  const n = audio.channels[0]?.length ?? 0;
  const bps = format === 'pcm16' ? 2 : 4;
  const dataBytes = n * C * bps;
  const buf = new ArrayBuffer(44 + dataBytes);
  const v = new DataView(buf);
  const str = (o: number, s: string): void => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  v.setUint32(4, 36 + dataBytes, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, format === 'pcm16' ? 1 : 3, true);
  v.setUint16(22, C, true);
  v.setUint32(24, audio.sampleRate, true);
  v.setUint32(28, audio.sampleRate * C * bps, true);
  v.setUint16(32, C * bps, true);
  v.setUint16(34, bps * 8, true);
  str(36, 'data');
  v.setUint32(40, dataBytes, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < C; c++) {
      const s = audio.channels[c]?.[i] ?? 0;
      if (format === 'pcm16') {
        const clamped = Math.max(-1, Math.min(1, s));
        v.setInt16(o, Math.round(clamped * 32767), true);
        o += 2;
      } else {
        v.setFloat32(o, s, true);
        o += 4;
      }
    }
  }
  return new Uint8Array(buf);
}

/** Decodes a RIFF WAVE file to planar float audio. */
export function decodeWav(bytes: Uint8Array): PlanarAudio {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o: number): string => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (bytes.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('not a WAV file');
  let o = 12;
  let fmt: { tag: number; channels: number; rate: number; bits: number } | null = null;
  while (o + 8 <= bytes.length) {
    const id = tag(o);
    const size = v.getUint32(o + 4, true);
    const body = o + 8;
    if (id === 'fmt ') {
      let fmtTag = v.getUint16(body, true);
      if (fmtTag === 0xfffe && size >= 26) fmtTag = v.getUint16(body + 24, true);
      fmt = { tag: fmtTag, channels: v.getUint16(body + 2, true), rate: v.getUint32(body + 4, true), bits: v.getUint16(body + 14, true) };
    } else if (id === 'data') {
      if (!fmt) throw new Error('WAV data before fmt chunk');
      const bps = fmt.bits / 8;
      const end = Math.min(bytes.length, body + size);
      const n = Math.floor((end - body) / (bps * fmt.channels));
      const channels = Array.from({ length: fmt.channels }, () => new Float32Array(n));
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < fmt.channels; c++) {
          const p = body + (i * fmt.channels + c) * bps;
          let s: number;
          if (fmt.tag === 3 && fmt.bits === 32) s = v.getFloat32(p, true);
          else if (fmt.tag === 1 && fmt.bits === 16) s = v.getInt16(p, true) / 32768;
          else if (fmt.tag === 1 && fmt.bits === 24) s = ((v.getUint8(p) | (v.getUint8(p + 1) << 8) | (v.getInt8(p + 2) << 16)) / 8388608);
          else if (fmt.tag === 1 && fmt.bits === 32) s = v.getInt32(p, true) / 2147483648;
          else throw new Error(`unsupported WAV format ${fmt.tag}/${fmt.bits}`);
          (channels[c] as Float32Array)[i] = s;
        }
      }
      return { sampleRate: fmt.rate, channels };
    }
    o = body + size + (size & 1);
  }
  throw new Error('WAV file has no data chunk');
}

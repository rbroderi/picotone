/**
 * The look-ahead peak limiter at the end of every decode (mirrors `limiter::Limiter` in the Rust library:
 * same constants, and the same f32 arithmetic in the same order, so equal input gives equal output).
 *
 * Instead of clipping samples over the ceiling, the gain is lowered around them: it ramps down over the
 * `2 * ATTACK` samples before an over, is held for `HOLD` samples after it, and ramps back up. Channels
 * share one gain. The gain at a sample depends only on the input from `LOOKBACK` samples before it to
 * `LOOKAHEAD` after it; where none of those is over the ceiling it is exactly 1.
 */
export const CEILING = Math.fround(0.99);
export const ATTACK = 110;
export const HOLD = 2205;
export const LOOKAHEAD = 2 * ATTACK;
export const LOOKBACK = 2 * ATTACK + HOLD;

const f32 = Math.fround;

export class Limiter {
  /** Input samples not returned yet, per channel. */
  private pending: Float32Array[];
  /** The gain each sample needs on its own, from LOOKBACK samples before the first pending one. */
  private need: Float32Array;

  constructor(channels: number) {
    this.pending = Array.from({ length: channels }, () => new Float32Array(0));
    this.need = new Float32Array(LOOKBACK).fill(1);
  }

  /** Takes the next samples (planar) and returns those no later input can change: all but the last LOOKAHEAD. */
  push(audio: readonly Float32Array[]): Float32Array[] {
    const count = audio[0]?.length ?? 0;
    const need = new Float32Array(count);
    for (let t = 0; t < count; t++) {
      let peak = 0;
      for (const ch of audio) peak = Math.max(peak, Math.abs(ch[t] ?? 0));
      need[t] = peak > CEILING ? f32(CEILING / peak) : 1;
    }
    this.pending = this.pending.map((p, c) => concat(p, audio[c] ?? new Float32Array(count)));
    this.need = concat(this.need, need);
    return this.emit(Math.max(0, this.pendingLength() - LOOKAHEAD));
  }

  /** Ends the stream and returns the samples still held back. */
  finish(): Float32Array[] {
    this.need = concat(this.need, new Float32Array(LOOKAHEAD).fill(1));
    return this.emit(this.pendingLength());
  }

  private pendingLength(): number {
    return this.pending[0]?.length ?? 0;
  }

  private emit(n: number): Float32Array[] {
    const out = this.pending.map((p) => p.slice(0, n));
    this.pending = this.pending.map((p) => p.slice(n));
    if (n === 0) return out;
    const span = this.need.subarray(0, n + LOOKBACK + LOOKAHEAD);
    if (span.some((g) => g < 1)) {
      const g = gains(span, n);
      for (let t = 0; t < n; t++) {
        const gt = g[t] ?? 1;
        if (gt < 1) {
          for (const ch of out) ch[t] = Math.min(CEILING, Math.max(-CEILING, f32((ch[t] ?? 0) * gt)));
        }
      }
    }
    this.need = this.need.slice(n);
    return out;
  }
}

/** The gains of `n` samples, the first at need[LOOKBACK]: the held minimum, then its mean over ATTACK either side. */
function gains(need: Float32Array, n: number): Float32Array {
  const width = HOLD + 2 * ATTACK + 1;
  const count = n + 2 * ATTACK;
  const held = new Float32Array(count);
  const window = new Int32Array(count + width);
  let head = 0;
  let tail = 0;
  for (let j = 0; j < count + width - 1; j++) {
    const v = need[j] ?? 1;
    while (tail > head && (need[window[tail - 1] ?? 0] ?? 1) >= v) tail--;
    window[tail++] = j;
    if (j + 1 >= width) {
      const k = j + 1 - width;
      while (tail > head && (window[head] ?? 0) < k) head++;
      held[k] = need[window[head] ?? 0] ?? 1;
    }
  }
  const taps = 2 * ATTACK + 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = i; k < i + taps; k++) sum = f32(sum + (held[k] ?? 1));
    out[i] = f32(sum / taps);
  }
  return out;
}

function concat(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length === 0) return b.slice();
  const out = new Float32Array(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

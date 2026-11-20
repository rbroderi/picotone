/**
 * Pre-skip, end trim and loop fold (mirrors `layout::Trimmer` in encodec-rs).
 *
 * A file with `"ps"` / `"tr"` codes `ps + n + tr` samples of which the middle
 * `n` are the audio. For a loop (`"loop": true`) the encoder coded the loop's
 * own end before it, and the last `f = min(ps, n, rate / 10)` output samples
 * cross-fade into that decoded pre-roll:
 * `out[n - f + t] = (1 - a) raw[ps + n - f + t] + a raw[ps - f + t]` with
 * `a = (t + 1) / (f + 1)`, so `out[n - 1]` runs into `out[0]` the way
 * `raw[ps - 1]` runs into `raw[ps]`.
 */
export class Trimmer {
  private readonly fold: number;
  private taken = 0;
  /** The `fold` raw samples just before the pre-skip ends, per channel. */
  private readonly preroll: Float32Array[];

  constructor(
    private readonly preSkip: number,
    readonly length: number,
    looped: boolean,
    sampleRate: number,
    channels: number,
  ) {
    this.fold = looped ? Math.min(preSkip, length, Math.floor(sampleRate / 10)) : 0;
    this.preroll = Array.from({ length: channels }, () => new Float32Array(this.fold));
  }

  /** Whether the trimmer changes nothing. */
  get identity(): boolean {
    return this.preSkip === 0 && this.fold === 0;
  }

  /** Takes the next decoded samples (planar) and returns the output samples they complete. */
  push(raw: readonly Float32Array[]): Float32Array[] {
    const count = raw[0]?.length ?? 0;
    const { preSkip: ps, length: n, fold: f } = this;
    const start = this.taken;
    this.taken += count;
    // Output covers raw [ps, ps + n).
    const lo = Math.max(start, ps);
    const hi = Math.min(start + count, ps + n);
    const out = raw.map((ch, c) => {
      const pre = this.preroll[c] as Float32Array;
      // Keep the fold's pre-roll, raw [ps - f, ps).
      for (let r = Math.max(start, ps - f); r < Math.min(start + count, ps); r++) pre[r - (ps - f)] = ch[r - start] ?? 0;
      if (hi <= lo) return new Float32Array(0);
      const seg = ch.slice(lo - start, hi - start);
      for (let r = Math.max(lo, ps + n - f); r < hi; r++) {
        const t = r - (ps + n - f);
        const a = (t + 1) / (f + 1);
        seg[r - lo] = (1 - a) * (ch[r - start] ?? 0) + a * (pre[t] ?? 0);
      }
      return seg;
    });
    return out;
  }

  /** Throws when fewer samples than the file needs went through. */
  finish(): void {
    if (this.taken < this.preSkip + this.length) {
      throw new Error(`decoded ${this.taken} samples, the file needs ${this.preSkip + this.length}`);
    }
  }
}

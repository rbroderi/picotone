"""The pilot's pass/fail rule (and the full run's target), on the picotone test clips (sets old + new).

Thresholds come from the DAC paper's own 512-wide decoder ablation (trained jointly, 400k steps): mel distance
1.09 -> 1.11 (+1.8 %) and SI-SDR 9.1 -> 8.7 dB (-0.4 dB) against the 1536-wide model. A distilled student
should land about there at the end of a full run; we allow:

- full run:  log-mel <= stock DAC x 1.03  (the paper's +1.8 % plus measurement slack on 53 clips), and
             SI-SDR >= stock - 0.7 dB     (the paper's -0.4 dB plus slack);
- pilot:     log-mel <= stock DAC x 1.08  and SI-SDR >= stock - 1.5 dB. At 50k of 400k steps a GAN-trained
             decoder is still improving; DAC-style training curves put the mel loss a few per cent above its
             final value there, so ~2.5x the final gap is "on track", while a larger gap means width 512 (or
             the recipe) is not enough.
- both:      log-mel below picotone music12 on the same clips (the product claim: better than picotone at a
             matched bitrate; music12 is 9.97 kbps, DAC q5 raw 8.61 kbps), and still improving (last eval better
             than the one at half the steps). Not music24: measured on these 53 clips, stock DAC q5 itself is at
             2.807 dB against music24's 2.795 (old/new mean), so that test would fail the teacher. music24 and
             music6 are printed for information.
- listening: the user's call: no metallic/ringing tones, buzz, pre-echo or GAN hiss in the listen/ WAVs.
"""
PILOT = {"logmel_rel": 1.08, "sisdr_db": 1.5}
FULL = {"logmel_rel": 1.03, "sisdr_db": 0.7}
SETS = ("old", "new")
BITRATE_MATCH = "music12"


def _mean(d, key):
    vals = [d[s][key] for s in SETS if s in d]
    return sum(vals) / len(vals) if vals else float("nan")


def verdict(student, stock, history, picotone=None, rule="pilot", n_q=None):
    """student/stock: {set: metrics}; history: list of (step, {set: metrics}); picotone: {preset: {set: metrics}}."""
    th = PILOT if rule == "pilot" else FULL
    s_lm, t_lm = _mean(student, "logmel"), _mean(stock, "logmel")
    s_si, t_si = _mean(student, "sisdr"), _mean(stock, "sisdr")
    lines, ok, border = [], True, True
    lim = t_lm * th["logmel_rel"]
    c1 = s_lm <= lim
    lines.append(f"  log-mel {s_lm:.3f} dB vs stock DAC {t_lm:.3f} (limit {lim:.3f} = x{th['logmel_rel']}): "
                 f"{'ok' if c1 else 'FAIL'}")
    border &= s_lm <= t_lm * (1 + 1.5 * (th["logmel_rel"] - 1))
    c2 = s_si >= t_si - th["sisdr_db"]
    lines.append(f"  SI-SDR  {s_si:.2f} dB vs stock {t_si:.2f} (limit {t_si - th['sisdr_db']:.2f}): "
                 f"{'ok' if c2 else 'FAIL'}")
    border &= s_si >= t_si - 1.5 * th["sisdr_db"]
    c3 = True
    if picotone and BITRATE_MATCH in picotone:
        p_lm = _mean(picotone[BITRATE_MATCH], "logmel")
        c3 = s_lm < p_lm
        lines.append(f"  log-mel vs picotone {BITRATE_MATCH} {p_lm:.3f} (bitrate-matched): "
                     f"{'ok (better)' if c3 else 'FAIL (worse)'}")
        info = [f"{p} {_mean(picotone[p], 'logmel'):.3f}" for p in ("music24", "music6") if p in picotone]
        if info:
            lines.append(f"  (for information, picotone log-mel: {', '.join(info)})")
    else:
        lines.append(f"  picotone {BITRATE_MATCH}: not scored yet (run compare_picotone.py, pass --picotone-json)")
    c4 = True
    if len(history) >= 2:
        last_step = history[-1][0]
        half = min(history[:-1], key=lambda h: abs(h[0] - last_step / 2))
        h_lm = _mean(half[1], "logmel")
        c4 = s_lm < h_lm
        lines.append(f"  trend: log-mel {h_lm:.3f} at step {half[0]} -> {s_lm:.3f}: "
                     f"{'improving' if c4 else 'FLAT/WORSE'}")
    ok = c1 and c2 and c3 and c4
    status = "PASS" if ok else ("BORDERLINE" if (border and c3) else "FAIL")
    head = f"{rule.upper()} VERDICT{f' at {n_q} codebooks' if n_q else ''}: {status}"
    tail = "  + listen to listen/*.wav: no metallic/ringing tones, buzz or hiss (your call)"
    return status, "\n".join([head] + lines + [tail])

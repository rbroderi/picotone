"""Parameters, MACs and Snake evaluations per second of audio (one channel) for the student widths against
stock DAC's decoder, measured with forward hooks; plus a 1-thread CPU timing and projected browser speeds.

    uv run --extra cuda python report_model.py [--widths 512 384] [--time-seconds 2] [--json out.json]
"""
import argparse
import json
import time

import torch

from student import STOCK_WIDTH, StudentDecoder, count_cost, fold_weight_norm, n_params

FPS = 44100 / 512
# Earlier study (scratchpad dac_study): stock DAC 44 kHz decode, stereo, measured/projected real-time factors.
STOCK_RTF_WASM_1T = 0.22
STOCK_RTF_WEBGPU = 0.78   # the study's 512-wide WebGPU projection (about 7x) over its MAC ratio (about 9x)
SNAKE_MAC_EQ = 4.0        # degree-3 polynomial Snake with range reduction, in MAC-equivalents (snake_bench)


def cost_row(model, frames=862):
    z = torch.zeros(1, 1024, frames)
    c = count_cost(model, z)
    sec = frames / FPS
    return {"params": n_params(model), "gmac_per_s": c["mac"] / sec / 1e9, "snake_per_s": c["snake"] / sec,
            "gmac_eq_per_s": (c["mac"] + SNAKE_MAC_EQ * c["snake"]) / sec / 1e9}


def time_1t(model, seconds):
    torch.set_num_threads(1)
    z = torch.randn(1, 1024, int(seconds * FPS))
    with torch.inference_mode():
        model(z[..., :8])
        t = time.perf_counter()
        model(z)
        return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[512, 384])
    ap.add_argument("--time-seconds", type=float, default=2.0, help="audio length for the 1-thread timing; 0 = skip")
    ap.add_argument("--json")
    a = ap.parse_args()
    torch.manual_seed(0)
    rows = {}
    for w in [STOCK_WIDTH] + a.widths:
        m = fold_weight_norm(StudentDecoder(w)).eval()
        r = cost_row(m)
        if a.time_seconds > 0:
            r["cpu_1t_s_per_audio_s"] = time_1t(m, a.time_seconds) / a.time_seconds
        rows[w] = r
    base = rows[STOCK_WIDTH]
    print(f"{'width':>6} {'params':>11} {'GMAC/s':>8} {'Snake M/s':>10} {'MAC ratio':>9} {'CPU 1T x':>9} "
          f"{'wasm 1T stereo':>15} {'WebGPU stereo':>14}")
    for w, r in rows.items():
        ratio = base["gmac_eq_per_s"] / r["gmac_eq_per_s"]
        r["mac_ratio_vs_stock"] = ratio
        r["proj_rtf_wasm_1t_stereo"] = STOCK_RTF_WASM_1T * ratio
        r["proj_rtf_webgpu_stereo"] = STOCK_RTF_WEBGPU * ratio
        cpu = base.get("cpu_1t_s_per_audio_s", 0) / r["cpu_1t_s_per_audio_s"] if "cpu_1t_s_per_audio_s" in r else 0
        r["cpu_1t_speedup_vs_stock"] = cpu
        tag = "stock" if w == STOCK_WIDTH else "student"
        print(f"{w:>6} {r['params']:>11,} {r['gmac_per_s']:>8.2f} {r['snake_per_s'] / 1e6:>10.2f} {ratio:>8.2f}x "
              f"{cpu:>8.2f}x {r['proj_rtf_wasm_1t_stereo']:>14.2f}x {r['proj_rtf_webgpu_stereo']:>13.2f}x  {tag}")
    print("MACs/Snake per second of audio for ONE channel; stereo costs 2x. Projections scale the study's stock "
          f"DAC figures ({STOCK_RTF_WASM_1T}x wasm 1-thread, {STOCK_RTF_WEBGPU}x WebGPU, stereo) by the MAC-equivalent "
          f"ratio, with Snake as a degree-3 polynomial ({SNAKE_MAC_EQ:g} MAC-eq).")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()

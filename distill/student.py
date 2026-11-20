"""The student: DAC's decoder architecture at a narrower width.

Same layout as `dac.model.dac.Decoder` (MIT, Copyright (c) 2023-present, Descript; see LICENSES/DAC-MIT.txt):
a k7 input conv to `width` channels, four DecoderBlocks (Snake, transposed conv with stride 8/8/4/2 halving the
width, then residual units with dilations 1/3/9), Snake, a k7 output conv and tanh. Stock DAC 44 kHz is width
1536; the student is 512 (or 384, ...).

Weight norm is a parametrization while training; `fold_weight_norm` removes it for export. Snake and the
residual sums run in float32 even under bf16 autocast, so the residual stream keeps full precision.
"""
import math

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

RATES = (8, 8, 4, 2)
LATENT = 1024
STOCK_WIDTH = 1536


class _SnakeFn(torch.autograd.Function):
    """y = x + sin(a x)^2 / a, saving only x and a for backward (autograd would keep ~4 fp32 tensors)."""

    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(x, a)  # x as given (bf16 conv output under autocast): upcasting it is exact
        x = x.to(torch.promote_types(x.dtype, torch.float32))
        return x + (a + 1e-9).reciprocal() * torch.sin(a * x).pow(2)

    @staticmethod
    def backward(ctx, gy):
        x, a = ctx.saved_tensors
        in_dtype = x.dtype
        x = x.to(torch.promote_types(x.dtype, torch.float32))
        ax = a * x
        s2 = torch.sin(2 * ax)  # d/dx sin^2(ax) / a = sin(2ax)
        gx = gy * (1 + s2)
        inv = (a + 1e-9).reciprocal()
        ga = gy * (x * s2 * inv - torch.sin(ax).pow(2) * inv.pow(2))
        return gx.to(in_dtype), ga.sum(dim=(0, 2), keepdim=True)


class Snake1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, ch, 1))

    def forward(self, x):
        return _SnakeFn.apply(x, self.alpha.float())


def wn_conv(*a, **k):
    return weight_norm(nn.Conv1d(*a, **k))


def wn_tconv(*a, **k):
    return weight_norm(nn.ConvTranspose1d(*a, **k))


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(Snake1d(dim), wn_conv(dim, dim, 7, dilation=dilation, padding=pad),
                                   Snake1d(dim), wn_conv(dim, dim, 1))

    def forward(self, x):
        return x.float() + self.block(x).float()


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(cin),
            wn_tconv(cin, cout, 2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            ResidualUnit(cout, 1), ResidualUnit(cout, 3), ResidualUnit(cout, 9))

    def forward(self, x):
        return self.block(x)


class StudentDecoder(nn.Module):
    """Input z_q [B, 1024, F] -> audio [B, 1, 512 F]."""

    def __init__(self, width=512, rates=RATES, latent=LATENT):
        super().__init__()
        self.width = width
        layers = [wn_conv(latent, width, 7, padding=3)]
        for i, s in enumerate(rates):
            layers.append(DecoderBlock(width // 2 ** i, width // 2 ** (i + 1), s))
        out = width // 2 ** len(rates)
        layers += [Snake1d(out), wn_conv(out, 1, 7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.parametrizations.weight.original1, std=0.02)
                nn.init.constant_(m.bias, 0)
                with torch.no_grad():
                    w = m.parametrizations.weight.original1
                    m.parametrizations.weight.original0.copy_(
                        torch.linalg.vector_norm(w, dim=(1, 2), keepdim=True))

    def forward(self, z):
        # The output head (Snake, last conv, tanh) always runs in fp32: a bf16 output rounds every sample to
        # 8 mantissa bits, a noise floor that costs stock DAC about 0.8 dB log-mel (fp32 head: 0.07 dB).
        h = self.model[:-3](z)
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.model[-3:](h.float()).float()


def fold_weight_norm(model):
    """In place: replace weight-norm parametrizations by plain weights (for export / fast inference)."""
    for m in model.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return model


# ---------------------------------------------------------------- warm start from the teacher

def _plain_weight(m):
    """Effective (folded) weight of a teacher conv, whichever weight-norm API it uses."""
    return m.weight.detach()


def _teacher_convs(dec):
    """Ordered (kind, module) for the teacher decoder's convs and snakes; same order as the student's."""
    out = []
    for m in dec.modules():
        n = type(m).__name__
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)) or n == "Snake1d":
            out.append(m)
    return out


@torch.no_grad()
def warm_start_slice(student, teacher_decoder):
    """Initialise the student by keeping, per stage, the teacher channels with the largest weight energy.

    Channel selection commutes with Snake (a per-channel function), so a sliced network computes exactly the
    teacher's function restricted to the kept channels. Each stage's residual stream shares one channel set,
    chosen by the product of incoming and outgoing weight norms summed over the stage's convs. The dropped
    channels' contributions are lost, so this is only a starting point (cheaper than PCA, which would mix
    channels and break Snake's per-channel form).
    """
    t_mods = _teacher_convs(teacher_decoder)
    s_mods = _teacher_convs(student)
    assert len(t_mods) == len(s_mods), (len(t_mods), len(s_mods))
    tw = [None if type(m).__name__ == "Snake1d" else _plain_weight(m) for m in t_mods]
    # Stage of every module: stage 0 = width (input conv .. first tconv input), stage i = after tconv i.
    stage_of, st = [], 0
    for m in t_mods:
        if isinstance(m, nn.ConvTranspose1d):
            stage_of.append(("t", st))
            st += 1
        else:
            stage_of.append(st)
    n_stages = st + 1
    t_width = [None] * n_stages
    imp = [None] * n_stages
    for i, m in enumerate(t_mods):
        w = tw[i]
        if w is None:
            continue
        so = stage_of[i]
        if isinstance(m, nn.ConvTranspose1d):  # weight [Cin, Cout, k]: in = stage so[1], out = so[1] + 1
            a, b = so[1], so[1] + 1
            e_in, e_out = w.pow(2).sum(dim=(1, 2)), w.pow(2).sum(dim=(0, 2))
        else:  # [Cout, Cin, k]
            a = b = so
            e_in, e_out = w.pow(2).sum(dim=(0, 2)), w.pow(2).sum(dim=(1, 2))
            if w.shape[1] == LATENT and i == 0:
                e_in = None
            if w.shape[0] == 1:
                e_out = None
        for stg, e in ((a, e_in), (b, e_out)):
            if e is None:
                continue
            imp[stg] = e if imp[stg] is None else imp[stg] + e
            t_width[stg] = e.numel()
    s_width = [student.width // 2 ** k for k in range(n_stages)]
    keep = [torch.sort(torch.topk(imp[k], s_width[k]).indices).values for k in range(n_stages)]

    def stage_in_out(i):
        so = stage_of[i]
        if isinstance(so, tuple):
            return so[1], so[1] + 1
        return so, so

    for i, (tm, sm) in enumerate(zip(t_mods, s_mods)):
        a, b = stage_in_out(i)
        if type(tm).__name__ == "Snake1d":
            sm.alpha.copy_(tm.alpha[:, keep[a]])
            continue
        w = tw[i]
        bias = tm.bias.detach()
        if isinstance(tm, nn.ConvTranspose1d):
            w = w[keep[a]][:, keep[b]]
            bias = bias[keep[b]]
        else:
            if w.shape[0] != 1:
                w = w[keep[b]]
                bias = bias[keep[b]]
            if w.shape[1] != LATENT:
                w = w[:, keep[a]]
        # Compensate the lost fan-in energy so activations keep roughly their teacher scale.
        fan_t = tw[i].shape[0 if isinstance(tm, nn.ConvTranspose1d) else 1]
        fan_s = w.shape[0 if isinstance(tm, nn.ConvTranspose1d) else 1]
        if fan_t != fan_s:
            w = w * math.sqrt(fan_t / fan_s)
        sm.weight = w.contiguous()  # weight-norm right_inverse sets g and v
        sm.bias.copy_(bias)
    return student


# ---------------------------------------------------------------- cost report

def count_cost(model, z):
    """MACs and Snake evaluations for one forward on z [1, C, F]. Measured with forward hooks."""
    tot = {"mac": 0, "snake": 0}
    hooks = []

    def conv_hook(m, inp, out):
        x = inp[0]
        if isinstance(m, nn.Conv2d):
            tot["mac"] += out[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.ConvTranspose1d):
            tot["mac"] += x.shape[-1] * m.in_channels * m.out_channels * m.kernel_size[0] // m.groups
        else:
            tot["mac"] += out.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]

    def snake_hook(m, inp, out):
        tot["snake"] += inp[0].numel()

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif type(m).__name__ == "Snake1d":
            hooks.append(m.register_forward_hook(snake_hook))
    with torch.inference_mode():
        model(z)
    for h in hooks:
        h.remove()
    return tot


def n_params(model):
    return sum(p.numel() for p in model.parameters())

"""Frozen DAC 44 kHz (descript-audio-codec, MIT): encoder + quantiser give the student's inputs, the stock
decoder gives the target line and the optional output-matching target."""
import torch
from torch.nn.utils import parametrize

HOP = 512
N_CB = 9


def load_teacher(device="cpu"):
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    import dac

    model = dac.DAC.load(dac.utils.download(model_type="44khz", model_bitrate="8kbps"))
    for m in model.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
        elif parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_codes(teacher, x):
    """x [B, 1, T] (T a multiple of 512, 44.1 kHz) -> codes [B, 9, T/512] (long). fp32, no autocast."""
    assert x.shape[-1] % HOP == 0
    with torch.autocast(device_type=x.device.type, enabled=False):
        z = teacher.encoder(x.float())
        _, codes, *_ = teacher.quantizer(z, n_quantizers=N_CB)
    return codes


@torch.no_grad()
def codes_to_z(teacher, codes, n_q):
    """codes [B, 9, F]; n_q int or LongTensor [B] -> z_q [B, 1024, F] summing the first n_q codebooks."""
    q = teacher.quantizer
    B = codes.shape[0]
    if isinstance(n_q, int):
        n_q = torch.full((B,), n_q, device=codes.device)
    z = 0.0
    for i in range(int(n_q.max())):
        zi = q.quantizers[i].out_proj(q.quantizers[i].decode_code(codes[:, i, :]))
        z = z + zi * (n_q > i).view(B, 1, 1).to(zi.dtype)
    return z


@torch.no_grad()
def stock_decode(teacher, z):
    return teacher.decoder(z)


def encoder_cost(teacher, seconds=4.0):
    from student import count_cost

    T = int(44100 * seconds) // HOP * HOP
    c = count_cost(teacher.encoder, torch.zeros(1, 1, T))
    return {k: v / (T / 44100) for k, v in c.items()}

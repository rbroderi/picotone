"""Small network helpers shared by the training and export scripts: weight-norm folding, a MAC/Snake cost count
and a parameter count."""
import torch
from torch import nn
from torch.nn.utils import parametrize


def fold_weight_norm(model):
    """In place: replace weight-norm parametrizations by plain weights (for export / fast inference)."""
    for m in model.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return model


def count_cost(model, z):
    """MACs and Snake evaluations for one forward on z [1, C, F]. Measured with forward hooks (convs and
    Linear layers; the Vocos head's inverse FFT is not counted, about 0.003 GMAC/s)."""
    tot = {"mac": 0, "snake": 0}
    hooks = []

    def conv_hook(m, inp, out):
        x = inp[0]
        if isinstance(m, nn.Linear):  # the Vocos decoder's pointwise layers and head (vocos_student.py)
            tot["mac"] += out.numel() // out.shape[-1] * m.in_features * m.out_features
        elif isinstance(m, nn.Conv2d):
            tot["mac"] += out[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.ConvTranspose1d):
            tot["mac"] += x.shape[-1] * m.in_channels * m.out_channels * m.kernel_size[0] // m.groups
        else:
            tot["mac"] += out.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]

    def snake_hook(m, inp, out):
        tot["snake"] += inp[0].numel()

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.Linear)):
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

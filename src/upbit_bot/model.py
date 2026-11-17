import torch
from torch import nn

from .model_names import ARCHITECTURES


class ReturnLSTM(nn.Module):
    """LSTM return forecaster.

    ``attention=False`` reproduces the original architecture (last hidden state only).
    ``attention=True`` normalizes inputs, adds learned attention pooling over all time steps
    and concatenates it with the last state, so short spikes earlier in the window are not lost.
    """

    def __init__(self, inputs, hidden=64, layers=2, dropout=0.15, attention=False):
        super().__init__()
        self.attention = attention
        self.norm = nn.LayerNorm(inputs) if attention else nn.Identity()
        self.lstm = nn.LSTM(inputs, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        if attention:
            self.score = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))
            width = hidden * 2
        else:
            width = hidden
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(self.norm(x))
        last = output[:, -1]
        if self.attention:
            weights = torch.softmax(self.score(output), dim=1)
            last = torch.cat([last, (weights * output).sum(dim=1)], dim=-1)
        return self.head(last).squeeze(-1) * 100.0


def _head(width, dropout):
    return nn.Sequential(
        nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 1)
    )


class ReturnGRU(nn.Module):
    """GRU variant: fewer parameters than LSTM, often equal accuracy on short noisy windows."""

    def __init__(self, inputs, hidden=64, layers=2, dropout=0.15, **_):
        super().__init__()
        self.norm = nn.LayerNorm(inputs)
        self.rnn = nn.GRU(inputs, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.score = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = _head(hidden * 2, dropout)

    def forward(self, x):
        output, _ = self.rnn(self.norm(x))
        weights = torch.softmax(self.score(output), dim=1)
        pooled = torch.cat([output[:, -1], (weights * output).sum(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1) * 100.0


class _CausalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout, kernel=3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = self.drop(y)
        y = self.conv2(nn.functional.pad(y, (self.pad, 0)))
        return self.act(x + self.drop(y))


class ReturnTCN(nn.Module):
    """Temporal convolution network: dilated causal convolutions (1..32) see the whole window in
    parallel, no recurrence. Strong at local multi-scale patterns (spikes, breakouts)."""

    def __init__(self, inputs, hidden=64, layers=2, dropout=0.15, **_):
        super().__init__()
        self.norm = nn.LayerNorm(inputs)
        self.proj = nn.Conv1d(inputs, hidden, 1)
        # Receptive field 1 + 2*2*(1+2+...+32) = 253 bars >= any configured lookback.
        self.blocks = nn.Sequential(*[_CausalBlock(hidden, 2**i, dropout) for i in range(6)])
        self.head = _head(hidden * 2, dropout)

    def forward(self, x):
        y = self.blocks(self.proj(self.norm(x).transpose(1, 2)))
        pooled = torch.cat([y[:, :, -1], y.mean(dim=2)], dim=-1)
        return self.head(pooled).squeeze(-1) * 100.0


class _AttentionBlock(nn.Module):
    """Pre-norm self-attention block written with plain ops so ONNX export is exact."""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, w // self.heads).permute(
            2, 0, 3, 1, 4
        )
        att = torch.softmax(q @ k.transpose(-2, -1) / (w // self.heads) ** 0.5, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(b, t, w)
        x = x + self.drop(self.out(y))
        return x + self.drop(self.ff(self.norm2(x)))


class ReturnTransformer(nn.Module):
    """Transformer encoder over 4-bar patches (PatchTST-style): 96 bars -> 24 tokens, which keeps
    attention cheap and denoises single-bar noise. Learned position embedding."""

    def __init__(self, inputs, hidden=64, layers=2, dropout=0.15, lookback=96, patch=4, heads=4, **_):
        super().__init__()
        if lookback % patch:
            patch = 1
        self.patch, self.tokens = patch, lookback // patch
        self.norm = nn.LayerNorm(inputs)
        self.embed = nn.Linear(inputs * patch, hidden)
        self.position = nn.Parameter(torch.zeros(1, self.tokens, hidden))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.Sequential(*[_AttentionBlock(hidden, heads, dropout) for _ in range(layers)])
        self.final = nn.LayerNorm(hidden)
        self.head = _head(hidden * 2, dropout)

    def forward(self, x):
        b = x.shape[0]
        z = self.norm(x).reshape(b, self.tokens, -1)
        z = self.final(self.blocks(self.embed(z) + self.position))
        pooled = torch.cat([z[:, -1], z.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1) * 100.0




def build_member(architecture, inputs, mc, attention=False):
    """One ensemble member. ``mc`` is the [model] config section."""
    kwargs = {k: mc[k] for k in ("hidden", "layers", "dropout")}
    if architecture == "lstm":
        return ReturnLSTM(inputs, **kwargs, attention=attention)
    if architecture == "gru":
        return ReturnGRU(inputs, **kwargs)
    if architecture == "tcn":
        return ReturnTCN(inputs, **kwargs)
    if architecture == "transformer":
        return ReturnTransformer(inputs, **kwargs, lookback=mc["lookback"])
    raise ValueError(f"Unknown architecture {architecture!r}; choose from {ARCHITECTURES}")


def member_architectures(mc, members):
    """``architecture`` may be one name or a list cycled over members (heterogeneous ensemble)."""
    value = mc.get("architecture", "lstm")
    names = [value] if isinstance(value, str) else list(value)
    if not names or any(n not in ARCHITECTURES for n in names):
        raise ValueError(f"model.architecture must be one of {ARCHITECTURES} or a list of them")
    return [names[i % len(names)] for i in range(members)]


class Ensemble(nn.Module):
    """Averages independently trained members. Second output is member disagreement (std, bps),
    a cheap epistemic-uncertainty estimate the runtime can penalize."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        preds = torch.stack([m(x) for m in self.members], dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0, unbiased=False) if len(self.members) > 1 else torch.zeros_like(mean)
        return mean, std


def export_onnx(model, inputs, lookback, destination):
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = model.cpu().eval()
    dummy = torch.randn(1, lookback, inputs)
    torch.onnx.export(
        model,
        dummy,
        str(destination),
        input_names=["features"],
        output_names=["return_bps", "uncertainty_bps"],
        dynamic_axes={"features": {0: "batch"}, "return_bps": {0: "batch"}, "uncertainty_bps": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    max_error = 0.0
    for size in (1, 5):
        x = torch.randn(size, lookback, inputs)
        with torch.inference_mode():
            reference = [t.numpy() for t in model(x)]
        actual = session.run(None, {"features": x.numpy()})
        for ref, act in zip(reference, actual):
            np.testing.assert_allclose(act, ref, atol=0.005, rtol=0.001)
            max_error = max(max_error, float(np.max(np.abs(ref - act))))
    return max_error

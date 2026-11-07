import torch
from torch import nn


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

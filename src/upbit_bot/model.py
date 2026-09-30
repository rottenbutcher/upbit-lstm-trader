import torch
from torch import nn


class ReturnLSTM(nn.Module):
    def __init__(self, inputs, hidden=64, layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(inputs, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).squeeze(-1) * 100.0


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
        output_names=["return_bps"],
        dynamic_axes={"features": {0: "batch"}, "return_bps": {0: "batch"}},
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
            reference = model(x).numpy()
        actual = session.run(None, {"features": x.numpy()})[0]
        np.testing.assert_allclose(actual, reference, atol=0.005, rtol=0.001)
        max_error = max(max_error, float(np.max(np.abs(reference - actual))))
    return max_error

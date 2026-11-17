"""Architecture names without importing torch (the inference server has no torch)."""

ARCHITECTURES = ("lstm", "gru", "tcn", "transformer")


def check_architecture(mc):
    value = mc.get("architecture", "lstm")
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, list) or not names or any(n not in ARCHITECTURES for n in names):
        raise ValueError(f"model.architecture must be one of {ARCHITECTURES} or a list of them")
    if "transformer" in names and int(mc["hidden"]) % 4:
        raise ValueError("model.hidden must be divisible by 4 for the transformer (4 heads)")

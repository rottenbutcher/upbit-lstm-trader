import zipfile
from pathlib import Path

from .runtime import Bundle


def package(root=".", artifacts="artifacts", destination="dist/oracle-upload.zip"):
    root, artifacts, destination = Path(root), Path(artifacts), Path(destination)
    Bundle(artifacts)  # Integrity and runtime loading check before packaging.
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        root / "pyproject.toml",
        root / "config.toml",
        root / "README.md",
        root / ".env.example",
        root / "requirements-runtime.txt",
    ]
    for directory in ("src", "deploy", "docs"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        for name in ("model.onnx", "manifest.json", "report.json"):
            archive.write(artifacts / name, "artifacts/" + name)
    return destination

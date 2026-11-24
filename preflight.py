"""Check a future GPU host without downloading models or making API calls."""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/gpu.example.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(__file__).parent
    checks = {name: shutil.which(name) is not None for name in ("ffmpeg", "ffprobe")}
    for name in ("torch", "torchaudio", "insightface", "onnxruntime", "speakerlab"):
        checks[name] = importlib.util.find_spec(name) is not None
    checks["cuda_available"] = False
    if checks["torch"]:
        try:
            import torch
            checks["cuda_available"] = torch.cuda.is_available()
        except Exception as error:
            checks["torch_import_error"] = str(error)
    for item in cfg["inputs"]:
        checks[item] = (root / item).is_file()
    face = root / cfg["face"]["model_dir"]
    checks["face_onnx_files_present"] = face.is_dir() and bool(list(face.glob("*.onnx")))
    checks["voice_checkpoint_present"] = (root / cfg["voice"]["checkpoint"]).is_file()
    # Do not claim inference readiness merely because files/imports are present.
    checks["inference_adapters_implemented"] = False
    print(json.dumps({"checks": checks, "inference_ready": False}, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Prepare reproducible AMI smoke-test clips; requires ffmpeg and ffprobe only."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "data/ami")
    args = parser.parse_args()
    root = args.root.resolve()
    video = root / "raw/IS1009a.C.avi"
    audio = root / "raw/IS1009a.Mix-Headset.wav"
    for path, kind in ((video, "video"), (audio, "audio")):
        if not any(s["codec_type"] == kind for s in probe(path)["streams"]):
            raise ValueError(f"Missing {kind} stream: {path}")
    out = root / "clips"
    out.mkdir(exist_ok=False)
    manifest = {"dataset": "AMI", "meeting": "IS1009a", "license": "CC BY 4.0",
        "attribution": "AMI Meeting Corpus, AMI Consortium",
        "source": "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1009a/",
        "documentation": "https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml",
        "note": "Low-resolution central camera; muxed with mixed headset audio. No manual AV-sync or identity ground-truth validation performed.",
        "raw": [{"file": str(p.relative_to(root)), "sha256": sha256(p)} for p in (video, audio)],
        "clips": []}
    for index, start in enumerate((60, 180), 1):
        path = out / f"IS1009a_{index:02d}.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-ss", str(start), "-i", str(video), "-ss", str(start), "-i", str(audio),
            "-t", "60", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(path)]
        subprocess.run(command, check=True)
        info = probe(path)
        if not {"video", "audio"}.issubset({s["codec_type"] for s in info["streams"]}):
            raise ValueError("Output missing audio or video")
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], check=True)
        manifest["clips"].append({"file": str(path.relative_to(root)), "source_start": start,
            "duration": float(info["format"]["duration"]), "sha256": sha256(path),
            "streams": [{k: s[k] for k in ("codec_type", "codec_name", "width", "height", "sample_rate", "channels") if k in s} for s in info["streams"]]})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["clips"], indent=2))


if __name__ == "__main__":
    main()

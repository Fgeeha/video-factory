#!/usr/bin/env python3
"""Assemble a vertical (9:16) VK Clip from still photos + text-to-speech + music.

Pipeline: photos -> Ken Burns segments (ffmpeg zoompan) -> concat -> voiceover
(piper) -> auto subtitles (faster-whisper) -> mux with background music.

Usage:
    uv run make_clip.py example.config.yaml
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

import comfy_client

WIDTH, HEIGHT, FPS = 1080, 1920, 30


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def split_durations(n: int, total: float, given: list[float] | None = None) -> list[float]:
    """Split `total` seconds across `n` images, or use per-image durations if given."""
    if given:
        if len(given) != n:
            raise ValueError(f"durations count {len(given)} != images count {n}")
        return list(given)
    base = round(total / n, 2)
    durations = [base] * n
    durations[-1] = round(total - base * (n - 1), 2)
    return durations


def build_zoompan_filter(duration: float, zoom_end: float = 1.15) -> str:
    """Ken Burns: slow zoom-in over the segment's duration."""
    frames = max(1, round(duration * FPS))
    zoom_step = (zoom_end - 1.0) / frames
    return (
        f"scale={WIDTH * 2}:{HEIGHT * 2}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH * 2}:{HEIGHT * 2},"
        f"zoompan=z='min(zoom+{zoom_step:.6f}\\,{zoom_end})':"
        f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"
    )


def synth_voiceover(text: str, voice_model: Path, out_wav: Path) -> None:
    subprocess.run(
        ["piper", "--model", str(voice_model), "--output_file", str(out_wav)],
        input=text, check=True, text=True,
    )


def render_image_segment(image: Path, duration: float, out_mp4: Path) -> None:
    run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(image),
        "-vf", build_zoompan_filter(duration),
        "-t", str(duration), "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4),
    ])


def normalize_segment(src: Path, out_mp4: Path) -> None:
    """Re-encode a generated video clip (e.g. Wan2.2's 480x832@16fps) to the standard canvas."""
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},fps={FPS}",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4),
    ])


def resolve_images(entries: list[dict], base: Path, tmp_dir: Path, comfy_url: str) -> list[dict]:
    """Turn each `images:` entry into {"kind": "photo"|"video", "path": Path, "duration": float|None}.

    "duration" is None for photos without an explicit duration (split evenly later),
    and always set for generated videos (their length is fixed by generation).
    """
    items = []
    for i, entry in enumerate(entries):
        if "file" in entry:
            items.append({"kind": "photo", "path": base / entry["file"], "duration": entry.get("duration")})
        elif "generate_image" in entry:
            spec = entry["generate_image"]
            path = tmp_dir / f"gen_img_{i:03d}.png"
            comfy_client.generate_image(spec["prompt"], path, comfy_url=comfy_url)
            items.append({"kind": "photo", "path": path, "duration": entry.get("duration")})
        elif "generate_video" in entry:
            spec = entry["generate_video"]
            path = tmp_dir / f"gen_vid_{i:03d}.mp4"
            comfy_client.generate_video(spec["prompt"], path, length=spec.get("length", 33), comfy_url=comfy_url)
            items.append({"kind": "video", "path": path, "duration": ffprobe_duration(path)})
        else:
            raise ValueError(f"images[{i}] needs one of: file, generate_image, generate_video")
    return items


def concat_segments(segments: list[Path], out_mp4: Path, tmp_dir: Path) -> None:
    list_file = tmp_dir / "concat.txt"
    list_file.write_text("".join(f"file '{s}'\n" for s in segments))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_mp4)])


def transcribe_to_srt(wav_path: Path, srt_path: Path, model_size: str) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav_path), language="ru")
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def _srt_ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def mux_final(video: Path, voice: Path, music: Path | None, music_volume: float,
              srt: Path | None, out: Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(voice)]
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]

    vf = f"subtitles={srt}" if srt else None
    if music:
        filt = f"[2:a]volume={music_volume}[bg];[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        maps = ["-map", "0:v", "-map", "[aout]"]
    else:
        filt = None
        maps = ["-map", "0:v", "-map", "1:a"]

    filter_complex = []
    if vf:
        filter_complex.append(f"[0:v]{vf}[v]")
        maps[1] = "[v]"
    if filt:
        filter_complex.append(filt)

    if filter_complex:
        cmd += ["-filter_complex", ";".join(filter_complex)]
    cmd += maps
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)]
    run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--whisper-model", default="small", help="faster-whisper model size")
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument("--comfy-url", default=os.environ.get("COMFYUI_URL", comfy_client.DEFAULT_URL),
                         help="ComfyUI server for generate_image/generate_video scenes")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base = args.config.parent

    out_path = base / cfg["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        voice_wav = tmp_dir / "voice.wav"
        synth_voiceover(cfg["voiceover_text"], base / cfg["voice_model"], voice_wav)
        voice_duration = ffprobe_duration(voice_wav)

        items = resolve_images(cfg["images"], base, tmp_dir, args.comfy_url)

        video_total = sum(it["duration"] for it in items if it["kind"] == "video")
        photo_items = [it for it in items if it["kind"] == "photo"]
        photo_durations = []
        if photo_items:
            given = [it["duration"] for it in photo_items if it["duration"] is not None] or None
            photo_durations = split_durations(len(photo_items), max(voice_duration - video_total, 0.1), given)

        segments = []
        photo_i = 0
        for i, it in enumerate(items):
            seg = tmp_dir / f"seg_{i:03d}.mp4"
            if it["kind"] == "video":
                normalize_segment(it["path"], seg)
            else:
                render_image_segment(it["path"], photo_durations[photo_i], seg)
                photo_i += 1
            segments.append(seg)

        concat_video = tmp_dir / "concat.mp4"
        concat_segments(segments, concat_video, tmp_dir)

        srt_path = None
        if not args.no_subtitles:
            srt_path = tmp_dir / "captions.srt"
            transcribe_to_srt(voice_wav, srt_path, args.whisper_model)

        music = base / cfg["music"] if cfg.get("music") else None
        mux_final(concat_video, voice_wav, music, cfg.get("music_volume", 0.15), srt_path, out_path)

    print(f"done: {out_path}")


if __name__ == "__main__":
    sys.exit(main())

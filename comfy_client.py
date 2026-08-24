"""Talks to a locally running ComfyUI server to generate stills (Z-Image Turbo)
and short video clips (Wan2.2 T2V + lightx2v 4-step LoRA) on a local GPU.

Requires ComfyUI running first: `cd ComfyUI && venv/bin/python main.py`.
Model files are the ones already staged under ComfyUI/models/ on this
machine — see README.md for the exact filenames expected.
"""
from __future__ import annotations

import json
import random
import time
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8188"

WAN_NEGATIVE_DEFAULT = (
    "low quality, blurry, static, oversaturated, overexposed, watermark, "
    "subtitles, deformed hands, extra fingers, ugly, worst quality"
)


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def submit_and_wait(workflow: dict, comfy_url: str = DEFAULT_URL, timeout: float = 1200) -> dict:
    """Submit a workflow (API-format {"prompt": {...}}) and block until it finishes.

    Returns the node outputs dict from ComfyUI's /history endpoint.
    """
    try:
        result = _post_json(f"{comfy_url}/prompt", workflow)
    except OSError as e:
        raise RuntimeError(f"ComfyUI not reachable at {comfy_url} — start it first: "
                            f"cd ComfyUI && venv/bin/python main.py") from e
    if result.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected workflow: {result['node_errors']}")
    prompt_id = result["prompt_id"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = _get_json(f"{comfy_url}/history/{prompt_id}")
        if prompt_id in history:
            entry = history[prompt_id]
            if entry["status"]["status_str"] != "success":
                raise RuntimeError(f"ComfyUI generation failed: {entry['status']}")
            return entry["outputs"]
        time.sleep(2)
    raise TimeoutError(f"ComfyUI generation did not finish within {timeout}s")


def download_output(outputs: dict, out_path: Path, comfy_url: str = DEFAULT_URL) -> None:
    """Pull the first image/video file referenced in `outputs` down to `out_path`."""
    for node_output in outputs.values():
        for key in ("images", "gifs", "videos"):
            for item in node_output.get(key, []):
                qs = f"filename={item['filename']}&subfolder={item['subfolder']}&type={item['type']}"
                urllib.request.urlretrieve(f"{comfy_url}/view?{qs}", out_path)
                return
    raise RuntimeError(f"no output file found in ComfyUI outputs: {outputs}")


def upload_image(path: Path, comfy_url: str = DEFAULT_URL) -> str:
    """Upload a local image into ComfyUI's input/ dir; returns the name for LoadImage."""
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{comfy_url}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return f"{result['subfolder']}/{result['name']}" if result["subfolder"] else result["name"]


def build_zimage_workflow(prompt: str, width: int, height: int, seed: int) -> dict:
    """Text-to-image via Z-Image Turbo (~15s on a 16GB GPU)."""
    return {"prompt": {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "TextEncodeZImageOmni", "inputs": {"clip": ["2", 0], "prompt": prompt, "auto_resize_images": True}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": seed, "steps": 9, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "video-factory/gen"}},
    }}


def build_wan_t2v_workflow(prompt: str, width: int, height: int, length: int, seed: int,
                            negative_prompt: str = WAN_NEGATIVE_DEFAULT) -> dict:
    """Text-to-video via Wan2.2 14B (MoE high/low-noise) + lightx2v 4-step LoRA.

    `length` is frame count at 16fps (33 ~= 2s, 81 ~= 5s), must be 4n+1.
    ~35-90s for a short clip on a 16GB GPU, depending on length.
    """
    return {"prompt": {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3", 0], "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors", "strength_model": 1.0}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors", "strength_model": 1.0}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["5", 0], "shift": 5.0}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["6", 0], "shift": 5.0}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": negative_prompt}},
        "11": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": width, "height": height, "length": length, "batch_size": 1}},
        "12": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["7", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["11", 0],
            "add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 2,
            "return_with_leftover_noise": "enable",
        }},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["8", 0], "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["12", 0],
            "add_noise": "disable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "start_at_step": 2, "end_at_step": 4,
            "return_with_leftover_noise": "disable",
        }},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["2", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": 16.0}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": "video-factory/gen", "format": "mp4", "codec": "h264"}},
    }}


def build_wan_i2v_workflow(image_name: str, prompt: str, width: int, height: int, length: int, seed: int,
                            negative_prompt: str = WAN_NEGATIVE_DEFAULT) -> dict:
    """Image-to-video: animates an existing photo via Wan2.2 14B (MoE) + lightx2v 4-step LoRA.

    `image_name` is a filename already uploaded to ComfyUI's input/ dir (see upload_image()).
    Same timing/step budget as build_wan_t2v_workflow.
    """
    return {"prompt": {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["3", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["5", 0], "shift": 5.0}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["6", 0], "shift": 5.0}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": negative_prompt}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "12": {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["9", 0], "negative": ["10", 0], "vae": ["2", 0],
            "width": width, "height": height, "length": length, "batch_size": 1,
            "start_image": ["11", 0],
        }},
        "13": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["7", 0], "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["12", 2],
            "add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "start_at_step": 0, "end_at_step": 2,
            "return_with_leftover_noise": "enable",
        }},
        "14": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["8", 0], "positive": ["12", 0], "negative": ["12", 1], "latent_image": ["13", 0],
            "add_noise": "disable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "start_at_step": 2, "end_at_step": 4,
            "return_with_leftover_noise": "disable",
        }},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["2", 0]}},
        "16": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": 16.0}},
        "17": {"class_type": "SaveVideo", "inputs": {"video": ["16", 0], "filename_prefix": "video-factory/gen", "format": "mp4", "codec": "h264"}},
    }}


def generate_image(prompt: str, out_path: Path, width: int = 768, height: int = 1344,
                    seed: int | None = None, comfy_url: str = DEFAULT_URL) -> None:
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    workflow = build_zimage_workflow(prompt, width, height, seed)
    outputs = submit_and_wait(workflow, comfy_url)
    download_output(outputs, out_path, comfy_url)


def generate_video(prompt: str, out_path: Path, width: int = 480, height: int = 832, length: int = 33,
                    seed: int | None = None, comfy_url: str = DEFAULT_URL) -> None:
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    workflow = build_wan_t2v_workflow(prompt, width, height, length, seed)
    outputs = submit_and_wait(workflow, comfy_url, timeout=1200)
    download_output(outputs, out_path, comfy_url)


def generate_video_from_image(source_image: Path, prompt: str, out_path: Path,
                               width: int = 480, height: int = 832, length: int = 33,
                               seed: int | None = None, comfy_url: str = DEFAULT_URL) -> None:
    """Animate an existing photo instead of generating a scene from scratch."""
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    image_name = upload_image(source_image, comfy_url)
    workflow = build_wan_i2v_workflow(image_name, prompt, width, height, length, seed)
    outputs = submit_and_wait(workflow, comfy_url, timeout=1200)
    download_output(outputs, out_path, comfy_url)

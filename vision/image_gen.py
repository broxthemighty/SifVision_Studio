# image_generator.py
import os
import subprocess
import re
import sys
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import torch
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from app_core.config_manager import ConfigManager

# current file's directory
base_dir = Path(__file__).parent

# go up one level
parent_dir = base_dir.parent

# path to compiled stable-diffusion.cpp folder
cfg = ConfigManager().load()
base_dir = Path(cfg["image_model"]["folder"])
SD_CPP_PATH = base_dir / "stable-diffusion.cpp" / "build_cli" / "bin" / "Release"
SD_EXE = os.path.join(SD_CPP_PATH, "sd.exe")

def _resolve_model_path(model_name: str) -> str:
    """
    Accepts either:
      - absolute path to a .gguf/.bin
      - filename in the sd.exe folder (SD_CPP_PATH)
      - filename in repo_root/models/
    Rejects HF repo ids like 'runwayml/stable-diffusion-v1-5'.
    """
    if "/" in model_name and not os.path.isabs(model_name):
        raise FileNotFoundError(
            f"'{model_name}' looks like a Hugging Face repo id. "
            "sd.exe needs a local .gguf/.bin file name."
        )

    # absolute path
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name

    # next to sd.exe
    cand1 = os.path.join(SD_CPP_PATH, model_name)
    if os.path.exists(cand1):
        return cand1

    # repo_root/models
    repo_root = os.path.abspath(os.path.join(SD_CPP_PATH, "..", "..", ".."))
    cand2 = os.path.join(repo_root, "models", model_name)
    if os.path.exists(cand2):
        return cand2

    raise FileNotFoundError(
        f"Model file '{model_name}' not found in:\n"
        f"  - {SD_CPP_PATH}\n"
        f"  - {os.path.join(repo_root, 'models')}\n"
        f"Provide a valid .gguf/.bin file."
    )

def list_available_models(base_dir=None):
    """Return all model folders/files inside the configured directory."""
    cfg = ConfigManager().load()
    base_dir = base_dir or cfg["image_model"]["folder"]
    model_list = []
    for item in os.listdir(base_dir):
        full = os.path.join(base_dir, item)
        if os.path.isdir(full) or item.endswith((".gguf", ".safetensors", ".ckpt")):
            model_list.append(full)
    model_list.sort()
    return model_list

def generate_image(
    prompt,
    output_dir="generated_images",
    steps=20,
    guidance=7.5,
    size=(512, 512),
    model_name="stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
    device="cuda",
    progress_callback=None,
    negative_prompt=None,
    seed=42,
    init_image=None,
    strength=0.55,
    style=None
):
    """
    Use stable-diffusion.cpp to generate an image locally, streaming progress.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    model_path = _resolve_model_path(model_name)

    # GPU availability check 
    gpu_available = torch.cuda.is_available()
    device_used = "cuda" if gpu_available else "cpu"
    print(f"[INFO] Image generation using device: {device_used}")

    # let sd.exe know you want CUDA if supported
    if gpu_available:
        cmd_device_flag = ["--rng", "cuda"]
    else:
        cmd_device_flag = ["--rng", "std_default"]

    cmd = [
        SD_EXE,
        "--prompt", prompt,
        "--output", out_path,
        "--steps", str(steps),
        "--cfg-scale", str(guidance),
        "--width", str(size[0]),
        "--height", str(size[1]),
        "--model", model_path,
        "--seed", str(seed)
    ]

    if init_image and os.path.exists(init_image):
        # cmd += ["--init-img", init_image, "--strength", "0.6"]
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(model_path, torch_dtype=torch.float16).to("cuda")
        result = pipe(prompt=prompt, image=Image.open(init_image), strength=strength, guidance_scale=guidance)
    else:
        pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16).to("cuda")
        result = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance)
    if negative_prompt:
        cmd += ["--negative-prompt", negative_prompt]

    cmd += cmd_device_flag

    print("[INFO] Running:", " ".join(cmd))

    try:
        # stream stdout so we can parse progress
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            # mirror CLI logs to console
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

            # match both: "[5/22]" and "5/22 - 6.2s/it"
            m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
            if m and progress_callback:
                step, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    progress_callback(int(step * 100 / total))

        process.wait()

        if process.returncode == 0 and os.path.exists(out_path):
            print(f"[INFO] Image saved: {out_path}")
            if progress_callback:
                progress_callback(100)
            if "network" in prompt.lower() or "firewall" in prompt.lower() or "diagram" in prompt.lower():
                out_path = annotate_image(out_path)
            return out_path
        else:
            raise RuntimeError("Stable Diffusion failed to produce output.")

    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        fail_path = os.path.join(output_dir, "generation_failed.png")
        Image.new("RGB", (512, 512), (255, 0, 0)).save(fail_path)
        if progress_callback:
            progress_callback(0)
        return fail_path

def generate_image_diffusers(
    prompt,
    output_dir="generated_images",
    steps=20,
    guidance=7.5,
    size=(512, 512),
    model_name="UncannyValley_VPred.safetensors",
    device="cuda",
    negative_prompt=None,
    seed=42,
    init_image=None,
    strength=0.55,
    style=None,
    progress_callback=None
):
    """Generate an image using Diffusers-based models (.safetensors or folders)."""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    generator = torch.Generator(device).manual_seed(seed)

    # --- Determine which pipeline to use ---
    model_name_lower = model_name.lower()

    # SDXL folder or model name detection
    if os.path.isdir(model_name) or "xl" in model_name_lower:
        if init_image and os.path.exists(init_image):
            print(f"[INFO] Loading SDXL Img2Img pipeline from {model_name}")
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
            image = Image.open(init_image).convert("RGB").resize(size)
            result = pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                generator=generator
            )
        else:
            print(f"[INFO] Loading SDXL pipeline from {model_name}")
            pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
            result = pipe(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                negative_prompt=negative_prompt,
                generator=generator
            )

    # SD 1.5 / 2.1 single safetensors
    else:
        if init_image and os.path.exists(init_image):
            print(f"[INFO] Loading SD 1.x Img2Img pipeline from {model_name}")
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(model_name, torch_dtype=torch_dtype).to(device)
            image = Image.open(init_image).convert("RGB").resize(size)
            result = pipe(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                generator=generator
            )
        else:
            print(f"[INFO] Loading SD 1.x pipeline from {model_name}")
            pipe = StableDiffusionPipeline.from_single_file(model_name, torch_dtype=torch_dtype).to(device)
            result = pipe(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                negative_prompt=negative_prompt,
                generator=generator
            )

    # --- Save output ---
    result.images[0].save(out_path)
    if progress_callback:
        progress_callback(100)
    print(f"[INFO] Image saved: {out_path}")
    return out_path

def annotate_image(image_path, labels=None, output_path=None):
    """
    Adds readable text labels to the generated image for clarity.
    """
    if not os.path.exists(image_path):
        print(f"[WARN] Image not found for annotation: {image_path}")
        return image_path

    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        # use a simple built-in font for now
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # default fallback labels if none provided
        default_labels = [
            ("Firewall", (img.width * 0.45, img.height * 0.40)),
            ("Internet", (img.width * 0.45, img.height * 0.05)),
            ("Home Devices", (img.width * 0.10, img.height * 0.85)),
            ("Router", (img.width * 0.65, img.height * 0.65)),
        ]
        labels = labels or default_labels

        # add semi-transparent black background behind text
        for text, (x, y) in labels:
            text_w, text_h = draw.textsize(text, font=font)
            padding = 6
            rect = [x - padding, y - padding,
                    x + text_w + padding, y + text_h + padding]
            draw.rectangle(rect, fill=(0, 0, 0, 150))
            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

        # save annotated image
        output_path = output_path or image_path.replace(".png", "_labeled.png")
        img.save(output_path)
        print(f"[INFO] Annotated image saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"[ERROR] Failed to annotate image: {e}")
        return image_path
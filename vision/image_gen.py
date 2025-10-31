# image_generator.py
import os
import subprocess
import re
import sys
from PIL import Image, ImageDraw, ImageFont, ImageOps
from datetime import datetime
import torch
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline
)
from diffusers.utils import load_image
from app_core.config_manager import ConfigManager
import random
import cv2
import gc
import numpy as np

# current file's directory
base_dir = Path(__file__).parent

# go up one level
parent_dir = base_dir.parent

# path to compiled stable-diffusion.cpp folder
cfg = ConfigManager().load()
base_dir = ConfigManager().get_model_dir("image_model")
SD_CPP_PATH = base_dir / "stable-diffusion.cpp" / "build_cli" / "bin" / "Release"
SD_EXE = os.path.join(SD_CPP_PATH, "sd.exe")

PIPE_CACHE = {}

def _get_cached_pipeline(model_name, pipe_class, **kwargs):
    key = f"{pipe_class.__name__}:{model_name}"
    if key not in PIPE_CACHE:
        print(f"[CACHE] Loading new pipeline: {key}")
        PIPE_CACHE[key] = pipe_class.from_pretrained(model_name, **kwargs)
    else:
        print(f"[CACHE] Using cached pipeline: {key}")
    return PIPE_CACHE[key]

def clear_pipeline_cache():
    """Free all cached pipelines and release GPU memory."""
    global PIPE_CACHE
    print(f"[CACHE] Clearing {len(PIPE_CACHE)} pipelines...")
    for key, pipe in list(PIPE_CACHE.items()):
        try:
            del pipe
        except Exception:
            pass
    PIPE_CACHE.clear()
    torch.cuda.empty_cache()
    print("[CACHE] Cleared successfully.")

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
    if base_dir is None:
        base_dir = ConfigManager().get_model_dir("image_model")
    model_list = []
    for item in os.listdir(base_dir):
        full = os.path.join(base_dir, item)
        if os.path.isdir(full) or item.endswith((".gguf", ".safetensors", ".ckpt")):
            model_list.append(full)
    model_list.sort()
    return model_list

def _pipeline_kwargs():
    """
    Returns the correct dtype argument for Diffusers pipelines,
    handling both old (torch_dtype) and new (dtype) versions.
    """
    import inspect
    import torch
    test_pipelines = [
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline
    ]

    # Determine which argument name is valid for the installed version
    for pipe_class in test_pipelines:
        sig = inspect.signature(pipe_class.from_pretrained)
        if "dtype" in sig.parameters:
            return {"dtype": torch.float16 if torch.cuda.is_available() else torch.float32}

    # fallback for older versions
    return {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}

def make_canny_edges(init_image, low_thresh=100, high_thresh=200):
    """Generate a ControlNet-compatible canny edge map from an init image."""
    np_img = np.array(init_image.convert("RGB"))
    edges = cv2.Canny(np_img, low_thresh, high_thresh)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)    

def generate_image(
    prompt,
    output_dir="generated_images",
    steps=20,
    guidance=7.5,
    size=(512, 768),
    model_name="stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
    device="cuda",
    progress_callback=None,
    negative_prompt=None,
    seed=None,
    init_image=None,
    strength=0.35,
    style=None
):
    """Generate an image using stable-diffusion.cpp or diffusers if available."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    model_path = _resolve_model_path(model_name)

    # use random seed if not specified
    seed = seed or random.randint(0, 999999)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if init_image and os.path.exists(init_image):
        print(f"[INFO] Running image-to-image using avatar: {init_image}")
        pipe = _get_cached_pipeline(model_name, StableDiffusionXLImg2ImgPipeline, **_pipeline_kwargs()).to(device)
        image = Image.open(init_image).convert("RGB")
        image.thumbnail(size, Image.LANCZOS)
        image = ImageOps.pad(image, size, color=(0, 0, 0))
        result = pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            negative_prompt=negative_prompt,
            generator=torch.Generator(device).manual_seed(seed)
        )
    else:
        print(f"[INFO] Running text-to-image (no init).")
        pipe = StableDiffusionPipeline.from_single_file(model_path, **_pipeline_kwargs()).to(device)
        result = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            strength=strength,
            height=size[1],
            width=size[0],
            negative_prompt=negative_prompt,
            generator=torch.Generator(device).manual_seed(seed)
        )

    result.images[0].save(out_path)
    if progress_callback:
        progress_callback(100)
    print(f"[INFO] Image saved: {out_path} | Seed: {seed}")
    return out_path

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
    strength=0.35,
    style=None,
    progress_callback=None,
    use_controlnet=False
):
    """
    Generate an image using Diffusers-based models (.safetensors or folders).
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    generator = torch.Generator(device).manual_seed(seed)

    # --- Determine which pipeline to use ---
    model_name_lower = model_name.lower()
    if progress_callback:
        progress_callback(25)

        model_name_lower = model_name.lower()

        # Heuristic: treat folder/model names containing "xl" as SDXL-family
        is_sdxl = ("xl" in model_name_lower) or (os.path.isdir(model_name) and "xl" in os.path.basename(model_name).lower())

    if use_controlnet:
        # Pick a matching ControlNet for the detected family
        if is_sdxl:
            controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"
            pipe_class = StableDiffusionXLControlNetPipeline
        else:
            controlnet_id = "lllyasviel/control_v11p_sd15_canny"
            # (If you later support SD 2.1 explicitly, switch to a v2 ControlNet here)
            from diffusers import StableDiffusionControlNetPipeline as SD15ControlNetPipe
            pipe_class = SD15ControlNetPipe

        controlnet = _get_cached_pipeline(
            controlnet_id,
            ControlNetModel,
            **_pipeline_kwargs()
        )
        
        torch.cuda.empty_cache()
        gc.collect()
        
        pipe = _get_cached_pipeline(
            model_name,
            pipe_class,
            controlnet=controlnet,
            **_pipeline_kwargs()
        ).to(device)
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(gen_device).manual_seed(seed)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_vae_tiling()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # --- Sanity check: cross-attention dims must match
        try:
            cn_dim = getattr(controlnet.config, "cross_attention_dim", None)
            unet_dim = getattr(pipe.unet.config, "cross_attention_dim", None)
            if (cn_dim is not None) and (unet_dim is not None) and (cn_dim != unet_dim):
                raise ValueError(
                    f"[Model mismatch] ControlNet cross_attention_dim={cn_dim} "
                    f"does not match UNet cross_attention_dim={unet_dim}. "
                    f"Pick a ControlNet that matches the base model family."
                )
        except Exception as e:
            print(f"[WARN] Could not validate dimensions: {e}")

        # Build conditioning (need an image for canny)
        if not init_image or not os.path.exists(init_image):
            raise ValueError("ControlNet (canny) requires an init image. Enable 'Use Current Image' or provide init_image.")

        base_img = Image.open(init_image).convert("RGB").resize(size)
        edge_img = make_canny_edges(base_img)

        result = pipe(
            prompt=prompt,
            image=base_img,
            control_image=edge_img,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=steps,
            negative_prompt=negative_prompt,
            generator=generator
        )
        if progress_callback:
            progress_callback(50)
    else:
        # SDXL folder or model name detection
        if os.path.isdir(model_name) or "xl" in model_name_lower:
            if init_image and os.path.exists(init_image):
                print(f"[INFO] Loading SDXL Img2Img pipeline from {model_name}")
                torch.cuda.empty_cache()
                gc.collect()
                pipe = _get_cached_pipeline(model_name, StableDiffusionXLImg2ImgPipeline, **_pipeline_kwargs()).to(device)
                try:
                    pipe.enable_model_cpu_offload()
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_tiling()
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
                    # If using IP-Adapter or ControlNet face reference
                    if "ip-adapter" in model_name.lower():
                        from diffusers import IPAdapter
                        pipe.load_ip_adapter(model_name)
                except Exception as e:
                    print(f"[WARN] Optional consistency modules not loaded: {e}")
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
                if progress_callback:
                    progress_callback(50)
            else:
                print(f"[INFO] Loading SDXL pipeline from {model_name}")
                torch.cuda.empty_cache()
                gc.collect()
                pipe = _get_cached_pipeline(model_name, StableDiffusionXLPipeline, **_pipeline_kwargs()).to(device)
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                pipe.enable_vae_tiling()
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
                result = pipe(
                    prompt=prompt,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    generator=generator
                )
                if progress_callback:
                    progress_callback(50)

        # SD 1.5 / 2.1 single safetensors
        else:
            if init_image and os.path.exists(init_image):
                print(f"[INFO] Loading SD 1.x Img2Img pipeline from {model_name}")
                torch.cuda.empty_cache()
                gc.collect()
                pipe = _get_cached_pipeline(model_name, StableDiffusionImg2ImgPipeline, **_pipeline_kwargs()).to(device)
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                pipe.enable_vae_tiling()
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
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
                if progress_callback:
                    progress_callback(50)
            else:
                print(f"[INFO] Loading SD 1.x pipeline from {model_name}")
                torch.cuda.empty_cache()
                gc.collect()
                pipe = _get_cached_pipeline(model_name, StableDiffusionPipeline, **_pipeline_kwargs()).to(device)
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                pipe.enable_vae_tiling()
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
                result = pipe(
                    prompt=prompt,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    negative_prompt=negative_prompt,
                    strength=strength,
                    generator=generator
                )
                if progress_callback:
                    progress_callback(50)

    # --- Save output ---
    if progress_callback:
        progress_callback(100)
    print(f"[INFO] Image saved: {out_path}")
    result.images[0].save(out_path)
    print(f"[INFO] Image saved with seed {seed}")
    return out_path

def generate_multilayer_image(
        prompt,
        init_image,
        model_base,
        model_refiner=None,
        output_dir="layered_outputs",
        steps=25,
        guidance=7.5,
        size=(512, 512),
        device="cuda",
        strength=0.35,
        negative_prompt=None,
        progress_callback=None,
    ):
        """
        Multi-stage image refinement: base SDXL -> ControlNet -> optional refiner.
        """
        from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel

        os.makedirs(output_dir, exist_ok=True)
        generator = torch.Generator(device).manual_seed(random.randint(0, 999999))
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        controlnet = _get_cached_pipeline("lllyasviel/sd-controlnet-canny",ControlNetModel,torch_dtype=torch.float16)
        base = _get_cached_pipeline(model_base, StableDiffusionXLImg2ImgPipeline, **_pipeline_kwargs()).to(device)
        control = _get_cached_pipeline(model_base, StableDiffusionXLControlNetPipeline, controlnet=controlnet, **_pipeline_kwargs()).to(device)
        refiner = _get_cached_pipeline(model_refiner, StableDiffusionXLImg2ImgPipeline, **_pipeline_kwargs()).to(device)
        # Stage 1: Base Img2Img
        pipe = _get_cached_pipeline(model_base, StableDiffusionXLImg2ImgPipeline, **_pipeline_kwargs()).to(device)
        base.enable_vae_tiling()
        img = Image.open(init_image).convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        img = ImageOps.pad(img, size, color=(0, 0, 0))
        result1 = base(prompt=prompt, image=img, strength=0.3, guidance_scale=guidance, negative_prompt=negative_prompt,
                    num_inference_steps=steps, generator=generator)
        if progress_callback: progress_callback(33)
        stage1 = os.path.join(output_dir, "stage1_base.png")
        result1.images[0].save(stage1)

        # Stage 2: ControlNet
        control = _get_cached_pipeline(model_base, StableDiffusionXLControlNetPipeline, controlnet=controlnet, **_pipeline_kwargs()).to(device)
        edge_img = make_canny_edges(result1.images[0])
        result2 = control(
            prompt=prompt,
            image=result1.images[0],
            control_image=edge_img,
            strength=0.35, 
            guidance_scale=guidance, 
            num_inference_steps=steps, 
            negative_prompt=negative_prompt
        )
        if progress_callback: progress_callback(66)
        stage2 = os.path.join(output_dir, "stage2_control.png")
        result2.images[0].save(stage2)

        # Stage 3: Optional Refiner
        if model_refiner:
            refiner = _get_cached_pipeline(model_refiner, StableDiffusionXLImg2ImgPipeline, **_pipeline_kwargs()).to(device)
            result3 = refiner(prompt=prompt, image=result2.images[0], strength=0.2, guidance_scale=guidance,
                            negative_prompt=negative_prompt, num_inference_steps=steps)
            if progress_callback: progress_callback(100)
            final_path = os.path.join(output_dir, "stage3_refined.png")
            result3.images[0].save(final_path)
            return final_path

        return stage2

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
    
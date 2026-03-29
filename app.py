"""ZPix Gradio app."""
# Based on https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo

import logging
from argparse import ArgumentParser
from io import BytesIO
from json import dumps as dump_json, load as load_json
from os import environ
from pathlib import Path
from random import randint
from re import search
from shutil import rmtree
from zipfile import ZIP_DEFLATED, ZipFile

import gradio as gr
from diffusers import ZImagePipeline
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from torch import Generator, bfloat16, cuda, xpu

from source.py.disclaimer import TERMS_OF_USE, TermsOfUse
from source.py.lora_model import LoraModel

logging.basicConfig(format="%(levelname)s: %(message)s")

# Path to Triton cache directory
# shortened by good measure to avoid too long path errors on Windows
# even if this has been fixed recently.
environ["TRITON_CACHE_DIR"] = str(Path.home() / ".triton")

app_dir = Path(__file__).parent
"""App directory."""

# We store temp files created by Gradio in app directory
# to ease visualization of disk space used by this app.
environ["GRADIO_TEMP_DIR"] = str(app_dir / "temp" / "GradioApp")

translation: dict[str, str] = {}
"""Translation."""

metadata: dict[str, str] = {}
"""Metadata."""

pipe: ZImagePipeline | None = None
"""Pipeline."""

optimized: bool = False
"""Pipeline is optimized?"""

pipe_is_busy: bool = False
"""Pipeline is busy? e.g. loading a LoRA."""

negative_prompt_supported: bool | None = None
"""Pipeline supports negative prompts?"""


def load_translation(locale: str) -> None:
    """Load translation for a given locale, if available."""
    global translation

    translation_file = app_dir / "translations" / f"{locale}.json"
    if not translation_file.exists():
        logging.warning(f"Translation for {locale} not found.")
        return

    with open(translation_file, "r", encoding="utf-8") as file:
        translation = load_json(file)


def t(string: str) -> str:
    """Translate a string."""
    return translation.get(string, string)


def get_metadata(filename: str) -> str:
    """Get metadata."""
    if filename not in metadata:
        file = app_dir / "metadata" / filename
        metadata[filename] = file.read_text()

    return metadata[filename]


def get_example_prompts() -> list[str]:
    """Get example prompts."""
    prompts_file = app_dir / "examples" / "prompts.json"

    with open(prompts_file, "r", encoding="utf-8") as file:
        prompts = load_json(file)

    return [prompt["text"] for prompt in prompts]


def get_theme():
    """Get customized theme."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#f7f6ff",
            c100="#efedff",
            c200="#d8d2ff",
            c300="#c0b7ff",
            c400="#a192ff",
            c500="#624aff",
            c600="#5843e6",
            c700="#4534b3",
            c800="#312580",
            c900="#1d164d",
            c950="#0a071a",
        )
    )


def on_app_load():
    """On app load."""
    if not optimized:
        gr.Warning(
            t(
                "Image generation may be slow because diffusion pipeline is not optimized."
            )
            + "<br>"
            + t(
                "Try upgrading your graphics card drivers, then reboot your PC and restart"
            )
            + f" {get_metadata('NAME')}.",
            duration=None,  # Until user closes it.
        )


def get_aspects_and_resolutions() -> tuple:
    """Get aspect ratios and resolutions,
    possibly translated.

    Returns:
        Tuple of (
            resolutions by aspect,
            default resolution choices,
            aspect ratio choices,
            default aspect ratio
        )
    """
    default_aspect_ratio = "{} (16:9)".format(t("Landscape"))

    resolutions_by_aspect = {
        "{} (1:1)".format(t("Square")): [
            "1024x1024",
            "1280x1280",
            "1440x1440",
        ],
        "{} (16:9)".format(t("Landscape")): [
            "1280x720",
            "1920x1088",
        ],
        "{} (9:16)".format(t("Portrait")): [
            "720x1280",
            "1088x1920",
        ],
        "{} (4:3)".format(t("Landscape")): [
            "1152x864",
            "1440x1088",
            "1920x1440",
        ],
        "{} (3:4)".format(t("Portrait")): [
            "864x1152",
            "1088x1440",
            "1440x1920",
        ],
        "{} (16:10)".format(t("Landscape")): [
            "1280x800",
            "1440x912",
            "1920x1200",
        ],
        "{} (10:16)".format(t("Portrait")): [
            "800x1280",
            "912x1440",
            "1200x1920",
        ],
        "{} (21:9)".format(t("Ultra Wide")): [
            "1344x576",
        ],
    }

    default_resolution_choices = resolutions_by_aspect[default_aspect_ratio]
    aspect_ratio_choices = list(resolutions_by_aspect.keys())

    return (
        resolutions_by_aspect,
        default_resolution_choices,
        aspect_ratio_choices,
        default_aspect_ratio,
    )


def parse_resolution(resolution):
    """Parse resolution string into width and height.

    Args:
        resolution: Resolution string in format "WIDTHxHEIGHT" or "WIDTH×HEIGHT".

    Returns:
        Tuple of (width, height) as integers. Defaults to (1024, 1024) if parsing fails.
    """
    match = search(r"(\d+)\s*[×x]\s*(\d+)", resolution)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1024, 1024


def update_trigger_word(trigger_words: list, prompt: str) -> str:
    """Update the trigger word in the prompt.

    Args:
        trigger_words: List of [previous, current] trigger words.
        prompt: The current prompt as a string.

    Returns:
        Updated prompt.
    """
    previous_tw, current_tw = trigger_words

    # Removes the previous trigger word from start of the prompt.
    if previous_tw and prompt.startswith(previous_tw):
        prompt = prompt[len(previous_tw) :].lstrip()

    # Adds the current trigger word to start of the prompt.
    if current_tw:
        prompt = f"{current_tw} {prompt}"

    return prompt


def remove_trigger_word(trigger_words: list, prompt: str) -> tuple:
    """Remove the current trigger word from the prompt.

    Args:
        trigger_words: List of [previous, current] trigger words.
        prompt: The current prompt as a string.

    Returns:
        Tuple of (empty trigger words list, updated prompt).
    """
    _, current_tw = trigger_words

    # Removes the current trigger word from start of the prompt.
    if current_tw and prompt.startswith(current_tw):
        prompt = prompt[len(current_tw) :].lstrip()

    return [None, None], prompt


def load_model(model: str, backup_model: str):
    """Load and configure the Z-Image pipeline.

    Args:
        model: Hugging Face (HF) model name.
        backup_model: HF backup model name.
    """
    global pipe
    global optimized

    try:
        pipe = ZImagePipeline.from_pretrained(
            model,
            torch_dtype=bfloat16,
        )
    except Exception:
        logging.warning(f"Can't load {model}, falling back to {backup_model}.")
        pipe = ZImagePipeline.from_pretrained(
            backup_model,
            torch_dtype=bfloat16,
        )

    # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
    if triton_is_available and (cuda.is_available() or xpu.is_available()):
        pipe.transformer = apply_sdnq_options_to_model(
            pipe.transformer, use_quantized_matmul=True
        )
        pipe.text_encoder = apply_sdnq_options_to_model(
            pipe.text_encoder, use_quantized_matmul=True
        )
        try:
            pipe.transformer.set_attention_backend("_sage_qk_int8_pv_fp16_triton")
            optimized = True
        except Exception as e:
            logging.warning(f"SageAttention is not available: {e}")

    pipe.enable_model_cpu_offload()


def swap_lora(path: str) -> str | None:
    """Swap or load a new LoRA model.

    Args:
        path: Path to a LoRA file.

    Returns:
        Trigger word of LoRA model.
    """
    global pipe_is_busy

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    if not path.endswith(".safetensors"):
        raise gr.Error(
            t("LoRA file extension must be .safetensors"),
            duration=20,
        )

    lora = LoraModel(path)

    try:
        if lora.base_model() != "zimage":
            gr.Warning(
                f"{t('This LoRA seems incompatible with')} Z-Image.<br>"
                f"{t('It might not work.')}",
                duration=5,
            )
    except Exception as e:
        logging.warning(f"Can't check LoRA compatibility: {e}")

    bfloat16_lora = lora.to_bf16()
    gr.Info(t("Loading LoRA..."), duration=2)

    try:
        pipe_is_busy = True
        pipe.unload_lora_weights()
        pipe.load_lora_weights(bfloat16_lora, adapter_name="lora_1")
    finally:
        pipe_is_busy = False

    trigger_word = lora.trigger_word()

    return trigger_word


def set_lora_strength(strength: float):
    """Set LoRA strength."""
    adapters = pipe.get_list_adapters()

    if "transformer" not in adapters or "lora_1" not in adapters["transformer"]:
        raise gr.Error("No LoRA loaded.")

    pipe.set_adapters("lora_1", strength)


def unload_lora():
    """Unload LoRA model."""
    global pipe_is_busy

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    try:
        pipe_is_busy = True
        pipe.unload_lora_weights()
    finally:
        pipe_is_busy = False


def generate_image(
    pipe,
    prompt,
    negative_prompt="",
    resolution="1024x1024",
    seed=42,
    num_inference_steps=8,
):
    """Generate one image using the Z-Image pipeline.

    Args:
        pipe: The loaded ZImagePipeline instance.
        prompt: Text prompt describing the desired image.
        negative_prompt: Text prompt describing what to avoid.
        resolution: Output resolution as "WIDTHxHEIGHT" string.
        seed: Random seed for reproducible generation.
        num_inference_steps: Number of denoising steps.

    Returns:
        Generated PIL Image.
    """
    global negative_prompt_supported
    width, height = parse_resolution(resolution)
    generator_device = "cpu"

    if cuda.is_available():
        generator_device = "cuda"
    elif xpu.is_available():
        generator_device = "xpu"

    generator = Generator(device=generator_device).manual_seed(seed)

    generation_kwargs = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 0.0,
        "generator": generator,
        "num_images_per_prompt": 1,
    }

    if negative_prompt_supported is not False and negative_prompt:
        generation_kwargs["negative_prompt"] = negative_prompt

    try:
        image = pipe(
            **generation_kwargs,
        ).images[0]
        if "negative_prompt" in generation_kwargs:
            negative_prompt_supported = True
    except TypeError as error:
        if "negative_prompt" not in generation_kwargs:
            raise
        if "negative_prompt" not in str(error):
            raise

        negative_prompt_supported = False
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
            num_images_per_prompt=1,
        ).images[0]

    return image


def build_image_metadata(
    prompt: str,
    negative_prompt: str,
    seed: int,
    resolution: str,
    steps: int,
    batch_index: int,
    batch_size: int,
) -> dict[str, int | str]:
    """Build exportable metadata for one generated image."""
    return {
        "app": get_metadata("NAME"),
        "app_version": get_metadata("VERSION"),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "resolution": resolution,
        "denoising_steps": steps,
        "batch_index": batch_index,
        "batch_size": batch_size,
    }


def export_latest_batch(latest_batch: list | None) -> str:
    """Export the latest generated batch to a zip archive."""
    if not latest_batch:
        raise gr.Error(t("Generate a batch before downloading it."), duration=4)

    export_dir = app_dir / "temp" / "Exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    first_seed = latest_batch[0]["metadata"]["seed"]
    last_seed = latest_batch[-1]["metadata"]["seed"]
    zip_path = export_dir / f"zpix_batch_{first_seed}_{last_seed}.zip"

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "batch_metadata.json",
            dump_json(
                [entry["metadata"] for entry in latest_batch],
                ensure_ascii=False,
                indent=2,
            ),
        )

        for entry in latest_batch:
            png_info = PngInfo()
            png_info.add_text(
                "zpix_metadata",
                dump_json(entry["metadata"], ensure_ascii=False),
            )

            image_buffer = BytesIO()
            entry["image"].save(image_buffer, format="PNG", pnginfo=png_info)
            archive.writestr(
                f"image_{entry['metadata']['batch_index']:02d}_seed_{entry['metadata']['seed']}.png",
                image_buffer.getvalue(),
            )

    return str(zip_path)


def format_favorites_status(favorite_indices: list[int]) -> str:
    """Format favorite selection status for display."""
    if not favorite_indices:
        return t("No favorites selected")

    favorites_text = ", ".join(str(index) for index in sorted(favorite_indices))
    return f"{t('Favorites')}: {favorites_text}"


def toggle_favorite(
    latest_batch: list | None,
    favorite_indices: list[int] | None,
    selected_gallery_index: int | None,
):
    """Toggle favorite status for the currently selected image."""
    if not latest_batch:
        raise gr.Error(t("Generate a batch before selecting favorites."), duration=4)

    if selected_gallery_index is None:
        raise gr.Error(t("Select an image before marking it as favorite."), duration=4)

    selected_batch_index = None
    for entry in latest_batch:
        if entry["metadata"]["gallery_index"] == selected_gallery_index:
            selected_batch_index = entry["metadata"]["batch_index"]
            break

    if selected_batch_index is None:
        raise gr.Error(t("Select an image from the latest batch to mark it as favorite."), duration=4)

    favorite_indices = set(favorite_indices or [])

    if selected_batch_index in favorite_indices:
        favorite_indices.remove(selected_batch_index)
    else:
        favorite_indices.add(selected_batch_index)

    updated_favorites = sorted(favorite_indices)
    return updated_favorites, format_favorites_status(updated_favorites)


def export_favorites(
    latest_batch: list | None,
    favorite_indices: list[int] | None,
) -> str:
    """Export selected favorites from the latest batch to a zip archive."""
    if not latest_batch:
        raise gr.Error(t("Generate a batch before downloading favorites."), duration=4)

    favorite_indices = set(favorite_indices or [])
    if not favorite_indices:
        raise gr.Error(t("Select favorites before downloading them."), duration=4)

    favorite_entries = [
        entry
        for entry in latest_batch
        if entry["metadata"]["batch_index"] in favorite_indices
    ]
    if not favorite_entries:
        raise gr.Error(
            t("Select favorites from the latest batch before downloading them."),
            duration=4,
        )

    export_dir = app_dir / "temp" / "Exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    first_seed = favorite_entries[0]["metadata"]["seed"]
    last_seed = favorite_entries[-1]["metadata"]["seed"]
    zip_path = export_dir / f"zpix_favorites_{first_seed}_{last_seed}.zip"

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "favorites_metadata.json",
            dump_json(
                [entry["metadata"] for entry in favorite_entries],
                ensure_ascii=False,
                indent=2,
            ),
        )

        for entry in favorite_entries:
            png_info = PngInfo()
            png_info.add_text(
                "zpix_metadata",
                dump_json(entry["metadata"], ensure_ascii=False),
            )

            image_buffer = BytesIO()
            entry["image"].save(image_buffer, format="PNG", pnginfo=png_info)
            archive.writestr(
                f"favorite_{entry['metadata']['batch_index']:02d}_seed_{entry['metadata']['seed']}.png",
                image_buffer.getvalue(),
            )

    return str(zip_path)


def export_contact_sheet(latest_batch: list | None) -> str:
    """Export the latest generated batch as a contact sheet image."""
    if not latest_batch:
        raise gr.Error(t("Generate a batch before previewing it."), duration=4)

    export_dir = app_dir / "temp" / "Exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    first_seed = latest_batch[0]["metadata"]["seed"]
    last_seed = latest_batch[-1]["metadata"]["seed"]
    sheet_path = export_dir / f"zpix_contact_sheet_{first_seed}_{last_seed}.png"

    image_count = len(latest_batch)
    columns = min(4, image_count)
    rows = (image_count + columns - 1) // columns

    thumbnail_size = (320, 320)
    padding = 20
    label_height = 30

    sheet_width = columns * thumbnail_size[0] + (columns + 1) * padding
    sheet_height = rows * (thumbnail_size[1] + label_height) + (rows + 1) * padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(245, 245, 250))
    draw = ImageDraw.Draw(sheet)

    for index, entry in enumerate(latest_batch):
        column = index % columns
        row = index // columns
        x = padding + column * (thumbnail_size[0] + padding)
        y = padding + row * (thumbnail_size[1] + label_height + padding)

        thumbnail = entry["image"].copy()
        thumbnail.thumbnail(thumbnail_size)

        thumb_x = x + (thumbnail_size[0] - thumbnail.width) // 2
        thumb_y = y + (thumbnail_size[1] - thumbnail.height) // 2

        draw.rectangle(
            [x - 1, y - 1, x + thumbnail_size[0] + 1, y + thumbnail_size[1] + 1],
            outline=(210, 210, 220),
            width=1,
        )
        sheet.paste(thumbnail, (thumb_x, thumb_y))

        draw.text(
            (x, y + thumbnail_size[1] + 6),
            f"#{entry['metadata']['batch_index']}  seed {entry['metadata']['seed']}",
            fill=(45, 45, 55),
        )

    sheet.save(sheet_path, format="PNG")
    return str(sheet_path)


def generate(
    prompt,
    negative_prompt="",
    resolution="1024x1024",
    seed=42,
    steps=8,
    image_count=1,
    random_seed=True,
    gallery_images=None,
    latest_batch=None,
    progress=gr.Progress(track_tqdm=False),
):
    """Gradio callback to generate images and update the gallery.

    Args:
        prompt: Text prompt for image generation.
        negative_prompt: Text prompt describing what to avoid.
        resolution: Resolution string (e.g. "1024x1024").
        seed: Seed value for reproducibility.
        steps: Number of inference (denoising) steps.
        image_count: Number of images to generate in one batch.
        random_seed: If True, generate a random seed ignoring the seed parameter.
        gallery_images: Existing gallery images to append to.
        latest_batch: Latest generated batch state.

    Returns:
        Generator yielding updated UI state while the batch is generated.

    Raises:
        gr.Error: If the pipeline is not loaded or busy.
    """
    global pipe_is_busy

    if pipe is None:
        raise gr.Error("Pipeline not loaded.")

    if pipe_is_busy:
        raise gr.Error(
            t("Pipeline is busy. Please try again shortly."),
            duration=4,
        )

    if random_seed:
        new_seed = randint(1, 1000000)
    else:
        new_seed = int(seed) if seed != -1 else randint(1, 1000000)

    if gallery_images is None:
        gallery_images = []
    else:
        gallery_images = list(gallery_images)

    latest_batch = []
    pipe_is_busy = True

    try:
        for index in range(int(image_count)):
            current_seed = new_seed + index
            progress(
                (index, int(image_count)),
                desc=t("Generating image") + f" {index + 1}/{int(image_count)}",
            )

            generation_args = {
                "pipe": pipe,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "resolution": resolution,
                "seed": current_seed,
                "num_inference_steps": int(steps + 1),
            }
            try:
                image = generate_image(**generation_args)
            except UnicodeDecodeError:
                # A corrupted Triton cache can cause an UnicodeDecodeError.
                rmtree(Path.home() / ".triton", ignore_errors=True)
                gr.Warning(t("Cleared Triton cache as it may be corrupted."), duration=6)

                gr.Info(t("Regenerating same image..."), duration=8)
                image = generate_image(**generation_args)

            image_metadata = build_image_metadata(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=current_seed,
                resolution=resolution,
                steps=int(steps + 1),
                batch_index=index + 1,
                batch_size=int(image_count),
            )
            image_metadata["gallery_index"] = len(gallery_images)
            latest_batch.append({"image": image, "metadata": image_metadata})

            # Prompt is added as image caption.
            gallery_images.append((image, prompt))

            yield (
                gallery_images,
                len(gallery_images) - 1,
                len(gallery_images) - 1,
                f"{new_seed}-{new_seed + int(image_count) - 1}"
                if int(image_count) > 1
                else str(new_seed),
                int(new_seed),
                latest_batch,
                None,
                [],
                format_favorites_status([]),
                None,
                t("Generating image") + f" {index + 1}/{int(image_count)}",
            )
    finally:
        pipe_is_busy = False

    progress(1.0, desc=t("Batch ready"))
    yield (
        gallery_images,
        len(gallery_images) - 1,
        len(gallery_images) - 1,
        f"{new_seed}-{new_seed + int(image_count) - 1}"
        if int(image_count) > 1
        else str(new_seed),
        int(new_seed),
        latest_batch,
        None,
        [],
        format_favorites_status([]),
        None,
        t("Batch ready"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backup-model", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--locale", type=str, required=False, default="en-US")
    args, _ = parser.parse_known_args()

    load_model(args.model, args.backup_model)

    if args.locale != "en-US":
        load_translation(args.locale)

    tou = TermsOfUse(app_dir / ".tou_accepted")

    (
        resolutions_by_aspect,
        default_resolution_choices,
        aspect_ratio_choices,
        default_aspect_ratio,
    ) = get_aspects_and_resolutions()

    with gr.Blocks(
        analytics_enabled=False,
    ) as app:
        with gr.Row(elem_classes=[] if tou.accepted() else ["blurred"]) as ui_row:
            with gr.Column(min_width=48, elem_classes=["sidebar"]):
                gr.Button(
                    "",
                    icon=app_dir / "assets" / "noto-emoji" / "emoji_u26a1.svg",
                    link=get_metadata("HOME_URL"),
                    link_target="_blank",  # Opens default browser. See app.js
                    elem_id="home-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("home-btn")
                        btn.title = "{t("Visit project homepage to check updates")}"
                    """
                )
                gr.Button(
                    "",
                    icon=app_dir / "assets" / "lora_grad.svg",
                    elem_id="swap-lora-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("swap-lora-btn")
                        btn.title = "{t("Load a LoRA file to apply a new style")}"
                    """
                )
                lora_path = gr.Textbox(
                    visible="hidden",  # See "portal" in app.js
                    elem_id="lora-path",
                )
                gr.Button(
                    "",
                    icon=app_dir / "assets" / "kerismaker" / "tech_13631866.png",
                    link=f"{get_metadata('HOME_URL')}/blob/main/docs/FAQ.md",
                    link_target="_blank",
                    elem_id="faq-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("faq-btn")
                        btn.title = "{t("Access the FAQ of this application")}"
                    """
                )
                gr.Button(
                    "",
                    icon=app_dir / "assets" / "kofi_symbol.svg",
                    link=get_metadata("DONATE_URL"),
                    link_target="_blank",
                    elem_id="donate-btn",
                )
                gr.HTML(
                    js_on_load=f"""
                        let btn = document.getElementById("donate-btn")
                        btn.title = "{t("Keep project developer awake with a coffee")} 😄"
                    """
                )
                gr.Button(
                    t("Quit App"),
                    variant="stop",
                    elem_id="quit-app-btn",
                )

            with gr.Column():
                trigger_words = gr.State(value=[None, None])
                """Trigger words (previous, current)."""

                prompt = gr.Textbox(
                    label=t("Prompt"),
                    lines=3,
                    placeholder=t("Enter your prompt here..."),
                    html_attributes=gr.InputHTMLAttributes(spellcheck=False),
                )

                negative_prompt = gr.Textbox(
                    label=t("Negative Prompt"),
                    lines=2,
                    placeholder=t("Describe what to avoid..."),
                    html_attributes=gr.InputHTMLAttributes(spellcheck=False),
                )

                with gr.Row(visible=False) as lora_row:
                    with gr.Column():
                        lora_strength = gr.Slider(
                            label=t("LoRA Strength"),
                            minimum=-2.5,
                            maximum=2.5,
                            step=0.1,
                            value=1.0,
                        )
                        lora_strength.change(
                            set_lora_strength,
                            inputs=lora_strength,
                        )
                    with gr.Column():
                        unload_lora_btn = gr.Button(t("Unload LoRA"))

                        # On "Unload LoRA" button click:
                        # - unload LoRA model,
                        # - remove trigger word from prompt,
                        # - empty trigger words history,
                        # - make LoRA row invisible.
                        unload_lora_btn.click(
                            unload_lora,
                        ).then(
                            remove_trigger_word,
                            inputs=[trigger_words, prompt],
                            outputs=[trigger_words, prompt],
                        ).then(
                            lambda: gr.update(visible=False),
                            outputs=lora_row,
                        )

                # When a LoRA path is selected:
                # - discard appended timestamp,
                # - unload any LoRA model,
                # - load selected LoRA model,
                # - shift trigger words history,
                # - update trigger word in prompt,
                # - make LoRA row visible.
                lora_path.change(
                    lambda p, tw: [tw[1], swap_lora(p)],
                    inputs=[lora_path, trigger_words],
                    outputs=trigger_words,
                    js="(p, tw) => [p.split('|')[0], tw]",
                ).then(
                    update_trigger_word,
                    inputs=[trigger_words, prompt],
                    outputs=prompt,
                ).then(
                    lambda: gr.update(visible=True),
                    outputs=lora_row,
                )

                with gr.Row():
                    with gr.Column():
                        aspect_ratio = gr.Dropdown(
                            value=default_aspect_ratio,
                            choices=aspect_ratio_choices,
                            label=t("Aspect Ratio"),
                        )
                    with gr.Column():
                        resolution = gr.Dropdown(
                            value=default_resolution_choices[0],
                            choices=default_resolution_choices,
                            label=t("Resolution"),
                        )

                with gr.Row():
                    with gr.Column():
                        advanced_checkbox = gr.Checkbox(
                            label=t("Advanced Settings"), value=False
                        )
                    with gr.Column():
                        generate_btn = gr.Button(t("Generate Image"), variant="primary")

                with gr.Row(visible=False) as seed_random_row:
                    seed = gr.Number(label=t("Seed"), value=42, precision=0)
                    random_seed = gr.Checkbox(label=t("Random"), value=True)

                with gr.Row(visible=False) as steps_row:
                    steps = gr.Slider(
                        label=t("Denoising Steps"),
                        minimum=4,
                        maximum=9,
                        value=8,
                        step=1,
                    )

                with gr.Row(visible=False) as image_count_row:
                    image_count = gr.Slider(
                        label=t("Image Count"),
                        minimum=1,
                        maximum=20,
                        value=1,
                        step=1,
                    )

                def advanced_rows_visibility(v):
                    return (
                        gr.update(visible=v),
                        gr.update(visible=v),
                        gr.update(visible=v),
                    )

                advanced_checkbox.change(
                    advanced_rows_visibility,
                    inputs=advanced_checkbox,
                    outputs=[seed_random_row, steps_row, image_count_row],
                )

                gr.Examples(
                    examples=get_example_prompts(),
                    inputs=prompt,
                    label=t("Example Prompts"),
                )

            with gr.Column():
                gallery_images = gr.Gallery(
                    label=t("Generated Images"),
                    columns=2,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    format="png",
                    buttons=["download", "fullscreen"],
                    interactive=True,
                )
                last_image_index = gr.State(value=None)
                selected_gallery_index = gr.State(value=None)
                latest_batch = gr.State(value=None)
                favorite_indices = gr.State(value=[])
                used_seed = gr.Textbox(
                    label=t("Seed Used"), interactive=False, visible=True
                )
                generation_status = gr.Textbox(
                    label=t("Batch Status"),
                    value=t("Ready"),
                    interactive=False,
                )
                download_batch_btn = gr.Button(t("Download Latest Batch ZIP"))
                preview_sheet_btn = gr.Button(t("Preview Contact Sheet"))
                toggle_favorite_btn = gr.Button(t("Toggle Favorite"))
                download_favorites_btn = gr.Button(t("Download Favorites ZIP"))
                latest_batch_zip = gr.File(
                    label=t("Latest Batch ZIP"),
                    interactive=False,
                )
                favorites_zip = gr.File(
                    label=t("Favorites ZIP"),
                    interactive=False,
                )
                contact_sheet = gr.Image(
                    label=t("Latest Contact Sheet"),
                    interactive=False,
                    type="filepath",
                )
                favorites_status = gr.Textbox(
                    label=t("Favorites"),
                    value=t("No favorites selected"),
                    interactive=False,
                )

        with gr.Row():
            # Add source model link to footer, after Gradio credit.
            gr.HTML(
                js_on_load=f"""
                    document.querySelector("footer").insertAdjacentHTML(
                        "beforeend",
                        `<a
                            href="https://huggingface.co/{args.model}"
                            target="_blank"
                        >
                            {t("Source model")} 🤗
                        </a>`
                    )
                """
            )

        with gr.Row(
            visible=not tou.accepted(),
            elem_id="tou-row",
        ) as tou_row:
            with gr.Column(elem_id="tou-card"):
                gr.Markdown(f"### {t('Terms of Use')}")
                gr.Markdown(t(TERMS_OF_USE))
                agree_tou_btn = gr.Button(t("I agree"), variant="primary")

                agree_tou_btn.click(tou.accept).then(
                    lambda: (gr.update(visible=False), gr.update(elem_classes=[])),
                    outputs=[tou_row, ui_row],
                )

        def update_resolution_choices(_aspect_ratio):
            resolution_choices = resolutions_by_aspect.get(
                _aspect_ratio, default_resolution_choices
            )
            return gr.update(value=resolution_choices[0], choices=resolution_choices)

        aspect_ratio.change(
            update_resolution_choices, inputs=aspect_ratio, outputs=resolution
        )
        generate_btn.click(
            generate,
            inputs=[
                prompt,
                negative_prompt,
                resolution,
                seed,
                steps,
                image_count,
                random_seed,
                gallery_images,
                latest_batch,
            ],
            outputs=[
                gallery_images,
                last_image_index,
                selected_gallery_index,
                used_seed,
                seed,
                latest_batch,
                latest_batch_zip,
                favorite_indices,
                favorites_status,
                favorites_zip,
                generation_status,
            ],
        ).then(
            # Select generated image in gallery:
            lambda imgs, idx: gr.Gallery(value=imgs, selected_index=idx),
            inputs=[gallery_images, last_image_index],
            outputs=gallery_images,
        )
        download_batch_btn.click(
            export_latest_batch,
            inputs=[latest_batch],
            outputs=[latest_batch_zip],
        )
        preview_sheet_btn.click(
            export_contact_sheet,
            inputs=[latest_batch],
            outputs=[contact_sheet],
        )
        gallery_images.select(
            lambda evt: evt.index,
            outputs=[selected_gallery_index],
        )
        toggle_favorite_btn.click(
            toggle_favorite,
            inputs=[latest_batch, favorite_indices, selected_gallery_index],
            outputs=[favorite_indices, favorites_status],
        )
        download_favorites_btn.click(
            export_favorites,
            inputs=[latest_batch, favorite_indices],
            outputs=[favorites_zip],
        )

        app.load(on_app_load)

    app.launch(
        server_port=args.port,
        footer_links=["gradio"],  # Credit
        theme=get_theme(),
        css_paths=[app_dir / "source" / "app.css"],
        js=(app_dir / "source" / "app.js").read_text(),
    )

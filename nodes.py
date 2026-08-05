"""Standalone text and vision-language generation with ComfyUI Qwen CLIPs."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's instructions accurately. "
    "Return only the requested answer unless an explanation is requested."
)

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def _tokenizer_identity(clip: Any) -> str:
    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer_type = type(tokenizer)
    return f"{tokenizer_type.__module__}.{tokenizer_type.__name__}"


def _is_qwen_vl_tokenizer(clip: Any) -> bool:
    identity = _tokenizer_identity(clip).casefold()
    return any(
        marker in identity for marker in ("qwen3vl", "qwen3_vl", "qwen35", "qwen3_5")
    )


def validate_qwen_clip(clip: Any) -> tuple[str, bool]:
    """Validate the public generation interface and reject known wrong CLIPs."""

    if clip is None:
        raise RuntimeError(
            "Qwen VL Generate Text needs a complete generation-capable CLIP input."
        )

    required = ("tokenize", "generate", "decode")
    missing = [name for name in required if not callable(getattr(clip, name, None))]
    if missing:
        raise RuntimeError(
            "The connected CLIP cannot generate text because it is missing callable "
            + ", ".join(missing)
            + ". Connect a complete instruction-tuned Qwen3-VL or Qwen3.5-VL CLIP."
        )

    identity = _tokenizer_identity(clip)
    recognized = _is_qwen_vl_tokenizer(clip)
    if identity.casefold().startswith("comfy.") and not recognized:
        raise RuntimeError(
            "The connected ComfyUI CLIP uses "
            f"{identity}, not a supported Qwen3-VL/Qwen3.5-VL tokenizer. "
            "A diffusion conditioning encoder or the truncated MiniMax H3 encoder is "
            "not a standalone language model."
        )
    return identity, recognized


def _image_count(image: Any) -> int:
    if image is None:
        return 0
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    try:
        return len(image)
    except TypeError as exc:
        raise ValueError("image must be a ComfyUI IMAGE batch.") from exc


def _evenly_spaced_indices(count: int, limit: int) -> list[int]:
    if count <= 0 or limit <= 0:
        return []
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return [round(index * (count - 1) / (limit - 1)) for index in range(limit)]


def select_images(image: Any, batch_mode: str, max_images: int) -> list[Any]:
    """Turn an IMAGE batch into ordered one-image tensors for Qwen's tokenizer."""

    count = _image_count(image)
    if count == 0:
        return []
    limit = max(1, int(max_images))
    if batch_mode == "first image only":
        indices = [0]
    elif batch_mode == "all images from start":
        indices = list(range(min(count, limit)))
    elif batch_mode == "evenly sample batch":
        indices = _evenly_spaced_indices(count, limit)
    else:
        raise ValueError(f"Unknown image_batch_mode: {batch_mode!r}.")
    return [image[index : index + 1] for index in indices]


def format_qwen_chat(
    system_prompt: str,
    user_prompt: str,
    visual_count: int,
    thinking: bool,
) -> str:
    """Build a complete Qwen chat with one visual token per attached image."""

    user = (user_prompt or "").strip()
    if not user:
        raise ValueError("prompt cannot be empty.")

    parts: list[str] = []
    system = (system_prompt or "").strip()
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    visual_prefix = VISION_BLOCK * max(0, int(visual_count))
    if visual_prefix:
        visual_prefix += "\n"
    parts.append(
        f"<|im_start|>user\n{visual_prefix}{user}<|im_end|>\n<|im_start|>assistant\n"
    )
    if not thinking:
        # Qwen3 convention for disabling reasoning. Because this node supplies a
        # complete chat template, the tokenizer will not append it for us.
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


def clean_generated_text(text: str) -> str:
    """Remove Qwen chat residue while preserving the requested answer format."""

    value = (text or "").strip()
    value = re.sub(r"^<\|im_start\|>assistant\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^assistant\s*:?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^<think>.*?</think>\s*", "", value, flags=re.DOTALL)
    value = re.sub(
        r"\s*(?:<\|im_end\|>|<\|endoftext\|>)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


class H3QwenVLGenerateText:
    """Generate general-purpose text with a complete Qwen vision-language CLIP."""

    CATEGORY = "H3 Qwen VL/Text"
    FUNCTION = "generate_text"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "generated_text",
        "raw_output",
        "chat_prompt",
        "system_prompt",
        "generation_report",
    )
    OUTPUT_TOOLTIPS = (
        "Clean answer text. Completed <think> reasoning and outer Qwen chat markers are removed when clean_output is enabled.",
        "Exact text decoded from the model, retained for debugging reasoning or tokenizer behavior.",
        "Exact textual Qwen chat sent to the tokenizer. Image pixels are supplied separately through visual tokens.",
        "The system prompt used for this generation, echoed so it can be saved or edited downstream.",
        "Detected tokenizer, image count, sampling mode, and model-residency behavior.",
    )
    DESCRIPTION = (
        "Standalone Qwen3-VL/Qwen3.5-VL language-model generation. Connect a complete "
        "generation-capable ComfyUI CLIP and optionally an IMAGE batch. This node has no "
        "MiniMax H3 dependency and never force-unloads the connected model."
    )

    @classmethod
    def INPUT_TYPES(cls):
        multiline = {"multiline": True, "dynamicPrompts": False}
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Connect a complete instruction-tuned Qwen3-VL or Qwen3.5-VL "
                            "model from a ComfyUI CLIP loader. Truncated diffusion text "
                            "encoders cannot generate text."
                        )
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        **multiline,
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "tooltip": (
                            "High-level behavior and output rules for Qwen. This is a real "
                            "system turn. Leave it blank to omit the system turn entirely."
                        ),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        **multiline,
                        "default": "Describe the image clearly and concisely.",
                        "tooltip": (
                            "The user request. If images are connected, state what Qwen "
                            "should inspect, compare, extract, or answer about them."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                        "tooltip": (
                            "Maximum number of tokens the model may generate. This does not "
                            "shorten the input prompt or visual tokens."
                        ),
                    },
                ),
                "sampling": (
                    ["deterministic", "sample"],
                    {
                        "default": "sample",
                        "tooltip": (
                            "Deterministic chooses the most likely next token. Sample uses "
                            "temperature and probability filters for varied answers."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.01,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Sampling creativity; used only when sampling is sample.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Keep only the K most likely tokens; 0 disables this filter.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus-sampling threshold; 1.0 disables this filter.",
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Relative minimum token probability; 0 disables this filter.",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.05,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Discourages repeated tokens; values near 1.0 are safest.",
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Additional penalty for tokens already present in the answer.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed used by sampled generation.",
                    },
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable Qwen reasoning mode. It is slower and consumes output "
                            "tokens. raw_output keeps it; generated_text removes a completed "
                            "<think> block when clean_output is enabled."
                        ),
                    },
                ),
                "image_batch_mode": (
                    [
                        "first image only",
                        "all images from start",
                        "evenly sample batch",
                    ],
                    {
                        "default": "all images from start",
                        "tooltip": (
                            "How to interpret a connected IMAGE batch. Each selected batch "
                            "item becomes a separate ordered Qwen image. Use evenly sample "
                            "batch for a short sequence of video frames."
                        ),
                    },
                ),
                "max_images": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Safety cap for images taken from the batch. More images increase "
                            "vision encoding time, prompt length, and VRAM use substantially."
                        ),
                    },
                ),
                "clean_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Remove completed reasoning and outer chat markers from "
                            "generated_text. raw_output is always unchanged."
                        ),
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional image or IMAGE batch for Qwen to analyze. Images remain "
                            "in their original order; max_images limits the selected items."
                        )
                    },
                )
            },
        }

    def generate_text(
        self,
        clip,
        system_prompt: str,
        prompt: str,
        max_new_tokens: int,
        sampling: str,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        seed: int,
        thinking: bool,
        image_batch_mode: str,
        max_images: int,
        clean_output: bool,
        image=None,
    ):
        identity, recognized = validate_qwen_clip(clip)
        images = select_images(image, image_batch_mode, max_images)
        chat_prompt = format_qwen_chat(
            system_prompt,
            prompt,
            visual_count=len(images),
            thinking=thinking,
        )
        tokenize_options = {
            "skip_template": True,
            "min_length": 1,
            "thinking": thinking,
        }
        if images:
            # Current ComfyUI Qwen3-VL and Qwen3.5 tokenizers accept `images`.
            # `image` is also supplied for compatible third-party wrappers that
            # only implement the native single/batched-image argument.
            tokenize_options["images"] = images
            tokenize_options["image"] = image

        tokens = clip.tokenize(chat_prompt, **tokenize_options)
        generation_options = {
            "do_sample": sampling == "sample",
            "max_length": int(max_new_tokens),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "repetition_penalty": float(repetition_penalty),
            "presence_penalty": float(presence_penalty),
            "seed": int(seed),
        }
        generated_ids = clip.generate(tokens, **generation_options)
        raw_output = str(clip.decode(generated_ids) or "")
        generated_text = (
            clean_generated_text(raw_output) if clean_output else raw_output
        )
        if not generated_text.strip():
            raise RuntimeError(
                "Qwen decoded an empty answer. Try more output tokens, deterministic "
                "sampling, or verify that the connected CLIP includes its final norm and LM head."
            )

        recognition = (
            "recognized Qwen VL tokenizer" if recognized else "custom tokenizer"
        )
        report = (
            f"Generation completed with {identity} ({recognition}); "
            f"{len(images)} image(s); sampling={sampling}; thinking={str(thinking).lower()}. "
            "The connected CLIP was not explicitly unloaded; ComfyUI manages model residency."
        )
        return (
            generated_text,
            raw_output,
            chat_prompt,
            (system_prompt or "").strip(),
            report,
        )


NODE_CLASS_MAPPINGS = {"H3QwenVLGenerateText": H3QwenVLGenerateText}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3QwenVLGenerateText": "H3 Qwen VL Generate Text (Standalone)"
}

"""Standalone text generation with MiniMax H3's base CLIP and Qwen tail."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's instructions accurately. "
    "Return only the requested answer unless an explanation is requested."
)

# Match the original H3 Guide pack so either dedicated tail-loader output can
# connect when both packs are installed. The descriptor shape is identical too.
TAIL_TYPE = "MINIMAX_H3_GENERATION_TAIL"
NO_TAIL_FOUND = "[no H3 generation_tail_50_63 file found]"


def _tokenizer_identity(clip: Any) -> str:
    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer_type = type(tokenizer)
    return f"{tokenizer_type.__module__}.{tokenizer_type.__name__}"


def _is_minimax_h3_tokenizer(clip: Any) -> bool:
    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer_type = type(tokenizer)
    return tokenizer_type.__name__ == "MiniMaxH3Tokenizer" or (
        tokenizer_type.__module__.endswith(".minimax")
    )


def validate_h3_base_clip(clip: Any) -> str:
    """Require the standard 50-layer MiniMax H3 conditioning CLIP."""

    if clip is None:
        raise RuntimeError(
            "H3 Qwen VL Generate Text needs the 50-layer MiniMax H3 CLIP input."
        )

    required = ("tokenize", "decode")
    missing = [name for name in required if not callable(getattr(clip, name, None))]
    if missing:
        raise RuntimeError(
            "The connected base CLIP is missing callable "
            + ", ".join(missing)
            + ". Connect MiniMax H3's normal Qwen3-VL-32B text encoder from "
            "ComfyUI's Load CLIP node."
        )

    identity = _tokenizer_identity(clip)
    if not _is_minimax_h3_tokenizer(clip):
        raise RuntimeError(
            f"The connected CLIP uses {identity}. This generator specifically needs "
            "MiniMax H3's 50-layer Qwen3-VL-32B conditioning CLIP; its dedicated "
            "tail input supplies layers 50-63, final norm, and the LM head."
        )
    return identity


def _tail_choices() -> list[str]:
    """List only the tail artifacts compatible with the H3 50-layer base."""

    try:
        import folder_paths
    except ImportError:
        return [NO_TAIL_FOUND]
    names = [
        name
        for name in folder_paths.get_filename_list("text_encoders")
        if "generation_tail_50_63" in name.casefold()
    ]
    return sorted(names, key=str.casefold) or [NO_TAIL_FOUND]


def _resolve_tail_name(tail_clip: Any) -> str:
    if isinstance(tail_clip, str):
        name = tail_clip
    elif isinstance(tail_clip, dict):
        name = tail_clip.get("tail_name")
    else:
        name = None
    if not isinstance(name, str) or not name or name == NO_TAIL_FOUND:
        raise ValueError(
            "tail_clip must come from H3 Qwen VL Generation Tail Loader. Place a "
            "compatible generation_tail_50_63 file in models/text_encoders first."
        )
    return name


def _generate_with_tail(clip, tail_name: str, tokens, generation_options: dict):
    """Lazy import keeps node discovery and isolated tests lightweight."""

    try:
        from .hybrid_tail import generate_with_tail
    except ImportError:
        from hybrid_tail import generate_with_tail
    return generate_with_tail(clip, tail_name, tokens, generation_options)


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
    thinking: bool,
) -> str:
    """Build Qwen chat text; the H3 tokenizer prepends attached image blocks."""

    user = (user_prompt or "").strip()
    if not user:
        raise ValueError("prompt cannot be empty.")

    parts: list[str] = []
    system = (system_prompt or "").strip()
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
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


class H3QwenVLGenerationTailLoader:
    """Select the exact H3 generation tail; the generator loads it lazily."""

    CATEGORY = "H3 Qwen VL/Loaders"
    FUNCTION = "select_tail"
    RETURN_TYPES = (TAIL_TYPE,)
    RETURN_NAMES = ("tail_clip",)
    OUTPUT_TOOLTIPS = (
        "Connect to H3 Qwen VL Generate Text.tail_clip. The selected layers "
        "50-63, final norm, and LM head are loaded only during generation.",
    )
    DESCRIPTION = (
        "Dedicated companion loader for MiniMax H3's Qwen3-VL-32B generation "
        "tail. The tail is not a complete CLIP: it reuses embeddings, vision, "
        "tokenizer, and layers 0-49 from the normal H3 CLIP input."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tail_name": (
                    _tail_choices(),
                    {
                        "tooltip": (
                            "Select the H3 Qwen3-VL-32B generation tail containing "
                            "exactly layers 50-63, model.norm, and model.lm_head. "
                            "Tail files are discovered in models/text_encoders."
                        )
                    },
                )
            }
        }

    def select_tail(self, tail_name: str):
        if tail_name == NO_TAIL_FOUND:
            raise FileNotFoundError(
                "No compatible H3 generation tail was found. Put the published "
                "generation_tail_50_63 safetensors file in models/text_encoders, "
                "then refresh ComfyUI."
            )
        return ({"tail_name": tail_name},)


class H3QwenVLGenerateText:
    """Generate text through H3 base layers 0-49 and a connected tail 50-63."""

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
        "Exact textual Qwen chat sent to the H3 tokenizer. The tokenizer prepends attached picture blocks separately.",
        "The system prompt used for this generation, echoed so it can be saved or edited downstream.",
        "Detected H3 base, selected tail, image count, sampling mode, and cleanup behavior.",
    )
    DESCRIPTION = (
        "Standalone language generation using MiniMax H3's normal 50-layer "
        "Qwen3-VL-32B conditioning CLIP plus the dedicated generation tail. Connect "
        "both loader outputs. The base CLIP is preserved; the temporary tail is "
        "unloaded after each generation."
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
                            "Connect MiniMax H3's normal 50-layer Qwen3-VL-32B text "
                            "encoder from ComfyUI's Load CLIP node. This supplies the "
                            "tokenizer, embeddings, vision tower, and layers 0-49."
                        )
                    },
                ),
                "tail_clip": (
                    TAIL_TYPE,
                    {
                        "tooltip": (
                            "Connect H3 Qwen VL Generation Tail Loader. It supplies "
                            "layers 50-63, final normalization, and the language-model "
                            "head only while text is generated."
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
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Maximum number of tokens the model may generate. This does not "
                            "shorten the input prompt or visual tokens. Higher values reserve "
                            "larger KV caches across all 64 language layers."
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
                        "default": 4,
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
        tail_clip,
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
        identity = validate_h3_base_clip(clip)
        tail_name = _resolve_tail_name(tail_clip)
        images = select_images(image, image_batch_mode, max_images)
        chat_prompt = format_qwen_chat(
            system_prompt,
            prompt,
            thinking=thinking,
        )
        tokenize_options = {
            "skip_template": True,
            "min_length": 1,
            "thinking": thinking,
        }
        if images:
            # MiniMaxH3Tokenizer prepends numbered picture blocks and replaces
            # each visual entry with the real image tensor. Do not also place
            # generic Qwen vision-pad text inside the chat or images duplicate.
            tokenize_options["images"] = images

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
        generated_ids = _generate_with_tail(clip, tail_name, tokens, generation_options)
        raw_output = str(clip.decode(generated_ids) or "")
        generated_text = (
            clean_generated_text(raw_output) if clean_output else raw_output
        )
        if not generated_text.strip():
            raise RuntimeError(
                "Qwen decoded an empty answer. Try more output tokens, deterministic "
                "sampling, or verify that the connected tail includes its final norm "
                "and LM head."
            )

        report = (
            f"Generation completed with H3 base {identity} and tail {tail_name}; "
            f"{len(images)} image(s); sampling={sampling}; thinking={str(thinking).lower()}. "
            "The temporary tail was unloaded after generation. The connected base CLIP "
            "was left under ComfyUI model-residency management."
        )
        return (
            generated_text,
            raw_output,
            chat_prompt,
            (system_prompt or "").strip(),
            report,
        )


NODE_CLASS_MAPPINGS = {
    "H3QwenVLGenerationTailLoader": H3QwenVLGenerationTailLoader,
    "H3QwenVLGenerateText": H3QwenVLGenerateText,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3QwenVLGenerationTailLoader": "H3 Qwen VL Generation Tail Loader",
    "H3QwenVLGenerateText": "H3 Qwen VL Generate Text (Standalone)",
}

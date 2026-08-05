# ComfyUI H3 Qwen3-VL Text Generation

A small standalone ComfyUI node pack for using a complete Qwen3-VL or Qwen3.5-VL text encoder as a general-purpose local language and vision-language model. It has no MiniMax H3 dependency.

## Node

`H3 Qwen VL Generate Text (Standalone)` accepts:

- a complete generation-capable Qwen3-VL or Qwen3.5-VL `CLIP`;
- separate editable system and user prompts;
- an optional `IMAGE` batch;
- deterministic or sampled decoding controls;
- optional Qwen thinking mode.

It returns cleaned text, the untouched decoded output, the exact textual chat prompt, the resolved system prompt, and a short generation report.

## Basic workflow

1. Load a complete instruction-tuned Qwen3-VL or Qwen3.5-VL checkpoint with a compatible ComfyUI CLIP loader.
2. Connect its `CLIP` output to this node's `clip` input.
3. Write the model's behavior in `system_prompt` and the current request in `prompt`.
4. Optionally connect one image or an IMAGE batch.
5. Read `generated_text`; use `raw_output` and `chat_prompt` when debugging.

The image batch can be interpreted as its first image, the first N images, or N evenly sampled images. Every selected item is attached as a separate visual input in batch order. The default cap is eight because visual tokens substantially increase runtime and VRAM use.

## Important model distinction

This node needs a complete language model with a final normalization layer and LM head. A truncated text encoder bundled only for diffusion conditioning cannot generate reliable text. In particular, the MiniMax H3 50-layer conditioning encoder and its H3-specific generation-tail workflow are intentionally outside this pack.

The node calls ComfyUI's public `clip.tokenize()`, `clip.generate()`, and `clip.decode()` flow. It never synchronously force-unloads the connected model; ComfyUI manages model residency.

## Installation

Place or link this folder inside ComfyUI's `custom_nodes` directory, then restart ComfyUI:

```text
ComfyUI-H3-Qwen3VL-TextGen/
```

No additional Python package is required beyond a current ComfyUI build with Qwen3-VL/Qwen3.5-VL support.

## Development

Run the isolated tests from this directory:

```bash
pytest -q
```

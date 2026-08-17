# ComfyUI H3 Qwen3-VL Text Generation

A standalone ComfyUI node pack that turns MiniMax H3's truncated Qwen3-VL-32B text encoder into a local text and vision-language generator by reconnecting its published generation tail.

This pack is independent of the MiniMax H3 prompt-guide workflow. It reuses the same proven tail model and cleanup path, but its output is unrestricted general-purpose text rather than an H3 video prompt.

## Nodes

### H3 Qwen VL Generation Tail Loader

Selects a compatible `generation_tail_50_63` file from `models/text_encoders`. The tail contains:

- Qwen3-VL-32B language layers 50–63;
- the final normalization layer;
- the language-model head.

It is deliberately a dedicated typed loader rather than a standard `CLIP` loader. The tail is not a complete CLIP and has no tokenizer, token embeddings, or vision tower of its own.

Its connection type and descriptor are compatible with the original H3 Guide pack's generation-tail loader, although this standalone pack does not require the Guide pack to be installed.

### H3 Qwen VL Generate Text (Standalone)

Combines:

- `clip`: MiniMax H3's normal 50-layer Qwen3-VL-32B text encoder, loaded with ComfyUI's standard `Load CLIP` node;
- `tail_clip`: output from `H3 Qwen VL Generation Tail Loader`;
- an optional generation-only Qwen3-VL-32B overlay and strength;
- separate editable system and user prompts;
- an optional `IMAGE` batch;
- deterministic or sampled decoding controls;
- selectable ComfyUI, SageAttention 2, Comfy Kitchen INT8, or PyTorch attention;
- optional Comfy Kitchen fixed-KV Flash Attention decoding;
- optional Qwen thinking mode.

It returns cleaned text, untouched decoded output, the textual chat prompt, the resolved system prompt, and a generation report.

## Required graph

```text
Load CLIP (MiniMax H3 TE) ── clip ──────┐
                                        ├─ H3 Qwen VL Generate Text
H3 Qwen VL Tail Loader ── tail_clip ────┘
Optional IMAGE batch ───── image ───────┘
```

The normal H3 CLIP supplies tokenization, embeddings, vision processing, and language layers 0–49. During generation the node loads the tail as a second ComfyUI-managed model, runs layers 0–49 and 50–63 sequentially for every token, and evaluates the tail's final norm and LM head.

Without an overlay, only the temporary tail is explicitly unloaded. With an
overlay selected, the temporary patched base clone is unloaded as well. The
original connected base CLIP remains under normal ComfyUI model-residency
management and its weights are unchanged.

## Optional learned generation overlay

The `generation_overlay` widget lists compatible files under
`models/loras` whose names contain both `h3` and `generation_overlay`.
Select
`minimax_h3_qwen3vl32b__prompt_generation_overlay__polaris_r16_plus_heretic_v2.safetensors`
and start with `overlay_strength = 1.0`.

The overlay is attached with ComfyUI bypass LoRA hooks only for the text
generation call. Its layers 0–49 run on a temporary clone of the connected
MiniMax H3 CLIP, while layers 50–63 run on the temporary generation tail.
Both sets of hooks are removed during cleanup. The connected CLIP weights used
later for H3 video conditioning are not rewritten or merged.

This overlay is a Qwen text-encoder artifact, not a MiniMax H3 diffusion LoRA.
Do not connect it to a normal H3 diffusion-model LoRA loader.

## Example workflow

Load
[`example_workflows/h3_qwen_vl_standalone_polaris_overlay.json`](example_workflows/h3_qwen_vl_standalone_polaris_overlay.json)
in ComfyUI for a minimal text-only graph. It selects the official H3 INT8
ConvRot encoder, its matching INT8 generation tail, and the experimental
Polaris r16 + Heretic-v2 generation overlay at strength `1.0`. The workflow
contains download metadata for all three artifacts and exposes both generated
text and the generation report.

The node's general default remains **no overlay**. The bundled workflow opts in
explicitly so official Qwen behavior remains the safe baseline for other
workflows.

## Images

The optional IMAGE batch can be interpreted as its first image, the first N images, or N evenly sampled images. MiniMax H3's tokenizer prepends each selected image as a numbered `<Picture N>` visual block before the Qwen chat. The default cap is four because additional images increase vision encoding time, context length, and VRAM use.

## Tail compatibility

The tail loader only lists filenames containing `generation_tail_50_63`. Runtime validation additionally requires exactly source layers 50–63 plus `model.norm.weight` and `model.lm_head.weight`.

The current quantized LM-head path supports the published ComfyUI `int8_tensorwise` ConvRot layout. Incompatible or incomplete tail artifacts fail with an explicit error instead of silently producing corrupted text.

## Attention acceleration

`attention_backend` controls the regular attention function used by Qwen's
language layers. `SageAttention 2` selects ComfyUI's registered `sage` backend;
on Blackwell this requires a SageAttention 2 build compiled for SM120. This
local selection is necessary because ComfyUI's Qwen implementation otherwise
requests its small-input attention path directly.

`Comfy Kitchen INT8` selects ComfyUI's registered INT8 attention kernel. It
requires a `comfy-kitchen` build that reports INT8 attention support. Missing
explicit backends produce a clear error instead of silently selecting an
unrelated implementation.

`decode_backend` is separate because one-token autoregressive decoding can use
Comfy Kitchen's fixed-KV Flash Attention kernel directly. Its default is
`auto (Comfy Kitchen if available)`, which enables that path on compatible
devices and falls back to the standard KV cache otherwise. Select `standard KV
cache` if you specifically want SageAttention 2 to handle decode calls too.
The generation report records the paths selected at runtime.

## Download the generation tails

The compatible tail files are published in the [Qwen3-VL-32B H3 ComfyUI Generation Tails repository](https://huggingface.co/ethanfel/Qwen3-VL-32B-H3-ComfyUI-Generation-Tails). The repository may populate progressively while the initial upload is still running.

Download the desired `generation_tail_50_63` `.safetensors` file into:

```text
ComfyUI/models/text_encoders/
```

Then restart ComfyUI or refresh its model list so the dedicated tail loader can discover it.

Place a compatible generation overlay under:

```text
ComfyUI/models/loras/
```

## Installation

Place or link this folder inside ComfyUI's `custom_nodes` directory, put the H3 base text encoder and generation tail in `models/text_encoders`, then restart ComfyUI:

```text
ComfyUI-H3-Qwen3VL-TextGen/
```

No dependency on the separate H3 Guide custom-node pack is required.

## Development

Run the isolated tests from this directory:

```bash
pytest -q
```

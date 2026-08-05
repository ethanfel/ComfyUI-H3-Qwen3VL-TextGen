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
- separate editable system and user prompts;
- an optional `IMAGE` batch;
- deterministic or sampled decoding controls;
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

After generation, only the temporary tail is explicitly unloaded. The connected base CLIP remains under normal ComfyUI model-residency management.

## Images

The optional IMAGE batch can be interpreted as its first image, the first N images, or N evenly sampled images. MiniMax H3's tokenizer prepends each selected image as a numbered `<Picture N>` visual block before the Qwen chat. The default cap is four because additional images increase vision encoding time, context length, and VRAM use.

## Tail compatibility

The tail loader only lists filenames containing `generation_tail_50_63`. Runtime validation additionally requires exactly source layers 50–63 plus `model.norm.weight` and `model.lm_head.weight`.

The current quantized LM-head path supports the published ComfyUI `int8_tensorwise` ConvRot layout. Incompatible or incomplete tail artifacts fail with an explicit error instead of silently producing corrupted text.

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

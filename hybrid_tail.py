"""Temporary Qwen3-VL-32B generation tail for standalone H3 text generation.

The connected standard MiniMax H3 CLIP owns the embedding, vision tower, and
language layers 0..49. This module loads only layers 50..63, the final norm,
and the LM head while text is generated. It never changes the connected CLIP.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
import re
import threading
import traceback

import torch

import folder_paths
import comfy.hooks
import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import comfy.weight_adapter
from comfy.text_encoders import llama


TAIL_FIRST_LAYER = 50
TAIL_LAYER_COUNT = 14
FULL_LAYER_COUNT = 64
RUNTIME_HEADROOM = 6 * 1024**3
LM_HEAD_CHUNK_ROWS = 4096
LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")
OVERLAY_PREFIX = "text_encoders.qwen3vl_32b.transformer.model"
OVERLAY_KEY = re.compile(rf"^{re.escape(OVERLAY_PREFIX)}\.layers\.(\d+)\.(.+)$")
EXPECTED_ADAPTERS_PER_LAYER = 7
ATTENTION_BACKEND_NAMES = {
    "auto": "ComfyUI automatic",
    "sage": "SageAttention 2",
    "comfy_kitchen_int8": "Comfy Kitchen INT8",
    "pytorch": "PyTorch SDPA",
}
DECODE_BACKENDS = {"auto", "comfy_kitchen", "standard"}
_ATTENTION_OVERRIDE_LOCK = threading.RLock()
_MISSING = object()


def _layout_name(weight):
    layout = getattr(weight, "_layout_cls", "")
    return layout if isinstance(layout, str) else getattr(layout, "__name__", "")


def _int8_scale_chunk(scale, start: int, end: int, row_count: int):
    """Return a scalar or row-broadcastable scale for an INT8 head chunk."""

    if not isinstance(scale, torch.Tensor):
        raise ValueError("The generation-tail LM-head INT8 scale is not a tensor.")
    if scale.numel() == 1:
        return scale
    if scale.shape[0] != row_count:
        raise ValueError(
            "The generation-tail LM-head INT8 scale must be scalar or have "
            f"one value per vocabulary row; got shape {tuple(scale.shape)} for "
            f"{row_count} rows."
        )
    chunk = scale[start:end]
    if chunk.ndim == 1:
        chunk = chunk.unsqueeze(-1)
    if chunk.ndim != 2 or chunk.shape[1] != 1:
        raise ValueError(
            "The generation-tail LM-head only supports tensor-wise or per-row "
            f"INT8 scales; got shape {tuple(scale.shape)}."
        )
    return chunk


def _validate_int8_head(weight, qdata, scale):
    """Reject quantized layouts that cannot use the explicit chunked head path."""

    layout = _layout_name(weight)
    if layout != "TensorWiseINT8Layout":
        raise ValueError(
            "The generation-tail LM head must use ComfyUI int8_tensorwise "
            "quantization; unsupported layout "
            f"{layout or '<unknown>'}. Select the published INT8 ConvRot tail."
        )
    if not isinstance(qdata, torch.Tensor) or qdata.ndim != 2:
        raise ValueError(
            "The generation-tail LM-head INT8 data must be a two-dimensional "
            "[vocabulary, hidden_size] tensor."
        )
    if qdata.dtype != torch.int8:
        raise ValueError(
            "The generation-tail LM-head int8_tensorwise data must have torch.int8 "
            f"storage, got {qdata.dtype}."
        )
    _int8_scale_chunk(scale, 0, min(1, qdata.shape[0]), qdata.shape[0])

    params = getattr(weight, "_params", None)
    if getattr(params, "convrot", False):
        group_size = int(getattr(params, "convrot_groupsize", 0))
        factor = group_size
        while factor > 1 and factor % 4 == 0:
            factor //= 4
        if group_size < 4 or factor != 1:
            raise ValueError(
                "The generation-tail ConvRot group size must be a positive power "
                f"of four; got {group_size}."
            )
        # _hadamard below additionally verifies a power of four. This early
        # divisibility check provides a clearer artifact-compatibility error.
        if qdata.shape[1] % group_size:
            raise ValueError(
                f"The LM-head hidden size {qdata.shape[1]} is not divisible by "
                f"its ConvRot group size {group_size}."
            )


def _hadamard(size, device, dtype=torch.float32):
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        device=device,
        dtype=dtype,
    )
    matrix = h4
    current = 4
    while current < size:
        matrix = torch.kron(matrix, h4)
        current *= 4
    if current != size:
        raise ValueError(f"ConvRot group must be a power of four, got {size}")
    return matrix / (size**0.5)


def _rotate_activation(x, group_size):
    groups, remainder = divmod(x.shape[-1], group_size)
    if remainder:
        raise ValueError(
            f"LM-head input {x.shape[-1]} is not divisible by {group_size}"
        )
    h = _hadamard(group_size, x.device, x.dtype)
    return torch.matmul(x.reshape(-1, groups, group_size), h).reshape(x.shape)


@dataclass
class Qwen3VL32BTailConfig(llama.Qwen3VL_32BConfig):
    num_hidden_layers: int = TAIL_LAYER_COUNT
    lm_head: bool = True
    final_norm: bool = True


class Qwen3VL32BGenerationTail(torch.nn.Module):
    """Source layers 50..63 represented locally as layers 0..13."""

    def __init__(self, dtype, device, operations):
        super().__init__()
        config = Qwen3VL32BTailConfig()
        self.model = llama.Llama2_(config, device=device, dtype=dtype, ops=operations)
        # The connected 50-layer CLIP supplies token embeddings. Avoid a
        # duplicate 151936x5120 table in the temporary tail.
        del self.model.embed_tokens

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        return self.model.init_kv_cache(batch, max_cache_len, device, execution_dtype)

    def forward(self, hidden, position_ids, past_key_values):
        return self.model(
            None,
            embeds=hidden,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def logits(self, hidden):
        """Evaluate the huge quantized head in rows to limit peak VRAM."""

        weight = self.model.lm_head.weight
        qdata = getattr(weight, "_qdata", None)
        params = getattr(weight, "_params", None)
        scale = getattr(params, "scale", None)
        if qdata is None and scale is None:
            return self.model.lm_head(hidden[:, -1:])
        if qdata is None or scale is None:
            raise ValueError(
                "The generation-tail LM head has an incomplete quantized layout: "
                "both INT8 data and scale are required."
            )

        _validate_int8_head(weight, qdata, scale)

        input_token = hidden[:, -1:].float()
        if getattr(params, "convrot", False):
            input_token = _rotate_activation(input_token, params.convrot_groupsize)

        output = []
        for start in range(0, qdata.shape[0], LM_HEAD_CHUNK_ROWS):
            end = min(qdata.shape[0], start + LM_HEAD_CHUNK_ROWS)
            q_chunk = qdata[start:end].to(device=input_token.device, non_blocking=True)
            weight_chunk = q_chunk.to(dtype=torch.float32)
            scale_chunk = _int8_scale_chunk(scale, start, end, qdata.shape[0])
            weight_chunk.mul_(
                scale_chunk.to(
                    device=input_token.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )
            output.append(torch.nn.functional.linear(input_token, weight_chunk))
            del q_chunk, weight_chunk
        return torch.cat(output, dim=-1)


def _load_tail_state(name):
    path = folder_paths.get_full_path_or_raise("text_encoders", name)
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    state, _ = comfy.utils.convert_old_quants(state, metadata=metadata)

    layers = {
        int(match.group(1))
        for key in state
        if (match := LAYER_KEY.match(key)) is not None
    }
    if layers != set(range(TAIL_FIRST_LAYER, FULL_LAYER_COUNT)):
        raise ValueError(
            "clip_tail must contain exactly Qwen3-VL layers 50..63; "
            f"found {min(layers, default='none')}.."
            f"{max(layers, default='none')} ({len(layers)} layers)"
        )
    for key in ("model.norm.weight", "model.lm_head.weight"):
        if key not in state:
            raise ValueError(f"clip_tail is missing {key}")

    remapped = {}
    for key, value in state.items():
        match = LAYER_KEY.match(key)
        if match is not None:
            local_layer = int(match.group(1)) - TAIL_FIRST_LAYER
            key = f"model.layers.{local_layer}." + key[match.end() :]
        remapped[key] = value
    return remapped


def _build_tail(name, load_device, offload_device):
    state = _load_tail_state(name)
    dtype = state["model.norm.weight"].dtype
    quantization = comfy.utils.detect_layer_quantization(state, "")
    operations = (
        comfy.ops.mixed_precision_ops(quantization, dtype, full_precision_mm=True)
        if quantization is not None
        else comfy.ops.manual_cast
    )
    parameters = comfy.utils.calculate_parameters(state)
    initial_device = comfy.model_management.text_encoder_initial_device(
        load_device,
        offload_device,
        parameters * comfy.model_management.dtype_size(dtype),
    )
    tail = Qwen3VL32BGenerationTail(dtype, initial_device, operations)
    patcher = comfy.model_patcher.CoreModelPatcher(
        tail, load_device=load_device, offload_device=offload_device
    )
    patcher.set_model_compute_dtype(torch.float32)
    patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
    patcher.is_clip = True

    missing, unexpected = tail.load_state_dict(
        state, strict=False, assign=patcher.is_dynamic()
    )
    if missing or unexpected:
        raise RuntimeError(
            "clip_tail state mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    comfy.model_management.archive_model_dtypes(tail)
    return tail, patcher


def _load_generation_overlay(name):
    path = folder_paths.get_full_path_or_raise("loras", name)
    state, metadata = comfy.utils.load_torch_file(
        path, safe_load=True, return_metadata=True
    )
    metadata = metadata or {}
    if metadata.get("overlay_type") != "qwen3vl32b_h3_generation_overlay":
        raise ValueError(
            "generation_overlay is not a Qwen3-VL-32B H3 generation overlay"
        )
    return comfy.lora_convert.convert_lora(state)


def _overlay_prefixes(state, first_layer, last_layer):
    prefixes = set()
    suffix = ".lora_A.weight"
    for key in state:
        if not key.endswith(suffix):
            continue
        prefix = key[: -len(suffix)]
        match = OVERLAY_KEY.match(prefix)
        if match is None:
            continue
        layer = int(match.group(1))
        if first_layer <= layer < last_layer:
            prefixes.add(prefix)
    expected = (last_layer - first_layer) * EXPECTED_ADAPTERS_PER_LAYER
    if len(prefixes) != expected:
        raise ValueError(
            f"generation_overlay needs {expected} adapters for layers "
            f"{first_layer}..{last_layer - 1}; found {len(prefixes)}"
        )
    return prefixes


def _attach_bypass_adapters(patcher, model, loaded, strength, label):
    manager = comfy.weight_adapter.BypassInjectionManager()
    state_keys = set(model.state_dict())
    attached = set()
    for key, adapter in loaded.items():
        if not isinstance(adapter, comfy.weight_adapter.WeightAdapterBase):
            raise TypeError(
                f"{label} supports LoRA adapters only; {key} is "
                f"{type(adapter).__name__}"
            )
        if key in state_keys:
            manager.add_adapter(key, adapter, strength=float(strength))
            attached.add(key)
    if len(attached) != len(loaded):
        missing = sorted(set(loaded) - attached)
        raise ValueError(f"{label} targets missing model weights: {missing[:8]}")
    injections = manager.create_injections(model)
    if manager.get_hook_count() != len(attached):
        raise RuntimeError(
            f"{label} created {manager.get_hook_count()} hooks for "
            f"{len(attached)} adapters"
        )
    patcher.set_injections(label, injections)
    return len(attached)


def _clone_base_with_overlay(clip, state, strength):
    prefixes = _overlay_prefixes(state, 0, TAIL_FIRST_LAYER)
    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
    key_map = {key: value for key, value in key_map.items() if key in prefixes}
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)
    if len(loaded) != len(prefixes):
        raise ValueError(
            f"generation_overlay loaded {len(loaded)} of {len(prefixes)} base adapters"
        )
    patched = clip.clone()
    _attach_bypass_adapters(
        patched.patcher,
        patched.cond_stage_model,
        loaded,
        strength,
        "h3_generation_overlay_base",
    )
    return patched


def _attach_tail_overlay(tail, patcher, state, strength):
    prefixes = _overlay_prefixes(state, TAIL_FIRST_LAYER, FULL_LAYER_COUNT)
    tail_keys = set(tail.state_dict())
    key_map = {}
    for prefix in prefixes:
        match = OVERLAY_KEY.match(prefix)
        global_layer = int(match.group(1))
        module = match.group(2)
        target = f"model.layers.{global_layer - TAIL_FIRST_LAYER}.{module}.weight"
        if target in tail_keys:
            key_map[prefix] = target
    loaded = comfy.lora.load_lora(state, key_map, log_missing=False)
    if len(loaded) != len(prefixes):
        raise ValueError(
            f"generation_overlay loaded {len(loaded)} of {len(prefixes)} tail adapters"
        )
    _attach_bypass_adapters(
        patcher,
        tail,
        loaded,
        strength,
        "h3_generation_overlay_tail",
    )


def _get_minimax_base(clip):
    cond_stage = getattr(clip, "cond_stage_model", None)
    clip_name = getattr(cond_stage, "clip", "qwen3vl_32b")
    wrapper = getattr(cond_stage, clip_name, None)
    base = getattr(wrapper, "transformer", None)
    model = getattr(base, "model", None)
    layers = getattr(model, "layers", None)
    if wrapper is None or base is None or layers is None or len(layers) != 50:
        raise ValueError(
            "clip_tail requires the normal 50-layer MiniMax H3 Qwen3-VL-32B "
            "CLIP loaded with ComfyUI's standard CLIPLoader. A complete "
            "generative CLIP must leave the enhancer's clip_tail input disconnected."
        )
    return wrapper, base


def _sample(base, logits, options, history, generator):
    do_sample = bool(options.get("do_sample", True))
    return base.sample_token(
        logits,
        float(options.get("temperature", 1.0)) if do_sample else 1.0,
        int(options.get("top_k", 64)),
        float(options.get("top_p", 0.95)),
        float(options.get("min_p", 0.0)),
        float(options.get("repetition_penalty", 1.0)),
        history,
        generator,
        do_sample=do_sample,
        presence_penalty=float(options.get("presence_penalty", 0.0)),
    )


def _tensor_is_finite(tensor):
    """Synchronize one small reduction before CUDA sampling can assert."""

    return bool(torch.isfinite(tensor).all().item())


def _validate_finite_logits(
    logits,
    base_hidden,
    full_hidden,
    step,
    overlay_name=None,
    overlay_strength=1.0,
):
    """Stop non-finite generation before torch.multinomial poisons CUDA."""

    if _tensor_is_finite(logits):
        return

    if not _tensor_is_finite(base_hidden):
        stage = "base language layers 0-49"
    elif not _tensor_is_finite(full_hidden):
        stage = "generation-tail language layers 50-63"
    else:
        stage = "generation-tail LM head"

    overlay = (
        f"overlay={overlay_name!r} at strength {float(overlay_strength):g}"
        if overlay_name
        else "overlay=none"
    )
    hint = (
        " Set generation_overlay to [none] to isolate the official NVFP4 "
        "base/tail. If that succeeds, retry the overlay below strength 1.0."
        if overlay_name
        else " Verify that the base encoder and generation tail use the same format."
    )
    raise FloatingPointError(
        "H3 text generation produced NaN or Inf at generated token "
        f"{int(step) + 1} in {stage} ({overlay}). Sampling was stopped before "
        "torch.multinomial to avoid a fatal CUDA device-side assertion." + hint
    )


def _resolve_attention_function(attention_backend):
    if attention_backend not in ATTENTION_BACKEND_NAMES:
        raise ValueError(
            "attention_backend must be one of "
            + ", ".join(sorted(ATTENTION_BACKEND_NAMES))
        )
    if attention_backend == "auto":
        return None

    try:
        from comfy.ldm.modules.attention import get_attention_function

        return get_attention_function(attention_backend)
    except (ImportError, KeyError) as exc:
        if attention_backend == "sage":
            guidance = (
                "Install a SageAttention 2 build compatible with your GPU "
                "(SM120 on Blackwell) in ComfyUI's Python environment."
            )
        elif attention_backend == "comfy_kitchen_int8":
            guidance = (
                "Install or update Comfy Kitchen to a build with INT8 attention "
                "support."
            )
        else:
            guidance = "Update ComfyUI to a build that registers this backend."
        raise RuntimeError(
            f"The requested {ATTENTION_BACKEND_NAMES[attention_backend]} backend "
            f"is unavailable. {guidance}"
        ) from exc


def _comfy_kitchen_decode_available(device):
    try:
        import comfy_kitchen

        checker = getattr(comfy_kitchen, "flash_attention_decode_is_available")
        return bool(checker(device))
    except (AttributeError, ImportError, RuntimeError, TypeError):
        return False


@contextmanager
def _use_attention_runtime(
    base,
    tail,
    load_device,
    attention_backend="auto",
    decode_backend="auto",
    runtime_report=None,
):
    """Temporarily select Qwen attention and KV-cache implementations."""

    if decode_backend not in DECODE_BACKENDS:
        raise ValueError(
            "decode_backend must be one of " + ", ".join(sorted(DECODE_BACKENDS))
        )

    attention_function = _resolve_attention_function(attention_backend)
    fixed_kv_available = _comfy_kitchen_decode_available(load_device)
    if decode_backend == "comfy_kitchen" and not fixed_kv_available:
        raise RuntimeError(
            "Comfy Kitchen fixed-KV Flash Attention decode was explicitly "
            "requested but is unavailable on this device/build."
        )
    fixed_kv_enabled = fixed_kv_available and decode_backend != "standard"

    models = (base.model, tail.model)

    with _ATTENTION_OVERRIDE_LOCK:
        old_fixed_kv = [getattr(model, "fixed_kv", _MISSING) for model in models]
        old_selector = getattr(llama, "optimized_attention_for_device", _MISSING)
        try:
            if attention_function is not None:
                if old_selector is _MISSING:
                    raise RuntimeError(
                        "This ComfyUI build does not expose Qwen's attention selector."
                    )

                def select_attention(_device, mask=False, small_input=False):
                    del mask, small_input
                    return attention_function

                llama.optimized_attention_for_device = select_attention

            for model in models:
                model.fixed_kv = fixed_kv_enabled

            if runtime_report is not None:
                runtime_report["attention"] = ATTENTION_BACKEND_NAMES[attention_backend]
                if fixed_kv_enabled:
                    runtime_report["decode"] = "Comfy Kitchen fixed-KV Flash Attention"
                elif decode_backend == "auto":
                    runtime_report["decode"] = (
                        "standard KV cache (Comfy Kitchen unavailable)"
                    )
                else:
                    runtime_report["decode"] = "standard KV cache"
            yield
        finally:
            if attention_function is not None and old_selector is not _MISSING:
                llama.optimized_attention_for_device = old_selector
            for model, old_value in zip(models, old_fixed_kv):
                if old_value is _MISSING:
                    delattr(model, "fixed_kv")
                else:
                    model.fixed_kv = old_value


def _generate_tail_tokens_inner(
    wrapper,
    base,
    tail,
    tokens,
    generation_options,
    load_device,
    overlay_name=None,
    overlay_strength=1.0,
):
    """Own all temporary CUDA generation state in a short-lived stack frame."""

    token_batches = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
    token_ids = [[token[0] for token in batch] for batch in token_batches]
    embeds, _, _, embeds_info = wrapper.process_tokens(token_ids, load_device)
    position_ids, visual_pos_masks, deepstack = base.build_image_inputs(
        embeds, embeds_info
    )

    execution_dtype = (
        torch.bfloat16
        if comfy.model_management.should_use_bf16(load_device)
        else torch.float32
    )
    embeds = embeds.to(execution_dtype)
    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)
    max_new_tokens = int(generation_options.get("max_length", 128))
    cache_len = embeds.shape[1] + max_new_tokens
    base_cache = base.init_kv_cache(
        embeds.shape[0], cache_len, load_device, execution_dtype
    )
    tail_cache = tail.init_kv_cache(
        embeds.shape[0], cache_len, load_device, execution_dtype
    )

    do_sample = bool(generation_options.get("do_sample", True))
    generator = (
        torch.Generator(device=load_device).manual_seed(
            int(generation_options.get("seed", 0))
        )
        if do_sample
        else None
    )
    generated = []
    progress = comfy.utils.ProgressBar(max_new_tokens)
    next_pos = (
        int(position_ids[:, -1].max()) + 1
        if position_ids is not None
        else embeds.shape[1]
    )

    for step in range(max_new_tokens):
        extra = {}
        if step == 0 and deepstack is not None:
            extra["deepstack_embeds"] = deepstack
            extra["visual_pos_masks"] = visual_pos_masks
        base_hidden, _, base_cache = base(
            None,
            embeds=embeds,
            position_ids=position_ids,
            past_key_values=base_cache,
            embeds_info=embeds_info if step == 0 else [],
            **extra,
        )
        full_hidden, _, tail_cache = tail(base_hidden, position_ids, tail_cache)
        logits = tail.logits(full_hidden)[:, -1]
        _validate_finite_logits(
            logits,
            base_hidden,
            full_hidden,
            step,
            overlay_name,
            overlay_strength,
        )
        next_token = _sample(base, logits, generation_options, generated, generator)
        token_id = next_token[0].item()
        generated.append(token_id)
        progress.update(1)
        if token_id in tail.model.config.stop_tokens:
            break

        embeds = base.model.embed_tokens(next_token).to(execution_dtype)
        position_ids = torch.tensor([[next_pos]], device=load_device)
        next_pos += 1
        comfy.model_management.throw_exception_if_processing_interrupted()
    return generated


def _generate_tail_tokens(
    wrapper,
    base,
    tail,
    tokens,
    generation_options,
    load_device,
    overlay_name=None,
    overlay_strength=1.0,
    attention_backend="auto",
    decode_backend="auto",
    runtime_report=None,
):
    with _use_attention_runtime(
        base,
        tail,
        load_device,
        attention_backend,
        decode_backend,
        runtime_report,
    ):
        return _generate_tail_tokens_inner(
            wrapper,
            base,
            tail,
            tokens,
            generation_options,
            load_device,
            overlay_name,
            overlay_strength,
        )


def generate_with_tail(
    clip,
    tail_name,
    tokens,
    generation_options,
    overlay_name=None,
    overlay_strength=1.0,
    attention_backend="auto",
    decode_backend="auto",
    runtime_report=None,
):
    """Generate through base layers 0..49 and a temporary tail 50..63."""

    working_clip = clip
    overlay_state = None
    load_device = clip.patcher.load_device
    offload_device = clip.patcher.offload_device
    tail = tail_patcher = None
    wrapper = base = None
    old_execution_device = None
    result = None
    primary_error = primary_traceback = None
    cleanup_errors = []

    def remember_cleanup_error(exc):
        traceback.clear_frames(exc.__traceback__)
        cleanup_errors.append(exc)

    try:
        if overlay_name:
            overlay_state = _load_generation_overlay(overlay_name)
            working_clip = _clone_base_with_overlay(
                clip, overlay_state, overlay_strength
            )
        wrapper, base = _get_minimax_base(working_clip)
        old_execution_device = wrapper.execution_device
        tail, tail_patcher = _build_tail(tail_name, load_device, offload_device)
        if overlay_state is not None:
            _attach_tail_overlay(tail, tail_patcher, overlay_state, overlay_strength)
        comfy.model_management.load_models_gpu(
            [working_clip.patcher, tail_patcher], memory_required=RUNTIME_HEADROOM
        )
        wrapper.execution_device = load_device
        # Keeping CUDA temporaries inside a separate call is deliberate: its KV
        # caches, embeddings, and logits are released before the managed tail
        # unload below asks PyTorch to return cached memory to the driver.
        result = _generate_tail_tokens(
            wrapper,
            base,
            tail,
            tokens,
            generation_options,
            load_device,
            overlay_name,
            overlay_strength,
            attention_backend,
            decode_backend,
            runtime_report,
        )
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
        # InterruptProcessingException inherits BaseException. Clear completed
        # inner forward frames before cleanup so their CUDA tensors do not stay
        # reachable through the traceback while the original error propagates.
        traceback.clear_frames(primary_traceback)
    finally:
        if wrapper is not None:
            try:
                wrapper.execution_device = old_execution_device
            except BaseException as exc:
                remember_cleanup_error(exc)
        if tail_patcher is not None:
            try:
                comfy.model_management.unload_model_and_clones(
                    tail_patcher,
                    unload_additional_models=False,
                    all_devices=True,
                )
            except BaseException as exc:
                remember_cleanup_error(exc)
        if working_clip is not clip:
            try:
                comfy.model_management.unload_model_and_clones(
                    working_clip.patcher,
                    unload_additional_models=False,
                    all_devices=True,
                )
            except BaseException as exc:
                remember_cleanup_error(exc)
        working_clip = overlay_state = None
        tail = tail_patcher = None
        try:
            gc.collect()
        except BaseException as exc:
            remember_cleanup_error(exc)
        try:
            comfy.model_management.soft_empty_cache(force=True)
        except TypeError:
            # Compatibility with older ComfyUI builds that lacked `force`.
            try:
                comfy.model_management.soft_empty_cache()
            except BaseException as exc:
                remember_cleanup_error(exc)
        except BaseException as exc:
            remember_cleanup_error(exc)

    if primary_error is not None:
        if cleanup_errors and hasattr(primary_error, "add_note"):
            primary_error.add_note(
                "Generation-tail cleanup also failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            )
        raise primary_error.with_traceback(primary_traceback)
    if cleanup_errors:
        raise cleanup_errors[0]
    return result

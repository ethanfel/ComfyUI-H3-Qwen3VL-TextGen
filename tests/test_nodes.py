from types import SimpleNamespace

import pytest

import nodes
from nodes import (
    DEFAULT_SYSTEM_PROMPT,
    H3QwenVLGenerateText,
    H3QwenVLGenerationTailLoader,
    TAIL_TYPE,
    clean_generated_text,
    format_qwen_chat,
    select_images,
    validate_h3_base_clip,
)


class FakeImageBatch:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values), 2, 2, 3)

    def __getitem__(self, item):
        return FakeImageBatch(self.values[item])

    def __len__(self):
        return len(self.values)


def minimax_tokenizer():
    tokenizer_type = type("MiniMaxH3Tokenizer", (), {})
    tokenizer_type.__module__ = "comfy.text_encoders.minimax"
    return tokenizer_type()


class FakeClip:
    def __init__(self, decoded="<think>private</think>\nFinal answer"):
        self.tokenizer = minimax_tokenizer()
        self.decoded = decoded
        self.tokenize_call = None

    def tokenize(self, prompt, **kwargs):
        self.tokenize_call = (prompt, kwargs)
        return {"qwen3vl_32b": [[(1, 1.0)]]}

    def decode(self, generated_ids):
        assert generated_ids == [101, 102]
        return self.decoded


def generation_kwargs(**overrides):
    values = {
        "clip": FakeClip(),
        "tail_clip": {"tail_name": "qwen_generation_tail_50_63.safetensors"},
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "prompt": "Compare the supplied images.",
        "max_new_tokens": 300,
        "sampling": "sample",
        "temperature": 0.7,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "presence_penalty": 0.0,
        "seed": 42,
        "thinking": False,
        "image_batch_mode": "all images from start",
        "max_images": 8,
        "clean_output": True,
        "image": None,
    }
    values.update(overrides)
    return values


def test_two_loader_schema_requires_base_clip_and_dedicated_tail():
    generator = H3QwenVLGenerateText.INPUT_TYPES()
    loader = H3QwenVLGenerationTailLoader.INPUT_TYPES()
    assert generator["required"]["clip"][0] == "CLIP"
    assert generator["required"]["tail_clip"][0] == TAIL_TYPE
    assert "tail_name" in loader["required"]
    assert H3QwenVLGenerationTailLoader.RETURN_TYPES == (TAIL_TYPE,)
    assert H3QwenVLGenerationTailLoader.RETURN_NAMES == ("tail_clip",)
    assert generator["required"]["system_prompt"][1]["default"] == (
        DEFAULT_SYSTEM_PROMPT
    )
    assert generator["optional"]["image"][0] == "IMAGE"
    assert len(H3QwenVLGenerateText.OUTPUT_TOOLTIPS) == len(
        H3QwenVLGenerateText.RETURN_TYPES
    )


def test_tail_loader_returns_a_lazy_typed_descriptor():
    name = "folder/qwen_generation_tail_50_63_int8.safetensors"
    assert H3QwenVLGenerationTailLoader().select_tail(name) == ({"tail_name": name},)


def test_chat_format_uses_qwen_roles_and_disables_thinking():
    chat = format_qwen_chat("Be exact.", "What is shown?", thinking=False)
    assert chat.startswith("<|im_start|>system\nBe exact.<|im_end|>")
    assert "<|im_start|>user\nWhat is shown?" in chat
    assert "<|vision_start|>" not in chat
    assert chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_empty_system_omits_system_turn():
    chat = format_qwen_chat("", "Answer this.", thinking=True)
    assert "<|im_start|>system" not in chat
    assert chat == ("<|im_start|>user\nAnswer this.<|im_end|>\n<|im_start|>assistant\n")


def test_image_batch_selection_supports_start_and_even_sampling():
    batch = FakeImageBatch(range(10))
    start = select_images(batch, "all images from start", 3)
    even = select_images(batch, "evenly sample batch", 4)
    assert [item.values for item in start] == [[0], [1], [2]]
    assert [item.values for item in even] == [[0], [3], [6], [9]]


def test_generation_routes_base_tokens_and_images_through_tail(monkeypatch):
    clip = FakeClip()
    batch = FakeImageBatch(["a", "b", "c"])
    tail_calls = []

    def generate_with_tail(base, tail_name, tokens, options):
        tail_calls.append((base, tail_name, tokens, options))
        return [101, 102]

    monkeypatch.setattr(nodes, "_generate_with_tail", generate_with_tail)
    result = H3QwenVLGenerateText().generate_text(
        **generation_kwargs(clip=clip, image=batch)
    )

    generated, raw, chat, system, report = result
    assert generated == "Final answer"
    assert raw == "<think>private</think>\nFinal answer"
    assert system == DEFAULT_SYSTEM_PROMPT
    assert "<|vision_start|>" not in chat
    token_prompt, token_options = clip.tokenize_call
    assert token_prompt == chat
    assert len(token_options["images"]) == 3
    assert "image" not in token_options
    assert token_options["skip_template"] is True
    assert tail_calls == [
        (
            clip,
            "qwen_generation_tail_50_63.safetensors",
            {"qwen3vl_32b": [[(1, 1.0)]]},
            {
                "do_sample": True,
                "max_length": 300,
                "temperature": 0.7,
                "top_k": 64,
                "top_p": 0.95,
                "min_p": 0.05,
                "repetition_penalty": 1.05,
                "presence_penalty": 0.0,
                "seed": 42,
            },
        )
    ]
    assert "3 image(s)" in report
    assert "temporary tail was unloaded" in report
    assert "base CLIP was left" in report


def test_deterministic_generation_disables_tail_sampling(monkeypatch):
    clip = FakeClip(decoded="answer")
    options_seen = []
    monkeypatch.setattr(
        nodes,
        "_generate_with_tail",
        lambda _clip, _tail, _tokens, options: (
            options_seen.append(options) or [101, 102]
        ),
    )
    result = H3QwenVLGenerateText().generate_text(
        **generation_kwargs(clip=clip, sampling="deterministic")
    )
    assert result[0] == "answer"
    assert options_seen[0]["do_sample"] is False
    assert not callable(getattr(clip, "generate", None))


def test_cleaning_only_removes_outer_chat_and_completed_reasoning():
    raw = (
        "<|im_start|>assistant\n<think>reasoning</think>\n"
        "```python\nprint('kept')\n```<|im_end|>"
    )
    assert clean_generated_text(raw) == "```python\nprint('kept')\n```"


def test_complete_qwen_clip_is_rejected_because_tail_needs_h3_base():
    tokenizer_type = type("Qwen35ImageTokenizer_", (), {})
    tokenizer_type.__module__ = "comfy.text_encoders.qwen35"
    clip = SimpleNamespace(
        tokenizer=tokenizer_type(),
        tokenize=lambda *_args, **_kwargs: None,
        generate=lambda *_args, **_kwargs: None,
        decode=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="50-layer Qwen3-VL-32B"):
        validate_h3_base_clip(clip)


def test_h3_base_does_not_need_native_generate_method():
    clip = FakeClip()
    assert validate_h3_base_clip(clip) == (
        "comfy.text_encoders.minimax.MiniMaxH3Tokenizer"
    )

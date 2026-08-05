from types import SimpleNamespace

import pytest

from nodes import (
    DEFAULT_SYSTEM_PROMPT,
    H3QwenVLGenerateText,
    VISION_BLOCK,
    clean_generated_text,
    format_qwen_chat,
    select_images,
    validate_qwen_clip,
)


class FakeImageBatch:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values), 2, 2, 3)

    def __getitem__(self, item):
        return FakeImageBatch(self.values[item])

    def __len__(self):
        return len(self.values)


def qwen_tokenizer(module="comfy.text_encoders.qwen35"):
    tokenizer_type = type("Qwen35ImageTokenizer_", (), {})
    tokenizer_type.__module__ = module
    return tokenizer_type()


class FakeClip:
    def __init__(self, decoded="<think>private</think>\nFinal answer"):
        self.tokenizer = qwen_tokenizer()
        self.decoded = decoded
        self.tokenize_call = None
        self.generate_call = None

    def tokenize(self, prompt, **kwargs):
        self.tokenize_call = (prompt, kwargs)
        return {"qwen": [[(1, 1.0)]]}

    def generate(self, tokens, **kwargs):
        self.generate_call = (tokens, kwargs)
        return [101, 102]

    def decode(self, generated_ids):
        assert generated_ids == [101, 102]
        return self.decoded


def generation_kwargs(**overrides):
    values = {
        "clip": FakeClip(),
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


def test_node_schema_is_standalone_and_exposes_system_prompt():
    schema = H3QwenVLGenerateText.INPUT_TYPES()
    assert schema["required"]["clip"][0] == "CLIP"
    assert schema["required"]["system_prompt"][1]["default"] == DEFAULT_SYSTEM_PROMPT
    assert schema["optional"]["image"][0] == "IMAGE"
    assert H3QwenVLGenerateText.RETURN_NAMES == (
        "generated_text",
        "raw_output",
        "chat_prompt",
        "system_prompt",
        "generation_report",
    )


def test_chat_format_places_visuals_in_user_turn_and_disables_thinking():
    chat = format_qwen_chat("Be exact.", "What is shown?", 2, thinking=False)
    assert chat.startswith("<|im_start|>system\nBe exact.<|im_end|>")
    assert f"<|im_start|>user\n{VISION_BLOCK * 2}\nWhat is shown?" in chat
    assert chat.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_empty_system_omits_system_turn():
    chat = format_qwen_chat("", "Answer this.", 0, thinking=True)
    assert "<|im_start|>system" not in chat
    assert chat == ("<|im_start|>user\nAnswer this.<|im_end|>\n<|im_start|>assistant\n")


def test_image_batch_selection_supports_start_and_even_sampling():
    batch = FakeImageBatch(range(10))
    start = select_images(batch, "all images from start", 3)
    even = select_images(batch, "evenly sample batch", 4)
    assert [item.values for item in start] == [[0], [1], [2]]
    assert [item.values for item in even] == [[0], [3], [6], [9]]


def test_generation_passes_images_and_sampling_to_native_clip_api():
    clip = FakeClip()
    batch = FakeImageBatch(["a", "b", "c"])
    result = H3QwenVLGenerateText().generate_text(
        **generation_kwargs(clip=clip, image=batch)
    )

    generated, raw, chat, system, report = result
    assert generated == "Final answer"
    assert raw == "<think>private</think>\nFinal answer"
    assert system == DEFAULT_SYSTEM_PROMPT
    assert chat.count(VISION_BLOCK) == 3
    token_prompt, token_options = clip.tokenize_call
    assert token_prompt == chat
    assert len(token_options["images"]) == 3
    assert token_options["image"] is batch
    assert token_options["skip_template"] is True
    _, generation_options = clip.generate_call
    assert generation_options == {
        "do_sample": True,
        "max_length": 300,
        "temperature": 0.7,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "presence_penalty": 0.0,
        "seed": 42,
    }
    assert "3 image(s)" in report
    assert "not explicitly unloaded" in report


def test_deterministic_generation_turns_sampling_off_without_forced_unload():
    clip = FakeClip(decoded="answer")
    result = H3QwenVLGenerateText().generate_text(
        **generation_kwargs(clip=clip, sampling="deterministic")
    )
    assert result[0] == "answer"
    assert clip.generate_call[1]["do_sample"] is False
    assert not hasattr(clip, "patcher")


def test_cleaning_only_removes_outer_chat_and_completed_reasoning():
    raw = (
        "<|im_start|>assistant\n<think>reasoning</think>\n"
        "```python\nprint('kept')\n```<|im_end|>"
    )
    assert clean_generated_text(raw) == "```python\nprint('kept')\n```"


def test_known_non_qwen_comfy_clip_is_rejected():
    tokenizer_type = type("CLIPTokenizer", (), {})
    tokenizer_type.__module__ = "comfy.text_encoders.clip_l"
    clip = SimpleNamespace(
        tokenizer=tokenizer_type(),
        tokenize=lambda *_args, **_kwargs: None,
        generate=lambda *_args, **_kwargs: None,
        decode=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="not a supported Qwen3-VL"):
        validate_qwen_clip(clip)


def test_unknown_custom_generation_wrapper_remains_compatible():
    clip = SimpleNamespace(
        tokenizer=object(),
        tokenize=lambda *_args, **_kwargs: None,
        generate=lambda *_args, **_kwargs: None,
        decode=lambda *_args, **_kwargs: None,
    )
    identity, recognized = validate_qwen_clip(clip)
    assert identity == "builtins.object"
    assert recognized is False

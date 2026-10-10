"""Chat templating for Qwen2.5 (ChatML), applied client-side so both backends receive the
identical prompt string. Matches the model's tokenizer_config.json chat_template for a single
user turn with the default system prompt and add_generation_prompt=True."""

from __future__ import annotations

QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def qwen_chat(user: str, system: str = QWEN_DEFAULT_SYSTEM) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")

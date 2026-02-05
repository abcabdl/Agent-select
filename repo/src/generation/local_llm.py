from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _require(pkg: str) -> None:
    try:
        __import__(pkg)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"Missing dependency '{pkg}'. Install with: pip install transformers peft bitsandbytes"
        ) from exc


_require("transformers")
_require("peft")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _resolve_dtype(value: str) -> Optional[torch.dtype]:
    value = (value or "").lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    return None


class LocalLLMClient:
    """Local transformers + LoRA inference client with OpenAI-like chat() API."""

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_4bit: bool = False,
        use_8bit: bool = False,
    ) -> None:
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = _resolve_dtype(dtype)
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if use_4bit or use_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                load_in_8bit=use_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype is torch.bfloat16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        elif torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path)

        if not (use_4bit or use_8bit):
            model = model.to(self.device)

        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def _build_prompt(self, messages: Iterable[Dict[str, str]]) -> str:
        messages_list = list(messages)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts = []
        for msg in messages_list:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if self.use_4bit or self.use_8bit:
            input_ids = input_ids.to(self.model.device)
        else:
            input_ids = input_ids.to(self.device)

        gen_temperature = temperature if temperature is not None else self.temperature
        max_new_tokens = max_tokens or self.max_new_tokens
        do_sample = gen_temperature > 0

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=gen_temperature if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


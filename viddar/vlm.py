"""Vision language model backends and a persistent response cache."""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import math
import os
import re
from pathlib import Path

from PIL import Image

# Short names accepted by the scripts. Any other string is treated as a Hugging Face model id.
PRESETS: dict[str, dict] = {
    "qwen3-vl-2b": {"model_id": "Qwen/Qwen3-VL-2B-Instruct"},
    "qwen3-vl-4b": {"model_id": "Qwen/Qwen3-VL-4B-Instruct"},
    "qwen2.5-vl-3b": {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    # The LLaVA model used in the paper; needs 4-bit quantisation (bitsandbytes) to fit 12 GB.
    "llava-next-8b-4bit": {"model_id": "llava-hf/llama3-llava-next-8b-hf", "load_in_4bit": True},
    "gpt-4o": {"backend": "openai", "model_id": "gpt-4o-2024-08-06"},
}


def resize_to_max_pixels(img: Image.Image, max_pixels: int | None) -> Image.Image:
    """Downscale (never upscale) so that width * height <= max_pixels, keeping aspect ratio."""
    if not max_pixels or img.width * img.height <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / (img.width * img.height))
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class HFVLM:
    """Local VLM through `transformers` (Qwen-VL, LLaVA, Gemma, SmolVLM, ... families)."""

    def __init__(
        self,
        model_id: str,
        max_pixels: int | None = 1_000_000,
        max_new_tokens: int = 512,
        load_in_4bit: bool = False,
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        kwargs: dict = {"dtype": getattr(torch, dtype), "device_map": "cuda"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype), bnb_4bit_quant_type="nf4"
            )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()

    def generate(self, images: list[Image.Image], prompt: str, temperature: float = 0.0, seed: int = 0) -> str:
        import torch

        images = [resize_to_max_pixels(im, self.max_pixels) for im in images]
        # Same ordering as the original OpenAI payloads: text first, then images.
        content = [{"type": "text", "text": prompt}] + [{"type": "image"} for _ in images]
        messages = [{"role": "user", "content": content}]
        template_kwargs = {}
        if "enable_thinking" in str(self.processor.chat_template or ""):
            template_kwargs["enable_thinking"] = False  # hybrid reasoning models: answer directly
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **template_kwargs
        )
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict = {"max_new_tokens": self.max_new_tokens}
        if temperature > 0:
            torch.manual_seed(seed)
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return _strip_thinking(decoded)

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class OpenAIVLM:
    """OpenAI chat-completions backend (the paper's GPT-4o setting). Needs OPENAI_API_KEY."""

    def __init__(self, model_id: str = "gpt-4o-2024-08-06", max_new_tokens: int = 600, max_pixels: int | None = None):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.api_key = os.environ["OPENAI_API_KEY"]

    def generate(self, images: list[Image.Image], prompt: str, temperature: float = 0.0, seed: int = 0) -> str:
        import requests

        content: list[dict] = [{"type": "text", "text": prompt}]
        for im in images:
            buf = io.BytesIO()
            resize_to_max_pixels(im, self.max_pixels).save(buf, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_new_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        pass


def load_vlm(name: str, max_pixels: int | None = 1_000_000, max_new_tokens: int = 512):
    spec = dict(PRESETS.get(name, {"model_id": name}))
    if spec.pop("backend", "hf") == "openai":
        return OpenAIVLM(spec["model_id"], max_new_tokens=max_new_tokens, max_pixels=max_pixels)
    return HFVLM(max_pixels=max_pixels, max_new_tokens=max_new_tokens, **spec)


def resolve_model_id(name: str) -> str:
    return PRESETS.get(name, {"model_id": name})["model_id"]


class ResponseCache:
    """Append-only JSONL cache of VLM answers, keyed by everything that affects the output."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec

    @staticmethod
    def make_key(**fields) -> str:
        return hashlib.sha1(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        rec = self._data.get(key)
        return rec["response"] if rec else None

    def put(self, key: str, response: str, **meta) -> None:
        rec = {"key": key, "response": response, **meta}
        self._data[key] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_SHA1_MEMO: dict[str, str] = {}


def _file_sha1(path: str) -> str:
    """Content hash, so the cache is independent of file names and folder locations."""
    if path not in _SHA1_MEMO:
        with open(path, "rb") as f:
            _SHA1_MEMO[path] = hashlib.sha1(f.read()).hexdigest()
    return _SHA1_MEMO[path]


class CachedVLM:
    """Wraps a lazily-loaded VLM so that cached answers never require loading the model."""

    def __init__(self, name: str, cache_dir: str | Path = "results/cache", max_pixels: int | None = 1_000_000,
                 max_new_tokens: int = 512, temperature: float = 0.0, seed: int = 0):
        self.name = name
        self.model_id = resolve_model_id(name)
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_id)
        self.cache = ResponseCache(Path(cache_dir) / f"{slug}.jsonl")
        self._vlm = None

    def ask(self, image_paths: list[str], prompt: str) -> str:
        key = ResponseCache.make_key(
            model=self.model_id, prompt=prompt, images=[_file_sha1(p) for p in image_paths],
            max_pixels=self.max_pixels, max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, seed=self.seed,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self._vlm is None:
            self._vlm = load_vlm(self.name, self.max_pixels, self.max_new_tokens)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        answer = self._vlm.generate(images, prompt, temperature=self.temperature, seed=self.seed)
        self.cache.put(key, answer, images=[str(p) for p in image_paths], prompt=prompt[:80])
        return answer

    def close(self) -> None:
        if self._vlm is not None:
            self._vlm.close()
            self._vlm = None

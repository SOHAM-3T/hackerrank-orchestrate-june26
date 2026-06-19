"""Provider adapter for structured vision-language calls."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .io_utils import image_data_url, load_json, save_json, stable_hash

PROMPT_VERSION = "evidence-agent-v1"


class MissingModelKeyError(RuntimeError):
    pass


class ModelClient:
    def __init__(self, cache_dir: Path, model: str | None = None, mode: str | None = None) -> None:
        load_dotenv()
        self.cache_dir = cache_dir
        self.model = model or os.getenv("MODEL_NAME", "gpt-4.1-mini")
        self.mode = (mode or os.getenv("EVIDENCE_AGENT_MODE", "openai")).strip().lower()
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.max_retries = int(os.getenv("MAX_MODEL_RETRIES", "2"))

    @property
    def is_heuristic(self) -> bool:
        return self.mode in {"heuristic", "offline", "mock"}

    def require_ready(self) -> None:
        if self.is_heuristic:
            return
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingModelKeyError(
                "OPENAI_API_KEY is not set. Set it in the environment or run with "
                "EVIDENCE_AGENT_MODE=heuristic for local smoke tests."
            )

    def structured_vision_json(
        self,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        images: list[tuple[Path, str]],
        extra_cache_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self.is_heuristic:
            return {}

        self.require_ready()
        cache_key = stable_hash(
            {
                "prompt_version": PROMPT_VERSION,
                "model": self.model,
                "task_name": task_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "images": [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path, _ in images],
                "extra": extra_cache_payload,
            }
        )
        cache_path = self.cache_dir / "model_responses" / f"{cache_key}.json"
        cached = load_json(cache_path)
        if cached is not None:
            return cached

        from openai import OpenAI

        client = OpenAI()
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        for path, mime in images:
            content.append({"type": "input_image", "image_url": image_data_url(path, mime)})

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.responses.create(
                    model=self.model,
                    temperature=self.temperature,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": content},
                    ],
                    text={"format": {"type": "json_object"}},
                )
                text = response.output_text
                payload = json.loads(text)
                save_json(cache_path, payload)
                return payload
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Model call failed after retries: {last_error}") from last_error

    def structured_text_json(
        self,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        extra_cache_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Text-only structured JSON call with caching and retries."""
        if self.is_heuristic:
            return {}

        self.require_ready()
        cache_key = stable_hash(
            {
                "prompt_version": PROMPT_VERSION,
                "model": self.model,
                "task_name": task_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "extra": extra_cache_payload,
            }
        )
        cache_path = self.cache_dir / "model_responses" / f"{cache_key}.json"
        cached = load_json(cache_path)
        if cached is not None:
            return cached

        from openai import OpenAI

        client = OpenAI()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.responses.create(
                    model=self.model,
                    temperature=self.temperature,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                    ],
                    text={"format": {"type": "json_object"}},
                )
                text = response.output_text
                payload = json.loads(text)
                save_json(cache_path, payload)
                return payload
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"Text model call failed after retries: {last_error}") from last_error

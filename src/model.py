"""Model construction helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .config import get_model_name


def build_model(env: Mapping[str, str] | None = None) -> Any:
    source = env or os.environ
    model_name = get_model_name(source)

    from langchain.chat_models import init_chat_model

    return init_chat_model(model_name, model_provider="openai", temperature=0)

"""
Prompt Engineering module for the AI Platform Demo.

Provides versioned Jinja2-based prompt templates with few-shot examples,
chain-of-thought patterns, and output format specifications.

Usage::

    from prompts import PromptLoader

    loader = PromptLoader()
    system_prompt = loader.load("researcher_system")
    report_prompt = loader.load("report_format", context={"topic": "AI trends"})
    print(loader.list_prompts())
"""

from .loader import PromptLoader
from .registry import PROMPT_REGISTRY, PromptEntry, PromptVersion

__all__ = ["PromptLoader", "PROMPT_REGISTRY", "PromptEntry", "PromptVersion"]

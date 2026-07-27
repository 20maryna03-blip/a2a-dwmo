"""
HuggingFace Inference API tools for the Custom MCP server.

Provides multimodal and NLP capabilities via the HuggingFace Hub Inference API:
  - Image analysis / captioning  (BLIP — Salesforce/blip-image-captioning-large)
  - Text summarization            (facebook/bart-large-cnn)
  - Zero-shot text classification (facebook/bart-large-mnli)

HF_API_KEY environment variable is required for the Inference API.
All models are free-tier compatible; no GPU needed.
"""

from __future__ import annotations

import json
import logging
import os
import base64

import httpx

logger = logging.getLogger(__name__)

HF_API_KEY = os.environ.get("HF_API_KEY", "")
HF_VISION_MODEL = os.environ.get("HF_VISION_MODEL", "Salesforce/blip-image-captioning-large")
_HF_BASE = "https://api-inference.huggingface.co/models"
_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}


def _hf_available() -> bool:
    return bool(HF_API_KEY and not HF_API_KEY.startswith("hf-your"))


async def _hf_post(model: str, payload: dict, timeout: float = 30.0) -> dict | list | str:
    """POST to HuggingFace Inference API and return parsed response."""
    if not _hf_available():
        return {"error": "HF_API_KEY not configured. Set it in .env to enable HuggingFace tools."}
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    url = f"{_HF_BASE}/{model}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
    ct = resp.headers.get("content-type", "")
    if "application/json" in ct:
        return resp.json()
    return resp.text


async def _hf_post_bytes(model: str, image_bytes: bytes, timeout: float = 30.0) -> dict | list:
    """POST raw image bytes to HuggingFace Inference API."""
    if not _hf_available():
        return {"error": "HF_API_KEY not configured. Set it in .env to enable HuggingFace tools."}
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/octet-stream"}
    url = f"{_HF_BASE}/{model}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, content=image_bytes, headers=headers)
        resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

async def analyze_image(image_url: str) -> str:
    """Analyze an image and generate a descriptive caption using BLIP.

    Uses the Salesforce BLIP image-captioning model via HuggingFace Inference
    API to generate a natural-language description of any publicly accessible
    image.  This enables multimodal research — agents can reference images
    (charts, diagrams, photos) and get textual descriptions.

    Args:
        image_url: Publicly accessible URL of the image to analyze
                   (JPEG, PNG, WEBP, GIF supported).

    Returns:
        JSON with the generated caption and model metadata.
    """
    if not _hf_available():
        return json.dumps({
            "status": "unavailable",
            "message": "HuggingFace API key not configured. Add HF_API_KEY to .env.",
            "image_url": image_url,
        })
    try:
        # Download image
        async with httpx.AsyncClient(timeout=20.0) as client:
            img_resp = await client.get(image_url, follow_redirects=True)
            img_resp.raise_for_status()
        image_bytes = img_resp.content
        result = await _hf_post_bytes(HF_VISION_MODEL, image_bytes, timeout=40.0)

        # BLIP returns [{"generated_text": "..."}]
        if isinstance(result, list) and result:
            caption = result[0].get("generated_text", str(result[0]))
        elif isinstance(result, dict):
            caption = result.get("generated_text", str(result))
        else:
            caption = str(result)

        return json.dumps({
            "image_url": image_url,
            "caption": caption,
            "model": HF_VISION_MODEL,
            "status": "success",
        })
    except Exception as exc:
        logger.exception("Error in analyze_image")
        return json.dumps({"error": str(exc), "image_url": image_url, "status": "failed"})


async def summarize_text(text: str, max_length: int = 150) -> str:
    """Summarize a long piece of text using a HuggingFace summarization model.

    Uses facebook/bart-large-cnn — a state-of-the-art abstractive summarizer
    trained on CNN/DailyMail data.  Ideal for condensing long research papers,
    news articles, or documents into concise summaries.

    Args:
        text:       Input text to summarize (recommended: 200–2000 words).
        max_length: Maximum token length of the summary (default 150).

    Returns:
        JSON with the summary text and word-count statistics.
    """
    if not _hf_available():
        return json.dumps({
            "status": "unavailable",
            "message": "HuggingFace API key not configured. Add HF_API_KEY to .env.",
            "text_length": len(text),
        })
    model = "facebook/bart-large-cnn"
    payload = {
        "inputs": text[:3000],  # BART context window limit
        "parameters": {
            "max_length": max(50, min(max_length, 300)),
            "min_length": 30,
            "do_sample": False,
        },
    }
    try:
        result = await _hf_post(model, payload, timeout=45.0)
        if isinstance(result, list) and result:
            summary = result[0].get("summary_text", str(result[0]))
        elif isinstance(result, dict):
            summary = result.get("summary_text", str(result))
        else:
            summary = str(result)
        return json.dumps({
            "summary": summary,
            "original_words": len(text.split()),
            "summary_words": len(summary.split()),
            "model": model,
            "status": "success",
        })
    except Exception as exc:
        logger.exception("Error in summarize_text")
        return json.dumps({"error": str(exc), "status": "failed"})


async def classify_text(text: str, candidate_labels: str) -> str:
    """Classify text into one or more of the given candidate labels (zero-shot).

    Uses facebook/bart-large-mnli for zero-shot classification — no training
    data needed.  Provide any labels and the model will score how well the
    text fits each one.  Useful for categorising research findings, news
    articles, or any text into custom taxonomies.

    Args:
        text:             Text to classify (up to ~500 words works best).
        candidate_labels: Comma-separated list of possible labels,
                          e.g. "machine learning, cybersecurity, climate change".

    Returns:
        JSON with labels ranked by confidence score.
    """
    if not _hf_available():
        return json.dumps({
            "status": "unavailable",
            "message": "HuggingFace API key not configured. Add HF_API_KEY to .env.",
        })
    model = "facebook/bart-large-mnli"
    labels = [l.strip() for l in candidate_labels.split(",") if l.strip()]
    if not labels:
        return json.dumps({"error": "candidate_labels must be a non-empty comma-separated string"})
    payload = {
        "inputs": text[:1000],
        "parameters": {"candidate_labels": labels, "multi_label": False},
    }
    try:
        result = await _hf_post(model, payload, timeout=30.0)
        if isinstance(result, dict) and "labels" in result:
            ranking = [
                {"label": lbl, "score": round(score, 4)}
                for lbl, score in zip(result["labels"], result["scores"])
            ]
            return json.dumps({
                "text_preview": text[:100] + ("…" if len(text) > 100 else ""),
                "classification": ranking,
                "top_label": ranking[0]["label"] if ranking else "",
                "model": model,
                "status": "success",
            })
        return json.dumps({"raw_result": result, "model": model, "status": "success"})
    except Exception as exc:
        logger.exception("Error in classify_text")
        return json.dumps({"error": str(exc), "status": "failed"})


async def generate_text_with_hf(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text using a HuggingFace language model (Mistral-7B).

    Calls the Mistral-7B-Instruct model via HuggingFace Inference API for
    open-source LLM text generation.  This provides an alternative to OpenAI
    and showcases multimodal / multi-provider agent architectures.

    Args:
        prompt:         Instruction or prompt for the model.
        max_new_tokens: Maximum tokens to generate (default 256, max 512).

    Returns:
        JSON with the generated text and model details.
    """
    if not _hf_available():
        return json.dumps({
            "status": "unavailable",
            "message": "HuggingFace API key not configured. Add HF_API_KEY to .env.",
        })
    model = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
    payload = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": min(max_new_tokens, 512),
            "temperature": 0.7,
            "return_full_text": False,
        },
    }
    try:
        result = await _hf_post(model, payload, timeout=60.0)
        if isinstance(result, list) and result:
            generated = result[0].get("generated_text", str(result[0]))
        elif isinstance(result, dict):
            generated = result.get("generated_text", str(result))
        else:
            generated = str(result)
        return json.dumps({
            "prompt": prompt[:100] + ("…" if len(prompt) > 100 else ""),
            "generated_text": generated,
            "model": model,
            "status": "success",
        })
    except Exception as exc:
        logger.exception("Error in generate_text_with_hf")
        return json.dumps({"error": str(exc), "status": "failed"})

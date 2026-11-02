"""Turns a rule-engine reason string into a vector, for `PostgresEmbeddingStore`.

Deliberately separate from `embedding_store.py` (storage) and `vlm.py` (triage), even
though this and `vlm.py` both wrap an OpenAI-compatible client: triage sends images and
carries a whole conversation's worth of prompt-building, this sends one string and gets one
vector back. Folding it into vlm.py would make triage's already-large surface responsible
for a feature that fails independently (an embeddings outage must not touch VLM triage).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

Embedder = Callable[[str], tuple[list[float], str] | None]


def build_embedder(settings) -> Embedder | None:  # noqa: ANN001 - Settings, avoided to dodge a config import cycle
    """Build a `text -> (vector, model_name)` function, or None if embeddings are off.

    Mirrors `vlm.build_triage_client`'s degrade-don't-fail shape: a missing key or a
    disabled flag returns None rather than raising, so a monitoring service built without
    embeddings configured still starts -- this is an optional annotation layer, not a
    dependency the rest of the app can be blocked on.
    """
    if not getattr(settings, "embeddings_enabled", False):
        return None
    if not settings.openai_api_key:
        logger.warning("EMBEDDINGS_ENABLED is true but OPENAI_API_KEY is empty; similar-incident lookup is disabled.")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("Install the openai package to use embeddings.")
        return None

    # Bounded the same way vlm.py's triage client is: the openai SDK's own default is a
    # 600s timeout with retries, which is effectively "hang" from a guard's point of view --
    # this call sits inside record_operator_verdict, the response the guard's "Confirm
    # real"/"False alarm" click is waiting on. Confirmed live: an unbounded client here is
    # what made the verdict buttons appear to freeze once embeddings were wired in, on the
    # same call shape as the original event-loop-blocking bug, just with a fixed cause this
    # time instead of an unbounded one.
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=getattr(settings, "openai_base_url", "") or None,
        timeout=10.0,
        max_retries=1,
    )
    model = settings.embedding_model

    def _embed(text: str) -> tuple[list[float], str] | None:
        if not text.strip():
            return None
        try:
            response = client.embeddings.create(model=model, input=text)
        except Exception:  # noqa: BLE001 - a flaky embeddings call must not break the verdict it is attached to
            logger.exception("Could not embed incident text; similar-incident lookup will skip this alert.")
            return None
        return response.data[0].embedding, model

    return _embed

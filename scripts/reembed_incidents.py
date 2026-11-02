"""Rebuild `incident_embeddings` after a change to what the similar-incident lookup embeds.

Why this exists
---------------
`MonitoringService.similar_incidents` compares a live alert against stored vectors. Both
sides go through `embedding_text`, so the comparison is only meaningful while every stored
vector was built by the *same* version of that function. When it changes — as it did when
the VLM's reading was added to the text and the zone name taken out — every row written
before the change sits somewhere else in the vector space, and neighbours retrieved from the
old rows are no longer the nearest ones in any useful sense.

Run this once after deploying such a change. Until it has run, the panel still works; it
just compares new alerts against a library described in a different language.

What it can and cannot restore
------------------------------
Rebuilt from `AlertHistory`, which is the only durable place the source text survives:

  - `[rule]`   from `reason`         — recorded, so restored exactly.
  - `[ai]`     from `vlm_rationale`  — recorded, so restored exactly.
  - `[frames]` from frame observations — NOT recorded anywhere. Unrecoverable for old rows.

So a rebuilt vector is very close to, but not byte-identical with, what the live path would
write for the same incident today. That is a real limitation and not worth hiding: rows
re-embedded here carry rule and AI text, and gain the frame section only if the incident is
ever judged again. Alerts judged after the deploy get the full text from the start.

`--dry-run` reports what would be embedded and how many API calls that costs, and is worth
running first: this spends one embeddings call per row, against a real billed API.

Usage
-----
    python -m scripts.reembed_incidents --dry-run
    python -m scripts.reembed_incidents
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import get_settings
from src.models.camera_schemas import Alert
from src.services.alert_history import AlertHistory
from src.services.embedding_store import create_embedding_store
from src.services.embeddings import build_embedder
from src.services.monitoring import embedding_text

logger = logging.getLogger("reembed")


def _row_to_text(row: dict) -> str:
    """Compose one history row's embedding text through the live composer.

    Goes through a real `Alert` rather than formatting the sections here so that this script
    cannot drift from `embedding_text`. If the composer changes again, this follows it; a
    second copy of the formatting would be a second thing to remember to update, and the
    failure mode of forgetting is silent — vectors that look fine and match badly.
    """
    return embedding_text(
        Alert(
            id=row.get("alert_id", ""),
            camera_id=row.get("camera_id", ""),
            camera_name=row.get("camera_id", ""),
            label=row.get("label", ""),
            confidence=0.0,
            created_at=row.get("recorded_at", ""),
            reason=row.get("reason"),
            triage_rationale=row.get("vlm_rationale"),
            # Never recorded in history -- see the module docstring.
            triage_frame_observations=[],
        )
    )


def _judged_rows() -> list[dict]:
    """The newest judged row per alert, which is what the live path would have embedded.

    Only judged alerts are ever embedded (the comparison library is guard-confirmed cases
    only), and an alert judged twice should contribute the verdict that stands, not the one
    it replaced. `all_rows` returns append order, so the last write wins.
    """
    latest: dict[str, dict] = {}
    for row in AlertHistory().all_rows():
        alert_id = row.get("alert_id")
        if alert_id and row.get("operator_verdict"):
            latest[alert_id] = row
    return list(latest.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be embedded, spend nothing.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = all).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = _judged_rows()
    candidates = [(row, _row_to_text(row)) for row in rows]
    # A row whose reason and rationale are both empty has nothing to compare on; embedding an
    # empty string would store a vector that matches everything equally badly.
    candidates = [(row, text) for row, text in candidates if text]
    if args.limit > 0:
        candidates = candidates[: args.limit]

    if not candidates:
        logger.info("No judged incidents with embeddable text. Nothing to do.")
        return 0

    settings = get_settings()
    with_ai = sum(1 for _, text in candidates if "[ai]" in text)
    logger.info(f"Judged incidents to re-embed: {len(candidates)}")
    logger.info(f"  with the VLM's reading:     {with_ai}")
    logger.info(f"  rule text only:             {len(candidates) - with_ai}")
    logger.info(f"Embedding model:              {settings.embedding_model}")
    logger.info(f"Billed API calls this run:    {len(candidates)}")

    if args.dry_run:
        logger.info("\n--dry-run: nothing was embedded and nothing was billed.")
        logger.info("Sample of the text that would be sent:\n")
        for row, text in candidates[:3]:
            logger.info(f"  {row.get('alert_id')}: {text!r}")
        return 0

    store = create_embedding_store()
    if store is None:
        logger.error(
            "Embeddings are off, or DATABASE_URL is not Postgres. "
            "Set EMBEDDINGS_ENABLED=true and point DATABASE_URL at Postgres, then re-run."
        )
        return 1

    embed = build_embedder(settings)
    if embed is None:
        logger.error("No embedder available: check EMBEDDINGS_ENABLED and OPENAI_API_KEY.")
        return 1

    written = 0
    skipped = 0
    for row, text in candidates:
        alert_id = row["alert_id"]
        embedded = embed(text)
        if embedded is None:
            # `build_embedder` already logged the cause. Keep going: a transient failure on
            # one row should not abandon the rest of a run that costs money per row.
            logger.warning(f"  skipped {alert_id}: could not embed")
            skipped += 1
            continue
        vector, model = embedded
        try:
            store.upsert(alert_id, text, vector, model)
        except Exception:  # noqa: BLE001 - one bad row must not end the run
            logger.exception(f"  skipped {alert_id}: could not store")
            skipped += 1
            continue
        written += 1
        if written % 25 == 0:
            logger.info(f"  {written}/{len(candidates)}…")

    logger.info(f"\nRe-embedded {written} incident(s); {skipped} skipped.")
    if skipped:
        logger.info("Re-run to retry the skipped rows -- upsert is idempotent, so rows already done cost nothing new.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

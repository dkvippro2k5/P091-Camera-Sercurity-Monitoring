"""Operator-labelled incident history: the HITL feedback loop's storage.

When a guard marks an alert real or false, that judgement is the most reliable signal this
system will ever get about what its own false alarms look like on a *particular* camera —
one camera watches a stairwell where people crouch, another a corridor where nobody should
be at all. This module persists those judgements and feeds the recent ones back into the
next triage prompt as few-shot examples.

It is also the deliverable. PRD §4 asks for a measured false-alarm filter rate; the rows
here carry the rule engine's `reason`, the model's verdict, and the operator's verdict for
every reviewed alert, which is exactly the labelled dataset that number is computed from.

Two things to be honest about, both of which shaped the "annotate only" decision in
`vlm.py`:

- **Below roughly 50 labels per camera this does approximately nothing.** A handful of
  examples is not enough to shift a vision model's judgement; the value early on is the
  dataset, not the in-loop effect.
- **A feedback loop only ever sees what it surfaced.** If the model suppressed alerts, the
  incidents it wrongly rejected would never be shown, never labelled, and never enter this
  file — the loop would confidently entrench its own blind spot. Keeping every alert visible
  to the operator is what stops that.

JSONL rather than a vector store: the PRD's architecture diagram names a vector DB, but
`chromadb` is not installed, semantic retrieval only beats a plain `(camera, label)` filter
at a scale this PoC will not reach, and a text file is trivially inspectable and exportable.
The migration path stays open — only `recent_examples` would change.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import UTC, datetime

from src.config import get_settings
from src.services.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "data", "vlm_history.jsonl")

# Only the tail is ever read. The file is append-only and grows for the life of the
# deployment, so reading all of it to find six examples would turn every alert into an
# ever-slower disk scan.
_MAX_ROWS_SCANNED = 500

_OPERATOR_LABEL = {"real": "a REAL incident", "false": "a FALSE ALARM"}


class AlertHistory:
    """Append-only log of operator judgements, with few-shot retrieval by camera and label.

    Images are deliberately not stored: they are large, and keeping a captured frame's
    lifetime short is worth doing regardless. Each row references its `alert_id` instead.
    """

    def __init__(self, path: str = DEFAULT_HISTORY_PATH, store: SQLiteStore | None = None) -> None:
        self.path = path
        self.store = store
        if self.store is None and path == DEFAULT_HISTORY_PATH:
            try:
                self.store = SQLiteStore(get_settings().database_url)
            except (OSError, ValueError):
                self.store = None
        # Writes happen on the request thread and reads on a triage worker thread, so the
        # two need to be kept off each other's toes.
        self._lock = threading.Lock()

    def record(
        self,
        alert_id: str,
        camera_id: str,
        label: str,
        operator_verdict: str,
        reason: str | None = None,
        vlm_verdict: str | None = None,
        vlm_confidence: float | None = None,
        vlm_rationale: str | None = None,
        judged_by: str | None = None,
        operator_note: str | None = None,
        actual_event_type: str | None = None,
    ) -> None:
        """Append one reviewed alert. Never raises — a full disk must not break the UI.

        `judged_by` names the operator who gave the judgement. Rows written before accounts
        existed simply lack the key; every read path here and in `eval_report.py` goes through
        `.get()`, so an old row reads back as unattributed rather than breaking the report.
        """
        row = {
            "alert_id": alert_id,
            "camera_id": camera_id,
            "label": label,
            "reason": reason,
            "vlm_verdict": vlm_verdict,
            "vlm_confidence": vlm_confidence,
            "vlm_rationale": vlm_rationale,
            "operator_verdict": operator_verdict,
            "judged_by": judged_by,
            "operator_note": operator_note,
            "actual_event_type": actual_event_type,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        if self.store is not None:
            try:
                self.store.insert_history(row)
                return
            except (OSError, sqlite3.Error):
                logger.exception("Could not write alert history to SQLite")
        try:
            with self._lock:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Could not append to alert history: {exc}")

    def recent_examples(self, camera_id: str, label: str, limit: int = 6) -> list[str]:
        """Rendered one-liners for the last `limit` judgements of this label on this camera.

        Filtered by both keys on purpose. A fall on the stairwell camera says nothing useful
        about an abandoned bag in the lobby, and mixing them would spend prompt tokens
        actively misleading the model.
        """
        if limit <= 0:
            return []

        source_rows = (self.store.history() + self._read_tail()) if self.store is not None else self._read_tail()
        rows = [
            row
            for row in source_rows
            if row.get("camera_id") == camera_id and row.get("label") == label and row.get("operator_verdict")
        ]
        # Oldest-first among the selected few: recency reads as "most relevant" to a model
        # scanning down the list, so the newest judgement should be the last thing it sees.
        return [self._render(row) for row in rows[-limit:]]

    def all_rows(self) -> list[dict]:
        """Every readable row — for computing the precision/recall the PRD asks for."""
        return (self.store.history() + self._read_tail(limit=None)) if self.store is not None else self._read_tail(limit=None)

    def _read_tail(self, limit: int | None = _MAX_ROWS_SCANNED) -> list[dict]:
        """Parse the file, skipping any corrupt line rather than losing the whole history."""
        if not os.path.exists(self.path):
            return []
        try:
            with self._lock, open(self.path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:
            logger.warning(f"Could not read alert history: {exc}")
            return []

        rows = []
        for line in lines if limit is None else lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially-written final line is normal if the process died mid-append.
                continue
        return rows

    @staticmethod
    def _render(row: dict) -> str:
        """One line the prompt can carry: what fired, and what the guard concluded."""
        verdict = _OPERATOR_LABEL.get(row.get("operator_verdict", ""), row.get("operator_verdict", "unknown"))
        reason = row.get("reason") or "no rule evidence recorded"
        note = row.get("operator_note")
        correction = row.get("actual_event_type")
        suffix = f" Note: {note}" if note else ""
        if correction:
            suffix += f" Actual event: {correction}."
        return f'Rule fired with "{reason}" — the guard judged it {verdict}.{suffix}'

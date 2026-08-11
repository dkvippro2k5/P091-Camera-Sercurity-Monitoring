"""Every alert that fired, reviewed or not — the operational counterpart to `alert_history`.

These two files look similar and answer opposite questions, so the split is worth stating:

- `data/vlm_history.jsonl` (`alert_history.py`) holds **only reviewed** alerts. That is
  deliberate — it is the dataset precision is computed from, and an unlabelled row would be
  counted as neither a hit nor a miss. `acknowledge` writes nothing to it for exactly this
  reason.
- `data/incidents.jsonl` (here) holds **every** alert at the moment it fires, before anyone
  has looked at it.

So "how many incidents did camera 3 raise last night" cannot be answered from the first file:
alerts nobody judged are absent from it, and the ones that are present are the ones a guard
had time for — which is a biased sample of exactly the quiet-shift/busy-shift difference a
manager is trying to see. Counting from there would silently under-report, most on the
busiest nights.

Rows are written twice: once when the alert fires, and once more when a guard judges it, to
stamp the verdict and who gave it. Rather than rewriting the file in place, the update is
appended as a second row with the same `alert_id` and the reader folds them together — an
append is atomic enough under a lock and cannot corrupt earlier rows, while a rewrite of a
growing file on the request path can.
"""

import json
import logging
import os
import threading
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INCIDENT_LOG_PATH = os.path.join(_PROJECT_ROOT, "data", "incidents.jsonl")


class IncidentLog:
    """Append-only log of every alert, with a fold-on-read for verdict updates."""

    def __init__(self, path: str = DEFAULT_INCIDENT_LOG_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record_alert(self, alert_id: str, camera_id: str, camera_name: str, label: str, confidence: float) -> None:
        """One row as the alert fires. Never raises — a full disk must not break alerting.

        Called from `_emit_alert` on the camera's processor thread, which must not block, so
        this stays a single small buffered write under a lock and nothing more.
        """
        self._append(
            {
                "kind": "alert",
                "alert_id": alert_id,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "label": label,
                "confidence": confidence,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

    def record_review(self, alert_id: str, verdict: str, judged_by: str | None) -> None:
        """A second row stamping the guard's judgement onto an existing alert."""
        self._append(
            {
                "kind": "review",
                "alert_id": alert_id,
                "operator_verdict": verdict,
                "judged_by": judged_by,
                "reviewed_at": datetime.now(UTC).isoformat(),
            }
        )

    def _append(self, row: dict) -> None:
        try:
            with self._lock:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Could not append to incident log: {exc}")

    def _read(self) -> list[dict]:
        """Every row, skipping corrupt lines rather than losing the whole log."""
        if not os.path.exists(self.path):
            return []
        try:
            with self._lock, open(self.path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:
            logger.warning(f"Could not read incident log: {exc}")
            return []

        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially-written final line is normal if the process died mid-append.
                continue
        return rows

    def incidents(self) -> list[dict]:
        """Alerts with their reviews folded in, oldest first.

        A review whose alert row is missing is dropped rather than invented: it means the log
        was truncated or hand-edited, and a synthetic alert with no camera or label would
        quietly corrupt every per-camera count computed from it.
        """
        alerts: dict[str, dict] = {}
        reviews: dict[str, dict] = {}
        for row in self._read():
            alert_id = row.get("alert_id")
            if not alert_id:
                continue
            if row.get("kind") == "review":
                # Last review wins — a guard correcting their own judgement should not be
                # counted twice.
                reviews[alert_id] = row
            else:
                alerts[alert_id] = row

        folded = []
        for alert_id, alert in alerts.items():
            review = reviews.get(alert_id)
            folded.append(
                {
                    **alert,
                    "operator_verdict": (review or {}).get("operator_verdict"),
                    "judged_by": (review or {}).get("judged_by"),
                    "reviewed_at": (review or {}).get("reviewed_at"),
                }
            )
        folded.sort(key=lambda row: row.get("created_at") or "")
        return folded

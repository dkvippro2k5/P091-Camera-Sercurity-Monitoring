"""Asynchronous incident notifications.

Zalo is deliberately an adapter here: credentials and recipient IDs stay in environment
configuration, and a failed notification is logged without affecting camera processing.
"""

import json
import logging
import queue
import threading
import time
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class ZaloNotifier:
    endpoint = "https://openapi.zalo.me/v3.0/oa/message/cs"

    def __init__(self, enabled: bool, access_token: str, recipients: str) -> None:
        self.enabled = enabled and bool(access_token)
        self.access_token = access_token
        self.recipients = [item.strip() for item in recipients.split(",") if item.strip()]

    def send_alert(self, alert: dict) -> bool:
        if not self.enabled or not self.recipients:
            return True
        message = (
            f"[{alert.get('label', 'incident')}] {alert.get('camera_name', alert.get('camera_id', 'camera'))} "
            f"at {alert.get('created_at', '')}. Confidence: {float(alert.get('confidence', 0)):.0%}."
        )
        succeeded = True
        for recipient in self.recipients:
            payload = json.dumps({"recipient": {"user_id": recipient}, "message": {"text": message}}).encode()
            request = Request(
                self.endpoint,
                data=payload,
                headers={"access_token": self.access_token, "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=8) as response:  # noqa: S310 - fixed HTTPS endpoint
                    if response.status >= 300:
                        raise RuntimeError(f"HTTP {response.status}")
            except Exception as exc:  # noqa: BLE001 - notification must never break monitoring
                succeeded = False
                logger.warning("Zalo notification failed for %s: %s", recipient, exc)
        return succeeded


class NotificationWorker:
    """Bounded retry worker so notification delivery never blocks detection."""

    def __init__(self, notifier: ZaloNotifier, publish, max_retries: int = 3, backoff_sec: float = 2.0) -> None:
        self.notifier = notifier
        self.publish = publish
        self.max_retries = max(0, max_retries)
        self.backoff_sec = max(0.1, backoff_sec)
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="notification-worker", daemon=True)
        self._thread.start()

    def submit(self, alert: dict) -> None:
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            logger.error("Notification queue full; dropping notification for %s", alert.get("id"))
            self.publish("notification.status", {"alert_id": alert.get("id"), "status": "dropped"})

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Notification worker did not stop within 5s")

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                alert = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            alert_id = alert.get("id")
            delivered = False
            attempts = 0
            for attempts in range(1, self.max_retries + 2):
                delivered = self.notifier.send_alert(alert)
                if delivered:
                    break
                if attempts <= self.max_retries:
                    time.sleep(self.backoff_sec * (2 ** (attempts - 1)))
            self.publish(
                "notification.status",
                {"alert_id": alert_id, "status": "delivered" if delivered else "failed", "attempts": attempts},
            )
            self._queue.task_done()

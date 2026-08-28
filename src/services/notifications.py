"""Asynchronous Telegram incident notifications.

Credentials and recipient IDs stay in environment configuration, and a failed notification
is logged without affecting camera processing.
"""

import json
import logging
import queue
import threading
import time
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send confirmed incidents to Telegram users and groups."""

    def __init__(self, enabled: bool, bot_token: str, chat_ids: str) -> None:
        self.enabled = enabled and bool(bot_token)
        self.bot_token = bot_token
        self.chat_ids = [item.strip() for item in chat_ids.split(",") if item.strip()]
        self.endpoint = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send_alert(self, alert: dict) -> bool:
        if not self.enabled or not self.chat_ids:
            return True

        label = alert.get("label", "incident")
        label_text, severity = {
            "fall_detection": ("PHÁT HIỆN NGÃ", "Nghiêm trọng"),
            "intrusion": ("PHÁT HIỆN XÂM NHẬP", "Cao"),
            "abandoned_object": ("PHÁT HIỆN ĐỒ VẬT BỊ BỎ LẠI", "Trung bình"),
        }.get(label, ("CẢNH BÁO AN NINH", "Cần kiểm tra"))
        rationale = str(alert.get("triage_rationale") or "").strip()
        observations = [
            str(item).strip()
            for item in (alert.get("triage_frame_observations") or [])
            if str(item).strip()
        ]
        # Keep the message useful on a phone without dumping every sampled frame note.
        evidence = "\n".join(f"• {item}" for item in observations[:6])
        context = f"\n\n*Mô tả:*\n{rationale}" if rationale else ""
        if evidence:
            context += f"\n\n*Diễn biến quan sát:*\n{evidence}"
        message = (
            f"🚨 *{label_text}*\n"
            f"📍 Camera: {alert.get('camera_name', alert.get('camera_id', 'không xác định'))}\n"
            f"🕒 Thời gian: {alert.get('created_at', 'không xác định')}\n"
            f"⚠️ Mức độ: *{severity}*\n"
            f"🤖 Độ tin cậy VLM: {float(alert.get('triage_confidence') or 0):.0%}\n"
            f"🔎 Độ tin cậy detector: {float(alert.get('confidence', 0)):.0%}"
            f"{context}\n\nVui lòng kiểm tra camera và xử lý nếu cần."
        )
        # Telegram Markdown requires escaping only when user/model text contains formatting
        # punctuation; use plain text for model-generated evidence to avoid broken messages.
        payload_text = message.replace("*", "")
        message = payload_text
        succeeded = True
        for chat_id in self.chat_ids:
            payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}).encode()
            request = Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(request, timeout=8) as response:  # noqa: S310 - fixed HTTPS endpoint
                    if response.status >= 300:
                        raise RuntimeError(f"HTTP {response.status}")
                    body = json.loads(response.read() or b"{}")
                    if not body.get("ok", False):
                        raise RuntimeError(body.get("description", "Telegram rejected message"))
            except Exception as exc:  # noqa: BLE001 - notification must never break monitoring
                succeeded = False
                logger.warning("Telegram notification failed for %s: %s", chat_id, exc)
        return succeeded


# Backward-compatible import for existing integrations.
ZaloNotifier = TelegramNotifier


def is_telegram_eligible(alert: dict) -> bool:
    return alert.get("triage_status") == "complete" and alert.get("triage_verdict") == "real"


def should_notify(notifier, alert: dict) -> bool:
    return bool(notifier.enabled and is_telegram_eligible(alert))


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

"""Asynchronous incident notifications.

Zalo is deliberately an adapter here: credentials and recipient IDs stay in environment
configuration, and a failed notification is logged without affecting camera processing.
"""

import json
import logging
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class ZaloNotifier:
    endpoint = "https://openapi.zalo.me/v3.0/oa/message/cs"

    def __init__(self, enabled: bool, access_token: str, recipients: str) -> None:
        self.enabled = enabled and bool(access_token)
        self.access_token = access_token
        self.recipients = [item.strip() for item in recipients.split(",") if item.strip()]

    def send_alert(self, alert: dict) -> None:
        if not self.enabled or not self.recipients:
            return
        message = (
            f"[{alert.get('label', 'incident')}] {alert.get('camera_name', alert.get('camera_id', 'camera'))} "
            f"at {alert.get('created_at', '')}. Confidence: {float(alert.get('confidence', 0)):.0%}."
        )
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
                logger.warning("Zalo notification failed for %s: %s", recipient, exc)

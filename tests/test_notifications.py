import time

from src.services.notifications import NotificationWorker


class _FlakyNotifier:
    enabled = True

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def send_alert(self, alert: dict) -> bool:
        self.calls += 1
        return self.calls > self.failures


def _wait_for(events: list[tuple[str, dict]], status: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        for event_type, payload in events:
            if event_type == "notification.status" and payload.get("status") == status:
                return payload
        time.sleep(0.01)
    raise AssertionError(f"No {status} notification event: {events}")


def test_notification_worker_retries_and_reports_delivery():
    notifier = _FlakyNotifier(failures=2)
    events: list[tuple[str, dict]] = []
    worker = NotificationWorker(notifier, lambda kind, payload: events.append((kind, payload)), max_retries=3, backoff_sec=0.01)
    worker.submit({"id": "alert-1"})
    try:
        payload = _wait_for(events, "delivered")
        assert payload["attempts"] == 3
        assert notifier.calls == 3
    finally:
        worker.close()


def test_notification_worker_reports_failure_after_retry_budget():
    notifier = _FlakyNotifier(failures=99)
    events: list[tuple[str, dict]] = []
    worker = NotificationWorker(notifier, lambda kind, payload: events.append((kind, payload)), max_retries=1, backoff_sec=0.01)
    worker.submit({"id": "alert-2"})
    try:
        payload = _wait_for(events, "failed")
        assert payload["attempts"] == 2
        assert notifier.calls == 2
    finally:
        worker.close()

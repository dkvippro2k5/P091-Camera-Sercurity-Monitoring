import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.services.notifications import (
    CompositeNotifier,
    EmailNotifier,
    NotificationWorker,
    ZaloNotifier,
    build_notifier,
)


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


def test_email_notifier_disabled_without_credentials_is_a_noop():
    notifier = EmailNotifier(True, host="", port=587, username="", password="", from_address="", to_addresses="")
    assert notifier.enabled is False
    assert notifier.send_alert({"id": "a1"}) is True


def test_email_notifier_sends_via_smtp():
    notifier = EmailNotifier(
        True,
        host="smtp.example.com",
        port=587,
        username="user",
        password="pw",
        from_address="alerts@example.com",
        to_addresses="ops@example.com, oncall@example.com",
    )
    assert notifier.enabled is True
    with patch("src.services.notifications.smtplib.SMTP") as smtp_cls:
        server = MagicMock()
        smtp_cls.return_value.__enter__.return_value = server
        assert notifier.send_alert({"id": "a1", "label": "fall", "camera_id": "cam1"}) is True
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("user", "pw")
        server.sendmail.assert_called_once()
        assert server.sendmail.call_args[0][1] == ["ops@example.com", "oncall@example.com"]


def test_email_notifier_returns_false_on_smtp_failure():
    notifier = EmailNotifier(
        True, host="smtp.example.com", port=587, username="", password="", from_address="a@b.com", to_addresses="c@d.com"
    )
    with patch("src.services.notifications.smtplib.SMTP", side_effect=OSError("connection refused")):
        assert notifier.send_alert({"id": "a1"}) is False


def test_composite_notifier_requires_all_providers_to_succeed():
    ok = MagicMock(spec=["send_alert"])
    ok.send_alert.return_value = True
    fail = MagicMock(spec=["send_alert"])
    fail.send_alert.return_value = False
    composite = CompositeNotifier([ok, fail])
    assert composite.send_alert({"id": "a1"}) is False
    ok.send_alert.assert_called_once()
    fail.send_alert.assert_called_once()


def test_composite_notifier_with_no_providers_is_a_noop():
    composite = CompositeNotifier([])
    assert composite.send_alert({"id": "a1"}) is True
    assert composite.enabled is False


def test_composite_notifier_enabled_if_any_provider_enabled():
    disabled = MagicMock(spec=["send_alert", "enabled"])
    disabled.enabled = False
    enabled = MagicMock(spec=["send_alert", "enabled"])
    enabled.enabled = True
    assert CompositeNotifier([disabled, enabled]).enabled is True
    assert CompositeNotifier([disabled]).enabled is False


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        notification_channels="zalo",
        zalo_enabled=False,
        zalo_access_token="",
        zalo_recipients="",
        smtp_enabled=False,
        smtp_host="",
        smtp_port=587,
        smtp_username="",
        smtp_password="",
        smtp_from_address="",
        smtp_to_addresses="",
        smtp_use_tls=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_notifier_only_wires_listed_channels():
    composite = build_notifier(_settings(notification_channels="email"))
    assert len(composite.providers) == 1
    assert isinstance(composite.providers[0], EmailNotifier)


def test_build_notifier_wires_multiple_channels():
    composite = build_notifier(_settings(notification_channels="email, zalo"))
    kinds = {type(p) for p in composite.providers}
    assert kinds == {EmailNotifier, ZaloNotifier}

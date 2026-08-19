"""Optional Redis relay for cross-process monitoring events.

The local queue remains the fast path. Redis mirrors each event to other API/worker
processes, while an instance ignores its own messages to avoid duplicate deliveries.
"""

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable

logger = logging.getLogger(__name__)


class NullEventRelay:
    def publish(self, message: dict) -> None:
        del message

    def close(self) -> None:
        return


class RedisEventRelay:
    def __init__(self, url: str, channel: str, on_remote: Callable[[dict], None]) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - only used in production images
            raise RuntimeError("Install redis to enable the distributed event relay") from exc

        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._channel = channel
        self._on_remote = on_remote
        self._source = uuid.uuid4().hex
        self._outbox: queue.Queue[dict] = queue.Queue(maxsize=1000)
        self._stop = threading.Event()
        self._publisher = threading.Thread(target=self._publish_loop, name="redis-event-publisher", daemon=True)
        self._subscriber = threading.Thread(target=self._subscribe_loop, name="redis-event-subscriber", daemon=True)
        self._publisher.start()
        self._subscriber.start()

    def publish(self, message: dict) -> None:
        envelope = {"source": self._source, "message": message}
        try:
            self._outbox.put_nowait(envelope)
        except queue.Full:
            logger.warning("Redis event relay queue full; dropping %s", message.get("type"))

    def close(self) -> None:
        self._stop.set()

    def _publish_loop(self) -> None:
        while not self._stop.is_set():
            try:
                envelope = self._outbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._redis.publish(self._channel, json.dumps(envelope))
            except Exception:  # noqa: BLE001
                logger.exception("Could not publish monitoring event to Redis")

    def _subscribe_loop(self) -> None:
        while not self._stop.is_set():
            pubsub = None
            try:
                pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(self._channel)
                while not self._stop.is_set():
                    item = pubsub.get_message(timeout=1.0)
                    if not item:
                        continue
                    envelope = json.loads(item["data"])
                    if envelope.get("source") != self._source:
                        self._on_remote(envelope["message"])
            except Exception:  # noqa: BLE001
                logger.exception("Redis event subscriber disconnected; retrying")
                time.sleep(2.0)
            finally:
                if pubsub is not None:
                    try:
                        pubsub.close()
                    except Exception:  # noqa: BLE001
                        pass


def build_event_relay(url: str, channel: str, on_remote: Callable[[dict], None]):
    return RedisEventRelay(url, channel, on_remote) if url else NullEventRelay()

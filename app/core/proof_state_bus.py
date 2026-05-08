from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Callable, DefaultDict


ProofStateCallback = Callable[[object], None]

TOPIC_LINE_PROOF_CHANGED = "line.proof_changed"
TOPIC_PAGE_PROGRESS = "page.progress"
TOPIC_PROJECT_STATS = "project.stats"


class ProofStateBus:
    """纯 Python 的全局校对事件总线。"""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, list[ProofStateCallback]] = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, topic: str, callback: ProofStateCallback) -> Callable[[], None]:
        with self._lock:
            self._subscribers[topic].append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                callbacks = self._subscribers.get(topic, [])
                if callback in callbacks:
                    callbacks.remove(callback)
                if not callbacks and topic in self._subscribers:
                    del self._subscribers[topic]

        return _unsubscribe

    def publish(self, topic: str, data: object) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(topic, ()))
        for callback in callbacks:
            callback(data)

    def subscriber_count(self, topic: str) -> int:
        with self._lock:
            return len(self._subscribers.get(topic, ()))

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()


_DEFAULT_BUS = ProofStateBus()


def get_proof_state_bus() -> ProofStateBus:
    return _DEFAULT_BUS

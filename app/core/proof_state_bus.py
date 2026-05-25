"""校对事件总线：轻量发布/订阅，用于跨面板通知状态变更。

用法（新式，推荐）：
    from app.core.proof_state_bus import TOPIC_LINE_PROOF_CHANGED, get_proof_state_bus

    bus = get_proof_state_bus()
    unsub = bus.subscribe(TOPIC_LINE_PROOF_CHANGED, my_handler)
    bus.publish(TOPIC_LINE_PROOF_CHANGED, {"page_id": 1, "line_id": 5, "status": "OK"})
    unsub()  # 取消订阅

用法（旧式，仍兼容）：
    from app.core.proof_state_bus import ProofStateBus

    bus = ProofStateBus.instance()
    bus.subscribe("line.proof_changed", my_handler)
    bus.publish("line.proof_changed", page_id=1, line_id=5, status="OK")
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from app.core.proof_state import (
    TOPIC_LINE_PROOF_CHANGED,
    TOPIC_PROBE_OBSERVED,
    ProbeObservation,
    ProofUpdateRequest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class ProofStateBus:
    """单例事件总线。

    publish 支持两种调用形式：
      bus.publish(topic, payload_dict)   → handler(payload_dict)
      bus.publish(topic, **kwargs)       → handler(**kwargs)
    """

    _instance: "ProofStateBus | None" = None

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[..., Any]]] = {}

    # ------------------------------------------------------------------ 单例

    @classmethod
    def instance(cls) -> "ProofStateBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """测试用：重置单例。"""
        cls._instance = None

    # ------------------------------------------------------------------ API

    def subscribe(self, event: str, handler: Callable[..., Any]) -> Callable[[], None]:
        """注册事件处理函数。

        返回一个无参 callable，调用后取消订阅。
        """
        self._subscribers.setdefault(event, []).append(handler)

        def _unsubscribe() -> None:
            self.unsubscribe(event, handler)

        return _unsubscribe

    def unsubscribe(self, event: str, handler: Callable[..., Any]) -> None:
        """取消注册。若 handler 未注册则忽略。"""
        bucket = self._subscribers.get(event)
        if bucket:
            try:
                bucket.remove(handler)
            except ValueError:
                pass

    def publish(self, event: str, payload: Optional[Any] = None, **kwargs: Any) -> None:
        """发布事件，同步调用所有订阅者。handler 异常不中止其他 handler。

        支持两种形式：
          publish(topic, {"key": "value"})  → handler({"key": "value"})
          publish(topic, key="value")       → handler(key="value")
        """
        handlers = list(self._subscribers.get(event, []))
        for handler in handlers:
            try:
                if payload is not None:
                    handler(payload)
                elif kwargs:
                    handler(**kwargs)
                else:
                    handler()
            except TypeError as exc:
                if payload is not None and hasattr(payload, "to_legacy_payload"):
                    try:
                        handler(**payload.to_legacy_payload())
                        continue
                    except Exception as legacy_exc:  # noqa: BLE001
                        logger.warning("ProofStateBus legacy handler error [%s]: %s", event, legacy_exc)
                        continue
                logger.warning("ProofStateBus handler error [%s]: %s", event, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ProofStateBus handler error [%s]: %s", event, exc)

    def publish_line_update(self, request: ProofUpdateRequest) -> None:
        """Publish a typed proof line update."""
        self.publish(TOPIC_LINE_PROOF_CHANGED, request)

    def publish_probe_observed(self, observation: ProbeObservation) -> None:
        """Publish a typed quality-probe observation."""
        self.publish(TOPIC_PROBE_OBSERVED, observation)

    def subscriber_count(self, event: str) -> int:
        """返回指定事件的当前订阅者数量。"""
        return len(self._subscribers.get(event, []))

    def clear(self) -> None:
        """清除所有订阅（测试用）。"""
        self._subscribers.clear()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_proof_state_bus() -> ProofStateBus:
    """返回全局单例 ProofStateBus。"""
    return ProofStateBus.instance()

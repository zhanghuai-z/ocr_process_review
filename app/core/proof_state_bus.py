"""校对事件总线：轻量发布/订阅，用于跨面板通知状态变更。

用法：
    from app.core.proof_state_bus import ProofStateBus

    # 订阅
    bus = ProofStateBus.instance()
    bus.subscribe("line.proof_changed", my_handler)

    # 发布
    bus.publish("line.proof_changed", page_id=1, line_id=5, status="OK")
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class ProofStateBus:
    """单例事件总线。"""

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

    def subscribe(self, event: str, handler: Callable[..., Any]) -> None:
        """注册事件处理函数。同一 handler 可重复订阅同一事件（会执行多次）。"""
        self._subscribers.setdefault(event, []).append(handler)

    def unsubscribe(self, event: str, handler: Callable[..., Any]) -> None:
        """取消注册。若 handler 未注册则忽略。"""
        bucket = self._subscribers.get(event)
        if bucket:
            try:
                bucket.remove(handler)
            except ValueError:
                pass

    def publish(self, event: str, **kwargs: Any) -> None:
        """发布事件，同步调用所有订阅者。handler 异常不中止其他 handler。"""
        for handler in list(self._subscribers.get(event, [])):
            try:
                handler(**kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ProofStateBus handler error [%s]: %s", event, exc)

    def clear(self) -> None:
        """清除所有订阅（测试用）。"""
        self._subscribers.clear()

"""每字 verdict —— 颜色 + 证据链（hproof-yaxis-verdicts 第三任务）。

设计原则
========

用户在本轮 TASK 明确要求：黑 / 绿 / 橙 / 红的每一档都要有"证据来源"，
不能因为"看起来像那种正确"就硬编码冒充。

具体到目前 codebase 可见的数据源：

  - ``Char.confidence``        OCR 自评（每字一个 [0,1] 分数）
  - ``Line.ocr_text``          OCR 原始识别结果（用于判断"用户是否改过"）
  - ``Line.text``              当前显示的文本（可能已被用户编辑覆写）

** 缺失但用户期望存在的判定链 **（写在 handoff，不在本模块里造假）：

  - "黑 = 校对过且可视作绝对正确" 当前 **没有** 可靠的证据来源：
      a) ``app.core.quality_probe`` 本轮禁止修改；
      b) ``Line`` 模型里没有"此字已被人工核验"的字段；
      c) ``proof_status`` 只是行级的标记，不到字粒度。
    → 本模块**不输出** "absolute_correct" 这一档。
      默认 fallback 是 ``SEVERITY_UNVERIFIED``，颜色用普通深灰（不冒充"已通过校对"）。

  - "绿 = 程序判断正确" 当前只有一条证据：``Char.confidence >= 0.95``。
    这只是 OCR 自评，并不保证人工复核会同意。
    → severity 标 ``likely_ok``（不是 ``correct``）；handoff 里说明这一点。

  - "橙 / 红 = 仍参与正确性判断不能因人工改过自动洗白" → 颜色 **不读** 用户是否修改过；
    `user_modified` 只是一个并列的行为标记（UI 里画下划线），不会把颜色从红/橙变成绿/黑。

颜色映射
========

============ ================================== ==========================
颜色         证据                                 severity
============ ================================== ==========================
绿 #2e7d32   ocr_conf >= 0.95                    likely_ok
默认深灰     0.80 <= ocr_conf < 0.95              unverified
橙 #e8801f   0.50 <= ocr_conf < 0.80              suspect
红 #c62828   ocr_conf < 0.50                     error
============ ================================== ==========================

`user_modified` 单独成一个 bool 字段：UI 在该字下方画一条细线提示"这字被改过"，
但 **不改变** color/severity —— 这是用户本轮显式要求的"不洗白"语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


COLOR_LIKELY_OK   = "#2e7d32"   # 绿 —— 程序判断（OCR conf）较高
COLOR_UNVERIFIED  = "#222222"   # 默认深灰 —— 既无明显问题、也无"绝对正确"证据链
COLOR_SUSPECT     = "#e8801f"   # 橙 —— 可疑
COLOR_ERROR       = "#c62828"   # 红 —— 高风险 / 双源不一致

SEVERITY_LIKELY_OK  = "likely_ok"
SEVERITY_UNVERIFIED = "unverified"
SEVERITY_SUSPECT    = "suspect"
SEVERITY_ERROR      = "error"


@dataclass(frozen=True)
class CharVerdict:
    """单字 verdict：颜色 + severity + 证据描述 + 是否被人工改过。"""
    color: str
    severity: str
    evidence: str
    user_modified: bool = False


def classify_char(
    *,
    confidence: Optional[float],
    text_char: str,
    ocr_char: Optional[str],
) -> CharVerdict:
    """按上文表格分类。

    Parameters
    ----------
    confidence : Optional[float]
        ``Char.confidence``。None / 缺失时归入 ``unverified``，证据描述写明缺失。
    text_char : str
        当前显示文本中该位置的字（用于和 OCR 原文比较）。
    ocr_char : Optional[str]
        OCR 原始字（``Line.ocr_text[i]``）。None 表示未取到；
        ``user_modified`` 仅在拿得到时才可能为 True。
    """
    user_modified = ocr_char is not None and text_char != ocr_char

    if confidence is None:
        return CharVerdict(
            color=COLOR_UNVERIFIED,
            severity=SEVERITY_UNVERIFIED,
            evidence="ocr_conf 缺失 —— 无法判定，按未核验处理",
            user_modified=user_modified,
        )

    if confidence < 0.50:
        ev = f"ocr_conf={confidence:.2f} <0.50（高风险）"
        return CharVerdict(COLOR_ERROR, SEVERITY_ERROR, ev, user_modified)

    if confidence < 0.80:
        ev = f"ocr_conf={confidence:.2f} <0.80（可疑）"
        return CharVerdict(COLOR_SUSPECT, SEVERITY_SUSPECT, ev, user_modified)

    if confidence < 0.95:
        ev = (
            f"ocr_conf={confidence:.2f}（0.80~0.95）；"
            "无可用证据链可上升为'绝对正确'"
        )
        return CharVerdict(COLOR_UNVERIFIED, SEVERITY_UNVERIFIED, ev, user_modified)

    # confidence >= 0.95
    ev = f"ocr_conf={confidence:.2f} ≥0.95（程序判断正确）"
    return CharVerdict(COLOR_LIKELY_OK, SEVERITY_LIKELY_OK, ev, user_modified)

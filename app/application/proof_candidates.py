"""Candidate-glyph suggestion policy for vertical proofing.

This is a business suggestion strategy (which glyphs are plausible
replacements and in what trust order), deliberately kept out of the UI:
panels consume ``ranked_candidates`` instead of owning the confusables
table or the ranking rules.
"""
from __future__ import annotations

DEFAULT_CONFUSABLE_CANDIDATES = {
    # 田/由/甲/申 系
    "田": ["由", "甲", "申", "曲"],
    "由": ["田", "甲", "申"],
    "甲": ["田", "由", "申"],
    "申": ["田", "由", "甲"],
    # 日/曰/目/口 系
    "日": ["曰", "目", "口"],
    "曰": ["日", "目", "口"],
    "目": ["日", "曰", "自", "首"],
    "口": ["日", "曰", "囗", "回"],
    "囗": ["口", "回", "国"],
    # 己/已/巳 系
    "己": ["已", "巳"],
    "已": ["己", "巳"],
    "巳": ["己", "已"],
    # 末/未 系
    "未": ["末", "朱"],
    "末": ["未", "朱"],
    "朱": ["未", "末"],
    # 人/入/八 系
    "人": ["入", "八", "个"],
    "入": ["人", "八"],
    "八": ["人", "入"],
    # 大/太/犬 系
    "大": ["太", "犬", "夫"],
    "太": ["大", "犬", "夫"],
    "犬": ["大", "太"],
    "夫": ["大", "天", "夭"],
    "天": ["夫", "夭", "无"],
    "夭": ["天", "夫"],
    # 干/千/午/壬 系
    "干": ["千", "午", "于"],
    "千": ["干", "午"],
    "午": ["干", "千", "牛"],
    "牛": ["午", "牟"],
    # 土/士/王/玉/主/王 系
    "土": ["士", "工"],
    "士": ["土", "仕"],
    "王": ["玉", "主", "壬"],
    "玉": ["王", "主"],
    "主": ["王", "玉", "住"],
    "壬": ["王", "工", "土"],
    # 工/匚 系
    "工": ["土", "士", "壬"],
    # 又/叉/义 系
    "又": ["叉", "义"],
    "叉": ["又"],
    "义": ["又", "乂"],
    # 力/刀/办 系
    "力": ["刀", "办"],
    "刀": ["力", "刃"],
    "刃": ["刀"],
    # 木/术/本/朩
    "木": ["术", "本", "朩"],
    "术": ["木", "朮"],
    "本": ["木", "未", "末"],
    # 水/氺/永/冰
    "水": ["氺", "永", "冰"],
    "永": ["水", "求"],
    "冰": ["水", "永"],
    # 火/灬
    "火": ["灬", "炎"],
    # 心/必
    "心": ["必", "忄"],
    "必": ["心"],
    # 巾/币/市
    "巾": ["币", "市", "布"],
    "币": ["巾", "市"],
    "市": ["巾", "币", "布"],
    "布": ["巾", "市"],
    # 戊/戌/戍/戎/成
    "戊": ["戌", "戍", "戎", "成"],
    "戌": ["戊", "戍", "戎"],
    "戍": ["戊", "戌", "戎"],
    "戎": ["戊", "戌", "戍"],
    "成": ["戊", "戌"],
    # 凡/几/丸
    "凡": ["几", "丸"],
    "几": ["凡", "九"],
    "丸": ["凡", "九"],
    "九": ["几", "丸"],
    # 北/比/此
    "北": ["比", "兆"],
    "比": ["北", "此"],
    "此": ["比"],
    # 卜/上/下/不
    "卜": ["上", "下", "不"],
    "上": ["卜", "下"],
    "下": ["卜", "上"],
    "不": ["卜", "丕"],
    # 千/午/牛/年
    "年": ["午", "牛"],
    # 自/白/百
    "自": ["白", "百", "目"],
    "白": ["自", "百"],
    "百": ["白", "自"],
    # 干/于/亏
    "于": ["干", "亏", "乎"],
    "亏": ["于", "夸"],
    # 风/凤/凡
    "风": ["凤", "凡"],
    "凤": ["风", "凡"],
    # 鸟/乌/马
    "鸟": ["乌"],
    "乌": ["鸟"],
    # 兔/免
    "兔": ["免", "兑"],
    "免": ["兔"],
    # 衣/农/表
    "衣": ["农", "表"],
    "农": ["衣"],
    "表": ["衣"],
    # 万/方
    "万": ["方"],
    "方": ["万"],
    # 历/厉
    "历": ["厉"],
    "厉": ["历"],
    # 京/亨/享
    "京": ["亨", "享", "亰"],
    "亨": ["京", "享"],
    "享": ["京", "亨"],
    # 长
    "长": ["镸"],
    # 兵/丘
    "兵": ["丘"],
    "丘": ["兵"],
    # 体/休
    "休": ["体"],
    "体": ["休"],
    # 化/华
    "化": ["华"],
    "华": ["化"],
    # 今/令/兮
    "今": ["令", "兮"],
    "令": ["今"],
    "兮": ["今"],
    # 半/羊/丰
    "半": ["羊", "丰"],
    "丰": ["半", "羊"],
    "羊": ["半", "丰"],
    # 兄/见/贝/页
    "兄": ["见", "克"],
    "见": ["兄", "贝"],
    "贝": ["见", "页"],
    "页": ["贝", "顶"],
    # 圆/园/团
    "圆": ["园", "团"],
    "园": ["圆"],
    "团": ["圆", "园"],
    # 句/旬/勺
    "句": ["旬", "勺"],
    "旬": ["句"],
    "勺": ["句"],
}


def ranked_candidates(text: str, ocr_char: str | None) -> tuple[str, ...]:
    """Aggregate candidates from high to low trust, de-duplicated.

    Priority: the current glyph itself, the OCR source glyph at the same
    position, then first- and second-level confusables.
    """

    ranked: list[str] = []

    def push(value: str | None) -> None:
        if value and value not in ranked:
            ranked.append(value)

    push(text)
    push(ocr_char)
    first_level = DEFAULT_CONFUSABLE_CANDIDATES.get(text, [])
    for value in first_level:
        push(value)
    for first in first_level:
        for second in DEFAULT_CONFUSABLE_CANDIDATES.get(first, []):
            push(second)
    return tuple(ranked)


def has_confusable_entry(text: str) -> bool:
    """Whether the confusables table covers one glyph."""

    return text in DEFAULT_CONFUSABLE_CANDIDATES


__all__ = [
    "DEFAULT_CONFUSABLE_CANDIDATES",
    "has_confusable_entry",
    "ranked_candidates",
]

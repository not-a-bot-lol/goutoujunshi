# -*- coding: utf-8 -*-
"""从当前可见对话读出这一轮的目标和对方态度。不另调模型，也不把态度写成成功率。"""
from __future__ import annotations

try:
    from .goutou import explicit_boundary
except ImportError:
    from goutou import explicit_boundary

_NOTE = "根据当前可见对话推测，不是对方的真实心情。"
_WARM = ("哈哈", "好呀", "好啊", "想见", "想你", "喜欢", "可爱", "么么", "❤", "😊", "😄")
_UPSET = ("烦", "算了", "不用了", "别说了", "不想理", "生气", "失望", "又忘", "你到底")
_MEET = ("见面", "出来", "明天", "周末", "有空", "约")


def _parts(message):
    if isinstance(message, dict):
        return message.get("from"), str(message.get("text") or "")
    return message[0], str(message[1] if len(message) > 1 else "")


def read_rapport(messages, answers=None, strategy=None) -> dict:
    """返回 {goal, favor, note}。消息太少就明确写看不出来。"""
    answers = answers or {}
    hers = [text.strip() for who, text in (_parts(m) for m in messages or [])
            if who == "her" and text.strip()]
    if explicit_boundary(messages or []):
        return {"goal": "先停止联系", "favor": "对方要求停止", "note": _NOTE}
    if not hers:
        return {"goal": "先听对方说", "favor": "还看不出来", "note": "可见对话里还没有对方的话。"}

    recent = hers[-4:]
    blob = "\n".join(recent)
    warm = any(word in blob for word in _WARM)
    upset = any(word in blob for word in _UPSET)
    cold = len(recent) >= 2 and all(len(text) <= 4 for text in recent)
    intent = (answers.get("true_intent") or {}).get("choice")
    danger = (answers.get("danger_level") or {}).get("score")
    angry = upset or intent == "vent_anger" or (
        isinstance(danger, (int, float)) and not isinstance(danger, bool) and danger >= 6)
    if angry:
        favor = "明显不满"
    elif warm and not cold:
        favor = "比较热络"
    elif cold and not warm:
        favor = "偏冷"
    else:
        favor = "有回应"
    return {"goal": _goal(intent, strategy, blob, angry), "favor": favor, "note": _NOTE}


def _goal(intent, strategy, blob, angry):
    if angry:
        return "先把气氛缓和"
    by_intent = {
        "request_action": "把对方要的事定下来",
        "seek_explanation": "把事实说清楚",
        "confirm_you_care": "让对方感到被放在心上",
        "casual_chat": "把天聊下去",
        "close_topic": "体面收住这轮",
    }
    if intent in by_intent:
        return by_intent[intent]
    by_strategy = {
        "约见": "约一个具体时间",
        "澄清": "把关键事实问清",
        "降压": "先把压力降下来",
        "收线": "停止推进",
        "调侃": "轻松接住这句",
        "轻推": "往前迈一小步",
        "承接": "接住对方刚才那句",
    }
    if strategy in by_strategy:
        return by_strategy[strategy]
    if any(word in blob for word in _MEET):
        return "把见面说具体"
    if "为什么" in blob or "怎么回事" in blob:
        return "把事实说清楚"
    return "接住对方刚才那句"

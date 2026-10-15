# -*- coding: utf-8 -*-
"""从已经枚举好的顶层窗口里挑微信或 QQ。不含 Win32 调用。"""
from collections import namedtuple

ChatHit = namedtuple("ChatHit", "hwnd app title area")

APPS = (
    {
        "id": "wechat",
        "label": "微信",
        "exes": ("weixin.exe", "wechat.exe"),
        "titles": ("微信",),
    },
    {
        "id": "qq",
        "label": "QQ",
        "exes": ("qq.exe", "qqnt.exe"),
        "titles": ("QQ", "腾讯QQ"),
    },
)

# 看图窗、工具窗。主窗口标题「微信」「QQ」不在这里。
_SKIP_TITLES = frozenset({"", "weixin", "图片和视频", "图片查看器", "qq影像"})
_MIN_AREA = 120_000


def app_for_exe(exe):
    name = (exe or "").lower()
    for app in APPS:
        if name in app["exes"]:
            return app
    return None


def choose_hit(windows, foreground_hwnd=None, prefer_hwnd=None):
    """windows 按 Z 序从上到下。前台是聊天窗口就跟它；点在悬浮窗上则留在上次那个。"""
    if not windows:
        return None

    def same(hwnd, wanted):
        return wanted is not None and int(hwnd) == int(wanted)

    foreground = next((hit for hit in windows if same(hit.hwnd, foreground_hwnd)), None)
    if foreground is not None:
        if _usable(foreground):
            return foreground
        sibling = _fallback([hit for hit in windows if hit.app["id"] == foreground.app["id"]])
        if sibling is not None:
            return sibling
    if prefer_hwnd is not None:
        kept = next((hit for hit in windows if same(hit.hwnd, prefer_hwnd) and _usable(hit)), None)
        if kept is not None:
            return kept
    return _fallback(windows)


def _usable(hit):
    return hit.area >= _MIN_AREA and (hit.title or "").strip().lower() not in _SKIP_TITLES


def _fallback(windows):
    usable = [hit for hit in windows if _usable(hit)]
    for hit in usable:
        if hit.title in hit.app["titles"]:
            return hit
    return usable[0] if usable else None

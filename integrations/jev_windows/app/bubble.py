# -*- coding: utf-8 -*-
"""自己发出的气泡底色。不依赖 OCR 引擎，方便单测。"""


def is_self_bubble(bg) -> bool:
    """微信默认绿泡，QQ 默认蓝/青泡。白底、灰底是对方或面板，不是自己。"""
    r, g, b = int(bg[0]), int(bg[1]), int(bg[2])
    if g > r + 40 and g > b + 40 and g > 80:
        return True
    # 蓝明显高于红，并且不比绿更「绿」。青泡允许绿通道接近蓝。
    if b > r + 30 and b > 90 and b + 15 >= g and (b - r) > (g - r) + 10:
        return True
    return False

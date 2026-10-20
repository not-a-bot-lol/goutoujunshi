# -*- coding: utf-8 -*-
"""消息区截图 → 谁说了什么。RapidOCR 吃 numpy，不落盘。"""
import difflib
import re
import time

import numpy as np

from app.bubble import is_self_bubble


_ENGINE = None


def _engine():
    """OCR 引擎全进程共用：一个实例 ~40MB，每个会话一个 Reader，不能各带一个。
    det_limit_type 默认 'min' 会把小图放大到短边 736，裁小反而更慢；必须 'max'。"""
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR(intra_op_num_threads=4, det_limit_type="max", det_limit_side_len=4000)
    return _ENGINE


def _bounds(box):
    """OCR 四个角的顺序不固定，用外接矩形，不拿第一个点当左上角。"""
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def reading_order(results):
    """先排成屏幕从上到下、同一行从左到右，方便再从底端倒着取。

    只拿框的某一个角排序时，右边半句只要略高一点就会排到左边前面。"""
    items = []
    for box, text, score in results or []:
        left, top, right, bottom = _bounds(box)
        items.append((top, bottom, left, right, box, text, score))
    items.sort(key=lambda item: (item[0], item[2]))
    rows = []
    for item in items:
        top, bottom = item[0], item[1]
        if rows:
            row_top, row_bottom, row = rows[-1]
            overlap = min(bottom, row_bottom) - max(top, row_top)
            shorter = min(bottom - top, row_bottom - row_top)
            row_left = min(old[2] for old in row)
            row_right = max(old[3] for old in row)
            gap = item[2] - row_right if item[2] >= row_right else row_left - item[3]
            # 左右两条气泡会在竖直方向碰上，但不能当成同一行，否则会按左右读乱聊天顺序。
            same_line = shorter > 0 and overlap >= shorter * 0.5 and gap <= max(48, shorter * 2)
            if same_line:
                row.append(item)
                rows[-1] = (min(row_top, top), max(row_bottom, bottom), row)
                continue
        rows.append((top, bottom, [item]))
    ordered = []
    for _, _, row in rows:
        row.sort(key=lambda item: item[2])
        ordered.extend(row)
    return ordered


def recent_from_bottom(lines, limit):
    """从屏幕底端往上取 limit 条有用消息。

    底端是最新的一条。往上遇见一条就压栈，取满就停；再把栈倒出来，
    所以结果是这几条里旧的在前、最新的在最后。空文本不算一条。"""
    stack = []
    try:
        keep = int(limit)
    except (TypeError, ValueError):
        keep = 0
    for line in reversed(lines or []):
        if keep <= 0 or len(stack) >= keep:
            break
        text = line[2] if len(line) > 2 else ""
        if not str(text).strip():
            continue
        stack.append(line)
    stack.reverse()
    return stack


def same_transcript(previous, current) -> bool:
    """两条原文是不是同一屏。OCR 抖一两个字仍算同一屏，避免原文被反复刷掉。"""
    if len(previous) != len(current):
        return False
    for old, new in zip(previous, current):
        if old[0] != new[0] or not similar(str(old[1]), str(new[1])):
            return False
    return True


def _richer(old, new):
    """对上的两条里留下更完整的那句，称呼哪边有就留哪边。"""
    old_text, new_text = str(old[1]).strip(), str(new[1]).strip()
    base = old if len(old_text) >= len(new_text) else new
    name = None
    if len(old) > 2 and old[2]:
        name = old[2]
    elif len(new) > 2 and new[2]:
        name = new[2]
    return (base[0], base[1], name)


def _same_msg(old, new) -> bool:
    """同一条消息。换行被拆开或粘在一起时，只要一边包含另一边也算对上。"""
    if old[0] != new[0]:
        return False
    left, right = str(old[1]).strip(), str(new[1]).strip()
    if similar(left, right):
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    return len(short) >= 4 and short in long


def _align(history, screen):
    """按顺序把这一屏和已有记录对上。返回 [(屏幕下标, 记录下标)]。"""
    pairs = []
    start = 0
    for index, item in enumerate(screen):
        for cursor in range(start, len(history)):
            if _same_msg(history[cursor], item):
                pairs.append((index, cursor))
                start = cursor + 1
                break
    return pairs


def _merge_scroll(history, screen, pairs):
    first_s, first_h = pairs[0]
    last_s, last_h = pairs[-1]
    merged = list(history[:first_h]) + list(screen[:first_s])
    pair_at = {screen_index: hist_index for screen_index, hist_index in pairs}
    hist_i = first_h
    for screen_index in range(first_s, last_s + 1):
        if screen_index in pair_at:
            hist_index = pair_at[screen_index]
            if hist_i < hist_index:
                merged.extend(history[hist_i:hist_index])
            merged.append(_richer(history[hist_index], screen[screen_index]))
            hist_i = hist_index + 1
        else:
            merged.append(screen[screen_index])
    if hist_i <= last_h:
        merged.extend(history[hist_i:last_h + 1])
    merged.extend(history[last_h + 1:])
    merged.extend(screen[last_s + 1:])
    return merged


def _cap_around(merged, screen, limit, upward):
    """最多留 limit 条。往上翻时从最新的一头顶掉，往下翻时从最旧的一头顶掉。"""
    if len(merged) <= limit:
        return list(merged)
    start = None
    seen = 0
    for index, item in enumerate(merged):
        if seen < len(screen) and _same_msg(item, screen[seen]):
            if start is None:
                start = index
            seen += 1
            if seen == len(screen):
                end = index + 1
                break
    else:
        return list(merged[:limit] if upward else merged[-limit:])
    if end - start >= limit:
        if upward:
            return merged[start:start + limit]
        return merged[end - limit:end]
    while end - start < limit and (start > 0 or end < len(merged)):
        if start > 0:
            start -= 1
        if end - start >= limit:
            break
        if end < len(merged):
            end += 1
    return merged[start:end]


def scroll_direction(previous, current):
    """同一条消息在屏幕上的位置变了，就能分出上翻还是下翻。

    y 变大：这句话往屏幕下方走，用户在往上翻更早的记录。
    y 变小：这句话往屏幕上方走，用户在往下翻更新的记录。
    previous / current 每条是 (who, text, name, y) 或至少含文字和下标 3 的 y。"""
    deltas = []
    used = set()
    for item in current or []:
        if len(item) < 4:
            continue
        for index, old in enumerate(previous or []):
            if index in used or len(old) < 4:
                continue
            if _same_msg(old, item):
                used.add(index)
                deltas.append(float(item[3]) - float(old[3]))
                break
    if not deltas:
        return ""
    shift = sum(deltas) / len(deltas)
    if shift >= 12:
        return "up"
    if shift <= -12:
        return "down"
    return ""


def fold_scroll(history, screen, limit, direction=""):
    """把往上或往下翻出来的消息按顺序接进记录，超过 limit 就把多出来的顶掉。

    direction 来自屏幕上同一句话的位移：up = 往上翻，新句子接到前面，顶掉最新的；
    down = 往下翻，新句子接到后面，顶掉最旧的。没有位移时，才看新句子出现在重叠的哪一侧。"""
    history = [item for item in (history or []) if len(item) > 1 and str(item[1]).strip()]
    screen = [item for item in (screen or []) if len(item) > 1 and str(item[1]).strip()]
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        return []
    if not screen:
        return list(history)
    pairs = _align(history, screen)
    if direction == "up":
        upward = True
    elif direction == "down":
        upward = False
    elif pairs:
        added_above = pairs[0][0]
        added_below = len(screen) - pairs[-1][0] - 1
        # 重叠贴着记录的开头，说明视口在看更早的部分。
        near_start = pairs[0][1] < len(history) - 1 - pairs[-1][1]
        upward = added_above > added_below or (added_above == added_below and near_start)
    else:
        upward = False
    fresh = [item for item in screen if not any(_same_msg(old, item) for old in history)]
    if not pairs:
        merged = (fresh + list(history)) if upward else (list(history) + fresh)
    else:
        merged = _merge_scroll(history, screen, pairs)
        if upward:
            # 往上翻时，屏幕上方新出现、却没嵌进重叠里的句子，补到最前。
            ahead = [item for item in fresh if not any(_same_msg(old, item) for old in merged)]
            merged = ahead + merged
    return _cap_around(merged, screen, limit, upward)


def should_replace_transcript(previous, current) -> bool:
    """当前这屏要不要覆盖原文。

    识别变少、但每条都能在原文里找到时，多半是这一帧漏认，先留着。
    有新句子，或同一批句子顺序变了，就整段换成这一屏从底端读出的结果。"""
    if not current or same_transcript(previous, current):
        return False
    if len(current) >= len(previous):
        return True
    used = [False] * len(previous)
    for who, text in current:
        for index, (old_who, old_text) in enumerate(previous):
            if not used[index] and old_who == who and similar(str(old_text), str(text)):
                used[index] = True
                break
        else:
            return True
    return False


def read_title(header):
    """面板头部那一条截图 → 会话名（numpy RGB）。取最靠上的一行，同一行里取最左的
    （右边是图标按钮，OCR 不出字；下面那行是公告）。群聊的成员数「(422)」去掉，只留名字当 key。
    认不出返回 ""。一次约 60ms，所以调用方只在头部像素变了时才问。"""
    res, _ = _engine()(header, use_cls=False)
    ordered = reading_order(res)
    if not ordered:
        return ""
    first = ordered[0]
    row_bottom = first[1]
    text = min((item for item in ordered if item[0] < row_bottom), key=lambda item: item[2])[5]
    return re.sub(r"\s*[（(]\d+[)）]\s*$", "", text.strip())


def who_said(chat, box):
    """按 OCR 框里的颜色分类，不看 x 坐标。返回 (谁, 底色, 墨高)：
    先看底色平不平：框里众数颜色占比 <45% 就是图片（头像/照片/表情包）里的字 → None 丢掉。
    绿底（微信）或蓝/青底（QQ）→ me；其余文字对底色对比度 ≥150 → her；
    其余（引用块、群里的发言人名、时间戳、系统提示、链接卡片描述——都是灰字，对比度 80~95）→ "gray"。
    实测：气泡正文对比度 178~208，me 绿泡 142~150，灰字 ≤ 93。深浅主题都靠这套。
    墨高 = 框里最长一段连续有字的行数（OCR 框对小字有固定 padding、还会蹭到上下行，不能拿框高比大小）。"""
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    reg = chat[int(min(ys)):int(max(ys)), int(min(xs)):int(max(xs))].astype(int)
    if reg.size == 0:
        return None, None, 0
    vals, cnt = np.unique(reg.reshape(-1, 3), axis=0, return_counts=True)
    bg = vals[cnt.argmax()]
    if cnt.max() / reg.shape[0] / reg.shape[1] < 0.45:
        # 文字必须落在平底色上：WGC 帧是精确像素，气泡/面板里众数颜色占 0.56~0.82，
        # 头像/照片/表情包里只有 0.1~0.3——那是图片里的字（头像上的「借仲夏夜之梦」之类），不是消息。
        # ponytail: 只对精确像素的帧成立；缩放/压缩过的截图（比如拿预览窗再截一次的图）底色会糊成几百种颜色，全会被当图片。
        return None, bg, 0
    diff = np.abs(reg @ [0.299, 0.587, 0.114] - bg @ [0.299, 0.587, 0.114])
    ink_h = best = 0
    for r in (diff > 60).any(axis=1):
        best = best + 1 if r else 0
        ink_h = max(ink_h, best)
    if is_self_bubble(bg):
        return "me", bg, ink_h
    return ("her" if diff.max() >= 150 else "gray"), bg, ink_h


def similar(a, b):
    """同一段像素挪个位置 OCR 会抖（「傻逼了」↔「傻逼」、「不好意思」↔「不好竟思」），按相似度判同一条。"""
    if a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.75:
        return True
    return len(a) == len(b) >= 3 and sum(x != y for x, y in zip(a, b)) <= 1  # 短句错一个字


class Reader:
    """一个会话一个 Reader：lh/seen 各自算各自的，切走再切回来不会把旧消息当新的重报一遍。"""

    def __init__(self):
        self.ocr = _engine()
        self.lh = None  # 正常气泡字高，头一帧定
        self.seen = []  # [(who, name, text)]，累计，封顶 500
        self.last_boxes = []  # 调试视图用：[(x0,y0,x1,y1,kind,text)]，消息区裁剪坐标
        self.last_ms = 0  # 上一帧 OCR 耗时

    def read(self, chat, pane_bg):
        """→ [(who, name, text, y)]，同一气泡的多行已合并。who ∈ me/her；name 群聊里是发言人，单聊 None。
        顺带把每个框的分类记进 self.last_boxes（调试视图画框用，几十个 tuple，不开也不亏）。"""
        t0 = time.perf_counter()
        res, _ = self.ocr(chat, use_cls=False)
        self.last_ms = int((time.perf_counter() - t0) * 1000)
        self.last_boxes = []
        W = chat.shape[1]
        # 群聊：每条 her 气泡上方一行灰色发言人名（靠左、短、不带冒号、印在面板底色上），从上往下扫，名字带给后面的气泡。
        # 引用块/时间戳/公告带冒号，链接卡片灰字印在气泡底色上，都不会被当成名字。
        # ponytail: 名字行被 OCR 漏掉时会挂到上一个人头上。
        name, raw = None, []
        for top, bottom, _left, _right, box, text, _ in reading_order(res):
            kind, bg, h = who_said(chat, box)
            rect = (int(_left), int(top), int(_right), int(bottom))
            if kind == "gray":
                on_pane = np.abs(bg - pane_bg).sum() <= 6
                taken = bool(on_pane and _left < 0.25 * W and len(text) <= 16
                             and not re.search("[:：]", text))
                if taken:
                    name = text
                self.last_boxes.append(rect + ("name" if taken else "gray", text))
                continue
            if kind is None or (self.lh and h < 0.6 * self.lh):
                # 字比正常气泡小得多 = 图片消息（截图/表情包）里的字，不是气泡
                self.last_boxes.append(rect + ("image" if kind is None else "tiny", text))
                continue
            self.last_boxes.append(rect + (kind, text))
            raw.append((kind, name if kind == "her" else None, text, top, bottom, h))
        if not self.lh and len(raw) >= 3:
            self.lh = float(np.median([r[5] for r in raw]))
        # 同一气泡的多行合并：同人、上一行底到这一行顶的间距不到半个字高（不同气泡之间至少隔一个字高）
        lines = []
        for who, nm, text, top, bottom, h in raw:
            if lines and lines[-1][0] == who and lines[-1][1] == nm and top - lines[-1][4] < 0.6 * (self.lh or h):
                lines[-1][2] += text
                lines[-1][4] = bottom
            else:
                lines.append([who, nm, text, top, bottom])
        return [(w, n, t, y) for w, n, t, y, _ in lines]

    def new_lines(self, lines):
        """去重（滚动不重复）→ 这一帧里真正新出现的 [(who, name, text)]。
        本帧有已知行时只要已知行下方的：往上滚翻出来的旧消息在已知行上方，不算。
        本帧一行已知的都没有（大图把旧文字全顶出去了、切了聊天、滚远了）：全算，宁可多算不能漏。
        ponytail: 同一人连发两句一模一样的会吞一句——对触发分析无害。"""
        known_y = [y for w, n, t, y in lines if self._seen(w, n, t)]
        floor = max(known_y) if known_y else -1
        new = [(w, n, t) for w, n, t, y in lines if y > floor and not self._seen(w, n, t)]
        self.seen.extend((w, n, t) for w, n, t, _ in lines if not self._seen(w, n, t))
        del self.seen[:-500]
        return new

    def _seen(self, who, name, text):
        # 名字不参与判重：名字行滚出画面后同一条消息会从 her(LO) 变成 her，不能算新消息
        return any(w == who and similar(t, text) for w, _, t in self.seen)

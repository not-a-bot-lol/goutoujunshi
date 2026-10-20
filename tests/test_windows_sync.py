import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from integrations.jev_windows.app.bubble import is_self_bubble
from integrations.jev_windows.app.cloud_ocr import CloudReader, parse_transcription
from integrations.jev_windows.app.targets import APPS, ChatHit, choose_hit
from integrations.jev_windows.core import experience
from integrations.jev_windows.core.deepseek_strategy import parse_choice, parse_evidence
from integrations.jev_windows.core.rapport import read_rapport
from integrations.jev_windows.core.trend import load_csv


class WindowsSyncTest(unittest.TestCase):
    def test_cloud_ocr_requires_explicit_speaker_and_dedupes(self):
        lines = parse_transcription(json.dumps({"messages": [
            {"side": "them", "text": "明天见？"}, {"side": "me", "text": "好"}]}))
        reader = CloudReader("deepseek", "deepseek-flash", "dummy")
        self.assertEqual(reader.new_lines(lines), [("her", None, "明天见？"), ("me", None, "好")])
        self.assertEqual(reader.new_lines(lines), [])
        with self.assertRaisesRegex(ValueError, "说话人"):
            parse_transcription('{"messages":[{"side":"unknown","text":"你好"}]}')

    def test_deepseek_evidence_and_token_weights_are_distinct(self):
        evidence = parse_evidence('{"strategy":"降压","confidence":0.62,"intent":"可能在忙",'
                                  '"facts":["对方说今天忙"],"unknowns":["何时有空"]}')
        self.assertEqual(evidence["method"], "deepseek_self_report")
        self.assertEqual(evidence["weights"], {})
        labels = dict(zip("ABCDEFG", ("承接", "降压", "调侃", "轻推", "约见", "澄清", "收线")))
        top = [{"token": c, "logprob": -.1 if c == "B" else -3.0} for c in "ABCDEFG"]
        weights = parse_choice("B", {"content": [{"token": "B", "top_logprobs": top}]}, labels)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["降压"], weights["承接"])

    def test_details_and_rewrite_use_verified_own_words(self):
        messages = [("me", "行，周六见"), ("her", "好呀")]
        result = {"candidates": ["那周六见吧", "好的，周六见"], "goutou": {"action": "承接"}}
        responses = [json.dumps({"intent": "愿意继续聊", "facts": ["对方说好呀"],
                                 "hypotheses": [], "unknowns": ["具体时间"], "next_step": "确定时间"}),
                     '["行，周六见","好，周六见"]',
                     '{"scores":[{"id":0,"score":60},{"id":1,"score":40}]}']
        with patch.object(experience, "_ask", side_effect=responses) as ask:
            detail = experience.details(messages, "朋友", result, provider="deepseek", model="m", key="dummy")
            rewritten, weights = experience.rewrite(messages, result, provider="deepseek", model="m", key="dummy")
        self.assertIn("对方说好呀", detail)
        self.assertEqual(rewritten, ["行，周六见", "好，周六见"])
        self.assertEqual(weights, [.6, .4])
        self.assertEqual(ask.call_args_list[1].args[1]["my_samples"], ["行，周六见"])

    def test_candidate_reason_includes_tradeoff(self):
        result = {"candidates": ["好，周六见"], "goutou": {"action": "承接"}}
        with patch.object(experience, "_ask", return_value='{"reason":"接住邀约",'
                         '"tradeoff":"还需要确认时间"}'):
            text = experience.explain([("her", "周六见吗")], "朋友", result, 0,
                                      provider="deepseek", model="m", key="dummy")
        self.assertIn("接住邀约", text)
        self.assertIn("还需要确认时间", text)

    def test_rapport_reads_goal_and_favor_from_the_visible_chat(self):
        cold = read_rapport([("her", "嗯"), ("me", "在吗"), ("her", "哦")])
        self.assertEqual(cold["favor"], "偏冷")
        self.assertEqual(cold["goal"], "接住对方刚才那句")
        warm = read_rapport([("her", "好呀，周末想见你😊")])
        self.assertEqual(warm["favor"], "比较热络")
        self.assertEqual(warm["goal"], "把见面说具体")
        asked = read_rapport([("her", "那你什么时候有空")],
                             {"true_intent": {"choice": "request_action"}})
        self.assertEqual(asked["goal"], "把对方要的事定下来")
        stopped = read_rapport([("her", "不要再联系我")])
        self.assertEqual((stopped["goal"], stopped["favor"]), ("先停止联系", "对方要求停止"))
        self.assertEqual(read_rapport([("me", "在吗")])["favor"], "还看不出来")

    def test_wechat_green_and_qq_blue_are_self_bubbles(self):
        self.assertTrue(is_self_bubble((149, 236, 105)))
        self.assertTrue(is_self_bubble((18, 183, 245)))
        self.assertTrue(is_self_bubble((201, 231, 255)))
        self.assertTrue(is_self_bubble((30, 100, 180)))
        self.assertFalse(is_self_bubble((255, 255, 255)))
        self.assertFalse(is_self_bubble((237, 237, 237)))

    @unittest.skipUnless(sys.platform == "win32", "消息区定位依赖 Windows 上的 numpy")
    def test_same_color_sidebar_is_cut_at_the_vertical_line(self):
        import sys
        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "jev_windows"))
        from app.capture import chat_area

        img = np.full((600, 900, 3), 34, np.uint8)
        img[:40] = 18
        img[:, 220:223] = 90
        img[80:140, 40:180] = 200
        area = chat_area(img)
        self.assertIsNotNone(area)
        self.assertGreaterEqual(area[0], 220)
        self.assertLess(area[0], 260)

    @unittest.skipUnless(sys.platform == "win32", "消息区定位依赖 Windows 上的 numpy")
    def test_different_sidebar_color_stays_on_the_color_edge(self):
        import sys
        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "jev_windows"))
        from app.capture import chat_area

        img = np.full((600, 900, 3), 237, np.uint8)
        img[:, :250] = (32, 32, 32)
        img[:40] = 20
        area = chat_area(img)
        self.assertIsNotNone(area)
        self.assertGreaterEqual(area[0], 240)
        self.assertLess(area[0], 280)

    def test_foreground_qq_wins_over_previous_wechat(self):
        wechat, qq = APPS
        windows = [
            ChatHit(2, qq, "QQ", 800 * 600),
            ChatHit(1, wechat, "微信", 900 * 700),
            ChatHit(3, qq, "图片查看器", 1200 * 800),
        ]
        self.assertEqual(choose_hit(windows, foreground_hwnd=2, prefer_hwnd=1).hwnd, 2)
        self.assertEqual(choose_hit(windows, foreground_hwnd=3, prefer_hwnd=1).hwnd, 2)
        self.assertEqual(choose_hit(windows, prefer_hwnd=1).hwnd, 1)
        self.assertEqual(choose_hit(windows).hwnd, 2)

    @unittest.skipUnless(sys.platform == "win32", "屏幕识别排序和 Windows 客户端一起跑")
    def test_screen_text_follows_chat_order(self):
        windows = Path(__file__).resolve().parents[1] / "integrations" / "jev_windows"
        sys.path.insert(0, str(windows))
        from app.ocr import reading_order

        # 右边半句的角点更靠上；上面那条消息的第一个点却是它的底角。只按第一个点排序会读反。
        same_line = [
            ([[120, 8], [200, 8], [200, 28], [120, 28]], "右边", 0.9),
            ([[10, 30], [10, 12], [90, 14], [90, 32]], "左边", 0.9),
            ([[10, 36], [90, 36], [90, 70], [10, 70]], "第二句", 0.9),
        ]
        stacked = [
            ([[10, 40], [10, 10], [80, 10], [80, 40]], "上面", 0.9),
            ([[10, 36], [90, 36], [90, 70], [10, 70]], "下面", 0.9),
        ]
        self.assertEqual([item[5] for item in reading_order(same_line)], ["左边", "右边", "第二句"])
        self.assertEqual([item[5] for item in reading_order(stacked)], ["上面", "下面"])
        side_by_side = [
            ([[400, 10], [520, 10], [520, 50], [400, 50]], "右侧", 0.9),
            ([[10, 20], [100, 20], [100, 55], [10, 55]], "左侧", 0.9),
        ]
        self.assertEqual([item[5] for item in reading_order(side_by_side)], ["右侧", "左侧"])
        from app.ocr import recent_from_bottom, should_replace_transcript
        visible = [("her", None, "更早", 10), ("me", None, "", 20), ("her", None, "早", 40),
                   ("me", None, "中", 70), ("her", None, "新", 100), ("me", None, "最新", 130)]
        self.assertEqual([row[2] for row in recent_from_bottom(visible, 3)], ["中", "新", "最新"])
        previous = [("her", "甲"), ("me", "乙"), ("her", "丙")]
        self.assertFalse(should_replace_transcript(previous, [("her", "甲"), ("me", "乙")]))
        self.assertFalse(should_replace_transcript(previous, previous))
        self.assertTrue(should_replace_transcript(previous, [("me", "乙"), ("her", "甲"), ("her", "丙")]))
        self.assertTrue(should_replace_transcript(previous, previous + [("me", "丁")]))
        from app.ocr import fold_scroll
        kept = fold_scroll(
            [("her", "C", None), ("me", "D", None), ("her", "E", None), ("me", "F", None)],
            [("her", "A", None), ("me", "B", None), ("her", "C", None), ("me", "D", None)], 4)
        self.assertEqual([item[1] for item in kept], ["A", "B", "C", "D"])
        kept = fold_scroll(
            [("her", "A", None), ("me", "B", None), ("her", "C", None), ("me", "D", None)],
            [("her", "C", None), ("me", "D", None), ("her", "E", None), ("me", "F", None)], 4)
        self.assertEqual([item[1] for item in kept], ["C", "D", "E", "F"])
        kept = fold_scroll(
            [("her", "A", None), ("me", "B", None), ("her", "C", None)],
            [("me", "B", None), ("her", "C", None)], 5)
        self.assertEqual([item[1] for item in kept], ["A", "B", "C"])
        kept = fold_scroll(
            [("me", "D", None), ("her", "E", None), ("me", "F", None), ("her", "G", None)],
            [("her", "A", None), ("me", "B", None), ("her", "C", None), ("me", "D", None), ("her", "E", None)], 4)
        self.assertEqual([item[1] for item in kept], ["A", "B", "C", "D"])
        kept = fold_scroll([("her", "已经记下", None)], [("me", "翻到新的", None)], 5)
        self.assertEqual([item[1] for item in kept], ["已经记下", "翻到新的"])
        kept = fold_scroll([("her", "不要死啊", None), ("me", "我尽量", None)],
                           [("her", "不要死", None)], 5)
        self.assertEqual([item[1] for item in kept], ["不要死啊", "我尽量"])
        from app.ocr import scroll_direction
        self.assertEqual(scroll_direction(
            [("her", "锚", None, 100)], [("her", "锚", None, 180)]), "up")
        self.assertEqual(scroll_direction(
            [("her", "锚", None, 180)], [("her", "锚", None, 80)]), "down")
        kept = fold_scroll([("her", "C", None), ("me", "D", None)],
                           [("her", "A", None), ("me", "B", None)], 4, "up")
        self.assertEqual([item[1] for item in kept], ["A", "B", "C", "D"])
        kept = fold_scroll([("her", "A", None), ("me", "B", None)],
                           [("her", "C", None), ("me", "D", None)], 4, "down")
        self.assertEqual([item[1] for item in kept], ["A", "B", "C", "D"])

    @unittest.skipUnless(sys.platform == "win32", "设置写入走 Windows 注册表")
    def test_settings_survive_a_fresh_read_and_keys_skip_the_broadcast(self):
        windows = Path(__file__).resolve().parents[1] / "integrations" / "jev_windows"
        sys.path.insert(0, str(windows))
        # Mac 测试会先把 jev_mac/core.py 装进 sys.modules，这里要让 Windows 的 core 包重新加载。
        for name in [key for key in sys.modules if key == "core" or key.startswith("core.") or key == "app" or key.startswith("app.")]:
            del sys.modules[name]
        from app import settings

        self.assertIsNone(settings.key_to_save("  "))
        self.assertIsNone(settings.key_to_save(settings.REMEMBERED))
        self.assertEqual(settings.key_to_save(" sk-abc "), "sk-abc")
        previous = settings._CONFIG
        os.environ.pop("GOUTOU_TEST_SETTINGS_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                settings._CONFIG = os.path.join(directory, "config.json")
                settings.save("friends", 8, jev_provider_text="openrouter", jev_model_text="jev-test",
                              draft_provider_text="deepseek", draft_model_text="deepseek-flash",
                              strategy_provider_text="jev", style_text="话少", reply_target_on=True,
                              auto_analyze_on=False, thinking_on=False, check_update_on=False,
                              ocr_provider_text="local")
                settings._CONFIG = os.path.join(directory, "config.json")
                self.assertEqual(settings.relationship(), "friends")
                self.assertEqual(settings.context(), 8)
                self.assertEqual(settings.style(), "话少")
                self.assertEqual(settings.jev_model(), "jev-test")
                self.assertTrue(settings.reply_target())
                self.assertFalse(settings.check_update())
            key = MagicMock()
            key.__enter__.return_value = key
            with patch("winreg.OpenKey", return_value=key), patch("winreg.SetValueEx") as set_value, \
                    patch("ctypes.windll.user32.SendMessageTimeoutW", side_effect=AssertionError("broadcast")):
                settings._set_key("GOUTOU_TEST_SETTINGS_KEY", "secret")
            set_value.assert_called_once()
            self.assertEqual(os.environ.get("GOUTOU_TEST_SETTINGS_KEY"), "secret")
            os.environ["GOUTOU_TEST_SETTINGS_KEY"] = "stale-not-typed"
            with patch("winreg.OpenKey", return_value=key), patch("winreg.QueryValueEx", return_value=("sk-typed", 1)):
                self.assertEqual(settings._read_env("GOUTOU_TEST_SETTINGS_KEY"), "sk-typed")
            self.assertEqual(os.environ.get("GOUTOU_TEST_SETTINGS_KEY"), "sk-typed")
        finally:
            settings._CONFIG = previous
            os.environ.pop("GOUTOU_TEST_SETTINGS_KEY", None)

    def test_csv_requires_dates_and_calculates_direction_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.csv"
            path.write_text("timestamp,sender,message\n2026-09-01 10:00:00,other,你好\n"
                            "2026-09-01 10:01:00,me,好\n2026-09-02 12:00:00,other,周末见\n",
                            encoding="utf-8")
            rows = load_csv(path)
        self.assertEqual([(r.open, r.high, r.low, r.close) for r in rows],
                         [(0, 1, 0, 0), (0, 1, 0, 1)])


if __name__ == "__main__":
    unittest.main()

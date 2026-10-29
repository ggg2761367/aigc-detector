# -*- coding: utf-8 -*-
"""watermark.scan / visible_text 的回归测试（全部使用合成文本）。"""

import unittest

from watermark import scan, visible_text


def tag_encode(s: str) -> str:
    """把 ASCII 编码进 Unicode Tag 字符。"""
    return "".join(chr(0xE0000 + ord(c)) for c in s)


def vs_encode(s: str) -> str:
    """把 UTF-8 字节按高低半字节编码成变体选择符（VS1–VS16）。"""
    out = []
    for b in s.encode("utf-8"):
        out.append(chr(0xFE00 + (b >> 4)))
        out.append(chr(0xFE00 + (b & 0x0F)))
    return "".join(out)


class TestWatermark(unittest.TestCase):
    def test_clean_text_has_no_findings(self):
        r = scan("这是一段完全正常的中文文本，没有任何隐藏字符。")
        self.assertEqual(r["risk"], "none")
        self.assertFalse(r["detected"])
        self.assertEqual(r["findings"], [])
        self.assertEqual(r["invisible_chars"], 0)

    def test_empty_and_non_string(self):
        for value in ("", None, 123):
            r = scan(value)
            self.assertEqual(r["risk"], "none")
            self.assertEqual(r["findings"], [])

    def test_tag_chars_decoded(self):
        r = scan("正文开始。" + tag_encode("leak"))
        self.assertTrue(r["detected"])
        self.assertEqual(r["risk"], "high")
        f = next(f for f in r["findings"] if f["kind"] == "tag_chars")
        self.assertEqual(f["count"], 4)
        self.assertEqual(f["decoded"], "leak")
        self.assertIn("U+E", f["codepoints"][0])

    def test_zero_width_inside_word(self):
        r = scan("hello​world")
        self.assertTrue(r["detected"])
        self.assertIn("zero_width", [f["kind"] for f in r["findings"]])

    def test_emoji_zwj_is_not_flagged(self):
        r = scan("\U0001F468‍\U0001F469‍\U0001F467")
        self.assertNotIn("zero_width", [f["kind"] for f in r["findings"]])

    def test_bidi_controls(self):
        r = scan("abc‮def")
        self.assertIn("bidi_controls", [f["kind"] for f in r["findings"]])
        self.assertEqual(r["risk"], "high")

    def test_variation_selector_payload(self):
        r = scan("前缀" + vs_encode("Hi"))
        f = next(f for f in r["findings"] if f["kind"] == "variation_selectors")
        self.assertEqual(f["decoded"], "Hi")

    def test_single_emoji_variation_selector_not_flagged(self):
        r = scan("❤️")  # 正常的 emoji VS16
        self.assertNotIn("variation_selectors", [f["kind"] for f in r["findings"]])

    def test_homoglyph_mixed_script(self):
        r = scan("pаypal")  # 西里尔 а(U+0430) 冒充拉丁 a
        f = next(f for f in r["findings"] if f["kind"] == "homoglyph")
        self.assertEqual(f["severity"], "medium")
        self.assertIn("paypal", f["detail"])

    def test_pure_cyrillic_not_flagged(self):
        r = scan("привет мир")
        self.assertNotIn("homoglyph", [f["kind"] for f in r["findings"]])

    def test_odd_space_is_low(self):
        r = scan("hello world")
        self.assertEqual(r["risk"], "low")
        self.assertFalse(r["detected"])

    def test_visible_projection(self):
        src = "ab​cd" + tag_encode("x") + "e f"
        self.assertEqual(visible_text(src), "abcde f")


if __name__ == "__main__":
    unittest.main()

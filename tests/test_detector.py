# -*- coding: utf-8 -*-
"""detector 的回归测试：分句、打分方向、校准、信道隔离。"""

import unittest

from detector import analyze, calibrate, split_sentences, tokenize

AI_TEXT = (
    "首先，人工智能技术在教育领域的应用具有重要意义。其次，人工智能能够显著提升教学效率，"
    "为学生提供个性化的学习体验。此外，人工智能还可以赋能教师，使教师能够更加聚焦于学生的"
    "全面发展。值得注意的是，随着技术的不断发展，人工智能在教育中的作用将日益凸显。"
    "综上所述，人工智能技术在教育领域扮演着重要角色，具有广阔的发展前景。"
)

HUMAN_TEXT = (
    "昨天下午我在楼下便利店买了瓶水，收银的阿姨正在看一部很老的家庭剧，声音开得挺大。"
    "她一边找零一边跟我抱怨说这剧重播第八遍了，但她还是每天准点守着。"
    "我拎着水走出去，外面下起了小雨，没带伞，就站在屋檐下站了十来分钟。"
    "后来雨小了，我骑车回家，路上想不起晚饭该吃什么。"
)


def tag_encode(s: str) -> str:
    return "".join(chr(0xE0000 + ord(c)) for c in s)


class TestSplit(unittest.TestCase):
    def test_chinese_sentences(self):
        self.assertEqual(
            split_sentences("第一句。第二句！第三句？"),
            ["第一句。", "第二句！", "第三句？"],
        )

    def test_decimal_not_split(self):
        self.assertEqual(split_sentences("圆周率约等于3.14159，不是4。"),
                         ["圆周率约等于3.14159，不是4。"])

    def test_abbreviation_not_split(self):
        self.assertEqual(split_sentences("See Dr. Smith today."), ["See Dr. Smith today."])

    def test_newline_is_hard_boundary(self):
        self.assertEqual(split_sentences("第一行\n第二行"), ["第一行", "第二行"])

    def test_tokenize_mixed(self):
        self.assertEqual(tokenize("AI技术2024"), ["ai", "技", "术", "2024"])


class TestAnalyze(unittest.TestCase):
    def test_ai_higher_than_human(self):
        ai = analyze(AI_TEXT)
        human = analyze(HUMAN_TEXT)
        self.assertTrue(ai["ok"] and human["ok"])
        self.assertGreater(ai["doc"]["probability"], human["doc"]["probability"])
        self.assertGreater(ai["doc"]["probability"], 0.6)
        self.assertLess(human["doc"]["probability"], 0.5)

    def test_schema(self):
        r = analyze(AI_TEXT)
        self.assertTrue(r["ok"])
        for key in ("chars", "probability", "verdict", "mixed_ratio",
                    "contributions", "metrics"):
            self.assertIn(key, r["doc"])
        self.assertIsInstance(r["sentences"], list)
        self.assertIn("watermark", r)

    def test_empty_text_returns_watermark(self):
        r = analyze("   ")
        self.assertFalse(r["ok"])
        self.assertIn("watermark", r)

    def test_hidden_chars_do_not_change_style_score(self):
        clean = analyze(AI_TEXT)
        hidden = analyze(AI_TEXT + tag_encode("leak"))
        self.assertEqual(clean["doc"]["percent"], hidden["doc"]["percent"])
        self.assertEqual(clean["doc"]["sentences"], hidden["doc"]["sentences"])
        self.assertTrue(hidden["watermark"]["detected"])
        self.assertEqual(hidden["watermark"]["findings"][0]["decoded"], "leak")

    def test_invisible_only_text_is_rejected_with_hint(self):
        r = analyze("​​​")
        self.assertFalse(r["ok"])
        self.assertIn("隐藏字符", r["error"])


class TestCalibration(unittest.TestCase):
    def test_short_text_shrinks(self):
        # 200 字以上完全采信；10 字时按 min_shrink=0.25 收缩
        long_p = calibrate(10.0, 500, 10)
        short_p = calibrate(10.0, 10, 1)
        self.assertGreater(long_p, 0.99)
        self.assertLess(abs(short_p - 0.625), 0.01)

    def test_monotonic(self):
        self.assertLess(calibrate(-2.0, 500, 10), calibrate(2.0, 500, 10))


if __name__ == "__main__":
    unittest.main()

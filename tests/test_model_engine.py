# -*- coding: utf-8 -*-
"""ONNX 引擎回归测试：缺模型/依赖时跳过，装了 `.venv` 后执行。"""

import unittest

from model_engine import get_engine

AI_ZH = (
    "首先，人工智能技术在教育领域的应用具有重要意义。其次，人工智能能够显著提升教学效率，"
    "为学生提供个性化的学习体验。此外，人工智能还可以赋能教师，使教师能够更加聚焦于学生的"
    "全面发展。值得注意的是，随着技术的不断发展，人工智能在教育中的作用将日益凸显。"
)

HUMAN_ZH = (
    "昨天下午我在楼下便利店买了瓶水，收银的阿姨正在看一部很老的家庭剧，声音开得挺大。"
    "她一边找零一边跟我抱怨说这剧重播第八遍了，但她还是每天准点守着。"
    "我拎着水走出去，外面下起了小雨，没带伞，就站在屋檐下站了十来分钟。"
)


class TestModelEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine()

    def setUp(self):
        if not self.engine:
            self.skipTest("ONNX 模型或 onnxruntime/tokenizers 不可用")

    def test_ai_higher_than_human(self):
        ai = self.engine.predict(AI_ZH)
        human = self.engine.predict(HUMAN_ZH)
        self.assertGreater(ai["probability"], human["probability"])
        self.assertGreater(ai["probability"], 0.5)
        self.assertLess(human["probability"], 0.2)

    def test_calibration_metadata(self):
        info = self.engine.info()
        self.assertTrue(info["calibrated"])
        self.assertTrue(0.0 < info["threshold"] < 1.0)

    def test_probability_range(self):
        out = self.engine.predict("")
        self.assertGreaterEqual(out["probability"], 0.0)
        self.assertLessEqual(out["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()

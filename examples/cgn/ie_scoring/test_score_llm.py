# -*- coding: utf-8 -*-
"""
测试 _score 函数的输出是否正常
"""
import json
import os
import tempfile
import unittest

from score_llm import _score, load_json


class TestScore(unittest.TestCase):
    """_score 函数测试用例"""

    def setUp(self):
        """准备测试数据"""
        # gt/pred 格式: {imgname: {field_key: [values]}}
        self.gt = {
            "img1": {
                "发票号码": ["12345678"],
                "金额": ["100.00"],
                "日期": ["2024年1月1日"],
            },
            "img2": {
                "发票号码": ["87654321"],
                "金额": ["200.50"],
                "日期": ["2024年2月15日"],
            },
        }
        # 预测完全正确
        self.pred_perfect = {
            "img1": {
                "发票号码": ["12345678"],
                "金额": ["100.00"],
                "日期": ["2024年1月1日"],
            },
            "img2": {
                "发票号码": ["87654321"],
                "金额": ["200.50"],
                "日期": ["2024年2月15日"],
            },
        }
        # 预测部分错误
        self.pred_partial = {
            "img1": {
                "发票号码": ["12345678"],  # 正确
                "金额": ["100"],  # 与 gt "100.00" 可能相似
                "日期": ["2024年01月01日"],  # 格式略有不同
            },
            "img2": {
                "发票号码": ["wrong_num"],  # 错误
                "金额": ["200.50"],  # 正确
                "日期": ["2024年2月15日"],  # 正确
            },
        }
        self.imgnames = ["img1", "img2"]

    def test_score_perfect_prediction(self):
        """测试完全正确的预测 - 应得到较高分数"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_file = f.name

        try:
            mfh, mh = _score(
                gt=self.gt,
                pred=self.pred_perfect,
                imgnames=self.imgnames,
                exclude_keys=[],
                save_file=save_file,
                metric="rouge-l",
            )
            # 完全正确时 mfh 和 mh 应接近 1.0
            self.assertGreaterEqual(mfh, 0.99, f"mFieldHmean 应接近 1.0, 实际: {mfh}")
            self.assertGreaterEqual(mh, 0.99, f"MethodHmean 应接近 1.0, 实际: {mh}")
            self.assertIsInstance(mfh, (int, float))
            self.assertIsInstance(mh, (int, float))

            # 验证 save_file 已生成且格式正确
            self.assertTrue(os.path.exists(save_file))
            table = load_json(save_file)
            self.assertIsInstance(table, list)
            self.assertGreater(len(table), 0)
            for row in table:
                self.assertIn("imgname", row)
                self.assertIn("key", row)
                self.assertIn("gt_text", row)
                self.assertIn("pred_text", row)
                self.assertIn("error_type", row)
                self.assertIn("correct", row)
        finally:
            if os.path.exists(save_file):
                os.remove(save_file)

    def test_score_partial_prediction(self):
        """测试部分正确的预测 - 分数应介于 0 和 1 之间"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_file = f.name

        try:
            mfh, mh = _score(
                gt=self.gt,
                pred=self.pred_partial,
                imgnames=self.imgnames,
                exclude_keys=[],
                save_file=save_file,
                metric="rouge-l",
            )
            self.assertGreaterEqual(mfh, 0, f"mFieldHmean 应 >= 0, 实际: {mfh}")
            self.assertLessEqual(mfh, 1, f"mFieldHmean 应 <= 1, 实际: {mfh}")
            self.assertGreaterEqual(mh, 0, f"MethodHmean 应 >= 0, 实际: {mh}")
            self.assertLessEqual(mh, 1, f"MethodHmean 应 <= 1, 实际: {mh}")
            # 部分正确时分数应低于完美预测
            self.assertLess(mh, 1.0, "部分错误时 MethodHmean 应小于 1.0")
        finally:
            if os.path.exists(save_file):
                os.remove(save_file)

    def test_score_with_exclude_keys(self):
        """测试 exclude_keys 排除字段"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_file = f.name

        try:
            mfh, mh = _score(
                gt=self.gt,
                pred=self.pred_perfect,
                imgnames=self.imgnames,
                exclude_keys=["日期"],  # 排除日期字段
                save_file=save_file,
                metric="rouge-l",
            )
            self.assertGreaterEqual(mfh, 0)
            self.assertLessEqual(mfh, 1)
            table = load_json(save_file)
            # 排除的 key 不应出现在结果中
            keys_in_table = set(row["key"] for row in table)
            self.assertNotIn("日期", keys_in_table)
        finally:
            if os.path.exists(save_file):
                os.remove(save_file)

    def test_score_different_metrics(self):
        """测试不同 metric 都能正常返回"""
        metrics = ["rouge-l", "bleu-4", "meteor", "rocr"]
        for metric in metrics:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                save_file = f.name
            try:
                mfh, mh = _score(
                    gt=self.gt,
                    pred=self.pred_perfect,
                    imgnames=self.imgnames,
                    exclude_keys=[],
                    save_file=save_file,
                    metric=metric,
                )
                self.assertIsInstance(mfh, (int, float), f"metric={metric} mfh 应为数值")
                self.assertIsInstance(mh, (int, float), f"metric={metric} mh 应为数值")
                self.assertGreaterEqual(mfh, 0)
                self.assertLessEqual(mfh, 1)
                self.assertGreaterEqual(mh, 0)
                self.assertLessEqual(mh, 1)
            finally:
                if os.path.exists(save_file):
                    os.remove(save_file)

    def test_score_single_image(self):
        """测试单张图片"""
        gt_single = {"img1": {"字段A": ["测试文本"]}}
        pred_single = {"img1": {"字段A": ["测试文本"]}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_file = f.name

        try:
            mfh, mh = _score(
                gt=gt_single,
                pred=pred_single,
                imgnames=["img1"],
                exclude_keys=[],
                save_file=save_file,
                metric="rouge-l",
            )
            self.assertEqual(mfh, 1.0)
            self.assertEqual(mh, 1.0)
        finally:
            if os.path.exists(save_file):
                os.remove(save_file)


def run_manual_test():
    """
    手动运行以查看详细输出，用于验证 _score 结果是否正常
    """
    from score_llm import _score, load_json

    gt = {
        "img1": {"发票号码": ["12345678"], "金额": ["100.00"]},
        "img2": {"发票号码": ["87654321"], "金额": ["200.50"]},
    }
    pred = {
        "img1": {"发票号码": ["12345678"], "金额": ["100.00"]},
        "img2": {"发票号码": ["87654321"], "金额": ["200.50"]},
    }
    imgnames = ["img1", "img2"]
    save_file = "test_score_output.json"

    print("=" * 50)
    print("_score 手动测试")
    print("=" * 50)
    mfh, mh = _score(gt, pred, imgnames, [], save_file, metric="rouge-l")
    print(f"\n返回结果: mFieldHmean={mfh}, MethodHmean={mh}")
    print(f"\n生成的表格文件: {save_file}")
    if os.path.exists(save_file):
        table = load_json(save_file)
        print(f"表格行数: {len(table)}")
        print("\n前 5 行:")
        for i, row in enumerate(table[:5]):
            print(f"  {i+1}. {row}")
    print("=" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        run_manual_test()
    else:
        unittest.main()

# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
from collections import defaultdict
from enum import Enum
from itertools import groupby
from typing import Dict, List

import Levenshtein
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from metric import Scorer
from scipy.optimize import linear_sum_assignment


def list_json(path):
    return [xx for xx in os.listdir(path) if xx.endswith("json")]


def load_json(file):
    with open(file) as f:
        return json.load(f)


def dump_json(x, file):
    with open(file, "w") as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
    return


def str2list(x):
    def flatten_to_values(data):
        if not data:
            return [""]
        if isinstance(data, dict):
            # 如果是字典，递归处理每个值
            values = []
            for value in data.values():
                values.extend(flatten_to_values(value))
            return values
        elif isinstance(data, (list, tuple)):
            # 如果是列表或元组，递归处理每个元素
            values = []
            for item in data:
                values.extend(flatten_to_values(item))
            return values
        elif isinstance(data, str):
            # 如果是其他类型，视为最内层的值
            return [data]
        elif isinstance(data, (int, float, bool)):
            return [str(data)]
        else:
            raise TypeError(f"val:{data}\n Unsupported val type: {type(data).__name__}")

    return flatten_to_values(x)


def load_yaml(conf_file):
    """
    :param conf_file: can be file path, or string, or bytes
    :return:
    """
    if os.path.isfile(conf_file):
        return yaml.load(open(conf_file), Loader=yaml.FullLoader)
    else:
        return yaml.load(conf_file, Loader=yaml.FullLoader)


def do_hungarian(x, y):
    # x, y are two list with equal size
    # if size == 1 or x or y contains identical element
    # then no need to do hungarian matching
    def is_contain_same(x):
        last_element = x[0]
        is_same = True
        for xx in x[1:]:
            if xx != last_element:
                is_same = False
                break
        return is_same

    assert len(x) == len(y), "do padding before calling hungarian matching algo"
    if len(x) == 1:
        return False

    if is_contain_same(x) or is_contain_same(y):
        return False

    return True


def align_two_dicts(x, y):
    for key in x:
        val_x = x[key]
        val_y = y.get(key, [""])
        # print(val_y)
        # if val_y is None:
        #     val_y = []
        val_y = str2list(val_y)
        N_x = len(val_x)
        N_y = len(val_y)
        N = max(N_x, N_y)
        # step1: padding
        if N_x > N_y:
            val_y += [None] * (N_x - N_y)
        elif N_x < N_y:
            val_x += [None] * (N_y - N_x)
        # step2: shall we do hungarian matching?
        if do_hungarian(val_x, val_y):
            # generate bipartite Graph with Levenshtein distance
            G = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if val_x[i] is None or val_y[j] is None:
                        distance = 1000
                    else:
                        # hungarian requires nonzero edge weight
                        distance = 1 + Levenshtein.distance(val_x[i], val_y[j])
                    G[i, j] = distance
            # step3 hungarian matching
            x_ind, y_ind = linear_sum_assignment(G)

            # step4 reorder
            val_x_ = []
            val_y_ = []
            for i, j in zip(x_ind, y_ind):
                val_x_.append(val_x[i])
                val_y_.append(val_y[j])
            val_x = val_x_
            val_y = val_y_
        x[key] = val_x
        y[key] = val_y

    for key in y:
        if key not in x:
            x[key] = [None] * len(y[key])
    return x, y


def determine_errortype(x, y):
    # x: gt_value, y: pred_value

    if x is None and y is None:
        return 0
    elif x is None:
        return 2
    elif y is None:
        return 1
    else:
        return 34


def hack_xy(text):
    def isdigit(x):
        for c in x:
            if not c.isdigit():
                return False
        return True

    if text is None:
        return 0, 0, None
    pure_text = None
    if len(text) > 6 and text.startswith("~$"):
        xy = text.split("~")[1][1:].split(".")
        if len(xy) == 2 and isdigit(xy[0]) and isdigit(xy[1]):
            x = int(xy[0])
            y = int(xy[1])
            symbol_length = len(f"~${x}.{y}~")
            pure_text = text[symbol_length:]

    if pure_text is None:
        x = 0
        y = 0
        pure_text = text
    return x, y, pure_text


class ScoreEngine:
    def __init__(
        self,
        gt: Dict[str, Dict[str, list]],
        pred: Dict[str, Dict[str, list]],
        imgnames: List[str],
        metric: str,
        exclude_keys=[],
        threshold=0.8,
        replace=False,
        merge=False,
    ):
        self.imgnames = imgnames
        self.gt = gt
        self.pred = pred
        self.scorer = Scorer(metric)
        self.threshold = threshold
        if merge:
            for key in self.gt.keys():
                for word in self.gt[key].keys():
                    self.gt[key][word] = ["".join(self.gt[key][word])]
            for key in self.pred.keys():
                for word in self.pred[key].keys():
                    self.pred[key][word] = ["".join(self.pred[key][word])]
        assert len(self.gt) == len(self.pred), "json files amount does not equal gt's"
        self.exclude_keys = exclude_keys

    def generate_bigtable(self, savepath):
        ret = []
        for imgname in self.gt:
            ret_image = self._table_from_one_image(imgname)
            ret += ret_image
        dump_json(ret, savepath)
        return ret

    def _table_from_one_image(self, imgname):
        gt = self.gt[imgname]
        pred = self.pred[imgname]
        gt, pred = align_two_dicts(gt, pred)
        ret = []
        for key in gt:
            if key in self.exclude_keys:
                continue
            val_gt = gt[key]
            val_pred = pred[key]
            if val_gt == ["##"] and val_pred == [None]:
                continue
            for vg, vp in zip(val_gt, val_pred):
                errortype = determine_errortype(vg, vp)
                if errortype == 34:
                    if vg == '' or vp == '':
                        correct = 1 if all([vg == '', vp == '']) else 0
                    else:
                        score_result = self.scorer.compute(vp, vg)
                        correct = 1 if score_result.score >= self.threshold else 0
                else:
                    correct = 0

                ret.append(
                    {
                        "imgname": imgname,
                        "key": key,
                        "gt_text": vg,
                        "pred_text": vp,
                        "error_type": errortype,
                        "correct": correct,
                    }
                )
        return ret

    def score_e2e(self, tablefile):
        assert os.path.isfile(tablefile)
        x = load_json(tablefile)
        N = 0
        N0 = 0
        for xx in x:
            N += 1
            N0 += xx["correct"]
        score_e2e = 0 if N == 0 else N0 / N
        print(f"score e2e: {score_e2e * 100:.2f} %")
        return score_e2e

    def cal_MethodHmean(self, tabelfile):
        assert os.path.isfile(tabelfile)
        x = load_json(tabelfile)
        FieldPrecision, FieldRecall, FieldHmean = self.cal_f_score(x)
        print(
            f"MethodPrecision: {FieldPrecision * 100:.2f} %, MethodRecall: {FieldRecall * 100:.2f} %, MethodHmean: {FieldHmean * 100:.2f} %"
        )
        return FieldPrecision, FieldRecall, FieldHmean

    def cal_mFieldHmean(self, tabelfile):
        assert os.path.isfile(tabelfile)
        x = load_json(tabelfile)
        res = defaultdict(list)
        for xx in x:
            res[xx["key"]].append(xx)

        mFieldPrecision, mFieldRecall, mFieldHmean = 0, 0, 0
        for field in res:
            p, r, h = self.cal_f_score(res[field])
            print(f"{field} p: {p}, r: {r}, h: {h}")
            mFieldPrecision += p
            mFieldRecall += r
            mFieldHmean += h

        Nfield = len(res)
        mFieldPrecision = 0 if Nfield == 0 else mFieldPrecision / Nfield
        mFieldRecall = 0 if Nfield == 0 else mFieldRecall / Nfield
        mFieldHmean = 0 if Nfield == 0 else mFieldHmean / Nfield
        print(
            f"mFieldPrecision: {mFieldPrecision * 100:.2f} %, mFieldRecall: {mFieldRecall * 100:.2f} %, mFieldHmean: {mFieldHmean * 100:.2f} %"
        )
        return mFieldPrecision, mFieldRecall, mFieldHmean

    def cal_f_score(self, tabel):
        N_pred, N_gt, N_right = 0, 0, 0
        for xx in tabel:
            N_pred += xx["pred_text"] is not None
            N_gt += xx["gt_text"] is not None
            N_right += xx["correct"]
        precision = 0 if N_pred == 0 else N_right / N_pred
        recall = 0 if N_gt == 0 else N_right / N_gt
        f_score = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f_score


def parse_json(json_string: str) -> dict:
    match = re.search(r"```(json)?(.*)```", json_string, re.DOTALL)
    if match is None:
        json_str = json_string
    else:
        json_str = match.group(2)

    json_str = json_str.strip()
    json_str = json_str.replace("```", "")

    match = re.search(r"{.*}", json_str, re.DOTALL)
    if match is None:
        json_str = json_str
    else:
        json_str = match.group(0)
    if json_str.endswith("}\n}"):
        json_str = json_str[:-2]
    if json_str.startswith("{\n{"):
        json_str = json_str.replace("{\n{", "{", 1)

    # logger.info(f'llm response after parse: {json_str}')
    extract_res = json.loads(json_str)
    return extract_res


def _score(gt, pred, imgnames, exclude_keys, save_file, metric):
    SE = ScoreEngine(gt, pred, imgnames, exclude_keys=exclude_keys, replace=False, merge=False, metric=metric)
    SE.generate_bigtable(save_file)  # 将结果保存到tabelfile文件内
    mfh = SE.cal_mFieldHmean(save_file)[-1]  # 按字段平均计算f1_score
    mh = SE.cal_MethodHmean(save_file)[-1]  # 总体平均计算f1_score
    logger.info(f"mFieldHmean: {mfh}, MethodHmean: {mh}")
    return mfh, mh


def run_score(pred_file, val_file, save_dir, metric, exclude_keys=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(val_file, "r") as f:
        val_data = json.load(f)

    with open(pred_file, "r") as f:
        pred_data = list(map(json.loads, f.readlines()))

    assert len(val_data) == len(pred_data), "val_data and pred_data length not equal"
    val_data.sort(key=lambda x: x['name'])
    pred_data.sort(key=lambda x: x['meta_data']['name'])
    zip_data = zip(val_data, pred_data)
    zip_data = sorted(zip_data, key=lambda x: x[0]["scene"])
    groupby_scene = groupby(zip_data, key=lambda x: x[0]["scene"])

    all_mfh = list()
    all_mh = list()
    all_imgnames = list()
    df = pd.DataFrame(columns=["scene", "样本数", "mFieldHmean", "MethodHmean", 'type'])
    for scene, group_data in groupby_scene:
        logger.info(f"scene: {scene}")
        scene_gt = {}
        scene_pred = {}
        scene_imgnames = []
        for val, pred in group_data:
            image_name = val["name"]
            assert val['output'] == pred['label'], f"{scene} {image_name} label not equal {pred['meta_data']['name']}"
            label = val['output']
            # label = pred['label']
            predict = pred["predict"]
            label = parse_json(label)
            for key in label.keys():
                if not len(label[key]):
                    label[key] = [""]
            try:
                predict = parse_json(predict)
                for key in predict.keys():
                    if not len(predict[key]):
                        predict[key] = [""]
                    elif isinstance(predict[key], str):
                        predict[key] = [predict[key]]
                    elif isinstance(predict[key], dict):
                        predict[key] = [str(predict[key])]
                    else:
                        predict[key] = list(set(predict[key]))
            except Exception as e:
                logger.error(f"{scene} {image_name} pred error: {e}")
                continue

            scene_gt[image_name] = label
            scene_pred[image_name] = predict
            scene_imgnames.append(image_name)
        save_file = os.path.join(save_dir, f"{scene}.json")
        scene_mfh, scene_mh = _score(scene_gt, scene_pred, scene_imgnames, exclude_keys, save_file, metric=metric)
        df.loc[len(df)] = [scene, len(scene_imgnames), scene_mfh, scene_mh, val.get('type', '-')]

        all_mfh.append(scene_mfh)
        all_mh.append(scene_mh)
        all_imgnames.extend(scene_imgnames)
    df.loc[len(df)] = ["综合", len(all_imgnames), sum(all_mfh) / len(all_mfh), sum(all_mh) / len(all_mh), '-']
    df.to_excel(os.path.join(save_dir, "score.xlsx"), index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--pred_file", type=str, required=True)
    args.add_argument("--val_file", type=str, required=True)
    args.add_argument("--save_dir", type=str, required=True)
    args.add_argument("--metric", type=str, required=True, choices=["rouge-l", "bleu-4", "meteor", "rocr"])
    args.add_argument("--exclude_keys", type=str, default="")
    args = args.parse_args()

    run_score(args.pred_file, args.val_file, args.save_dir, args.metric, args.exclude_keys)

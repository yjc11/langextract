# **coding: utf-8**
import json
import re
import sys
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union

sys.path.append(str(Path(__file__).parent))


import jieba
import opencc
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from pydantic import BaseModel
from rocr_scoring.analyse import compare
from rouge_chinese import Rouge

UNIT_SET = {'$', '元', '元整', '圆整', '圆', '整', '美元', '人民币', 'm2', 'm', '万元整', '万元'}


def normalize_answer(s):
    s = s.lower()
    cc = opencc.OpenCC('t2s')
    s = cc.convert(s)
    pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
    cleaned_text = pattern.sub('', s)
    return cleaned_text


class Metrics(Enum):
    ROCR = 'rocr'
    ROUGE_L = 'rouge-l'
    BLEU_4 = 'bleu-4'
    METEOR = 'meteor'


class Metric(ABC):
    @abstractmethod
    def compute(self, pred: str, label: str, **kwargs) -> Union[int, float]: ...


class ROUGE_L(Metric):
    def compute(self, pred: str, label: str, **kwargs) -> float:
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge(metrics=['rouge-l'], **kwargs)
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        result = scores[0]
        return result['rouge-l']['f']


class BLEU_4(Metric):
    def compute(self, pred: str, label: str) -> float:
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        bleu_score = sentence_bleu(
            [hypothesis],
            reference,
            smoothing_function=SmoothingFunction().method3,
        )
        return bleu_score


class METEOR(Metric):
    def compute(self, pred: str, label: str) -> float:
        if pred == label:
            return 1.0

        if ''.join(set(pred) ^ set(label)) in UNIT_SET:
            return 1.0
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        meteor_s = single_meteor_score(reference, hypothesis)
        return meteor_s


class ROCR(Metric):
    def compute(self, pred: str, label: str) -> int:
        rocr_score = compare(
            pred,
            label,
            with_punctuation=True,
            use_dp=True,
            keep_space=False,
            upper=False,
        )
        return rocr_score[0]


class ScorerOutput(BaseModel):
    score: Union[int, float]


class Scorer:
    metrics = {
        Metrics.ROUGE_L: ROUGE_L,
        Metrics.BLEU_4: BLEU_4,
        Metrics.METEOR: METEOR,
        Metrics.ROCR: ROCR,
    }

    def __init__(self, metric: str):
        self.metric = self.metrics[Metrics(metric)]()

    def compute(self, pred: str, label: str, **kwargs) -> ScorerOutput:
        pred = normalize_answer(pred)
        label = normalize_answer(label)
        # print(pred,label)
        return ScorerOutput(
            score=self.metric.compute(pred=pred, label=label, **kwargs),
        )


if __name__ == '__main__':
    pair = {
        "gt_text": "this 14th day of February, 2020",
        "pred_text": "14thdayofFebruary,2020",
    }

    scorer = Scorer(metric='rocr')
    print(scorer.compute(pair['pred_text'], pair['gt_text']))
    # pair = [
    #     "【84】个月自【2021】年【12】月【20】日(注:筹备期广场填写装修期届满次日,营运期广场填写房产交付日)至【2028】年【12】月【19】日止",
    #     "自【2021】年【12】月【20】日(注:筹备期广场填写装修期届满次日,营运期广场填写房产交付日)至【2028】年【12】月【19】日止",
    #     "2010年5月15日至2020年4月30日止",
    #     "2010年5月15日至2020年4月30日",
    #     "双方签字盖章后生效",
    #     "双方签字盖章后生效,传真件有效",
    #     "IDC咨询(北京)有限公司(IDCChina)",
    #     "IDC咨询(北京)有限公司",
    #     "叁拾壹万捌仟零玖拾伍",
    #     "叁拾壹万捌仟零玖拾伍圆整",
    #     "本合同自双方盖章之日起生效,至保修期到期完毕后失效",
    #     "本合同自双方盖章之日起生效",
    #     "15000.00",
    #     "15000",
    #     "2020年1月20日",
    #     "2020年01月20日",
    #     "3年",
    #     "有效期为3年",
    # ]

    # for i in range(0, len(pair), 2):
    #     print(pair[i], pair[i + 1])
    #     print("ROUGE-L")
    #     print(Scorer(metric='rouge-l').compute(pair[i + 1], pair[i]))
    #     print("METEOR")
    #     print(Scorer(metric='meteor').compute(pair[i + 1], pair[i]))

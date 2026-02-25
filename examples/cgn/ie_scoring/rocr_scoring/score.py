import json
import os
from shutil import copy

import pandas as pd

from .core.std_recog_score import compare


def load_json(filepath):
    with open(filepath) as f:
        x = json.load(f)
    return x


def return_dict(img_list, image_dict):
    ret_dict = dict()
    for img_name in img_list:
        ret_dict[img_name] = image_dict[img_name]
    return ret_dict


def gen_label_json(imgnames, wrong_image_dict, img_dir, save_dir):
    text = {
        'ImageQuality': [],
        'ErrorLevel': ['\u6b63\u786e'],
        'ErrorTag': [],
        'predict_label': '',
        'gt_label': '##'
    }
    for image_name in imgnames:
        label, pred = wrong_image_dict[image_name]['label'], wrong_image_dict[
            image_name]['pred']
        copy(os.path.join(img_dir, image_name),
             os.path.join(save_dir, image_name))
        text['gt_label'] = label
        text['predict_label'] = pred
        image_name_ex = image_name.split('.')[0]
        with open(os.path.join(save_dir, image_name_ex + '.json'), 'w') as f:
            json.dump(text, f)


def compare_two_files(pred,
                      label,
                      mode='normal',
                      exclude_tags=[],
                      use_dp=False,
                      keep_space=False,
                      upper=False):
    assert mode in ['normal', 'baidu']
    memo = {}
    memo_rare = {}
    memo_CN_TW = {}
    wrong_punc_image = dict()
    wrong_depunc_image = dict()
    for key in label:
        items = label[key]
        label_val = items['value'][-1]
        scene = items['scene'][-1]
        tags = items['tags']

        exclude = False
        for tag in exclude_tags:
            if tag in tags:
                exclude = True
        if exclude:
            continue

        try:
            pred_val = pred[key]['value'][-1]
        except:
            # print('%s not found in submission file' % key)
            pred_val = ''

        # baidu special treatment
        if (mode == 'baidu') and (pred_val == '' or pred_val == '##'):
            continue

        score_punc, cer = compare(pred_val, label_val, True, use_dp,
                                  keep_space, upper)
        score_depunc, _ = compare(pred_val, label_val, False, use_dp,
                                  keep_space, upper)

        if 'rare' in tags:
            if scene not in memo_rare:
                memo_rare[scene] = {
                    'N': 0,
                    'N_punc': 0,
                    'N_depunc': 0,
                    'cer': 0,
                    'result': []
                }
            memo_rare[scene]['N'] += 1
            memo_rare[scene]['N_punc'] += score_punc
            memo_rare[scene]['N_depunc'] += score_depunc
            memo_rare[scene]['cer'] += cer
            memo_rare[scene]['result'].append([label_val, pred_val])
        elif 'CN-TW' in tags:
            if scene not in memo_CN_TW:
                memo_CN_TW[scene] = {
                    'N': 0,
                    'N_punc': 0,
                    'N_depunc': 0,
                    'cer': 0,
                    'result': []
                }
            memo_CN_TW[scene]['N'] += 1
            memo_CN_TW[scene]['N_punc'] += score_punc
            memo_CN_TW[scene]['N_depunc'] += score_depunc
            memo_CN_TW[scene]['cer'] += cer
            memo_CN_TW[scene]['result'].append([label_val, pred_val])
        else:
            if scene not in memo:
                memo[scene] = {
                    'N': 0,
                    'N_punc': 0,
                    'N_depunc': 0,
                    'cer': 0,
                    'result': []
                }
            memo[scene]['N'] += 1
            memo[scene]['N_punc'] += score_punc
            memo[scene]['N_depunc'] += score_depunc
            memo[scene]['cer'] += cer
            memo[scene]['result'].append([label_val, pred_val])

        if not score_punc:
            wrong_punc_image[key] = dict()
            wrong_punc_image[key]['label'] = label_val
            wrong_punc_image[key]['pred'] = pred_val
            wrong_punc_image[key]['scene'] = scene
            wrong_punc_image[key]['tag'] = tags
        if not score_depunc:
            wrong_depunc_image[key] = dict()
            wrong_depunc_image[key]['label'] = label_val
            wrong_depunc_image[key]['pred'] = pred_val
            wrong_depunc_image[key]['scene'] = scene
            wrong_depunc_image[key]['tag'] = tags
    return wrong_punc_image, wrong_depunc_image, memo, memo_rare, memo_CN_TW


def pandas_table(memo):
    panda = []
    N = 0
    N_punc = 0
    N_depunc = 0
    N_cer = 0
    for key in memo:
        N_key = memo[key]['N']
        N_punc_key = memo[key]['N_punc']
        N_depunc_key = memo[key]['N_depunc']
        N_cer_key = memo[key]['cer']
        N += N_key
        N_punc += N_punc_key
        N_depunc += N_depunc_key
        N_cer += N_cer_key
        line = [key, N_key, N_punc_key, N_depunc_key, N_cer_key]
        panda.append(line)
    panda.append(['总和', N, N_punc, N_depunc, N_cer])
    df = pd.DataFrame(panda)
    df.columns = ['scene', 'N', 'N_punc', 'N_depunc', 'N_cer']
    df['acc_punc'] = df['N_punc'] / df['N']
    df['acc_depunc'] = df['N_depunc'] / df['N']
    df['avg_cer'] = df['N_cer'] / df['N']
    return df


def score(pred,
          label,
          mode,
          exclude_tags=[],
          show_result=False,
          use_dp=False,
          keep_space=False,
          upper=False):
    wrong_punc_image, wrong_depunc_image, memo, memo_rare, memo_CN_TW = compare_two_files(
        pred, label, mode, exclude_tags, use_dp, keep_space, upper)
    df = pandas_table(memo)
    df_rare = pandas_table(memo_rare)
    df_CN_TW = pandas_table(memo_CN_TW)

    if show_result:
        if len(memo) > 0:
            print('*' * 25)
            print('常见字')
            print(df)

        if len(memo_rare) > 0:
            print('*' * 25)
            print('生僻字')
            print(df_rare)

        if len(memo_CN_TW) > 0:
            print('*' * 25)
            print('繁体字')
            print(df_CN_TW)
    return wrong_punc_image, wrong_depunc_image, df, df_rare, df_CN_TW

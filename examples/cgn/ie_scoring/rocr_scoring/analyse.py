# @Time    : 2020/8/10 2:44 下午
# @Author  : ZhangYue

import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core.std_recog_score import compare
from .score import gen_label_json, load_json, return_dict, score


class BadCase:
    def __init__(self,
                 pred_path_list,
                 label_path,
                 mode='normal',
                 exclude_tags=[],
                 show_result=False,
                 use_dp=False,
                 keep_space=False,
                 upper=False):
        self.mode = mode
        self.exclude_tags = exclude_tags
        self.use_dp = use_dp
        self.label = load_json(label_path)
        self.predictions = []
        self.wrong_punc_image = [] 
        self.wrong_depunc_image = []
        self.df = []
        self.df_rare = []
        self.df_CN_TW = []

        if isinstance(pred_path_list, str):
            pred_path_list = [pred_path_list]

        for i, pred_path in enumerate(pred_path_list):
            print("*" * 25)
            print(pred_path)
            print()
            prediction = load_json(pred_path)
            wrong_punc_image, wrong_depunc_image, df, df_rare, df_CN_TW = score(
                prediction, self.label, mode, exclude_tags, show_result,
                use_dp, keep_space, upper)
            self.predictions.append(prediction)
            self.wrong_punc_image.append(wrong_punc_image)
            self.wrong_depunc_image.append(wrong_depunc_image)
            self.df.append(df)
            self.df_rare.append(df_rare)
            self.df_CN_TW.append(df_CN_TW)
            print()

    @staticmethod
    def scene_count(df):
        """分场景统计错误数量.

        Args:
            wrong_image: Dict. 需要统计的图像信息

        Returns:
            List. 按数量从多到少排序后的场景-数量对
        """
        scene_punc_count = dict()
        scene_depunc_count = dict()
        for i in range(len(df)):
            scene_punc_count[df['scene'][i]] = df['N'][i] - df['N_punc'][i]
            scene_depunc_count[df['scene'][i]] = df['N'][i] - df['N_depunc'][i]
        return sorted(scene_punc_count.items(),
                      key=lambda x: x[1],
                      reverse=True), sorted(scene_depunc_count.items(),
                                            key=lambda x: x[1],
                                            reverse=True)

    def show_image(self,
                   img_dir,
                   prediction,
                   scene=[],
                   gray=False,
                   max_show=50):
        """

        Args:
            img_dir: Str. 图像的文件夹地址
            wrong_image: Dict. Dict[image name]={'label': xx, 'pred': xx, 'scene': xx}
            scene: List. 要显示的图像场景
            gray: Bool. 是否显示灰度图

        Returns:

        """
        image_names = list(self.label.keys())
        if scene:
            image_names = list(
                filter(lambda x: self.label[x]['scene'][0] in scene,
                       image_names))
        print('共有{}张图片'.format(len(image_names)))
        if max_show:
            random.shuffle(image_names)
            image_names = image_names[:max_show]
        for img_name in image_names:
            label_val, pred_val = self.label[img_name]['value'][
                -1], prediction[img_name]['value'][-1]

            print('label:{}\nprediction:{}'.format(label_val, pred_val))
            image = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)),
                                 cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            w_resize = int(32 * w / h)
            image = cv2.resize(image, (w_resize, 32))
            new_image = 255 * np.ones(
                shape=[32, max(500, w_resize), 3]).astype(np.uint8)
            new_image[:, :w_resize, :] = image
            fig_w = max(500, w_resize)
            plt.figure(figsize=(4, int(fig_w // 8)), dpi=200)
            if gray:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(new_image)
            plt.tick_params(labelsize=5)
            plt.title(img_name + '，h=' + str(h), fontsize='small')
            plt.show()

    def show_wrong_image(self,
                         img_dir,
                         wrong_image,
                         scene=[],
                         tag=[],
                         gray=False,
                         max_show=None):
        """

        Args:
            img_dir: Str. 图像的文件夹地址
            wrong_image: Dict. Dict[image name]={'label': xx, 'pred': xx, 'scene': xx}
            scene: List. 要显示的图像场景
            tag: List. []为全部，['common']为常规字，['rare']为生僻字， ['CN-TW']为繁体字
            gray: Bool. 是否显示灰度图

        Returns:

        """
        wrong_image_name = list(wrong_image.keys())
        if scene:
            wrong_image_name = list(
                filter(lambda x: wrong_image[x]['scene'] in scene,
                       wrong_image_name))
        if tag:
            wrong_image_name = list(
                filter(
                    lambda x: set(wrong_image[x]['tag']).intersection(set(
                        tag)), wrong_image_name))

        print('共有{}张图片'.format(len(wrong_image_name)))
        if max_show:
            random.shuffle(wrong_image_name)
            wrong_image_name = wrong_image_name[:max_show]
        for img_name in wrong_image_name:
            label_val, pred_val = wrong_image[img_name]['label'], wrong_image[
                img_name]['pred']
            print(img_name)

            print('label:{}\nprediction:{}'.format(label_val, pred_val))

            image = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)),
                                 cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            w_resize = int(32 * w / h)
            image = cv2.resize(image, (w_resize, 32))
            new_image = 255 * np.ones(
                shape=[32, max(500, w_resize), 3]).astype(np.uint8)
            new_image[:, :w_resize, :] = image
            fig_w = max(500, w_resize)
            plt.figure(figsize=(4, int(fig_w // 8)), dpi=200)
            if gray:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(new_image)
            plt.tick_params(labelsize=5)
            plt.title(img_name + '，h=' + str(h), fontsize='small')
            plt.show()
        return wrong_image_name

    @staticmethod
    def get_compare_pandas_table(df_list):
        col_name_list = [
            'N_punc', 'N_depunc', 'acc_punc', 'acc_depunc', 'avg_cer'
        ]
        col_names = ['scene', 'N']
        for col_name in col_name_list:
            col_names.extend([col_name + str(i) for i in range(len(df_list))])
        df = pd.DataFrame(columns=col_names)
        df['scene'] = df_list[0]['scene']
        df['N'] = df_list[0]['N']
        for i in range(len(df_list)):
            for col_name in col_name_list:
                df[col_name + str(i)] = df_list[i][col_name]
        return df

    def compare_scene(self, df1, df2, key='acc_depunc'):
        """分场景对比两个结果的好坏, 返回df1好于df2，df1差于df2的场景.

        Returns:
            List.
        """
        good_scenes = list()
        bad_scenes = list()
        for i, scene in enumerate(df1['scene']):
            if df1[key][i] < df2[key][i]:
                good_scenes.append([scene, df2[key][i] - df1[key][i]])
            elif df1[key][i] > df2['acc_depunc'][i]:
                bad_scenes.append([scene, df1[key][i] - df2[key][i]])
        print('1比2表现好的场景：\n')
        bad_scenes = sorted(bad_scenes, key=lambda x: x[-1], reverse=True)
        for scene in bad_scenes:
            print(scene)
        print('1比2表现差的场景：\n')
        good_scenes = sorted(good_scenes, key=lambda x: x[-1], reverse=True)
        for scene in good_scenes:
            print(scene)

    def compare_wrong_prediction(self, wrong_image1, wrong_image2, tag=[]):
        """返回两个结果相同和不同的图像名.

        Returns:
            List,List,List: 1错2对，1对2错，1错2错的图像名
        """
        wrong_right, right_wrong, wrong_wrong = list(set(wrong_image1).difference(wrong_image2)),\
                                                list(set(wrong_image2).difference(wrong_image1)),\
                                                list(set(wrong_image1.keys()).intersection(wrong_image2.keys()))
        if tag:
            wrong_right = list(
                filter(
                    lambda x: set(wrong_image1[x]['tag']).intersection(set(
                        tag)), wrong_right))
            right_wrong = list(
                filter(
                    lambda x: set(wrong_image2[x]['tag']).intersection(set(
                        tag)), right_wrong))
            wrong_wrong = list(
                filter(
                    lambda x: set(wrong_image1[x]['tag']).intersection(set(
                        tag)), wrong_wrong))
        return return_dict(wrong_right, wrong_image1), return_dict(
            right_wrong, wrong_image2), return_dict(wrong_wrong, wrong_image1)

    def generate_label_json(self, pic_num, wrong_image_dict, img_dir,
                            save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        imgnames = list(wrong_image_dict.keys())
        random.shuffle(imgnames)
        imgnames = imgnames[:pic_num]
        gen_label_json(imgnames, wrong_image_dict, img_dir, save_dir)
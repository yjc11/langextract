import math

import numpy as np


class RCIndex:
    def __init__(self, y, x, idx, row_index):
        self.y = y
        self.x = x
        self.idx = idx
        self.row_index = row_index

    def __repr__(self):
        return f"RCIndex(y={self.y:.2f}, x={self.x:.2f}, idx={self.idx}, row={self.row_index})"


class BBoxOps:

    @staticmethod
    def l2_norm(p1, p2):
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def compute_point_line_distance(p1, p2, direction):
        """
        p1: point on line
        p2: query point
        direction: line direction vector
        """
        a = p2 - p1
        b = direction
        m = np.dot(a, b)
        n = np.dot(b, b)
        c = b * (m / n)
        p = a - c
        e = np.sqrt(np.dot(p, p))

        # 右手旋转为正，左手旋转为负
        tmp = a[1] * b[0] - b[1] * a[0]
        return e if tmp < 0 else -e

    @staticmethod
    def normalize_point(p):
        norm = math.sqrt(p[0] * p[0] + p[1] * p[1])
        if norm == 0 or math.isnan(norm):
            return np.array([1.0, 0.0], dtype=np.float32)
        p_norm = p / norm
        if np.isnan(p_norm).any():
            return np.array([1.0, 0.0], dtype=np.float32)
        return p_norm

    @staticmethod
    def compute_bbox_stats(bbox):
        """
        bbox: numpy array (n, 4, 2)
        return: h_mean, hori_vec
        """
        n = bbox.shape[0]

        hori_xs = []
        hori_ys = []
        h_values = []

        for i in range(n):
            p0 = bbox[i, 0]
            p1 = bbox[i, 1]
            p2 = bbox[i, 2]
            p3 = bbox[i, 3]

            p_hori = (p1 + p2) / 2 - (p0 + p3) / 2

            hori_xs.append(p_hori[0])
            hori_ys.append(p_hori[1])

            h1 = BBoxOps.l2_norm(p0, p3)
            h2 = BBoxOps.l2_norm(p1, p2)
            h_values.append((h1 + h2) / 2)

        hori_xs = sorted(hori_xs)
        hori_ys = sorted(hori_ys)

        start = max(0, int(n * 0.25))
        end = min(n - 1, int(n * 0.75))
        count = end - start + 1

        mean_x = sum(hori_xs[start : end + 1]) / count
        mean_y = sum(hori_ys[start : end + 1]) / count

        hori_vec = BBoxOps.normalize_point(np.array([mean_x, mean_y], dtype=np.float32))
        h_mean = sum(h_values) / n

        return h_mean, hori_vec

    @staticmethod
    def rearrange_boxes(bbox, text, align_mode):
        """
        Args:
            bbox: (n, 4, 2)
            text: list of string (ignored in sorting logic)
            align_mode: 0 — center = (p0+p2)/2   其它 — (p2+p3)/2
        Returns:
            rc_indexes (list of RCIndex)
        """
        h_mean, hori_vec = BBoxOps.compute_bbox_stats(bbox)

        x_direction = hori_vec
        # rotate 90 deg CCW
        y_direction = np.array([x_direction[1], -x_direction[0]], dtype=np.float32)
        root = np.array([0.0, 0.0], dtype=np.float32)

        n = bbox.shape[0]
        rc_values = []

        for i in range(n):
            p0 = bbox[i, 0]
            p2 = bbox[i, 2]
            p3 = bbox[i, 3]

            # 选择中心
            if align_mode == 0:
                center = (p0 + p2) / 2
            else:
                center = (p2 + p3) / 2

            y = BBoxOps.compute_point_line_distance(root, center, x_direction)
            x = BBoxOps.compute_point_line_distance(root, center, y_direction)
            rc_values.append(RCIndex(y, x, i, 0))

        # 按 y 排序（从大到小）
        rc_values.sort(key=lambda v: v.y, reverse=True)

        # 行分段
        y_last = -1e9
        row_index = -1
        for v in rc_values:
            if abs(v.y - y_last) > h_mean * 0.5:
                row_index += 1
                y_last = v.y
            v.row_index = row_index

        # 按 (row, x 从大到小) 排序
        rc_values.sort(key=lambda v: (v.row_index, -v.x))

        return rc_values

    @staticmethod
    def process_ocr_to_text(bboxes, texts, align_mode=0):
        """
        将 OCR 结果处理成按行排列的文本，每个文本分配对应的 bbox

        Args:
            bboxes: numpy array, 形状为 (n, 4, 2)，表示 n 个文本框的 4 个顶点坐标
            texts: list of string, 长度为 n，表示每个文本框对应的文本内容
            align_mode: int, 对齐模式，0 表示 center = (p0+p2)/2，其它值表示 (p2+p3)/2

        Returns:
            reordered_texts: str，按行排列的文本内容
            reordered_bboxes: list，按行排列的 bbox，空格和换行符为 None

        Example:
            >>> bboxes = np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...])
            >>> texts = ['文本1', '文本2', ...]
            >>> reordered_texts, reordered_bboxes = BBoxOps.process_ocr_to_text(bboxes, texts)
            >>> # reordered_texts = '文本1 文本2\n文本3 文本4'
            >>> # reordered_bboxes = [bbox1, None, bbox2, None, bbox3, None, bbox4, None]
        """
        if len(bboxes) == 0:
            return []
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        rc_values = BBoxOps.rearrange_boxes(bboxes, texts, align_mode)

        reordered_texts = ''
        reordered_bboxes = []
        previous_row_index = -1
        for rc_value in rc_values:
            if previous_row_index == -1:
                reordered_texts = texts[rc_value.idx]
                reordered_bboxes = [bboxes[rc_value.idx]] * len(texts[rc_value.idx])
            elif rc_value.row_index != previous_row_index:
                reordered_texts += '\n' + texts[rc_value.idx]
                reordered_bboxes += [None] + [bboxes[rc_value.idx]] * len(texts[rc_value.idx])
            else:
                reordered_texts += ' ' + texts[rc_value.idx]
                reordered_bboxes += [None] + [bboxes[rc_value.idx]] * len(texts[rc_value.idx])
            previous_row_index = rc_value.row_index
        assert len(reordered_texts) == len(
            reordered_bboxes
        ), f"reordered_texts: {len(reordered_texts)}, reordered_bboxes: {len(reordered_bboxes)}"
        return reordered_texts, reordered_bboxes

    @staticmethod
    def xyxy_to_four_points(xyxy):
        """
        将两点坐标 (xyxy格式) 转换为四点坐标

        Args:
            xyxy: list or array, 格式为 [x1, y1, x2, y2] 或 [[x1, y1], [x2, y2]]
                  表示矩形框的左上角和右下角坐标

        Returns:
            numpy array: 形状为 (4, 2)，表示四个顶点坐标
                        顺序为: 左上(p0), 右上(p1), 右下(p2), 左下(p3)

        Example:
            >>> xyxy = [10, 20, 100, 80]
            >>> points = BBoxOps.xyxy_to_four_points(xyxy)
            >>> # points = [[10, 20], [100, 20], [100, 80], [10, 80]]
        """
        if isinstance(xyxy, (list, tuple, np.ndarray)):
            if len(xyxy) == 4:
                # 格式: [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            elif len(xyxy) == 2:
                # 格式: [[x1, y1], [x2, y2]]
                x1, y1 = xyxy[0]
                x2, y2 = xyxy[1]
            else:
                raise ValueError(f"Invalid xyxy format, expected length 2 or 4, got {len(xyxy)}")
        else:
            raise TypeError(f"xyxy must be list, tuple or numpy array, got {type(xyxy)}")

        # 四个顶点: 左上, 右上, 右下, 左下
        points = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32  # p0: 左上  # p1: 右上  # p2: 右下  # p3: 左下
        )

        return points

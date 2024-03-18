#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
#
#         Copyright (C) 2022 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : plot_utils.py
#   Author      : Hardy
#   Created date: 2022/11/16 19:30
#   Description : 绘制框和标签
#
# ======================================================================

import random

import cv2
import numpy as np
from ultralytics.utils.plotting import colors

#from data_class import Rect, Drawer, LabelBox
from data_class import  LabelBox
# from ..core.settings import settings as s


# def plot_drawer(image: np.ndarray, drawer: Drawer, width: int = 2560, space_mark: str = '0') -> np.ndarray:
#     """
#     绘制抽屉框
#     Args:
#         image: 待绘制图片
#         drawer: 抽屉框
#         width: 图片宽度
#         space_mark: 0-冷藏室 1-冷冻室
#
#     Returns:
#         是否保存
#     """
#     for d in [drawer.c_left, drawer.c_right, drawer.c_top, drawer.d_top, drawer.d_bottom]:
#         image = plot_box(image, LabelBox(*d.xyxy, ename='drawer', prob=d.prob)) if d else image
#     if space_mark == '0':
#         image = plot_line(image=image, dp=drawer.c_part, width=width)
#         image = plot_line2(image=image, dp=drawer.c_part, width=width)
#     return image


def plot_line(image: np.ndarray, dp: int, width: int = 2560) -> np.ndarray:
    """
    绘制进出冷藏室边界
    Args:
        image: 原始图片数组
        dp: 边界线
        width: 线宽度

    Returns:
        绘制后图像数组
    """
    cv2.line(image, (0, dp), (width, dp), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    return image


def plot_line2(image: np.ndarray, dp: int, width: int = 2560) -> np.ndarray:
    """
    绘制进出冷藏室边界
    Args:
        image: 原始图片数组
        dp: 边界线
        width: 线宽度

    Returns:
        绘制后图像数组
    """
    for x in range(width):
        cv2.circle(image, (x, dp - int((0.01 * (x - width / 2)) ** 2)), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    return image


def plot_box(image: np.ndarray, x: LabelBox) -> np.ndarray:
    """
    在图片数据上绘制框线及 label
    Args:
        image: 原始图片数组
        x: 绘制框信息

    Returns:
        绘制后图像数组
    """
    if x is None:
        return image
    tl = round(0.001 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    random.seed(x.ename)  # 设置随机数种子，保证每个 label 颜色一样
    color = [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (x.box.x1, x.box.y1), (x.box.x2, x.box.y2)
    image=np.int32(image)
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    label = f'{x.name if cv2.__version__ >= "5" else x.ename} {round(x.prob, 2)}'
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3 if c1[1] > t_size[1] + 3 else c1[1] + t_size[1] + 3
    cv2.rectangle(image, c1, c2, color, thickness=-1, lineType=cv2.LINE_AA)  # filled
    ct = (c1[0], c1[1] - 2 if c1[1] > t_size[1] + 3 else c1[1] + t_size[1] + 1)
    cv2.putText(image, label, ct, 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    #c_point = (x.cx, x.cy + x.box.height // s.hand_center_m if 'hand' in x.ename else x.cy)
    c_point = (x.cx, x.cy)
    cv2.circle(image, c_point, 3, color, -1, lineType=cv2.LINE_AA)
    return image


skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


def plot_kpts(image: np.ndarray, kpts: np.ndarray, width: int = 2560, height: int = 1440) -> np.ndarray:
    """
    绘制人体姿态

    0: 'nose',
    1: 'left_eye',  2: 'right_eye',
    3: 'left_ear',  4: 'right_ear',
    5: 'left_shoulder',  6: 'right_shoulder',
    7: 'left_elbow',  8: 'right_elbow',
    9: 'left_wrist',  10: 'right_wrist',
    11: 'left_hip',  12: 'right_hip',
    13: 'left_knee',  14: 'right_knee',
    15: 'left_ankle',  16: 'right_ankle'

    Args:
        image: 原始图片
        kpts: 预测关键点 [17, 3]. Each keypoint has (x, y, confidence)
        width: 图片宽度
        height: 图片高度

    Returns:
        绘制后图片
    """
    for i, k in enumerate(kpts):  # 绘点
        if k[2] < s.person_kpt_prob:
            continue
        color_k = [int(x) for x in kpt_color[i]]
        cv2.circle(image, (int(k[0]), int(k[1])), 5, color_k, -1, lineType=cv2.LINE_AA)
        cv2.putText(image, f'{i}', (int(k[0]) + 3, int(k[1])), 0, 1, color_k, lineType=cv2.LINE_AA)
    for i, sk in enumerate(skeleton):  # 绘线
        pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
        pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
        if kpts[(sk[0] - 1), 2] < s.person_kpt_prob or kpts[(sk[1] - 1), 2] < s.person_kpt_prob:
            continue
        if pos1[0] % width == 0 or pos1[1] % height == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % width == 0 or pos2[1] % height == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(image, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
    return image


def plot_polygon(image: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    绘制多边形框
    Args:
        image: 原始图片数组
        x: 绘制框信息

    Returns:
        绘制后图像数组
    """
    tl = round(0.001 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    cv2.polylines(image, [x], True, [23, 215, 95], thickness=tf, lineType=cv2.LINE_AA)
    return image


def put_name(image: np.ndarray, text: str) -> np.ndarray:
    """
    在图片上添加文字
    Args:
        image: 原始图片数组
        text: 文本内容

    Returns:
        绘制后图像数组
    """
    c = round(0.01 * (image.shape[0] + image.shape[1]) / 2) + 20
    tf = max(round(0.001 * (image.shape[0] + image.shape[1]) / 2), 1)  # line/font thickness
    cv2.putText(image, text, (c, c), 0, 1, [0, 0, 255], thickness=tf, lineType=cv2.LINE_AA)
    return image


# def plot_mosaic(image: np.ndarray, rects: list[Rect], neighbor: int = 9) -> np.ndarray:
#     """
#     给图片 rects 以外的区域打上马赛克
#     Args:
#         image: opencv frame
#         rects: 保留区域
#         neighbor: 马赛克每一块的宽
#
#     Returns:
#         打码后图片
#     """
#     frame_mosaic = np.zeros(image.shape, dtype='uint8')
#     for i in range(0, image.shape[0], neighbor):  # 全局打码
#         for j in range(0, image.shape[1], neighbor):
#             frame_mosaic[i:i + neighbor, j:j + neighbor] = image[i, j]
#     for r in rects:  # 部分恢复
#         frame_mosaic[r.y1:r.y2, r.x1:r.x2] = image[r.y1:r.y2, r.x1:r.x2]
#     return frame_mosaic

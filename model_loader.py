#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
#
#         Copyright (C) 2024 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : model_loader.py
#   Author      : Hardy
#   Created date: 2024/02/20 10:28
#   Description : 模型加载
#
# ======================================================================


import json

import cv2
import numpy as np
import torch
from torchvision import transforms as tf

from .resnet import *
from .transformer import *
from ...core.settings import settings as s
from ...detect.data_class import LabelBox
from ...utils.decorator import time_it

__all__ = ['load_classify_model_bound', 'food_classify', 'food_classify_one']

ModelBound = tuple[ResNet | SwinTransformer, dict, tf.Compose] | tuple[None, None, None]


def load_classify_model_bound() -> ModelBound:
    """
    加载分类识别资源包
    Returns:
        分类模型
        分类标签文件
        图片格式转化方法
    """
    if not s.use_classify:
        return None, None, None
    models = [
        'resnet34', 'resnet50', 'resnet101', 'resnet50_32x4d', 'resnet101_32x8d',
        'swin_tiny_patch4_window7_224',
        'swin_small_patch4_window7_224',
        'swin_base_patch4_window7_224',
        'swin_base_patch4_window12_384',
        'swin_base_patch4_window7_224_in22k',
        'swin_base_patch4_window12_384_in22k',
        'swin_large_patch4_window7_224_in22k',
        'swin_large_patch4_window12_384_in22k'
    ]
    model = eval(s.use_classify) if isinstance(s.use_classify, str) and s.use_classify in models else resnet50
    json_file = s.model_classify.with_suffix('.json').read_text()
    class_indict = json.loads(json_file)

    device = f'cuda:{s.device}' if s.device.isdigit() else s.device
    c_model = model(num_classes=len(class_indict)).to(device)
    c_model.load_state_dict(torch.load(f'{s.model_classify}', map_location=device))
    c_model.eval()

    size = 256 if isinstance(model, ResNet) else int(224 * 1.14)

    data_transform = tf.Compose([
        tf.ToTensor(),
        tf.Resize(size, antialias=True),
        tf.CenterCrop(224),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return c_model, class_indict, data_transform


@time_it
def food_classify(image: np.ndarray, boxes: list[LabelBox], classify_model_bound: ModelBound) -> list[LabelBox]:
    """
    批量执行目标分类，将每张图片的全部食材框打包一起放入模型进行分类
    Args:
        image: 完整图片
        boxes: 图片中的标签
        classify_model_bound: 分类识别资源包

    Returns:
        分类后识别结果
    """
    if len(boxes) == 0:
        return boxes
    model, class_indict, data_transform = classify_model_bound

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_classify = [x.cut_from_numpy(image) for x in boxes]
    img_list = [data_transform(img) for img in image_classify]
    boxes_label = []

    with torch.no_grad():
        batch_img = torch.stack(img_list, dim=0)
        device = f'cuda:{s.device}' if s.device.isdigit() else s.device
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)

        for pro, cla, box in zip(probs, classes, boxes):
            ename = class_indict[str(cla.numpy())] if box.ename not in ['drawer', 'hand'] else box.ename
            boxes_label.append(LabelBox(*box.box.xyxy, ename=ename, prob=box.prob))

    return boxes_label


@time_it
def food_classify_one(image: np.ndarray, box: LabelBox, classify_model_bound: ModelBound) -> LabelBox:
    """
    目标分类，将每张图片的全部食材框打包一起放入模型进行分类
    Args:
        image: 完整图片
        box: 图片中的标签
        classify_model_bound: 分类识别资源包

    Returns:
        分类后识别结果
    """
    model, class_indict, data_transform = classify_model_bound
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = box.cut_from_numpy(image)
    img = torch.unsqueeze(data_transform(image), dim=0)

    with torch.no_grad():
        device = f'cuda:{s.device}' if s.device.isdigit() else s.device
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        ename = class_indict[str(predict_cla)] if box.ename not in ['drawer', 'hand'] else box.ename
        box_label = LabelBox(*box.box.xyxy, ename=ename, prob=box.prob)

    return box_label

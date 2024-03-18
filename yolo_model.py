#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
#
#         Copyright (C) 2023 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : yolo_model.py
#   Author      : Hardy
#   Created date: 2023/02/10 17:48
#   Description : yolo 模型加载及初始化
#
# ======================================================================

from pathlib import Path
from queue import Queue

import numpy as np
from munch import Munch
from torch import Tensor
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from .data_class import LabelBox
from .person import Persons, Person
from ..core.settings import settings as s
from ..utils.decorator import time_it
from ..utils.log_util import logger

__all__ = ['ModelQueue', 'YOLOModel']


class YOLOModel(YOLO):
    def __init__(self, m='yolov8n.yaml'):
        super().__init__(model=m)
        self._init_or_update_model()

    def predict(self, source=None, stream=False, **kwargs) -> Tensor:
        """使用 YOLO 预测食材位置"""
        self._init_or_update_model(**kwargs)
        return self.predictor(source=source, stream=stream)

    def _init_or_update_model(self, **kwargs) -> None:
        """初始化识别模型"""
        overrides = self.overrides.copy()
        custom_setting = {'conf': 0.25, 'mode': 'predict', 'save': False, 'device': s.device, 'half': s.half}
        overrides.update(kwargs, **custom_setting)
        if not self.predictor:
            self.predictor = self.task_map[self.task]['predictor'](overrides=overrides, _callbacks=None)
            self.predictor.setup_model(model=self.model)
            self.predictor.run_callbacks = lambda x: None
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)

    def __repr__(self):
        return f'Model: {self.task}'


class ModelQueue(object):
    def __init__(self, maxsize=1):
        self.model_queue: Queue = Queue(maxsize=maxsize)
        if maxsize <= 0:
            self.model_queue.put((0, None, None, None, None))
        for ind in range(maxsize):
            f_model_lc, f_model_ld, p_model = self._init_models()
            self.model_queue.put((ind, f_model_lc, f_model_ld, p_model))

    @classmethod
    def _init_models(cls) -> tuple[YOLOModel, YOLOModel, YOLOModel]:
        """初始化一个模型"""
        # 冷藏室模型
        if not s.model_lc.exists():
            raise FileNotFoundError(f'{s.model_lc} not found, please initial a model in config file.')
        f_model_lc = YOLOModel(s.model_lc)
        # 冷冻室模型
        if not s.model_ld.exists():
            logger.warning(f'{s.model_ld} not found, use {s.model_lc} instead.')
            f_model_ld = f_model_lc
        elif s.model_ld == s.model_lc:
            f_model_ld = f_model_lc
        else:
            f_model_ld = YOLOModel(s.model_ld)
        # 人体识别模型
        if not s.model_pose.exists():
            raise FileNotFoundError(f'{s.model_pose} not found, please initial a model in config file.')
        p_model = YOLOModel(s.model_pose)
        return f_model_lc, f_model_ld, p_model

    def predict(
            self, image: np.ndarray, im: Munch = None, name: Path = None, space_mark: str = '0', save: bool = False
    ) -> tuple[list[LabelBox], Person | None]:

        """
        获取预测结果
        Args:
            image: 原始图片数组
            im: 图片基本信息
            name: 结果保存路径
            space_mark: 0-冷藏室 1-冷冻室
            save: 是否保存推理结果

        Returns:
            预测框
        """
        ind, f_model_lc, f_model_ld, p_model = self.model_queue.get()
        logger.info(f'Model index: {ind}.')
        try:
            if space_mark == '0':  # 冷藏室
                food = self.food_predict(image, f_model_lc, name, save=save)
                pose = self.pose_predict(image, p_model, im, name, save=save)
            else:  # 冷冻室
                food, pose = self.food_predict(image, f_model_ld, name, save=save), None
            return food, pose
        except Exception as e:
            logger.exception(e)
            return [], None
        finally:
            self.model_queue.put((ind, f_model_lc, f_model_ld, p_model))

    @time_it
    def food_predict(
            self, image: np.ndarray, model: YOLOModel, name: Path = None, save: bool = False
    ) -> list[LabelBox]:
        """
        获取预测结果
        Args:
            image: 原始图片数组
            model: 模型
            name: 结果保存路径
            save: 是否保存推理结果

        Returns:
            预测框
        """
        result_tmp = name.parent.joinpath(s.label_path, f'{name.name}.food') if name else None
        if model is None:
            texts = [x.strip() for x in result_tmp.read_text().split('\n') if x.strip()] \
                if result_tmp and result_tmp.exists() else []
            box_arr = np.array([[float(x.strip()) for x in text.split()] for text in texts])
        else:
            det = model.predict(image, stream=False, device=s.device, verbose=False)[0].cpu()
            box_arr = np.asarray(det.boxes.data)
            del det
        if save and result_tmp:
            texts = [' '.join([f'{x}' for x in arr]) for arr in box_arr]
            result_tmp.parent.mkdir(parents=True, exist_ok=True)
            result_tmp.write_text('\n'.join(texts))
        boxes = [LabelBox(*[x[i] for i in range(4)], model.names[int(x[5])], x[4]) for x in box_arr]
        return boxes

    @time_it
    def pose_predict(
            self, image: np.ndarray, model: YOLOModel, im: Munch, name: Path = None, save: bool = False
    ) -> Person | None:
        """
        获取预测结果
        Args:
            image: 原始图片数组
            model: 模型
            im: 图片基本信息
            name: 结果保存路径
            save: 是否保存推理结果

        Returns:
            预测框
        """
        result_tmp = name.parent.joinpath(s.label_path, f'{name.name}.pose') if name else None
        if model is None:
            txt_file = result_tmp if result_tmp and result_tmp.exists() else None
            persons = Persons(person=None, kpts=None, im=im, txt_file=txt_file)
        else:
            det = model.predict(image, stream=False, device=s.device, verbose=False)[0].cpu()
            persons = Persons(person=det.boxes.data, kpts=det.keypoints.data, im=im)
            del det
        if save and result_tmp:
            persons.save_txt(result_tmp)
        return persons.best_person

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
#
#         Copyright (C) 2023 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : data_class.py
#   Author      : Hardy
#   Created date: 2023/03/08 10:10
#   Description : 模型中用到的数据类
#
# ======================================================================

import math
from enum import Enum
from functools import cached_property
from typing import Final

import numpy as np
#from munch import Munch

#from .data_load import CATEGORY_DATA

__all__ = ['FridgeP', 'DoorStatus', 'Point', 'Rect', 'RectP', 'LabelBox', 'DrawerBox', 'Drawer']


class FridgeP(Enum):
    """
    冰箱位置信息，使用 7 位二进制表示
    第 1 位：冷藏或冷冻 0-冷藏 1-冷冻
    第 2 位：左右位置 0-左 1-右
    第 3-4 位：冰箱部件 00-门体 01-抽屉 10-搁架 11-附件
    第 5-7 位：序号 000-全部 001至111-从下至上编码
    例如：冷藏室第二层左抽屉编码为 0001001
    """
    c_left: Final[int] = 9  # 左抽屉 0001001
    c_right: Final[int] = 41  # 右抽屉 0101001
    c_top: Final[int] = 10  # 上层抽屉 0001010
    c_part: Final[int] = 16  # 搁架 0010000
    d_top: Final[int] = 74  # 冷冻室上层抽屉 1001010
    d_bottom: Final[int] = 73  # 冷冻室下层抽屉 1001001
    outside: Final[int] = 0  # 外部


class DoorStatus(object):
    """门开关状态
    bit0: 冷藏室门开关
    bit1: 冷冻室门开关
    bit2: 吧台门开关
    bit3: 出冰口分配器开关
    bit4: 变温室门开关
    bit5: 抑嘌呤抽屉开关
    bit6: 红蓝光抽屉开关
    bit7: 待分配
    """

    def __init__(self, status: int):
        self._status_str: str = f'{status:08b}'

    @cached_property
    def door_lc(self):
        """冷藏室电控信号"""
        return int(self._status_str[-1])

    @cached_property
    def door_ld(self):
        """冷冻室电控信号"""
        return int(self._status_str[-2])

    @cached_property
    def door_bw(self):
        """变温室电控信号"""
        return int(self._status_str[-5])

    @cached_property
    def door_lc_only(self):
        """是否仅开启冷藏室"""
        return not self.door_ld and not self.door_bw and self.door_lc

    @cached_property
    def door_nor_lc(self):
        """冷藏室是否未开启"""
        return (self.door_ld or self.door_bw) and self.door_lc

    @cached_property
    def door_closed(self):
        """是否各仓室全部关闭"""
        return not (self.door_lc or self.door_ld or self.door_bw)

    @cached_property
    def drawer_left(self):
        """左抽屉是否开启"""
        return int(self._status_str[-6]) and int(self._status_str[-1])

    @cached_property
    def drawer_right(self):
        """忧愁啼是否开启"""
        return int(self._status_str[-7]) and int(self._status_str[-1])


def _paras_repr(cls, repr_fields: list[str]) -> str:
    """
    __repr__ 展示内容（避免展示属性过多）
    Args:
        cls: 类对象
        repr_fields: 需要展示的属性

    Returns:
        展示结果
    """
    cls_name = cls.__class__.__qualname__
    fields = ', '.join(f'{x}={getattr(cls, x)}' for x in repr_fields)
    return f'{cls_name}({fields})'


class Point(object):
    """坐标点"""

    def __init__(self, x, y):
        self.x: int = int(x)
        self.y: int = int(y)

    @property
    def xy(self) -> tuple[int, int]:
        """点坐标"""
        return self.x, self.y

    def __repr__(self):
        return _paras_repr(self, ['x', 'y'])


class Rect(object):
    """矩形框"""

    def __init__(self, x1, y1, x2, y2):
        self.x1: int = int(x1)
        self.y1: int = int(y1)
        self.x2: int = int(x2)
        self.y2: int = int(y2)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """xyxy 格式点"""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        """xywh 格式点"""
        return self.x1, self.y1, self.width, self.height

    @property
    def center(self) -> Point:
        """矩形框中心点坐标"""
        return Point((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def p_3_1(self) -> Point:
        """Point(x1, y1)"""
        return Point((self.x1 * 2 + self.x2) // 3, (self.y1 * 2 + self.y2) // 3)

    @property
    def p_3_2(self) -> Point:
        """Point(x1, y2)"""
        return Point((self.x1 + self.x2 * 2) // 3, (self.y1 + self.y2 * 2) // 3)

    @property
    def width(self) -> int:
        """矩形框宽度"""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """矩形框高度"""
        return self.y2 - self.y1

    @property
    def s(self) -> int:
        """矩形框面积"""
        return self.width * self.height

    @property
    def cx(self) -> int:
        """中心横坐标"""
        return self.center.x

    @property
    def cy(self) -> int:
        """中心纵坐标"""
        return self.center.y

    def __repr__(self):
        return _paras_repr(self, ['x1', 'y1', 'x2', 'y2'])


class RectP(Rect):
    """带有概率的矩形框"""

    def __init__(self, x1, y1, x2, y2, prob=1.0):
        super().__init__(x1, y1, x2, y2)
        self.prob: float = prob

    def __repr__(self):
        return _paras_repr(self, ['x1', 'y1', 'x2', 'y2', 'prob'])


class LabelBox(object):
    """标准标注框"""

    def __init__(self, x1, y1, x2, y2, ename, prob=1.0):
        self.ename: str = ename
        self.box: Rect = Rect(x1, y1, x2, y2)
        self.prob: float = round(float(prob), 2)
        # self.class_info: tuple[int, str] = CATEGORY_DATA.get(ename, (0, None))
        self._position: int = 0
        self._full: bool = False
        self._type: str | None = None
        self._available: bool = True
        self._move_distance: float = math.inf
        self.unique_id: str = f'{ename}{self.box.x1}{self.box.y1}{self.box.x2}{self.box.y2}'
        self.on_edge_area: tuple[float, DrawerBox | None] = (0.0, None)

    def cut_from_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        从原图上按照识别结果截取对应的图片
        Args:
            image: 原始图片

        Returns:
            截取部分图片
        """
        x1, y1, x2, y2 = self.box.xyxy
        return image[y1:y2, x1:x2].copy()

    def copy_box(self) -> Rect:
        """拷贝 box 属性"""
        return Rect(*self.box.xyxy)

    def set_move_distance(self, distance: float = 0) -> None:
        """
        设置目标移动距离，根据匈牙利匹配算法计算得到
        Args:
            distance: 移动距离
        """
        self._move_distance = distance

    @property
    def move_distance(self) -> float:
        """获取移动距离"""
        return self._move_distance

    def set_position(self, pos: int, full: bool = False) -> None:
        """
        设置标注框对应的位置
        Args:
            pos: 食材框对应位置
            full: 是否全部在抽屉内部
        """
        self._position = pos
        self._full = full

    @property
    def position(self) -> int:
        """获取标注框对应位置"""
        return self._position

    @property
    def is_full_in(self) -> bool:
        """是否全部包含"""
        return self._full

    def set_type(self, l_type: str) -> None:
        """
        设置标注框的属性
        Args:
            l_type: 标注框属性 食材、手、头、容器中的一种
        """
        self._type = l_type

    @property
    def l_type(self) -> str:
        """获取标注框属性"""
        return self._type

    def set_disable(self) -> None:
        """框设置为无效"""
        self._available = False

    @property
    def is_available(self) -> bool:
        """判断标注框是否为有效框"""
        return self._available

    @property
    def food_id(self) -> int:
        """食材 id"""
        return self.class_info[0]

    @property
    def name(self) -> str:
        """列别名称"""
        return self.class_info[1] or self.ename

    @property
    def cx(self) -> int:
        """中心点 x 坐标"""
        return self.box.center.x

    @property
    def cy(self) -> int:
        """中心点 y 坐标"""
        return self.box.center.y

    def __repr__(self):
        return _paras_repr(self, ['box', 'prob', 'name'])


# class DrawerBox(RectP):
#     """抽屉框"""
#
#     def __init__(self, x1, y1, x2, y2, prob=1.0, ds: DoorStatus = None, border: int = 5, im: Munch = None):
#         super().__init__(x1, y1, x2, y2, prob=prob)
#         self._ds: DoorStatus = ds if ds else DoorStatus(status=1)
#         self._im: Munch = im
#         if border:
#             self.boundary_expansion(border=border)
#
#     def boundary_expansion(self, border: int = 5) -> None:
#         """
#         识别会有一定误差，关门时可能会造成食材与抽屉边缘相交，此处通过扩充抽屉边缘达到避免相交的目的
#         Args:
#             border: 扩充像素
#         """
#         self.x1 -= border
#         self.y1 -= 3 * border
#         self.x2 += border
#         self.y2 += 3 * border
#
#         if self.location_id == FridgeP.c_top.value:  # 宽抽屉不适用
#             self.y1 += 2 * border
#             self.y2 -= 3 * border
#
#         if self.location_id in [FridgeP.d_top.value, FridgeP.d_bottom.value]:  # 冷冻室抽屉边界调整
#             self.y1 += self._im.drawer_t
#             self.x1 += self._im.drawer_t // 3
#             self.x2 -= self._im.drawer_t // 5
#
#     @property
#     def location(self) -> FridgeP:
#         """抽屉位置，当冷冻室有开启信号时需要结合抽屉开关信号区分是冷藏室抽屉还是冷冻室抽屉"""
#         if self._ds.door_closed:  # 抽屉全部关闭
#             return FridgeP.outside
#         if not self._ds.door_lc:
#             return FridgeP.d_top if self._ds.door_bw == 1 else FridgeP.d_bottom
#         if self.p_3_1.x > self._center_line and self.p_3_2.x > self._center_line:
#             if self._ds.door_lc_only or self._ds.drawer_left:
#                 return FridgeP.c_left  # 左抽屉
#         if self.p_3_1.x < self._center_line and self.p_3_2.x < self._center_line:
#             if self._ds.door_lc_only or self._ds.drawer_right:
#                 return FridgeP.c_right  # 右抽屉
#         if self.y2 < self._im.drawer_in + 50 and self._ds.door_nor_lc:  # 冷冻室抽屉与上层宽抽屉区分
#             return FridgeP.d_top if self._ds.door_bw == 1 else FridgeP.d_bottom
#         return FridgeP.c_top  # 宽抽屉
#
#     @property
#     def location_id(self) -> int:
#         """抽屉的位置 id"""
#         return self.location.value
#
#     @property
#     def location_name(self) -> str:
#         """抽屉位置"""
#         return self.location.name
#
#     @property
#     def status(self) -> int:
#         """抽屉状态 0-关闭 1-半开 2-全开 """
#         return 2 if self.height > self._im.drawer_f else (1 if self.height > self._im.drawer_o else 0)
#
#     @property
#     def _center_line(self) -> float:
#         """图片中心线"""
#         return self._im.img_w / 2
#
#     @property
#     def _drawer_height(self) -> int:
#         """抽屉高度"""
#         return self.y2 - self.y1
#
#     def __repr__(self):
#         return _paras_repr(self, ['x1', 'y1', 'x2', 'y2', 'location_id', 'status'])
#
#
# class Drawer(object):
#     """抽屉大类，包含全部三个抽屉"""
#
#     def __init__(
#             self,
#             cl: str | Rect = None, cr: str | Rect = None, ct: str | Rect = None, cp: int = None,
#             dt: str | Rect = None, db: str | Rect = None, im: Munch = None, ds: DoorStatus = None
#     ):
#         self.c_left: DrawerBox = self._create_drawer(drawer=cl, ds=ds, im=im)
#         self.c_right: DrawerBox = self._create_drawer(drawer=cr, ds=ds, im=im)
#         self.c_top: DrawerBox = self._create_drawer(drawer=ct, ds=ds, im=im)
#         self.d_top: DrawerBox = self._create_drawer(drawer=dt, ds=ds, im=im)
#         self.d_bottom: DrawerBox = self._create_drawer(drawer=db, ds=ds, im=im)
#         self.c_part: int = self._create_line(cp=cp, im=im)  # 抽屉基线不代表判断抽屉进出搁架的线，该线要综合判断
#
#     def add(self, drawer: DrawerBox) -> bool:
#         """
#         添加抽屉框
#         Args:
#             drawer: 抽屉框
#
#         Returns:
#             是否添加成功
#         """
#         if drawer.status == 0:
#             return False
#         has_drawer = getattr(self, drawer.location_name)
#         if has_drawer and has_drawer.prob > drawer.prob:  # 检测多个同位置抽屉，选取置信度高的一个
#             return False
#         setattr(self, drawer.location_name, drawer)
#         return True
#
#     @classmethod
#     def _create_drawer(cls, drawer: str | Rect = None, ds: DoorStatus = None, im: Munch = None) -> DrawerBox | None:
#         """
#         从配置参数创建抽屉框（默认值）
#         Args:
#             drawer: 抽屉框
#             ds: 抽屉开关状态
#             im: 配置参数
#
#         Returns:
#             创建后抽屉信息
#         """
#         if not drawer:
#             return None
#         if isinstance(drawer, str):
#             drawer = RectP(*[x.strip() for x in drawer.split(',')])
#         prob = drawer.prob if isinstance(drawer, RectP) else 1.0
#         return DrawerBox(*drawer.xyxy, prob=prob, ds=ds, border=0, im=im)
#
#     def _create_line(self, cp: int = None, im: Munch = None) -> int:
#         """
#         寻找搁架参考线
#         Args:
#             cp: 搁架边界
#             im: 配置参数
#
#         Returns:
#             搁架边界
#         """
#         if self.c_top:
#             return self.c_top.y2
#         if self.c_left and self.c_right:
#             return max(self.c_left.y2, self.c_right.y2)
#         if self.c_left:
#             return self.c_left.y2
#         if self.c_right:
#             return self.c_right.y2
#         return cp if cp else im.drawer_in
#
#     def __repr__(self):
#         return _paras_repr(self, ['c_left', 'c_right', 'c_top', 'd_top', 'd_bottom'])

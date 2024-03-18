import math

import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from model import swin_tiny_patch4_window7_224 as create_model
# videospath='./videos'
# videolist=os.listdir(videospath)
# print(videolist)
from plot_utils import plot_box
from data_class import LabelBox
yolo_model = YOLO('./swin-3-12.json')
yolo_model = YOLO("./weights/990-five-class.pt")  # load a pretrained model (recommended for training)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
swintrans_model = create_model(num_classes=37).to(device)
swintrans_model_weight_path = "./weights/swin-3-12.pth"
swintrans_model.load_state_dict(torch.load(swintrans_model_weight_path, map_location=device))
swintrans_model.eval()


img_size = 224
data_transform = transforms.Compose(
    [transforms.Resize(int(img_size * 1.14)),
     transforms.CenterCrop(img_size),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def swin_trans_predict(img) -> str:
    # 将抠出的食材图片进行识别，将识别到的类别名称返回
    # img： 从yolo识别结果的food框中抠出的图像矩阵 ndarray
    # clsNprob_ename： 分类网络推理出的具体食材类别，e.g.: pear


    # plt.imshow(img)
    img = Image.fromarray(img.astype(np.uint8))
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    json_path = './swin-3-12.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(swintrans_model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())

    # clsNprob_result="{} {:.3}".format(class_indict[str(predict_cla)],predict[predict_cla].numpy())
    clsNprob_ename = class_indict[str(predict_cla)]
    clsNprob_prob = predict[predict_cla].numpy()

    return clsNprob_ename
    # plt.title(print_res)
    #
    # plt.show()


def single_img_process(frame):
    # 将单张图片转换为yolo和分类模型标好框的图片
    # frame： 用opencv读取的图片矩阵 ndarray
    # drawedimg: 画好框的图片矩阵 ndarray


    # frame=Image.open('./images/16947452442933.jpg')
    # yoloframe= torch.from_numpy(frame)
    results = yolo_model.predict(source=frame, save=False)
    boxes = results[0].boxes
    food_dict = results[0].names
    boxnum = len(boxes.cls)
    # print(boxnum)
    # IMG_frame=Image.fromarray(frame)
    drawed_img = frame  # 最终回执单结果,先转换为array类型
    for i in range(boxnum):
        foodlist = []  # 存放食材的索引和识别结果
        foodbox = tuple([int(x) for x in boxes.xyxy[i]])  # 剪裁区域转换为整数元组
        if boxes.cls[i] == 248:  # 如果类别为食材，将其抠出；

            # foodimg=IMG_frame.crop(foodbox) #剪裁出食物图像
            foodimg = frame[foodbox[1]:foodbox[3], foodbox[0]:foodbox[2]]

            ename = swin_trans_predict(foodimg)  # pred_result 格式eg: pear 1.0
        else:
            ename = food_dict[int(boxes.cls[i])]

        prob = boxes.conf[i]
        x = LabelBox(foodbox[0], foodbox[1], foodbox[2], foodbox[3], ename, prob)
        drawed_img = plot_box(drawed_img, x)

        # foodlist.append([i,pred_result])

    return drawed_img

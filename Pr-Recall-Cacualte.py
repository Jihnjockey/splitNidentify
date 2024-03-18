#从制作好的数据集中计算召回率和精确率
#先将图片送入网络中计算，然后根据label中的数据计算
import json
import os.path
import copy
import cv2
from tqdm import *
from ultralytics import YOLO
from pathlib import Path
from twostep_predict import single_img_process,swin_trans_predict
datasets_path='./datasets'
imagesets_path=os.path.join(datasets_path,'images')
labelsets_path=os.path.join(datasets_path,'labels')

treshhold=0.5

sigcls_pred=0.0


MODEL = './weights/990-five-class.pt'  # test spaces in path
CFG = "yolov8n.yaml"
#数据集存放格式：clsfctionimg--orange
                      #|_apple
                     # |_pear
                     #.....
clsfction_image_path='./clsfction'
clsfction_ids='./swin-3-12.json'

data='./food.yaml'
def YOLO_test_val():
    #测试yolo网络的性能指标
    """Test the validation mode of the YOLO model."""
    #print(Path(data).read_text(encoding='utf-8'))
    YOLO(MODEL).val(data=data, imgsz=640, save_hybrid=True)

def tramsformer_val():
    #测试分类网络的性能指标
    datalist=os.listdir(clsfction_image_path) # e.g. apple,pear,orange,....

    with open(clsfction_ids, 'r') as file:
        clsfciton_dict = json.load(file)

    food_countdict={value:0 for key, value in clsfciton_dict.items()}#统计各类食材分别的个数，即真例个数  同时初始化为0. e.g.：{apple：0，pear：0，....}
    TP_dict=copy.deepcopy(food_countdict) #真正例个数
    P_dict=copy.deepcopy(food_countdict)#真例个数
    #datalist_bar=tqdm(datalist)
    for foodname in datalist:
        #datalist_bar.set_description("Processing %s" % foodname)
        food_path=os.path.join(clsfction_image_path,foodname) #e.g. 苹果的文件夹路径
        imglist=os.listdir(food_path) #e.g. 苹果 图片的文件名列表
        imglist_bar=tqdm(imglist)
        food_countdict[foodname]=len(imglist)
        for imgname in imglist_bar:
            imglist_bar.set_description("Processing %s" % foodname)
            frame=cv2.imread(os.path.join(food_path,imgname))
            ename=swin_trans_predict(frame)  #得到分类网络推理得到的类别结果
            finalimgname=imgname.replace(foodname,ename)                                   #将识别结果添加到图片文件名中
            os.rename(os.path.join(food_path,imgname),os.path.join(food_path,finalimgname))
            P_dict[ename]+=1
            if ename==foodname:
                TP_dict[ename]+=1


    PR_dict=copy.deepcopy(food_countdict) #存放各类的精确率
    PR_dict['average']=0
    RC_dict=copy.deepcopy(food_countdict) #存放各类的召回率
    RC_dict['average'] = 0

    for foodname in TP_dict.keys(): #计算精确率
        try:
            pr=TP_dict[foodname]/P_dict[foodname]
        except:
            pr=0
        finally:
            PR_dict[foodname]=pr
            PR_dict['average'] +=pr
    PR_dict['average'] /= (len(PR_dict)-1)

    for foodname in TP_dict.keys(): #计算召回率
        try:
            rc=TP_dict[foodname]/food_countdict[foodname]
        except:
            rc=0
        finally:
            RC_dict[foodname]=rc
            RC_dict['average'] +=rc
    RC_dict['average'] /= (len(RC_dict)-1)

    print("food_countdict:")
    for key, value in food_countdict.items():
        print(key,":",value)

    print("TP_dict:")
    for key, value in TP_dict.items():
        print(key, ":", value)

    print("P_dict:")
    for key, value in P_dict.items():
        print(key, ":", value)



    print("精确率信息：")
    print(PR_dict)
    print("召回率信息：")
    print(RC_dict)

if __name__ == '__main__':
    # YOLO_test_val()
    tramsformer_val()
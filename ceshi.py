import os.path

import cv2

from twostep_predict import single_img_process
from PIL import Image

savedir='./images'
imgdir=r'D:\Bone_Joint_Identity\ultralytics-main\dataset\images'
imglist=os.listdir(imgdir)

for imgname in imglist:
    imgpath=os.path.join(imgdir,imgname)
    #img=Image.open(imgpath)
    img=cv2.imread((imgpath))
    drawedimg= single_img_process(img)
    cv2.imwrite(os.path.join(savedir,imgname),drawedimg)
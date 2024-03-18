#将一系列视频拆分为若干图片，放入yolo中，再将识别的food抠出放入分类网络识别具体食材，结果返回yolo识别的图片，将识别的食材框出，保存图片

import os

import cv2

from twostep_predict import single_img_process

videospath='./videos'   #视频组存放路径
videolist=os.listdir(videospath)
print(videolist)
img_savepath='./images' #图片结果保存路径




if __name__ == '__main__':

    for video in videolist:


        # 创建视频捕获对象
        cap = cv2.VideoCapture(os.path.join('./videos',video))

        # 初始化帧计数器
        frame_count = 0

        # 检查视频是否打开
        if cap.isOpened():
            # 循环直到视频结束
            while True:
                # 读取下一帧
                ret, frame = cap.read() #对应cv2.imwrite() 类型：numpy ndarray

                # 如果成功读取帧
                if ret:
                    # 保存图片，格式为 frame_XXXX.png，其中 XXXX 是帧编号
                    #cv2.imwrite(f'frame_{frame_count:04d}.png', frame)
                    # Load a model
                    img_savename = video.split('.')[0] + '_' + str(frame_count) + '.jpg'
                    #plt.imshow(frame)
                    #cv2.imshow(img_savename,frame)
                    try:
                        # frame=cv2.imread('./16947452442933.jpg')
                        drawedimg=single_img_process(frame)
                        # drawedimg=Image.fromarray(frame)
                    except:
                        print("出错帧为：")
                        print(img_savename)

                    #drawedimg.save(os.path.join(img_savepath,img_savename))

                    cv2.imwrite(os.path.join(img_savepath,img_savename), drawedimg)
                    frame_count += 1

                    # 按下 'q' 退出循环
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # 没有帧可读，退出循环
                    break
        print(frame_count)
        # 释放视频捕获对象
        cap.release()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()

import cv2
from PIL import Image
import numpy as np
import os

# cv2.imshow("out",img)

def get_imgNameList(path):
    """
    获取验证码图像的名字列表
    :param path:
    :return:
    """
    list = os.listdir(path)
    namelist=[os.path.join(path,name) for name in list]
    namelist.sort()
    return namelist
def split_data(img_list,csv_path,out_path):
    """
    拆分验证码图像，将数据集4个字符的验证码图像进行拆分，拆分为单个
    :return:
    """
    nnamelist = get_imgNameList("/home/chengyangkj/data/code")
    label_list = np.loadtxt(csv_path, delimiter=',').astype('int').astype('str')
    label_list.sort()

    #拆分1000个验证码图像
    for i in range(1000):

def opencv_handel_split(img_path):
    """
    使用opencv处理，切割图像
    :param img:
    :return: 一张验证码图像中的四个字符图像
    """
    #不知何原因 opencv读不入图像 采用PIL读取 转灰度
    im = Image.open(img_path).convert('L')
    #print(str(im.width),str(im.height))
    #PIL对象转opencv
    src = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
    # 使用中值滤波 降噪
    blur = cv2.medianBlur(src, 5)  # 模板大小3*3

def dataReader():
    """
    自定义数据集Reader可迭代对象,将验证码的数据和对应的label存在迭代器里面
    :return:
    """

if __name__ == '__main__':
    nameList = get_imgNameList("/home/chengyangkj/data/code")
    split_data(nameList, "/home/chengyangkj/data/code/code.csv",'/home/chengyangkj/data/code_split')

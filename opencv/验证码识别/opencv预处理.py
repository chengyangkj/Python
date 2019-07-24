import cv2
import numpy as np
src = cv2.imread("78187.png",0)

#获取名字中的验证码信息并存下
name="78187.png"
name_list = name.split(".")
number_list=list()
for i in name_list[0]:
    number_list.append(i)


if src is None:
    print("图片读取失败 ")
cv2.imshow("input",src)

#使用中值滤波 降噪
blur = cv2.medianBlur(src, 5)  # 模板大小3*3
cv2.imshow("blur",blur)


#阈值化处理 阈值设为80
ret,thresh1 = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)
cv2.imshow("thre",thresh1)
thresh = thresh1.copy()


#垂直投影
(h, w) = thresh1.shape  # 返回高和宽
# print(h,w)#s输出高和宽
a = [0 for z in range(0, w)]
high = np.zeros((h,w),np.uint8)
# 记录每一列的波峰
for j in range(0, w):  # 遍历一列
    for i in range(0, h):  # 遍历一行
        if thresh1[i, j] == 0:  # 如果改点为黑点
            a[j] += 1  # 该列的计数器加一计数
            thresh1[i, j] = 255  # 记录完后将其变为白色
    # print (j)

#
for j in range(0, w):  # 遍历每一列
    for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
        thresh1[i, j] = 0  # 涂黑

cv2.imshow('high',thresh1)
cv2.imshow("thre",thresh)



#寻找分割点
is_split=False
split_x =list()
cols=0
for i in thresh1[-1]:
    #等于0 为黑像素
    if i ==0:
        if not is_split:
            split_x.append(cols-1)
            is_split=True
    else:
        if is_split==True:
            split_x.append(cols - 1)
        is_split=False
    cols+=1


#根据分割点进行分割
img_list=list()
j=0
for i in split_x:
    print(i)
    #判断是否为最后一个
    if j%2==1:
        j+=1
        continue
    #将拆分结果加入列表
    img_list.append(thresh[0:-1, i:split_x[j+1]])
    j+=1

#显示所有的结果
print(number_list)
j=0
for i in number_list:
    cv2.imshow(str(j)+":"+i,img_list[j])
    j+=1
cv2.waitKey(0)
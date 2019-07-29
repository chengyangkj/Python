from PIL import Image
import os
import numpy as np



def divide_img(path):
    im = Image.open(path)
    im = im.convert('RGB')
    img1 = im.crop((10, 0, 20, 30))
    img2 = im.crop((20, 0, 30, 30))
    img3 = im.crop((30, 0, 40, 30))
    img4 = im.crop((42, 0, 52, 30))

    return img1,img2,img3,img4


if __name__ == '__main__':
    img_path = "/home/chengyangkj/data/codechar/code"
    label_path = "/home/chengyangkj/data/codechar/label.csv"
    lists = os.listdir(img_path)

    img_list = [os.path.join(img_path, name) for name in lists]
    img_list.sort()

    #获取label数据
    label_list = np.loadtxt(label_path,delimiter=',').astype('int').astype('str')
    label = [number[1].rjust(4,'0') for number in label_list]
    flag = 1
    list_t=[]
    for i in range(1000):
        print(img_list[i])
        img = divide_img(img_list[i])

        list_t.append([str(flag).rjust(4,'0'),label[i][0]])
        img[0].save("/home/chengyangkj/data/splitcode/code/"+str(flag).rjust(4,'0')+".jpg")
        flag+=1
        print(label[i][1])
        list_t.append([str(flag).rjust(4,'0'), label[i][1]])
        img[1].save("/home/chengyangkj/data/splitcode/code/" + str(flag).rjust(4,'0') + ".jpg")
        flag += 1
        list_t.append([str(flag).rjust(4,'0'), label[i][2]])
        img[2].save("/home/chengyangkj/data/splitcode/code/" + str(flag).rjust(4,'0') + ".jpg")
        flag += 1
        list_t.append([str(flag).rjust(4,'0'), label[i][3]])
        img[3].save("/home/chengyangkj/data/splitcode/code/" + str(flag).rjust(4,'0') + ".jpg")
        flag += 1
    print(list_t)
    np.savetxt("/home/chengyangkj/data/splitcode/label.csv",list_t,delimiter=',', fmt ='%s,%s')
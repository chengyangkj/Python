import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cv2
#打开图片
src = cv2.imread("78187.png",0)

#获取名字中的验证码信息并存下
name="78187.png"
name_list = name.split(".")
number_list=list()
for i in name_list[0]:
    number_list.append(int(i))


if src is None:
    print("图片读取失败 ")

#使用中值滤波 降噪
blur = cv2.medianBlur(src, 5)  # 模板大小3*3


#阈值化处理 阈值设为80
ret,thresh1 = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)
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
    #判断是否为最后一个
    if j%2==1:
        j+=1
        continue
    #将拆分结果加入列表
    img_list.append(thresh[0:-1, i:split_x[j+1]])
    j+=1







# 定义data的数据域
with tf.variable_scope("data"):
    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.int32, [None, 10])
# 建立一个全连接层的神经网络 w[784, 10] b [10]
with tf.variable_scope("fc_model"):
    # 随机初始化权重和偏置
    weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="w")
    bias = tf.Variable(tf.constant(0.0, shape=[10]))
    # 预测None样本的输出结果 矩阵相乘 [None,784]*[784,10] + [10] = [None, 10]
    y_predict = tf.matmul(x, weight) + bias

# 计算损失
with tf.variable_scope("soft_cross"):
    # 求平均交叉熵损失  reduce_mean 函数对一个列表所有数相加除以数量
    # 先通过softmax方法将结果处理 之后再进行 nn交叉熵损失计算
    # 交叉熵损失计算：损失值=(目标值on_hot)*log(训练值的softmax格式)相加=1*log(y_p1)+1*log(y_p2)+1*log(y_p3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
# 梯度下降优化
with tf.variable_scope("optimizer"):
    # 对loss的值进行梯度下降优化
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 求出准确率
with tf.variable_scope("acc"):
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
# 收集变量
tf.summary.scalar("losses", loss)
tf.summary.scalar("acc", accuracy)

# 高纬度变量收集
tf.summary.histogram("weidth", weight)
tf.summary.histogram("biases", bias)

# 定义初始化变量的op
init_op = tf.global_variables_initializer()
# 定义一个合并收集到的变量的op (可视化学习步骤)
merged = tf.summary.merge_all()

# 创建一个saver 用于保存训练完成后的模型
saver = tf.train.Saver()
mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)
# 开启会话去训练
with tf.Session() as sess:
     # 打开保存的模型文件
     saver.restore(sess, "./tmp/训练结果/手写模型")
     x_test, y_test = mnist.test.next_batch(1)


     x_test = cv2.resize(img_list[0], (28, 28))
     x_test=x_test.flatten()
     y_test = sess.run(tf.one_hot(number_list[0], 10, 1, 0))
     j = 0
     for i in number_list:
         #取出一个测试数据集
         x_test = cv2.resize(img_list[j],(28,28))
         x_test=[x_test.flatten()]
         y_test = [sess.run(tf.one_hot(i, 10, 1, 0))]
         print("第%d张图片，手写数字目标是：%d,预测结果是：%d,预测准确性%f" % (
                 j,
                 # tf.argmax 将one-hont编码转换为数据形式
                 tf.argmax(y_test, 1).eval(),
                 sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}),
                 1 - sess.run(loss, feed_dict={x: x_test, y_true: y_test})
             ))
         j+=1
#coding:utf-8
#对图片进行卷积，卷积之后再进行全连接层处理，卷积相当于数据预处理，获取图片的关键特征，之后再进行训练识别
#重要公式：
#问题：输入的特征图片为 =[H1,W1,D]=[28,28,1],卷积层：k = 100个filter， F = 步长1， p = padding（0填充）=1
#求一次卷积后输出的图片大小：
#公式 H2 = (H1 - F+2 p)/s +1 ; W2 = (W1 - F +2p)/s +1; D2 = k

#答案：H2 = (28 - 5+2*1)/1 + 1 =26 w2 = (28 - 5 +2*1)/1 + 1 =26

#1.卷积层 tf.nn.conv2d(input,filter,strides=,padding=,name=None)
    #input:给定的输入张量，具有[batch, height, width, channel],类型为float32,类型为float32,64
    #filter:指定过滤器的大小 [filter_height,filter_width,in_channle,out_channle]
    #strides:strides=[1,stride,stride,1],步长
    #panding:"SAME","VALID",使用的填充算法的类型 "SAME"表示填充使得变化后的长宽一样大 "valid"表示超出滑动的部分舍弃
# 个数，大小，步长，0填充
#卷积层输出深度和宽带 深度由过滤器的个数决定 宽带 H2*w2*d2


#2.激活函数 (relu) tf.nn.relu(features,name=None)
# 在深度神经网络中是必须的 增加网络的非线性分割能力
#函数原型：f(x) = max(0,x)
#一个数小于0 的话就是0 大于0的话就是他的本身


#3.池化(pooling) tf.nn.max_pool(value,ksize=,strides=,padding=,name=None)
    #value: [batch, height, width, channel]
    #ksize:池化窗口的大小 [1,ksize,ksize,1]
    #strides:步长大小[1，strides,strides,1]
    #padding:"SAME","VALID",使用的填充的算法类型,默认使用"SAME"
#polling的作用主要就是特征提取，去掉不重要的样本 常用的是MaxPooling 2*2 2步长

#4.全连接层运算，对数据进行分类
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def model():
    """
    自定义卷积模型
    :return:
    """
    #1.准备数据的占位符
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32,[None, 784])
        y_true = tf.placeholder(tf.int32,[None, 10])
    #2,一层卷积 窗口大小5*5*1 32个过滤器 步数strides为1 激活 池化
    with tf.variable_scope("conv1"):
        #初始化权重 filter  [filter_height,filter_width,in_channle,out_channle] out_channle 填写过滤器的个数
        w_conv1 = weight_variables([5, 5, 1, 32])
        #初始化偏置 传入过滤器的个数
        b_conv1 = bias_variables([32])
        #改变图片的形状 [None,784] -> [None,28, 28, 1]
        x_reshape = tf.reshape(x,[-1, 28, 28, 1])
        #卷积操作(构建模型 卷积函数+偏置范围)  [None, 28, 28, 1] -> [None, 28, 28, 32]:
        x_conv1 = tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
        #激活   [None, 28, 28 32] -> [None, 28, 28 32]
        x_relu1 = tf.nn.relu(x_conv1)

        #池化 2*2的范围 步长2 结构变化 [None, 28, 28 32] -> [None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME")


    # 3,二层卷积 窗口大小5*5*32 64个过滤器 步数strides为1 激活 池化
    with tf.variable_scope("conv2"):
        # 初始化权重 filter  [filter_height,filter_width,in_channle,out_channle] out_channle 填写过滤器的个数
        w_conv2 = weight_variables([5, 5, 32, 64])
        # 初始化偏置 传入过滤器的个数
        b_conv2 = bias_variables([64])
        # 改变图片的形状 [None,784] -> [None,28, 28, 1]
        # 卷积操作(构建模型 卷积函数+偏置范围)  [None, 14, 14, 32] -> [None, 14, 14, 64]:
        x_conv2 = tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
        # 激活   [None, 14, 14, 64] -> [None, 14, 14, 64]
        x_relu2 = tf.nn.relu(x_conv2)

        # 池化 2*2的范围 步长2 结构变化 [None, 14, 14, 64] -> [None, 7, 7, 64]

        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


    # 4,全连接层 [None, 7, 7, 64] -->[None,7*7*64]*【7*7*64,10+[10]  = [None,10]
    with tf.variable_scope("full_connect"):
        #初始化权重和偏置
        w_fc = weight_variables([7*7*64, 10])
        b_fc = bias_variables([10])

        #修改形状 [None, 7, 7, 64] -->[None,7*7*64]
        x_fc_reshape = tf.reshape(x_pool2,[-1,7*7*64])

        #进行矩阵运算得到每个样本的十个结果
        y_predict = tf.matmul(x_fc_reshape, w_fc)+b_fc

    return x,y_true,y_predict

def conv_fc():
    # 获取真实的数据：
    mnist = input_data.read_data_sets("./data/mnist/input_data", one_hot=True)
    x, y_true, y_predict = model()
    # 计算损失
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失  reduce_mean 函数对一个列表所有数相加除以数量
        # 先通过softmax方法将结果处理 之后再进行 nn交叉熵损失计算
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    # 梯度下降优化
    with tf.variable_scope("optimizer"):
        # 对loss的值进行梯度下降优化
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 求出准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    #定义一个初始化会话op
    init_op = tf.global_variables_initializer()
    #开启会话运行
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)
        # 建立events文件，然后写入
        filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)
        # 迭代步数去训练
        for i in range(2000):
            # 取出数据集的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
            print("训练第%d次，准确率为%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))
def weight_variables(shape):
    """
    初始化权重函数
    :param shape:
    :return:
    """
    #随机初始化
    w = tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))
    return w
def bias_variables(shape):
    """
    初始化偏置函数
    :param shape:
    :return:
    """""
    b = tf.Variable(tf.constant(0.0,shape=shape))
    return b
if __name__ =="__main__":
    conv_fc()
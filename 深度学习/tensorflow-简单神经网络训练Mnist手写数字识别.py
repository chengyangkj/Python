#coding:utf-8

#调用mnist的数据集
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#1.定义数据占位符
#2.建立模型 随机初始化权重和偏置 全连接结果计算
#3.计算损失
#4.梯度下降优化

#定义神经网络全连接层处理函数
choose = input("请输入你要执行的（1.训练,2.预测）")
def full_connected():
    #1.建立数据的占位符 x[None,784] y_true [None,10]
    #x（特征值）: None是图片的数量不确定, 784是一张手写图片的所有像素大小（28*28=784）
    #y_true （正确的目标值）: None是图片的数量不确定，10是一张图片的目标值的on—hot编码格式，（on-hot）0-9 当为目标值的地方改为1 其他为0

    #获取真实的数据
    mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)

    #定义data的数据域
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])
    #建立一个全连接层的神经网络 w[784, 10] b [10]
    with tf.variable_scope("fc_model"):
        #随机初始化权重和偏置
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0 ), name="w")
        bias = tf.Variable(tf.constant(0.0, shape=[10]))
        #预测None样本的输出结果 矩阵相乘 [None,784]*[784,10] + [10] = [None, 10]
        y_predict = tf.matmul(x, weight) + bias

    #计算损失
    with tf.variable_scope("soft_cross"):
        #求平均交叉熵损失  reduce_mean 函数对一个列表所有数相加除以数量
        #先通过softmax方法将结果处理 之后再进行 nn交叉熵损失计算
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
    #梯度下降优化
    with tf.variable_scope("optimizer"):
        #对loss的值进行梯度下降优化
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    #求出准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
    #收集变量
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    #高纬度变量收集
    tf.summary.histogram("weidth", weight)
    tf.summary.histogram("biases", bias)

    #定义初始化变量的op
    init_op = tf.global_variables_initializer()
    #定义一个合并收集到的变量的op (可视化学习步骤)
    merged = tf.summary.merge_all()

    # 创建一个saver 用于保存训练完成后的模型
    saver = tf.train.Saver()
    #开启会话去训练
    with tf.Session() as sess:
        #当为训练时
        if choose == 1:
            #初始化变量
            sess.run(init_op)
            #建立events文件，然后写入
            filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)
            #迭代步数去训练
            for i in range(2000):
                #取出数据集的特征值和目标值
                mnist_x,mnist_y = mnist.train.next_batch(50)
                sess.run(train_op, feed_dict={x:mnist_x, y_true:mnist_y})
                #写入每步训练的值（可视化学习）
                summary = sess.run(merged, feed_dict={x:mnist_x, y_true:mnist_y})
                filewriter.add_summary(summary, i)
                print("训练第%d次，准确率为%f"%(i,sess.run(accuracy, feed_dict={x:mnist_x, y_true:mnist_y})))
            #在训练结束后保存模型
            saver.save(sess, "./tmp/训练结果/手写模型")
        else:
            #打开保存的模型文件
            saver.restore(sess, "./tmp/训练结果/手写模型")
            for i in range(100):
                #取出一个测试数据集
                x_test, y_test = mnist.test.next_batch(1)
                print ("第%d张图片，手写数字目标是：%d,预测结果是：%d,预测准确性%f"%(
                    i,
                    #tf.argmax 将one-hont编码转换为数据形式
                    tf.argmax(y_test, 1).eval(),
                    tf.argmax(sess.run(y_predict,feed_dict={x:x_test, y_true:y_test}), 1).eval(),
                    1-sess.run(loss,feed_dict={x:x_test, y_true:y_test})
                ))

    return None
# tensorboard --logdir="./tmp/summary/test/" 启动tensorboard


if __name__ == "__main__":
    full_connected()


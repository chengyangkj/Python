#测试程序，测试模型的识别率
import tensorflow as tf
import numpy as np
import os

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer("batch_size", 100, "每批次训练的样本数")
tf.app.flags.DEFINE_integer("label_num", 4, "每个样本的目标值数量")
tf.app.flags.DEFINE_integer("letter_num", 10, "每个目标值取的字母的可能心个数")
tf.app.flags.DEFINE_string("tfrecords_dir", "./tfrecords/captcha.tfrecords", "验证码tfrecords文件")
tf.app.flags.DEFINE_string("captcha_dir", "/home/chengyangkj/data/codechar/code", "验证码图片路径")
tf.app.flags.DEFINE_string("captlabel_dir", "/home/chengyangkj/data/codechar/label.csv", "验证码标签路径")
tf.app.flags.DEFINE_string("letter", "0123456789", "验证码字符的种类")

def picread(img_path):
    """
    读取图片
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    #1,构造文件名队列
    file_quqe = tf.train.string_input_producer([img_path])

    #2,构建图片阅读器读取队列数据 （默认读取一张图片）
    reader = tf.WholeFileReader()
    key,value = reader.read(file_quqe)
    #3,解码 成uint8
    image = tf.image.decode_jpeg(value)

    #固定图片的形状 （通道数）因为批处理不支持不固定形状的图片 使用静态固定方法
    image.set_shape([30,80,3])

    return image



def fc_model(image):
    """
    构建网络模型，进行预测结果
    :param image: 传入的图片数据 [100,80,30,3]
    :return: 预测的结果 [100,4*10]
    """
    with tf.variable_scope("model"):
        #将图片数据转为二维形状
        img_reshape = tf.reshape(image,[-1,80*30*3])

        #随机初始化权重
        weigth = tf.Variable(tf.random_normal(shape=[80*30*3,4*10],mean=0.0, stddev=1.0))

        #初始化偏置为0
        bias= tf.Variable(tf.constant(0.0,shape=[4*10]))

        #全连接层计算 [100,4*10]
        y_predict = tf.matmul(tf.cast(img_reshape,tf.float32),weigth) +bias
    return y_predict


def conver_to_onhot(label):
    """
    将标签转为onhot
    :param label: 传入的2维标签值 [100,4] [[2,1,6,9],[4,3,1,5]]
    :return: [100,4,10]
    """

    label_onhot = tf.one_hot(label,depth=10,on_value=1.0,axis=2)
    return label_onhot


if __name__ == '__main__':

    img = picread('7808.jpg')
    y_predict = fc_model(img)
    reshape = tf.reshape(y_predict, [4, 10])
    label = tf.argmax(reshape,1)

    #初始化变量
    init_op = tf.global_variables_initializer()
    # 创建一个saver 用于保存训练完成后的模型
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init_op)

        # 定义线程协调器和开启线程（有数据在文件当中读取提供给模型）
        coord = tf.train.Coordinator()

        # 开启线程去运行读取文件操作
        threads = tf.train.start_queue_runners(sess, coord=coord)
        #读取保存的模型文件
        saver = tf.train.import_meta_graph("./tmp/result/codechar.meta")
        saver.restore(sess,"./tmp/result/codechar")
        print(sess.run(label))
        # 回收线程
        coord.request_stop()

        coord.join(threads)

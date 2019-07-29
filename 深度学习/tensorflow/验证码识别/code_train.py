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
def deal_with_label(label):
    """
    将喇叭恶劣进行处理，【4321，4563，21343】 ——>[[4,3,2,1],[4,5,6,3]]
    :param label_batch:
    :return:
    """
    for label in label_batch:
        label_list=[]
        print(label)


def get_and_decode():
    """
    将图片和label数据从tfrecorder文件中读出，并且放入batcch中
    :return:
    """
    #创建文件队列
    file_qeue = tf.train.string_input_producer(["./tfrecords/captcha.tfrecords"])

    #创建tf文件阅读器
    reader = tf.TFRecordReader()
    key,value = reader.read(file_qeue)

    #创建文件解码器解码文件

    # tfrecords格式example,需要解析
    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string),
    })

    # 解码内容，字符串内容
    # 1、先解析图片的特征值
    image = tf.decode_raw(features["image"], tf.uint8)
    # 1、解析图片的目标值
    label = tf.decode_raw(features["label"], tf.uint8)


    #改变形状
    image_reshape = tf.reshape(image,[30,80,3])
    label_reshape = tf.reshape(label,[4])

    print("img,label")
    print(image_reshape)
    print(label_reshape)

    #创建batch 进行批处理
    img_batch,label_batch = tf.train.batch(
        [image_reshape,label_reshape],batch_size=100,num_threads=1,capacity=100,
    )

    print(img_batch,label_batch)
    return img_batch,label_batch




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
    img_batch,label_batch = get_and_decode()
    y_predict = fc_model(img_batch)

    print("y_pre")
    print(y_predict)

    y_value = label_batch
    #将label转换为onhot
    y_true = conver_to_onhot(label_batch)
    #softmax 交叉商损失
    with tf.variable_scope("soft_cross"):
        #改变真实值label的形状 [100,4,10] ->[100,4*10]
        y_true_reshape = tf.reshape(y_true,[100,4*10])
        mean_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = y_true_reshape,
                logits = y_predict
            )
        )
        # 5、梯度下降优化损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(mean_loss)

    #计算准确率
    with tf.variable_scope("vcc"):
        #tf.arg_max() 求最大项的索引 返回的纬度和被求的相同 要指出求的维度
        #tf.equal() 对比两个变量是否相同 相同为1 返回的结果和求的纬度相同
        # 比较每个预测值和目标值是否位置一样    y_predict [100, 4 * 26]---->[100, 4, 26]
        equal_list = tf.equal(tf.argmax(y_true, 2),
                              tf.argmax(tf.reshape(y_predict, [100,4,10]),2))

        # equal_list  100个样本   [1, 0, 1, 0, 1, 1,..........]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

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

        # 训练识别程序
        for i in range(20000):
            sess.run(train_op)

            print("第%d批次的准确率为：%f,损失%f" % (i, accuracy.eval(), mean_loss.eval()))
            # 在训练结束后保存模型
        saver.save(sess, "./tmp/result/codechar")

        # 回收线程
        coord.request_stop()

        coord.join(threads)

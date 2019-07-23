import tensorflow as tf
import os
#读取二进制训练样本文件

# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cifar_dir", "./data/cifar10/cifar-10-batches-bin/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "./tmp/cifar.tfrecords", "存进tfrecords的文件")

class CifarRead(object):
    """完成读取二进制文件， 写进tfrecords，读取tfrecords
    """
    def __init__(self, filelist):
        # 文件列表
        self.file_list = filelist

        # 定义读取的图片的一些属性
        self.height = 32
        self.width = 32
        self.channel = 3
        # 二进制文件每张图片的字节
        #每个图片的标签占的字节大小
        self.label_bytes = 1
        # 每个图片的占的字节大小
        self.image_bytes = self.height * self.width * self.channel

        #每个样本占的字节大小
        self.bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self):

        # 1、构造文件队列
        file_queue = tf.train.string_input_producer(self.file_list)

        # 2、构造二进制文件读取器，读取内容, 每个样本的字节数 设置二进制文件每次读取的字节数（即每个样本的字节大小）
        reader = tf.FixedLengthRecordReader(self.bytes)

        key, value = reader.read(file_queue)

        # 3、解码内容, 二进制文件内容的解码
        label_image = tf.decode_raw(value, tf.uint8)

        print(label_image)

        # 4、分割出图片和标签数据，切除特征值和目标值
        #slice（源数据,开始位置,分割大小） 分割 cast 改变元素数据类型
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)

        image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

        # 5、可以对图片的特征数据进行形状的改变 [3072] --> [32, 32, 3]
        image_reshape = tf.reshape(image, [self.height, self.width, self.channel])

        print(label, image_reshape)
        # 6、批处理数据
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=10)

        print(image_batch, label_batch)
        return image_batch, label_batch

if __name__ == "__main__":
    # 1、找到文件，放入列表   路径+名字  ->列表当中
    file_name = os.listdir(FLAGS.cifar_dir)

    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]

    # print(file_name)
    cf = CifarRead(filelist)

    # image_batch, label_batch = cf.read_and_decode()

    image_batch, label_batch = cf.read_from_tfrecords()

    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 打印读取的内容
        print(sess.run([image_batch, label_batch]))

        # 回收子线程
        coord.request_stop()

        coord.join(threads)
import tensorflow as tf
import os

#读取验证码图片进行处理

def picread(filelist):
    """
    读取csv文件
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    #1,构造文件名队列
    file_quqe = tf.train.string_input_producer(filelist)

    #2,构建图片阅读器读取队列数据 （默认读取一张图片）
    reader = tf.WholeFileReader()
    key,value = reader.read(file_quqe)
    #3,解码 成uint8
    image = tf.image.decode_jpeg(value)
    #4,统一图片的大小 像素uint8转float32
    image_resize = tf.image.resize_images(image,[200,200])

    #固定图片的形状 （通道数）因为批处理不支持不固定形状的图片 使用静态固定方法
    image_resize.set_shape([200,200,3])

    #进行批处理 图片批处理时一定要固定形状（长 宽 通道数）一次取出20个改变大小后的图片
    image_batch = tf.train.batch([image_resize],batch_size=20,num_threads=2,capacity=20)
    return image_batch

if __name__ == "__main__":
    #找到文件 放到列表
    filename = os.listdir("./data/code")
    filelist = [os.path.join("./data/code",name) for name in filename]

    image_resize = picread(filelist)
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()
        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        #回收子线程
        coord.request_stop()
        coord.join()
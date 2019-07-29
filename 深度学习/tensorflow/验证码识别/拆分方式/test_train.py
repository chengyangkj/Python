import tensorflow as tf
from PIL import Image

def divide_img(path):
    im = Image.open(path)
    im = im.convert('RGB')
    img1 = im.crop((0, 0, 20, 30))
    img2 = im.crop((20, 0, 35, 30))
    img3 = im.crop((35, 0, 50, 30))
    img4 = im.crop((50, 0, 70, 30))

    img1 = img1.resize((10,30))
    img2 = img2.resize((10, 30))
    img3 = img3.resize((10, 30))
    img4 = img4.resize((10,30))
    img2.save("3.jpg")
    img1.save("1.jpg")
    img3.save("8.jpg")
    img4.save("5.jpg")

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
    image.set_shape([10,30,3])

    return image
def fc_model(img_batch):
    with tf.variable_scope("model"):
        #将图片转为2维
        img_reshape = tf.reshape(img_batch,[-1,30*10*3])
        print(img_reshape)
        # 随机初始化权重
        weigth = tf.Variable(tf.random_normal(shape=[10 * 30 * 3,  10], mean=0.0, stddev=1.0))

        # 初始化偏置为0
        bias = tf.Variable(tf.constant(0.0, shape=[10]))

        # 全连接层计算 [100,4*10]  mutmul矩阵相乘 要求第一个矩阵第二维和第二个矩阵的第一维的维度相同
        y_predict = tf.matmul(tf.cast(img_reshape, tf.float32), weigth) + bias
    return y_predict

if __name__ == '__main__':
    #divide_img("1358.jpg")
    img_batch = picread("0095.jpg")
    y_pre = fc_model(img_batch)
    reshape = tf.reshape(y_pre, [10])
    label = tf.argmax(reshape, 0)

    #初始化变量
    init_op = tf.global_variables_initializer()

    #创建一个save保存训练模型
    # 初始化变量
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #设置进程管理器
        coor = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coor,sess=sess)
        sess.run(init_op)
        saver.restore(sess,"./tmp/codechar")
        print(label.eval())
        coor.request_stop()

        coor.join(threads)



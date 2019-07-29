import tensorflow as tf
import os
#预处理图片 将训练集处理为 tfrecoder文件格式
#改变tensorflow报错级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("tfrecords_dir", "./tfrecords/captcha.tfrecords", "验证码tfrecords文件")
tf.app.flags.DEFINE_string("captcha_dir", "/home/chengyangkj/data/codechar/code", "验证码图片路径")
tf.app.flags.DEFINE_string("captlabel_dir", "/home/chengyangkj/data/codechar/label.csv", "验证码标签路径")
tf.app.flags.DEFINE_string("letter", "0123456789", "验证码字符的种类")

def deal_label(label):
    """
    处理标签 将标签拆分单个
    :param label:
    :return:
    """
    array = []
    for byte in label:
        array2 = []
        #label里面的数据为[b'6732' b'7808' b'5236'] 需要解码  .rjust(4,'0')字符串向右对齐 前面不足四位补0
        for l in byte.decode('utf-8').rjust(4,'0'):
            array2.append(int(l,10))
        array.append(array2)

     #将array转为tensor类型
    label = tf.constant(array)
    print(label)
    return label
def get_imgBatch(img_path):
    """
    将图片数据集转为管道数据
    :return:
    """
    lists = os.listdir(img_path)
    img_list = [os.path.join(img_path,name) for name in lists]

    #排序图片
    img_list.sort()
    #构建图片文件名队列
    file_name_quque = tf.train.string_input_producer(img_list)

    #构建文件阅读器读取队列数据
    reader = tf.WholeFileReader()
    key,value = reader.read(file_name_quque)

    #解码图片 成uint8
    img = tf.image.decode_jpeg(value)

    # 4,统一图片的大小 像素uint8转float32 不然batch会报错 不知道为什么 这里reshape不要弄错了 是先高后宽
    image_resize = tf.image.resize_images(img, [30, 80])

    #固定图片形状(通道数)，因为下一步要进行批处理 批处理不支持形状不固定的数据，所以需要固定形状
    image_resize.set_shape([30,80,3])

    #创建管道数据 设置线程数等
    img_batch = tf.train.batch([image_resize],batch_size=4000,num_threads=2,capacity=4000)


    return img_batch

def get_labelBatch(label_path):
    """
    获取label标签的批处理文件
    :return:
    """
    #创建csv文件名的队列  path参数添的是列表 传入文件列表
    file_name_quqe = tf.train.string_input_producer([label_path],shuffle=False)
    #构建csv阅读器读取队列数据 一次读一行
    reader = tf.TextLineReader()
    key,value = reader.read(file_name_quqe)
    #解码csv文件
    #设置解码数据的缺省值和类型
    recorder = [[1],["None"]]
    number,label= tf.decode_csv(value,record_defaults=recorder)

    print(label)

    #构建管道数据 返回传入构建的列表 这里只构建了label

    label_batch = tf.train.batch([label],batch_size=4000,num_threads=2,capacity=4000)
    return label_batch

def save_to_recorder(img_batch,label_batch):
    """
    将对应的img和label保存为recorder文件
    :return:
    """

    # 转换类型
    label_batch = tf.cast(label_batch, tf.uint8)
    print(label_batch)
    # 建立TFRecords 存储器
    writer = tf.io.TFRecordWriter(FLAGS.tfrecords_dir)

    for i in range(3000):
        print("写第：%d"%(i))
        # 取出第i个图片数据，转换相应类型,图片的特征值要转换成字符串形式

        print(label_batch[i])
        #转换一下类型 float32——>uint8 避免转为string后长度变化
        img_retype = tf.cast(img_batch[i].eval(), tf.uint8)
        print(img_retype)
        img_string = img_retype.eval().tostring()
        print(len(img_string))
        #取出label数据
        # 标签值，转换成整型
        label_string = label_batch[i].eval().tostring()

        # 构造协议块
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_string])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))
        }))

        writer.write(example.SerializeToString())
    writer.close()
    return None
if __name__ == '__main__':
    img_batch =get_imgBatch(FLAGS.captcha_dir)
    label_batch = get_labelBatch(FLAGS.captlabel_dir)

    with tf.Session() as sess:
        #设置进程管理器
        coor = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess,coord=coor)

        label_arry = sess.run(label_batch)
        label_deal = deal_label(label_arry)

        save_to_recorder(img_batch,label_deal)

        coor.request_stop()
        coor.join()
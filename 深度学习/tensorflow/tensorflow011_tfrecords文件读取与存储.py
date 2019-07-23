import tensorflow as tf
import os

#tfrecords的优点：
#1,方便移动 2,类字典格式 格式规范

#建立数据等FLAG
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("code_dir","../data/code/","验证码文件的目录")
tf.app.flags.DEFINE_string("code_tfrecords","../tmp/code.tfrecords/","验证码样本tfrecords保持路径")
def write_to_tfrecords(image_batch,label_batch):
    """
    讲图片的特征值和目标值存入tfrecords
    :param image_batch: 100张图片特征值
    :param label_batch: 100张图片目标值
    :return: None
    """
    #1,构造一个tfrecords存储器
    writer = tf.python_io.TFRecordWriter(FLAGS.code_tfrecords)

    #2,循环讲所有的样本写入文件

    for i in range(100):
        #取出第i个样本的特征值和目标值 eval()获取张量的值 feature接收的是string类型的，所以需要转string
        image = image_batch[i].eval().tostring()
        #label_batch 为二维 需获取里面的具体的值
        label = label_batch[i].eval()[0]
        #构造一个example
        example = tf.train.Example(features=tf.train.Feature(feature={
            "image":tf.train.Feature(byte_list=tf.train.BytesList(value=[image])),
            "label":tf.train.Feature(byte_list=tf.train.Int64List(value=[label])),
        }
        ))

        #写入单独的样本 字符串序列化为example
        writer.write(example.SerializeToString())

    writer.close()
def read_from_tfrecords():
    """
    从tfrecords文件读取出数据
    :return:
    """
    #1,构建文件队列
    file_queue = tf.train.string_input_producer([FLAGS.code_tfrecords])

    #2,构建tfrecods文件阅读器,读取内容example value为一个样本序列化example

    reader = tf.TFRecordReader()
    key,value = reader.read(file_queue)

    #3,解析example
    features = tf.parse_single_example(value,features={
        "image":tf.FixedLenFeature([],tf.string),
        "label": tf.FixedLenFeature([], tf.int64)
    })
    #4,解码内容 如果读取的内容是string需要解码 int64 float32 不需要解码
    image = tf.decode_raw(features["image"],tf.uint8)
    label = tf.cast(features["label"],tf.int32)

    #5,固定图片形状 方便批处理

    image_reshape = tf.reshape(image,[200,200,3])

    #6,进行批处理
    image_batch,label_bash = tf.train.batch([image,label],batch_size=100,num_threads=10,capacity=100)

    return image_batch,label_bash
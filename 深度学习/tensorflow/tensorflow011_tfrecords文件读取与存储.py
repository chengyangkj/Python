import tensorflow as tf
import numpy
import scipy.misc as misc
import os
import cv2
def write_binary():
    cwd = os.getcwd()
    classes=['ym','zly','lyf']
    writer = tf.python_io.TFRecordWriter('./dara/tfrecords/data.tfrecord')
    for index, name in enumerate(classes):
        class_path = os.path.join(cwd,name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path , img_name)
            img = misc.imread(img_path)
            img1 = misc.imresize(img,[250,250,3])
            img_raw = img1.tobytes()              #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))}
                ))

    #序列化
            serialized = example.SerializeToString()
    #写入文件
            writer.write(serialized)
    writer.close()


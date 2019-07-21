import tensorflow as tf
import os
#1,创建文件队列 将文件名读入队列

#2
def csvread(filelist):
    """
    读取csv文件
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    #1,构造文件名队列
    file_quqe = tf.train.string_input_producer(filelist)

    #2,构建csv文件阅读器读取队列数据 (按一行)
    reader = tf.TextLineReader()

    key,value = reader.read(file_quqe)

    #3，对每行内容进行解码 record_defaults 指定每一列的默认类型和，默认缺省值
    #这里我们的csv文件为两列，因此需要指定两列缺省值 指定当这一列为空时默认的值为None 并且为string类型的
    record = [["None"],["None"]]
    example,label = tf.decode_csv(value,record_defaults=record)

    #4,想要一次读取多个数据，就要进行批处理 管道（将解码后的文件再放入队列中） 批处理的大小 和批处理的队列大小 和数据大小无影响 只决定这一次取多少个数据
    #tf.train.batch(tensor,batch_size,num_threads,capacity)
    # tensor指批处理队列中要存放的tensor数据，
    # batch_size 指批处理每次取出的数据量
    # capacity指管道队列的大小
    example_batch,label_batch = tf.train.batch([example,label],batch_size=9,num_threads=1,capacity=9)
    #这里的example_batc 和example 不一样，example只有一个数据，而example_batch 是经过管道处理后的数据，有9个数据
    return example_batch,label_batch
if __name__ == "__main__":
    #找到文件 放到列表
    filename = os.listdir("./data/csvdata")
    filelist = [os.path.join("./data/csvdata",name) for name in filename]
    example_batch ,label_batch = csvread(filelist)

    #开启会话运行结果
    with tf.Session() as sess:
        #定义一个线程协调器
        coord = tf.train.Coordinator()

        #开启读文件的线程 在上面的TextLineReader decode_csv等op都会被自动加入线程中去运行
        thread = tf.train.start_queue_runners(sess,coord=coord)
        print(sess.run([example_batch,label_batch]))

        #使用线程协调器 回收子线程
        coord.request_stop()
        coord.join(thread)


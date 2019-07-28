#tensorflow验证码识别
##一，主要思路
###1,数据预处理将训练集数据处理为tfrecorder模式
###2，在训练程序中，通过管道数据读取tfrecorder文件，获取训练值和标签值
###3，通过构建简单神经网络，损失值和优化方法进行训练
##二，数据预处理
如果不进行数据预处理，会造成训练时频繁的IO操作，造成训练效果低下，因此我们先将数据处理为tfrecorder格式，
然后在训练时开启线程，通过管道读取数据，传入网络

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
    
        for i in range(5000):
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
            
            
这里tensorflow中不知道为什么tfrecorder文件写入的特别慢，我写入了5000张图片，用了4个小时
##三，数据训练

训练的过程通过搭建简单神经网络，读取格式化后的tfrecorder文件，解码图片数据和label数据，通过批处理，一次处理100张
并计算损失值等
相关代码说明在注释里面有，这里就不详细讲了



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
            for i in range(2000):
                sess.run(train_op)
    
                print("第%d批次的准确率为：%f,损失%f" % (i, accuracy.eval(), mean_loss.eval()))
                # 在训练结束后保存模型
            saver.save(sess, "./tmp/result/codechar")
            # 回收线程
            coord.request_stop()
    
            coord.join(threads)

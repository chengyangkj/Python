#验证码识别（拆分方法）
这里就没有将数据集合成为tfrecorder格式，而是直接通过队列编解码图片和标签数据
对于自己制作的数据集一定要注意读入的数据的图片和标签是否对应
将一张图片四个验证码进行拆分，制作训练集，建立的为简单神经网络
##一，训练
train.py

相关说明在注释
        
        import tensorflow as tf
        import os
        from PIL import Image
        def get_label_batch(path):
            """
            获取label标签的批处理函数
            :param path: 传入label的位置
            :return: bath
            """
            #构建文件名队列 shuffle是否打乱顺序
            file_name = tf.train.string_input_producer([path])
            #打开csv文件阅读器，阅读队列数据，一次一行
            reader = tf.TextLineReader()
            key, value = reader.read(file_name)
        
            #解码数据
            #构建默认值列表
            recorder = [[1],[1]]
            number,label = tf.decode_csv(value,record_defaults=recorder)
        
            #构建管道数据
            label_batch = tf.train.batch([label],batch_size=100,num_threads=2,capacity=100)
        
            return label_batch
        def get_img_batch(path):
            """
            获取图片数据的批处理函数
            :param path:
            :return:
            """
            lists = os.listdir(path)
            img_list = [os.path.join(path, name) for name in lists]
            #将图片名按照升序排序
            img_list.sort()
            print(img_list[2])
            #shuffle表示是否打乱图片顺序
            file_name = tf.train.string_input_producer(img_list,shuffle=False)
        
            #打开图片阅读器
            reader = tf.WholeFileReader()
            number,value = reader.read(file_name)
        
            #解码图片 为uint8
            img = tf.image.decode_jpeg(value)
        
            #固定图片的形状
            img_reshape = tf.reshape(img,[30,10,3])
        
            #构建管道数据
            img_batch = tf.train.batch([img_reshape],batch_size=100,num_threads=2,capacity=100)
        
            return img_batch
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
        def conv_to_onthot(label_batch):
            #label_batch [100]
            label_one = tf.one_hot(label_batch,depth=10,on_value=1,axis=1)
        
            return label_one
        def conv_to_number(onehot_batch):
            number_batch = tf.argmax(onehot_batch,axis=1)
            return number_batch
        if __name__ == '__main__':
            label_batch = get_label_batch("/home/chengyangkj/data/splitcode/label.csv")
            img_batch = get_img_batch("/home/chengyangkj/data/splitcode/code/")
        
            y_true = conv_to_onthot(label_batch)
            writer = tf.summary.FileWriter('logs')
            y_pre = fc_model(img_batch)
        
            #softmax计算损失值
            with tf.variable_scope("softmax"):
                mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_true,
                    logits=y_pre
                )
               )
        
            # 5、梯度下降优化损失
            with tf.variable_scope("optimizer"):
                train_op = tf.train.GradientDescentOptimizer(0.001).minimize(mean_loss)
        
            #计算准确率
            with tf.variable_scope("acc"):
        
                equal_list =tf.equal(tf.argmax(y_true,axis=1),tf.argmax(y_pre,axis=1))
                # equal_list  100个样本   [1, 0, 1, 0, 1, 1,..........]
                accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))
        
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
                for i in range(2000):
                    sess.run(train_op)
                    print("第%d批次的准确率为：%f,损失%f" % (i, accuracy.eval(), mean_loss.eval()))
        
                saver.save(sess,"./tmp/codechar")
                coor.request_stop()
                coor.join()
        
        
            
  ##二，测试
    test_train.py
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
    
    
    
            
        

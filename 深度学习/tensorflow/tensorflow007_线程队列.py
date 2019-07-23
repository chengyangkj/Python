#要训练的图片保存在磁盘上，进行训练时进行读取时会造成速度减慢，因此我们需要把数据保存的内存中（队列）
#并且tensorflow有 1 ，多线程机制，可以并行的去执行任务 2，文件的改善（tfrecords）
#在子线程读取数据 主线程训练数据 子线程和主线程通过队列进行通信
#队列 先进先出队列： tf.FIFOQueue (capacity,dtypes,name='' ) capacity 队列大小，dtype 存入数据的对象列表
import tensorflow as tf

#模拟异步 子线程存入样本 主线程 读取样本
#1，定义大小为1000 类型为float32 的队列
Q = tf.FIFOQueue(1000,tf.float32)
#2，定义子线程要干的事情 循环值+1 放入队列
var = tf.Variable(0.0,tf.float32)
#实现一个自增 tf.assign_add(var,1) 这个op将值进行加1,并且返回+1后的var本身
data = tf.assign_add(var,tf.constant(1.0))
en_q = Q.enqueue(data)
#3,定义队列管理器 op，指定多少个子线程，子线程该干什么事情 2个线程执行en_q op 因为tensorflow具有依赖性 执行en_q后，就会自动执行上面的相关操作
#存的数据是 变量的为tensor类型(var,data) 为操作的是op（en_q,Q）
qr = tf.train.QueueRunner(Q,enqueue_ops=[en_q]*2)
#初始化变量的op
init_op = tf.global_variables_initializer();

with tf.Session() as sess:
    #初始化变量
    sess.run(init_op)

    #开启线程管理器
    coord = tf.train.Coordinator()
    #开启真正子线程 并且指定session和线程协调器
    threads = qr.create_threads(sess,coord=coord,start=True)

    #主线程，不断读取数据，训练数据
    for i in range(300):
        print(sess.run(Q.dequeue()))

    #在主线程结束后 使用线程协调器关闭子线程
    coord.request_stop()
    coord.join(threads)
#这种是最基础的线程操作 在下面的csv读取操作中，可以使用start_queue_runners自动将相关文件操作op加入线程中，就没有这么麻烦了
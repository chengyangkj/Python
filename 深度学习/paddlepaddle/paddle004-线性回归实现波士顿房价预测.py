import paddle.fluid as fluid
import paddle
import paddle.dataset.uci_housing as uci_housing

#定义简单线性网络
x = fluid.layers.data(name='x',shape=[13],dtype='float32')
#全连接层
hidden = fluid.layers.fc(input=x,size=100,act='relu')
#输出层 输出预测结果
net = fluid.layers.fc(input=hidden,size=1,act=None)

#定义损失函数
y = fluid.layers.data(name='y',shape=[1],dtype='float32')
cost = fluid.layers.square_error_cost(input=net,label=y)
avg_cost =fluid.layers.mean(cost)

#复制一个主程序
test_programe = fluid.default_main_program().clone(for_test=True)

#定义优化方法
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
opts = optimizer.minimize(avg_cost)

#创建一个使用cpu的执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
#进行参数初始化
exe.run(fluid.default_startup_program())

#从接口中获取房价数据 使用管道批处理 一次获得128个
train_reader = paddle.batch(reader=uci_housing.train(),batch_size=128)
train_test = paddle.batch(reader=uci_housing.test(),batch_size=128)

#定义输入数据维度 避免进行训练集拆分
feeder = fluid.DataFeeder(place=place,feed_list={x,y})

#开始训练和测试

#训练次数100次
for pass_id in range(100):
    #训练管道中的每一批数据 （128）
    #输出每一批最后一个损失值
    train_cost = 0
    for batch_id,data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost]
                             )
    print("第：%d批，训练的损失值为：%0.5f"%(pass_id,train_cost[0][0]))


    #开始测试并输出最后一个batch损失值
    test_cost=0
    for batch_id,data in enumerate(train_test()):
        test_cost=exe.run(program=test_programe,
                          feed=feeder.feed(data),
                          fetch_list=[avg_cost]
                          )
    print("test:%d,cost:%0.5f"%(pass_id,test_cost[0][0]))

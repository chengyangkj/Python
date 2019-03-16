#coding:utf-8
#加载曳尾花的数据
from sklearn.datasets import load_iris, fetch_20newsgroups
#加载拆分api
from sklearn.model_selection import train_test_split
#实例化数据集对象
li = load_iris()

print("获取特征值")
print(li["data"])
print("目标值")
print(li.target)
print(li.DESCR)
#拆分 第一个参数传入特征值 第一个目标值 第三个为训练集的大小
#返回值训练集特征 测试集特征 训练集目标，测试值目标
#注意返回值 训练集 train 训练集特征x_train 训练集目标 y_train 测试集：test x_test y_test
x_train, x_test,  y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
print("训练集特征和目标", x_train, y_train)
print("测试集特征和目标", x_test, y_test)

#加载新闻数据集分类的数据集 subset参数传入要不的类型 train 训练集 test 测试集 all全部
news =fetch_20newsgroups(subset="all")
print(news.data)
print(news.target)
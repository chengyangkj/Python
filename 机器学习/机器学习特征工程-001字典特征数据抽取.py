#coding:utf-8
from sklearn.feature_extraction import DictVectorizer
#特征工程目的：将数据处理为机器可以识别到的数据
#字典数据抽取
#实例化DIctVectorizer
dict = DictVectorizer(sparse=False) #sparse 是否转换为三元组形式 默认为true false的话就显示数组形式
#调用fit_transform
data =  dict.fit_transform([{"city":"北京","temperature":100},{"city":"上海","temperature":86}])
print(dict.get_feature_names())  #打印矩阵的列标
#打印结果 矩阵形式
print(data)

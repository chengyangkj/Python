#coding:utf-8
#归一化处理：将数据转换到某一特定区间内
#步骤：
#1.实例化 minmaxscaler
#2.调用fit_transfrom方法
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler(feature_range=(2, 3))  #分区里面填写转化的区间 这里转换的范围为2-3
data = mm.fit_transform([[90, 2, 10, 40, ], [60, 4, 15, 45, ],[75, 3, 13, 45]])
print(data)
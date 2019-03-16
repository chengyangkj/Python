#coding:utf-8
#和countveercter差不多，都是需要统计词的数量
#TF term frequency:词的频率
#idf：逆文档频率 log(总文档数量/该词出现的文档的数量)
#tf*idf 称为词在文档中的重要性
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

#实例化tf 分析
tf = TfidfVectorizer()

print("*"*100)
#使用jieba分词 处理中文文本 并进行特征处理
c1 = jieba.cut("人生苦短，我用python")
c2 = jieba.cut("人生漫长我不用python")
#处理分词结果 转换为列表结果
content1 = list(c1)
content2 = list(c2)
#转换为字符串结果 join方法
c1 = " ".join(content1)
c2 = " ".join(content2)
#进行tf特征处理 得到的结果为词的重要性系数
data = tf.fit_transform([c1, c2])
print(tf.get_feature_names())
print(data.toarray())
















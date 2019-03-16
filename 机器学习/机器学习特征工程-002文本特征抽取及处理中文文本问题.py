#coding:utf-8
from sklearn.feature_extraction.text import CountVectorizer
#jieba分词模块
import jieba
cv = CountVectorizer()
data = cv.fit_transform(["life is short i choose python", "life is to long i didnt like python"])
#统计每个词的出现次数，并保存为矩阵模式，对单个的英文字母不做统计，因为一个英文字母不能反应文章的内容
print(cv.get_feature_names())
print(data.toarray())   #toarray 转换为数组形式
print("*"*100)

#处理中文文本 但是这种模式不支持获取每个词，因为他默认是以空格作为词的分格
data = cv.fit_transform(["人生苦短，我用python", "人生漫长我不用python"])
print(cv.get_feature_names())
print(data.toarray())
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
#进行特征处理
data = cv.fit_transform([c1, c2])
print(cv.get_feature_names())
print(data.toarray())
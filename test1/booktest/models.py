from django.db import models
#设计和表模型类的数据
# Create your models here.
class BookInfoManager(models.Manager):  #继承models.Manager
    def all(self):  #重写all方法
        books=super(BookInfoManager, self).all()  #调用父类的all方法（super为父类）
        book=books.filter(id=2)  #获得id为2的
        return book

#图书lei

#图书类
class BookInfo(models.Model):  #继承自模型类
    "图书模型类"
    #charfiled 说明是一个字符串 maxlence最大长度
    btitle=models.CharField(max_length=20)
    #datefiled说明是一个日期类型
    bpub_date=models.DateField()
    #模型类生产
    objects=BookInfoManager()
    def __str__(self):
        return self.btitle
    class Meta :
        db_table="bookinfo"


class HeroInfo(models.Model):
    hname=models.CharField(max_length=20) #英雄名称
    hgender=models.BooleanField(default=True)  #性别 为bool类型，且默认值为true true代表女 false代表男
    hcomment=models.CharField(max_length=128)
    hbook =models.ForeignKey('BookInfo')  #外键，生成一对多的关系对应表的字段名 关系属性名_id
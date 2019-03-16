#encoding:utf-8
import requests
response=requests.get("https://www.baidu.com/img/baidu_jgylogo3.gif")
 #保存
with open("b.gif","wb") as f:  #保存的文件名 保存的方式（wb 二进制  w 字符串）
     f.write(response.content)
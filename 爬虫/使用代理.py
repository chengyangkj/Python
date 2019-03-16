# coding=utf-8
import requests
proxies={"http":"http://115.159.116.37:60372"} #设置代理地址
headers={
    "User-Agent":"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:63.0) Gecko/20100101 Firefox/63.0"
}
response=requests.get("https://www.baidu.com", headers=headers,proxies=proxies)
assert response.status_code==200 #检测是否使用成功 不成功会报错
print (response.status_code)  #200代表正常


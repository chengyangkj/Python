#encoding:utf-8
import requests
url="https://fanyi.baidu.com/basetrans"
while 1:
    i=raw_input("请输入你要进行翻译的中文：")
    text=str(i)
    #print (text)
    data={"from":"zh","to":"en","query":text}
    header={"User-Agent" : "Mozilla/5.0 (iPhone; CPU iPhon….0 Mobile/15A372 Safari/604.1"}
    response=requests.post(url, headers=header,data=data)
    #print (response.status_code)
    print (response.json()['trans'][0]['dst'])
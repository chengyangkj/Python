#-*-coding:utf-8-*-
import requests
import io
headers={
     "User-Agent":"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:63.0) Gecko/20100101 Firefox/63.0",
     "Cookie":"JSESSIONID=95D6853FC472F7BF1AF598B5432830CA.kingo154",
}
url="http://xk.henu.edu.cn/MainFrm.html"
r=requests.get(url, headers=headers)
with open("index.html", "w", encoding="utf-8") as f:
    f.write(r.content.decode())
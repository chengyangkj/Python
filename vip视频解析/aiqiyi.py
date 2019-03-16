import requests
import json
def find(name):
    url = "https://search.video.iqiyi.com/o?if=html5&key="+name+"&pageNum=1"
    response = requests.get(url)
    content_json = json.loads(response.text)
    if content_json["data"] == "search result is empty": #换线
        print("暂未查找到！")
        pass
    else:
        content_list = content_json["data"]["docinfos"]
        print("查找结果为：")
        result = list()
        for i, data in enumerate(content_list):
            try:
                result.append(data["albumDocInfo"]["albumLink"])
                print(str(i) + ":" + data["albumDocInfo"]["albumTitle"])
            except Exception:
                break
    return  result


def analysis(index,result):
    urls = result[index]



def main():
    name = input("请输入你要查找的电影：")
    result = find(name)
    index = input("请输入你要解析的标号：")
    analysis(index,result)

if __name__ == "__main__":
    main()
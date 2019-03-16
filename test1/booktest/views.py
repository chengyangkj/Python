from django.shortcuts import render

from django.http import HttpResponse,JsonResponse,HttpResponseRedirect
from django.views.decorators.csrf import csrf_protect
from django.template import loader,RequestContext #引入模板类
# Create your views here.
#http://127.0.0.1:8000/index
#一个地址定义一个处理函数
from booktest.models import BookInfo


def index (request):
    #进行处理，和M，T进行交互
    # # 1.获取模板文件对象
    # temp=loader.get_template("booktest/index.html")
    # #2.定义模板上下文：给模板文件传数据
    # context=RequestContext(request,{})
    # #3.模板渲染：产生标准的HTML内容
    # res_html=temp.render(context)
    # return HttpResponse(res_html)
    return render(request, "booktest/index.html", {"content":"hello world"})
def show_books(request):
    books=BookInfo.objects.all()
    return render(request, 'booktest/show_books.html',{"books":books})
def detail(request,bid):
    book=BookInfo.objects.get(id=bid)
    heros=book.heroinfo_set.all()
    return render(request, 'booktest/detail.html', {'book':book, 'heros':heros, })
def test(request,num):
    return HttpResponse(num)
def load(request):
    return render(request, 'booktest/load.html')
def ajax_test(request):
    return  JsonResponse({"msg": "ok"})
def set_cookie(request):
    response=HttpResponse('设置cookie')
    #设置cookie信息，名字为num值为1
    response.set_cookie('num',1)
    return response
def get_cookie(request):
    num=request.COOKIES['num']
    return HttpResponse(num)
def login(request):
    if "username" in request.COOKIES :
        return render(request,"booktest/login.html",{"name":request.COOKIES["username"]})
    else:
        return render(request,"booktest/login.html")
@csrf_protect
def login_check(request):
    name = request.POST.get("username")
    password = request.POST.get("password")
    remember =request.POST.get("remember")
    if name == "chengyangkj" and password == "123":
        # 页面跳转
        response = HttpResponseRedirect("/index")
        # 设置cookie
        if request.POST.get("remember")=="on":
            print ("ok")
            response.set_cookie("username",name,max_age=7 * 24 * 3600)
        return response
    else:
        return HttpResponseRedirect("/login")
def guolvqi(request):
    
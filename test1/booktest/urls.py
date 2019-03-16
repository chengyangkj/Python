#引入url
from django.conf.urls import url
#引入视图类
from booktest import views
urlpatterns=[
    #通过url函数设置url路由配置项
    url(r'^index$',views.index), #引入url/index和视图index之间的关系
    url(r'^books$',views.show_books),
    url(r'^books/(\d+)$',views.detail),  #(\d+)为组，django会自动把组传给detail的函数参数
    url(r'^showtest(?P<num>\d+)$',views.test),
    url(r'^load$',views.load),
    url(r'^ajax_handle$',views.ajax_test),
    url(r'^set_cookie$',views.set_cookie),
    url(r'^get_cookie$',views.get_cookie),
    url(r'^login$',views.login),
    url(r'^login_check$', views.login_check),
    url(r'^guolvqi$',views.guolvqi),
]
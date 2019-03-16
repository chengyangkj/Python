import socket
def main():
    #udp 是面向报文的，只需绑定本机端口和ip即可开启本机的udp，即可接收到其他udp发送的请求
    # 1,创建套接字
    udp_socket = socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
    #2，绑定本地端口
    localaddr = ("",7788)  # 必须绑定自己电脑的ip和port
    udp_socket.bind(localaddr)
    send_data = input("请输入你要发送的内容：")
    # 发送数据 第一个参数写内容 第二个参数写地址
    udp_socket.sendto(send_data.encode("utf-8"), ("192.168.100.8", 8080))
    #接受数据
    udp_data = udp_socket.recvfrom(1024)  # 024表示一次接受的最大值 此方法会堵塞 直到收到消息为止
    # #udp_data的数据格式为元祖类型的需要进行转换
    recv_msg = udp_data[0] # 存储发送方的信息
    recv_ip = udp_data[1][0]
    recv_port = udp_data[1][1]
    print(recv_ip+":"+str(recv_port)+":"+recv_msg.decode("gbk")) #需要对发送的消息进行解码
    #5，关闭套接字

    udp_socket.close()


if __name__ == "__main__":
    main()
import socket
def main():
    #创建套接字
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #绑定端口
    socketaddr=("",7788)
    tcp_server_socket.bind(socketaddr)
    #讲套接字设为监听状态
    tcp_server_socket.listen(128)
    #等待别的用户接入 堵塞
    client_socket,client_addr = tcp_server_socket.accept() #用两个变量对返回的元祖数据进行拆包,第一个为链接的客户端变量的套接字对象，第二个存的是套接字的地址
    print(client_addr[0]+":"+client_addr[1])
    #接收客户端发送的信息
    recv_data = client_socket.recv(1024)
    print(recv_data.decode("gbk"))
    #发送消息
    client_socket.send(recv_data)
    client_socket.close()
    tcp_server_socket.close()
if __name__ == "__main__":
    main()
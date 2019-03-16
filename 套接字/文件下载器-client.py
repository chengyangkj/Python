import socket

def main():
    #创建socket实例对象
    tcp_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    #绑定端口和ip
    ip = input("请输入你要连接的服务器ip：")
    port = int(input("请输入你要连接的服务器端口："))
    tcp_addr = (ip,port)
    #连接
    tcp_socket.connect(tcp_addr)
    FileName = input("请输入你要下载的文件名：")
    #将文件名发给服务端
    tcp_socket.send(FileName.encode("utf-8"))
    recv_data =  tcp_socket.recv(1024)  #1024==1k 1024*1024-->1k*1024==1M 1024 *1024 *1024=1G
    with open("[新]"+FileName,"wb") as f:
        f.write(recv_data)
    #关闭套接字
    tcp_socket.close()
if __name__ == "__main__":
    main()
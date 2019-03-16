import socket

def main():
    tcp_client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server_ip = input("请输入要链接的服务器的ip")
    server_port = int(input("请输入要链接的port"))
    server_addr = (server_ip,server_port)
    #和服务端建立链接
    tcp_client.connect(server_addr)   #注意这里接收的类型为元祖类型
    send_data = input("请输入要发送的数据：")
    #发送数据
    tcp_client.send(send_data.encode("utf-8"))
    #接收数据
    recv_data =  tcp_client.recv(1024)
    print(recv_data.decode("gbk"))
    #关闭链接
    tcp_client.close()

if __name__ == "__main__":
    main()
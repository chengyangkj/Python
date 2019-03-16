import socket

def send_file_toclient(socket_client):
    filecontent = None
    FIleName = socket_client.recv(1024).decode("utf-8")
    print("客户端要接收的文件是" + FIleName)
    try:
        f = open(FIleName,"rb")
        filecontent = f.read()
    except Exception as e:
        print(e)
    if filecontent:
        socket_client.send(filecontent)
    else:
        socket_client.send("文件不存在！".encode("utf-8"))
def main():
    socket_server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    socketaddr = ("",7789)
    socket_server.bind(socketaddr)
    socket_server.listen(128)
    while True:
        socket_client, clent_addr = socket_server.accept()
        print(clent_addr[0]+"登录成功！")
        send_file_toclient(socket_client)

if __name__ == "__main__":
    main()
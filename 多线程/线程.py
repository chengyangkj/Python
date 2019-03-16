import time
import threading
def Thread1():  # 子线程1
        for i in range(5):
                print("你好我是1号线程")
                time.sleep(1)
def Thread2():  # 子线程2
        for i in range(5):
                print("我是二号线程")
                time.sleep(1)

def main():  # 主线程
	t1 = threading.Thread(target=Thread1)
	t2 = threading.Thread(target=Thread2)
	t1.start()
	t2.start()
	while True:
		print(threading.enumerate())	
		if len(threading.enumerate())<=1:
			break
	

if __name__ == "__main__":
	main()

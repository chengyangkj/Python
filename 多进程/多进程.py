import time
import multiprocessing


def test():
	for i in range(5):
		print("我是子进程1！")
		time.sleep(1)

def test1():
	
	for i in range(5):
		print("我是子线程2！")
		time.sleep(1)
def main():
	p1 = multiprocessing.Process(target=test)
	p1.start()
	p2 = multiprocessing.Process(target=test1)
	p2.start()
if __name__ == "__main__":
	main()

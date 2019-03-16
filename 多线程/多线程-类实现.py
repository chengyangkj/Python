import threading
import time
class MyThread(threading.Thread):
	def run(self):
		for i in range(5):
			print("我是子线程")
		time.sleep(1)


def main():
	t1 = MyThread()
	t1.start()
	for i in range(5):
		print("我是主线程")
	time.sleep(1)

if __name__ == "__main__":
	main() 

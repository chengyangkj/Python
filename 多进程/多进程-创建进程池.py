import multiprocessing
import os,time
def worker(msg):
	time_start = time.time()
	print("开始执行，进程号为：%d"%(os.getpid()))
	time.sleep(random.random()*2)
	time_end = time.time()
	print("执行完毕，经历时间为：%0.2f"%(time_end-time_start))
def main():
	print("--------start----------")
	po = multiprocessing.Pool(3)  # 定义容量为3的进程池
	#向进程池添加10个数据但是此时进程此=池的容量只有3 所以进程池会先执行前3个任务，当前三个任务执行完成后再继续向下执行下面的7个任务
	for i in range(0,10):
		po.apply_async(worker,(i,))  # 向进程池添加数据 第一代表要执行的进程的函数 第二个代表要传递的参数元祖
		print("%d加入进程池------------"%(i))
	po.close()  # 关闭进程池 此时进程池不会接收新的请求
	po.join()  # 该方法会堵塞，等待所有的子进程执行完后执行下面的代码
	print("--------end-----------")
if __name__ == "__main__":
	main()

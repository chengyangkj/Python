import multiprocessing

def download(q):
	#模拟数据下载
	data = [11,22,33,44]
	for t in data:
		q.put(t) #向队列添加数据
	print("数据一下载完成并写入队列")
def analyse(q):
	#模拟数据处理
	waiting_data = list()  #创建空的列表
	while True:
		data = q.get() #获取队列的第一个元素
		waiting_data.append(data)
		if q.empty():
			break
	print ("下载到的数据为:")
	for i in waiting_data:
		print(i)


def main():
	q = multiprocessing.Queue() #创建队列
	p1 = multiprocessing.Process(target=download,args=(q,)) #传入队列的引用
	p2 = multiprocessing.Process(target=analyse,args = (q,))
	p1.start()
	p2.start()
if __name__ == "__main__":
	main(`)

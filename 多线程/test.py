num=1
def test1():
	global num
	num=2
def main():
	print (num)
	test1()
	print (num)
if __name__ == "__main__":
	main()	

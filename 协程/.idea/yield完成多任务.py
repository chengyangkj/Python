#encoding:utf-8
import time

def task_1():
    while True:
        print ("子线程1")
        time.sleep(1)
        yield


def task_2():
    while True:
        print ("子线程2")
        time.sleep(1)
        yield



def main():
    t1 = task_1()
    t2 = task_2()
    while True:
        next(t1)
        next(t2)
if __name__ == "__main__":
    main()
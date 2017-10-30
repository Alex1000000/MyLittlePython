import numpy as np
import time
from numpy import random
import threading
import multiprocessing as mp
from multiprocessing import Process, Lock, freeze_support

def lineOneT(x, y, z, i, j):
    for r in range(0, len(x[0])):
        z[i][j] = z[i][j] + x[i][r]*y[r][j]
    return

def lineOne(x,y,q,i,j):
    # global x,y,z,i,j
    value=0
    for r in range(0, len(x[0])):
        value = value + x[i][r]*y[r][j]
    q.put(value)
    return


def main():
    n=4
    m=5
    k=7
    x=random.random((n,k))
    y=random.random((k,m))
    z=random.random((n,m))
    # 3x3 matrix
    # x = [[12,7,3],
    #     [4 ,5,6],
    #     [7 ,8,9]]
    # # 3x4 matrix
    # y = [[5,8,1,2],
    #     [6,7,3,0],
    #     [4,5,9,1]]
    # # result is 3x4
    # z = [[0,0,0,0],
    #          [0,0,0,0],
    #          [0,0,0,0]]

    lock = Lock()
    jobs = []
    print(len(x),len(x[0]),len(y),len(y[0]))
    for j in range(0,len(y[0])):
        for i in range(0,len(x)):
            z[i][j]=0
            q = mp.Queue()
            p = mp.Process(target=lineOne,args=(x,y, q,i,j))
            jobs.append(p)
            p.start()
            z[i][j] = q.get()
            # t = threading.Thread(target=lineOneT, args=(x, y, z, i, j))
            # t.start()
            # t.join()
            # for r in range(0,len(x[0])):
            #     z[i][j]=z[i][j]+x[i][r]*y[r][j]
    print(z)
    print(np.dot(x,y))


if __name__ == '__main__':
    time1=time.time()
    freeze_support()
    main()
    time2=time.time()
    print("time=",time2-time1)
    #time= 3.7714178562164307
    # time = 0.031249523162841797
    #time= 0.015642404556274414
    #
    #
    #
    #
    #time= 0.14577984809875488
    #time= 2.124798536300659

    #time= 123.42744278907776
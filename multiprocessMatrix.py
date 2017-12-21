import numpy as np
import time
from numpy import random
import threading
import multiprocessing as mp
from multiprocessing import Process, Lock, freeze_support


def lineOneT(x, y, z, i, j): #forTreading
    for r in range(0, len(x[0])):
        z[i][j] = z[i][j] + x[i][r]*y[r][j]
    return

def lineOne(x,y,q,i,j): #forProcessing
    # global x,y,z,i,j
    value=0
    for r in range(0, len(x[0])):
        value = value + x[i][r]*y[r][j]
    q.put(value)
    return
def colOneT(Z, row, A, B):
    # print("len(A)=",len(A))
    # print(A[1,0])
    # print(A)
    # print("len(B)=",len(B[0]))
    # print("len(Z)=",len(Z),"," ,len(Z[0]))
    # Z[row, :] = A[row, :].dot(B)
    # for j in range(0, len(y[0])):
    # # for i in range(0, len(x)):
    # #         z[i][j] = 0
    # #         for r in range(0, len(x[0])):
    # #             z[i][j] = z[i][j] + x[i][r] * y[r][j]
    for i in row:
        for j in range(0,len(B[0])):
            Z[i, j]=0
            for k in range(0,len(A[0])):
                Z[i,j] = Z[i,j] + A[i,k] * B[k,j]

    return
def colOneM(Z, row, A, B,q):
    for i in row:
        for j in range(0,len(B[0])):
            Z[i, j]=0
            for k in range(0,len(A[0])):
                Z[i,j] = Z[i,j] + A[i,k] * B[k,j]
    q.put([Z[row, :], row])
    # q.put([A[row, :].dot(B), row])
    # Z[row, :] = A[row, :].dot(B)
    return

def threadOne(A, B, C, K, N, n1, n2):
    # print("n1=",n1,"  n2=",n2)
    for i in range(n1, n2):
        for j in range(0, K):
            C[i, j]=0
            for k in range(0, N):
                C[i, j] += A[i, k] * B[k, j]


def processOne( A, B, K, N, n1, n2, q):
    C_new = 0
    list_of_C = []
    for i in range(n1, n2):
        for j in range(0, K):
            C_new = 0
            for l in range(0, N):
                C_new += A[i, l] * B[l, j]
            list_of_C.append([C_new, i, j])
    q.put(list_of_C)


def main():
    def matraxMult_treading():
        treads = []
        n_rows_for_t=int(np.ceil((len(x))/number_of_cpus))
        for i in range(0, int(number_of_cpus)):
            if (n_rows_for_t+i*n_rows_for_t)<=len(x):
                row = np.arange(n_rows_for_t)+i*n_rows_for_t;
                t = threading.Thread(target=colOneT, args=(z, row, x, y))
                treads.append(t)
            else:
                if (n_rows_for_t+(i-1)*n_rows_for_t)<=len(x):
                    row=np.arange(n_rows_for_t+(i-1)*n_rows_for_t,len(x))
                    t = threading.Thread(target=colOneT, args=(z, row, x, y))
                    treads.append(t)
        time_start = time.time()
        for tre in treads:
            tre.start()
        for tre in treads:
            tre.join()
        time_stop = time.time()
        print("mulTread=", time_stop - time_start)
        # print("finished")
        return


    def matraxMult_processing():
        process = []
        n_rows_for_t=int(np.ceil((len(x))/number_of_cpus))
        q_main=mp.Queue()
        ques=[]
        for i in range(0, number_of_cpus):
            if (n_rows_for_t+i*n_rows_for_t)<=len(x):
                q = mp.Queue()
                row = np.arange(n_rows_for_t)+i*n_rows_for_t;
                p = mp.Process(target=colOneM, args=(z, row, x, y, q))
                process.append(p)
                ques.append(q)
            else:
                if (n_rows_for_t+(i-1)*n_rows_for_t)<=len(x):
                    q = mp.Queue()
                    row=np.arange(n_rows_for_t+(i-1)*n_rows_for_t,len(x))
                    p = mp.Process(target=colOneM, args=(z, row, x, y, q))
                    process.append(p)
                    ques.append(q)
        time_start = time.time()
        for pro in process:
            pro.start()
        for pro in ques:
            value_from_q = pro.get()
            z[value_from_q[1], :] = value_from_q[0]
        time_stop = time.time()
        print("mulProc=",time_stop-time_start)
        return

    n=1000
    m=1000
    k=100
    x=random.random((n,k))
    y=random.random((k,m))
    z=random.random((n,m))
    # print(np.dot(x, y))
    print("--------------")
    # A = np.zeros((M, N)) M=n, k=N
    # B = np.zeros((N, K))K=m
    # C = np.zeros((M, K))
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
    number_of_cpus = mp.cpu_count()
    print("number of cpu=",number_of_cpus)


    print(len(x), len(x[0]), len(y), len(y[0]))
    # time_n1=time.time()
    # number_of_cpus = number_of_cpus * 2
    matraxMult_treading()
    # print(z)
    treads=[]
    process = []
    ques=[]
    for i in range(0, 4):
        treads.append(
            threading.Thread(target=threadOne, args=(x, y, z, m, k, int(i * n / 4), int(i * n / 4 + n / 4))))
        q  = mp.Queue()
        ques.append(q)
        process.append(
            Process(target=processOne, args=( x, y, m, k, int(i * n / 4), int(i * n / 4 + n / 4), q)))

    time_start_tread = time.time()
    for elem in treads:
        elem.start()
    for elem in treads:
        elem.join()
    time_stop_tread = time.time()
    print("_treading=",time_stop_tread-time_start_tread)
    # print(z)
    time_start_proc = time.time()
    for elem in process:
        elem.start()
    for elem in ques:
        data = elem.get()
        for vij in data:
            z[vij[1], vij[2]] = vij[0]
    time_stop_proc = time.time()
    print("_processing=",time_stop_proc-time_start_proc)
    # print(z)
    # time_n2=time.time()
    # number_of_cpus = number_of_cpus *10
    matraxMult_processing()
    time_n3=time.time()
    for j in range(0, len(y[0])):
        for i in range(0, len(x)):
            z[i][j] = 0
            for r in range(0, len(x[0])):
                z[i][j] = z[i][j] + x[i][r] * y[r][j]
    time_n4 = time.time()
    #
    # # print("matraxMult_treading=",time_n2-time_n1)
    # # print("matraxMult_processing=",time_n3-time_n2)
    print("just mult=", time_n4 - time_n3)

    # for j in range(0,len(y[0])):
    #     for i in range(0,len(x)):
    #         z[i][j]=0
    #         for r in range(0,len(x[0])):
    #             z[i][j]=z[i][j]+x[i][r]*y[r][j]
    # print(z)
    # print(np.dot(x,y))


if __name__ == '__main__':
    time1=time.time()
    freeze_support()
    main()
    time2=time.time()
    # # print("time=",time2-time1)
    # C:\Users\Саша\AppData\Local\Programs\Python\Python35\python.exe
    # C: / Users / Саша / Desktop / MyLittlePython / multiprocessMatrix.py
    # --------------
    # number
    # of
    # cpu = 4
    # 1000
    # 100
    # 100
    # 1000
    # mulTread = 104.54187417030334
    # _treading = 83.6598756313324
    # _processing = 38.22137260437012
    # mulProc = 57.156723976135254
    # just
    # mult = 132.82513856887817
    #
    # Process
    # finished
    # with exit code 0

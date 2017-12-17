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
    Z[row, :] = A[row, :].dot(B)
    return
def colOneM(Z, row, A, B,q):
    q.put([A[row, :].dot(B), row])
    # Z[row, :] = A[row, :].dot(B)
    return


def main():
    def matraxMult_treading():
        treads = []
        # print("to_round=",(len(x))/number_of_cpus)
        n_rows_for_t=int(np.ceil((len(x))/number_of_cpus))
        # print("n_rows_for_t=",n_rows_for_t)
        for i in range(0, number_of_cpus):
            if (n_rows_for_t+i*n_rows_for_t)<=len(x):
                row = np.arange(n_rows_for_t)+i*n_rows_for_t;
                # print ("row=",row)
                t = threading.Thread(target=colOneT, args=(z, row, x, y))
                treads.append(t)
                t.start()
            else:
                if (n_rows_for_t+(i-1)*n_rows_for_t)<=len(x):
                    row=np.arange(n_rows_for_t+(i-1)*n_rows_for_t,len(x))
                    # print("row_2=", row)
                    t = threading.Thread(target=colOneT, args=(z, row, x, y))
                    treads.append(t)
                    t.start()
        for tre in treads:
            tre.join()
        # print("finished")
        return


    def matraxMult_processing():
        # lock = Lock()
        # jobs = []
        process = []
        # print("to_round=",(len(x))/number_of_cpus)
        n_rows_for_t=int(np.ceil((len(x))/number_of_cpus))
        # print("n_rows_for_t=",n_rows_for_t)
        q = mp.Queue()
        for i in range(0, number_of_cpus):
            if (n_rows_for_t+i*n_rows_for_t)<=len(x):
                row = np.arange(n_rows_for_t)+i*n_rows_for_t;
                # print ("row=",row, "proc=", i)
                p = mp.Process(target=colOneM, args=(z, row, x, y, q))
                process.append(p)
                p.start()
            else:
                if (n_rows_for_t+(i-1)*n_rows_for_t)<=len(x):
                    row=np.arange(n_rows_for_t+(i-1)*n_rows_for_t,len(x))
                    # print("row_2=", row, "proc=", i)
                    p = mp.Process(target=colOneM, args=(z, row, x, y, q))
                    process.append(p)
                    p.start()
        # print("len(process)=",len(process))
        for pro in process:
        # for j in range(0, len(process)):
        #     print("get from ", temp[1])
            value_from_q = q.get()
            # print("get from ", temp[1])
            # print("get from ", j, "get=", temp)
            z[value_from_q[1], :] = value_from_q[0]
        for pro in process:
            # print("join ", pro)
            pro.join()
        # print("finished")
        return

    n=50  #4
    m=600
    k=70
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
    number_of_cpus = mp.cpu_count()
    print("number of cpu=",number_of_cpus)


    print(len(x), len(x[0]), len(y), len(y[0]))
    time_n1=time.time()
    matraxMult_processing()
    time_n2=time.time()
    matraxMult_treading()
    time_n3=time.time()
    for j in range(0, len(y[0])):
        for i in range(0, len(x)):
            z[i][j] = 0
            for r in range(0, len(x[0])):
                z[i][j] = z[i][j] + x[i][r] * y[r][j]
    time_n4 = time.time()

    print("matraxMult_processing=",time_n2-time_n1)
    print("matraxMult_treading=",time_n3-time_n2)
    print("just mult=", time_n4 - time_n3)

    for j in range(0,len(y[0])):
        for i in range(0,len(x)):
            z[i][j]=0
            #miltProcess
            # q = mp.Queue()
            # p = mp.Process(target=lineOne,args=(x,y, q,i,j))
            # jobs.append(p)
            # p.start()
            # z[i][j] = q.get()
            #trading
            # t = threading.Thread(target=lineOneT, args=(x, y, z, i, j))
            # t.start()
            # t.join()
            #--just mult
            for r in range(0,len(x[0])):
                z[i][j]=z[i][j]+x[i][r]*y[r][j]
    print(z)
    print(np.dot(x,y))


if __name__ == '__main__':
    time1=time.time()
    freeze_support()
    main()
    time2=time.time()
    print("time=",time2-time1)
    # 50
    # 70
    # 70
    # 600
    # matraxMult_processing = 1.39772367477417
    # matraxMult_treading = 0.008013010025024414
    # just
    # mult = 2.675844669342041
import random
# from datetime import time
import threading
import time

import math
import numpy as np
import multiprocessing as mp

def acc_get(m, coordOfplanets):
    G = 6.67408 * (10 ** -11);
    numbOfPlanets = coordOfplanets.size // m.size
    acc = np.zeros(coordOfplanets.size)
    for i in range(m.size):
        for j in range(m.size):
            if i != j:
                rDiff = coordOfplanets[j * numbOfPlanets:(j + 1) * numbOfPlanets] - coordOfplanets[
                                                                              i * numbOfPlanets:(i + 1) * numbOfPlanets]
                dist=np.linalg.norm(rDiff)
                acc[i * numbOfPlanets:(i + 1) * numbOfPlanets] += G * m[j] * rDiff / (dist ** 3)
    return acc

def verlet_nBody(m,ip,iv,delta_t,n):
    # Plots the trajectories of 3 equal masses

    G = 6.67408 * (10 ** -11);
    times = np.arange(n) * delta_t
    numberOfPlanets = ip.size
    coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    coordinatesAndVel[0] = np.concatenate((ip, iv))
    acc = acc_get(m, coordinatesAndVel[0, : numberOfPlanets])
    for j in np.arange(n - 1) + 1:
        #r
        coordinatesAndVel[j, :numberOfPlanets] = coordinatesAndVel[j - 1, :numberOfPlanets] + coordinatesAndVel[j - 1, numberOfPlanets:] * delta_t + 0.5 * acc * delta_t ** 2
        acc_next = acc_get(m, coordinatesAndVel[j, :numberOfPlanets])
        #vel
        coordinatesAndVel[j, numberOfPlanets:] = coordinatesAndVel[j - 1, numberOfPlanets:] + 0.5 * (acc + acc_next) * delta_t
        acc = acc_next
    # tspan = sp.linspace(0, T, n + 1)
    return  coordinatesAndVel



def acceleration_no_np(body, masses, positions):
    G = 6.67408 * (10 ** -11);
    dimension = len(positions) // len(masses)

    res = np.zeros(dimension)
    displacement = np.empty(dimension)
    print("displacement=",displacement)
    for j in range(masses.size):
        if body != j:
            for k in range(dimension):
                displacement[k] = positions[j * dimension + k] - positions[body * dimension + k]
            res += G * masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res




def acceleration_no_np_proc(body, masses, positions, procId, iter):
    G = 6.67408 * (10 ** -11);
    dimension = len(positions) // len(masses)

    res = np.zeros(dimension)
    displacement = np.empty(dimension)
    # print("positions in A=",positions, "procId=", procId, "iter=",iter, "body num=",body)
    for j in range(masses.size):
        # print("positions in A=", positions[j], "procId=", procId, "iter=", iter, "body num=", body)
        if body != j:
            for k in range(dimension):
                displacement[k] = positions[j * dimension + k] - positions[body * dimension + k]
                if displacement[k] ==0:
                    print("disp=0:", positions[j * dimension + k] ,"-", positions[body * dimension + k],"procId=", procId, "iter=", iter, "body num=", body )
            # if (math.isnan(np.linalg.norm(displacement, 2))):
            #     print("NNNNan")
            # elif np.linalg.norm(displacement, 2)==0:
            #     print("Zero!!")
            res += G * masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    # print("displacement=", displacement, "procId=", procId, "iter=", iter, "body num=", body)
    return res



def mProc_newVersi(q, q_out, init_vel, shared_pos, body_row, events1, events2,m,delta_t,n, procId):
    dimension = len(shared_pos) // len(m)
    result_array=[]
    cur_acceleration=[]
    my_cur_pos=[]
    temp=[]
    ind_f = body_row[0]
    len_row = len(body_row)
    for body in body_row:
        my_cur_pos = np.concatenate((my_cur_pos,np.array(shared_pos[body*dimension:(body+1)*dimension])))
        cur_acceleration = np.concatenate((cur_acceleration,acceleration_no_np_proc(body, m, shared_pos,procId,0)))
        temp=np.concatenate((temp, my_cur_pos, init_vel[body*dimension:(body+1)*dimension]))
    print("my_cur_pos=",my_cur_pos)
    print("init_vel[row]=",init_vel[ind_f*dimension:(ind_f+1+len_row)*dimension])
    print("row=",body_row,"[",ind_f,",",(ind_f+1+len_row),"]","len_row=",len_row)
    result = np.empty((n, dimension * 2*len(body_row)))
    result[0, :] = np.concatenate((my_cur_pos, init_vel[ind_f*dimension:(ind_f+1+len_row)*dimension]))

    for j in np.arange(n - 1) + 1:
        for body in body_row:
            result[j, :dimension] = (my_cur_pos + result[j-1, dimension:] * delta_t
                 + 0.5 * cur_acceleration * delta_t**2)
        for body in body_row:
            # print(body, "body add to p", procId)
            q.put([body, result[j, :dimension]])
        events1[procId].set()
        # print("it=",j,"procId=",procId)
        if procId == 0:
            for i in range(0,len(events1)):
                # print("wait For=",i)
                events1[i].wait()
                # print("waitED For=", i)
                events1[i].clear()
            # print("geting from q, ", procId)
            for i in range(0, len(m)):
                # print(procId, "geting from q, mi=",i , "it=",j)
                tmp = q.get()
                # print(procId, "getED from q, mi=", i, "it=",j)
                shared_pos[tmp[0]*dimension:(tmp[0]+1)*dimension] = tmp[1]
            # print(procId, "get from q")
            for i in range(0,len(events2)):
                    events2[i].set()
        else:
            events2[procId].wait()
            events2[procId].clear()
        # print(j)
        my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
        # print("shared_pos=", shared_pos, "proc", procId, "it=", j)
        for k in range(len(shared_pos)):
            if math.isnan(shared_pos[k]):
                print("nan, proc", procId,"k=", k, "it=", j)
        for body in body_row:
            next_acceleration = acceleration_no_np_proc(body, m, shared_pos,procId,j)

            result[j, dimension:] = \
                (result[j-1, dimension:]
                 + 0.5 * (cur_acceleration + next_acceleration) * delta_t)
            cur_acceleration = next_acceleration
            result_array.append([body, result])
    for body in body_row:
        q_out.put(result_array)
    return


def mProc(q, q_out, init_vel, shared_pos, body_row, events1, events2,m,delta_t,n, procId):
    dimension = len(shared_pos) // len(m)
    result_array=[]
    result = np.empty((n, dimension * 2))
    my_cur_pos_lists=[]
    cur_acceleration_lists=[]
    net_acceleration_lists = []
    result_lists=[]
    result_next_lists = []
    temp_res=[]
    for body in body_row:
        my_cur_pos=np.array(shared_pos[body*dimension:(body+1)*dimension])
        my_cur_pos_lists.append(my_cur_pos)
        cur_acceleration = acceleration_no_np_proc(body, m, shared_pos,procId,0)
        cur_acceleration_lists.append(cur_acceleration)
        # temp_res=np.concatenate((my_cur_pos, init_vel[body * dimension:(body + 1) * dimension]))
        result_lists.append(np.concatenate((my_cur_pos, init_vel[body * dimension:(body + 1) * dimension])))

    result[0, :] = np.concatenate((my_cur_pos, init_vel[body * dimension:(body + 1) * dimension]))
    # print("my_cur_pos=", my_cur_pos)
    # print("result[0, :]", result[0, :])
    shared_pos_allInProc=[]

    for j in np.arange(n - 1) + 1:
        list_to_zeroP=[]
        for body in body_row:
            print("result_lists[body-body_row[0]]",result_lists[body-body_row[0]], "proc",procId)
            my_cur_pos=my_cur_pos_lists[body-body_row[0]]
            result[j-1,:]=result_lists[body-body_row[0]]
            cur_acceleration=cur_acceleration_lists[body-body_row[0]]
            # if j==1:
            #     my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
            #     cur_acceleration = acceleration_no_np_proc(body, m, shared_pos, procId, 0)
            #     result[0, :] = np.concatenate((my_cur_pos, init_vel[body * dimension:(body + 1) * dimension]))
            x2 = (my_cur_pos + result[j-1, dimension:] * delta_t
                 + 0.5 * cur_acceleration * delta_t**2)
            result[j, :dimension]=np.array(x2)
            # print([body, result[j, :dimension]]," put  ", result[j, :dimension], "to ",  "body=", body, "it=",j , "Proc=", procId)
            # for body in body_row:
            x=[body, np.array(x2), procId]
            q.put(x)
            list_to_zeroP.append(x)
            # x.clear()
            # print("PUT", x)
        # print("list_to_zeroP=",list_to_zeroP)
        events1[procId].set()
        # print("it=",j,"procId=",procId)
        if procId == 0:
            for i in range(0,len(events1)):
                # print("wait For=",i)
                events1[i].wait()
                # print("waitED For=", i)
                events1[i].clear()
            # print("geting from q, ", procId)
            for i in range(0, len(m)):
                # print(procId, "geting from q, mi=",i , "it=",j)
                tmp = q.get()
                # print(procId, "getED from q, mi=", i, "it=",j)
                shared_pos[tmp[0]*dimension:(tmp[0]+1)*dimension] = tmp[1]
                # shared_pos_allInProc.append()
                # print(tmp, "get   ind=( ", (tmp[0]*dimension),"," ,((tmp[0]+1)*dimension), ") body=",tmp[0], "it=",j, "Proc=", procId)
            # print(procId, "get from q")
            for i in range(0,len(events2)):
                    events2[i].set()
        else:
            events2[procId].wait()
            events2[procId].clear()
        # print(j)

        for body in body_row:
            my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
        # print("shared_pos=", shared_pos, "proc", procId, "it=", j)
        # for k in range(len(shared_pos)):
        #     if math.isnan(shared_pos[k]):
        #         print("nan, proc", procId,"k=", k, "it=", j)
        # for body in body_row:
            next_acceleration = acceleration_no_np_proc(body, m, shared_pos,procId,j)
            net_acceleration_lists.append(net_acceleration_lists)


            result[j - 1, :] = result_lists[body - body_row[0]]
            cur_acceleration = cur_acceleration_lists[body - body_row[0]]
            # result[j, dimension:] = \
            y2=  (result[j-1, dimension:]
                 + 0.5 * (cur_acceleration + next_acceleration) * delta_t)
            # cur_acceleration = next_acceleration

            result_next_lists.append(np.array(y2))

            result_array.append([body, result])
        # cur_acceleration = next_acceleration
        cur_acceleration_lists.clear()
        cur_acceleration_lists = net_acceleration_lists
        # result_lists.clear()
        result_lists=result_next_lists
        print("END,Proc",procId," of it",j,":",result_lists)

    for body in body_row:
        q_out.put(result_array)
    return







def mcProc(m,ip,iv,delta_t,n):
    number_of_cpus = mp.cpu_count()
    # print("mul proc start")

    G = 6.67408 * (10 ** -11);

    times = np.arange(n) * delta_t
    numberOfPlanets = ip.size

    coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    coordinatesAndVel[0] = np.concatenate((ip, iv))
    #

    times = np.arange(n) * delta_t

    coordinatesAndVel = np.empty((times.size, ip.size * 2))
    shared_pos = mp.Array('d', ip)
    print("len(m)=",len(m))
    print("m=",m)
    n_body_for1procc = int(np.ceil((len(m)) / number_of_cpus))

    events1 = []
    events2 = []
    for i in range(0, number_of_cpus):
        if (n_body_for1procc + i * n_body_for1procc) <= len(m):
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[len(events1)-1].clear()
            events2[len(events1)-1].clear()
        elif (n_body_for1procc + (i - 1) * n_body_for1procc) < len(m):
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[len(events1)-1].clear()
            events2[len(events1)-1].clear()

    q = mp.Queue()
    q_out = mp.Queue()
    processes = []
    print("n_body_for1procc=",n_body_for1procc)
    for i in range(0, number_of_cpus):
        if (n_body_for1procc + i * n_body_for1procc) <= len(m):
            row = np.arange(n_body_for1procc) + i * n_body_for1procc;
            print("row", row, "prc=", i)
            p=mp.Process(target=mProc_newVersi, args=(q, q_out, iv, shared_pos, row, events1, events2, m,delta_t,n,i))
            processes.append(p)
            p.start()
        elif (n_body_for1procc + (i - 1) * n_body_for1procc) < len(m):
            row = np.arange(n_body_for1procc + (i - 1) * n_body_for1procc, len(m))
            print("row_=", row, "prc=", i)
            p=mp.Process(target=mProc_newVersi, args=(q, q_out, iv, shared_pos,row,  events1, events2, m,delta_t,n,i))
            processes.append(p)
            p.start()

    dim = ip.size // len(m)
    # for i in processes:
    #     resFromq = q_out.get()
    #     for tmp in resFromq:
    #         print(tmp[0])
    #         coordinatesAndVel[:, tmp[0] * dim:(tmp[0] + 1) * dim] = tmp[1][:, :dim]
    #         coordinatesAndVel[:, ip.size + tmp[0] * dim: ip.size + (tmp[0] + 1) * dim] = tmp[1][:, dim:]

    for i in range(m.size):
        resFromq = q_out.get()
        for tmp in resFromq:
            # print("tmp[0]=",tmp[0])
            coordinatesAndVel[:, tmp[0] * dim:(tmp[0] + 1) * dim] = tmp[1][:, :dim]
            coordinatesAndVel[:, ip.size + tmp[0] * dim: ip.size + (tmp[0] + 1) * dim] = tmp[1][:, dim:]

    for process in processes:
        process.join()
    # print(coordinatesAndVel)
    # print("mul proc end")
    return











def solveForOneBodyM(q, q_out, init_vel, shared_pos, body_row, events1, events2,m,delta_t,n, proc_id):
    dimension = len(shared_pos) // len(m)
    result_array=[]
    for body in body_row:
        my_cur_pos = np.array(shared_pos[body*dimension:(body+1)*dimension])
        cur_acceleration = acceleration_no_np(body, m, shared_pos)

        result = np.empty((n, dimension * 2))
        result[0, :] = np.concatenate((my_cur_pos, init_vel[body*dimension:(body+1)*dimension]))

        for j in np.arange(n - 1) + 1:
            result[j, :dimension] = \
                (my_cur_pos
                 + result[j-1, dimension:] * delta_t
                 + 0.5 * cur_acceleration * delta_t**2)

            q.put([body, result[j, :dimension]])
            # print("body=",body)
            events1[proc_id].set()

            if proc_id == 0:
                for i in range(len(events1)):
                    events1[i].wait()
                    events1[i].clear()
                for i in range(len(m)):
                    tmp = q.get()
                    shared_pos[tmp[0]*dimension:(tmp[0]+1)*dimension] = tmp[1]
                for i in range(len(events2)):
                    events2[i].set()
            else:
                events2[proc_id].wait()
                events2[proc_id].clear()

            my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
            next_acceleration = acceleration_no_np(body, m, shared_pos)

            result[j, dimension:] = \
                (result[j-1, dimension:]
                 + 0.5 * (cur_acceleration + next_acceleration) * delta_t)
            cur_acceleration = next_acceleration
            result_array.append([body, result])
    q_out.put(result_array)
    return


def solveForOneBodyM2(q, q_out, init_vel, shared_pos, body_row, events1, events2,m,delta_t,n,procID):
    dimension = len(shared_pos) // len(m)
    result_array=[]
    for body in body_row:
        print(body," from ",procID,"started")
        my_cur_pos = np.array(shared_pos[body*dimension:(body+1)*dimension])
        cur_acceleration = acceleration_no_np(body, m, shared_pos)

        result = np.empty((n, dimension * 2))
        result[0, :] = np.concatenate((my_cur_pos, init_vel[body*dimension:(body+1)*dimension]))

        for j in np.arange(n - 1) + 1:
            result[j, :dimension] = \
                (my_cur_pos
                 + result[j-1, dimension:] * delta_t
                 + 0.5 * cur_acceleration * delta_t**2)

            q.put([body, result[j, :dimension]])
            events1[body].set()

            if body == 0:
                for i in range(len(m)):
                    events1[i].wait()
                    events1[i].clear()
                for i in range(len(m)):
                    tmp = q.get()
                    shared_pos[tmp[0]*dimension:(tmp[0]+1)*dimension] = tmp[1]
                for i in range(len(m)):
                    events2[i].set()
            else:
                events2[body].wait()
                events2[body].clear()

            my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
            next_acceleration = acceleration_no_np(body, m, shared_pos)

            result[j, dimension:] = \
                (result[j-1, dimension:]
                 + 0.5 * (cur_acceleration + next_acceleration) * delta_t)
            cur_acceleration = next_acceleration
            result_array.append([body, result])
            print(body, " from ", procID, "n=",n)
    q_out.put(result_array)
    return


def verlet_nBody_multiprocessing(m,ip,iv,delta_t,n):
    number_of_cpus = mp.cpu_count()
    print("mul proc start")
    # Plots the trajectories of 3 equal masses
    G = 6.67408 * (10 ** -11);
    times = np.arange(n) * delta_t
    numberOfPlanets = ip.size
    coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    coordinatesAndVel[0] = np.concatenate((ip, iv))
    times = np.arange(n) * delta_t
    coordinatesAndVel = np.empty((times.size, ip.size * 2))
    shared_pos = mp.Array('d', ip)

    n_body_for1procc = int(np.ceil((len(m)) / number_of_cpus))

    events1 = []
    events2 = []
    for i in range(0, len(m)): #number_of_cpus):
        events1.append(mp.Event())
        events2.append(mp.Event())
        events1[len(events1)-1].clear()
        events2[len(events1)-1].clear()

    q = mp.Queue()
    q_out = mp.Queue()
    processes = []
    print("n_body_for1procc=",n_body_for1procc)
    for i in range(0, number_of_cpus):
        if (n_body_for1procc + i * n_body_for1procc) <= len(m):
            row = np.arange(n_body_for1procc) + i * n_body_for1procc;
            print("row ", row,",",i)
            p=mp.Process(target=solveForOneBodyM, args=(q, q_out, iv, shared_pos, row, events1, events2, m,delta_t,n,i))
            processes.append(p)
            p.start()
        elif (n_body_for1procc + (i - 1) * n_body_for1procc) < len(m):
            row = np.arange(n_body_for1procc + (i - 1) * n_body_for1procc, len(m))
            print("row ", row, ",", i)
            p=mp.Process(target=solveForOneBodyM, args=(q, q_out, iv, shared_pos,row,  events1, events2, m,delta_t,n,i))
            processes.append(p)
            p.start()

    dim = ip.size // len(m)
    for i in processes:
        resFromq = q_out.get()
        for tmp in resFromq:
            coordinatesAndVel[:, tmp[0] * dim:(tmp[0] + 1) * dim] = tmp[1][:, :dim]
            coordinatesAndVel[:, ip.size + tmp[0] * dim: ip.size + (tmp[0] + 1) * dim] = tmp[1][:, dim:]

    for process in processes:
        process.join()
    print("mul proc end")
    return



def solveForOneBodyM2(q, q_out, init_vel, shared_pos, body, events1, events2,m,delta_t,n):
    dimension = len(shared_pos) // len(m)
    my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
    cur_acceleration = acceleration_no_np(body, m, shared_pos)

    result = np.empty((n, dimension * 2))
    result[0, :] = np.concatenate((my_cur_pos, init_vel[body * dimension:(body + 1) * dimension]))

    for j in np.arange(n - 1) + 1:
        result[j, :dimension] = \
            (my_cur_pos
             + result[j - 1, dimension:] * delta_t
             + 0.5 * cur_acceleration * delta_t ** 2)

        q.put([body, result[j, :dimension]])
        events1[body].set()

        # print(body,"wait", j)

        if body == 0:
            for i in range(len(m)):
                events1[i].wait()
                events1[i].clear()
            for i in range(len(m)):
                tmp = q.get()
                shared_pos[tmp[0] * dimension:(tmp[0] + 1) * dimension] = tmp[1]
            for i in range(len(m)):
                events2[i].set()
        else:
            events2[body].wait()
            events2[body].clear()

        my_cur_pos = np.array(shared_pos[body * dimension:(body + 1) * dimension])
        next_acceleration = acceleration_no_np(body, m, shared_pos)

        result[j, dimension:] = \
            (result[j - 1, dimension:]
             + 0.5 * (cur_acceleration + next_acceleration) * delta_t)
        cur_acceleration = next_acceleration
        # print(body, "not wait", j)
    q_out.put([body, result])


def verlet_nBody_multiprocessing2(m,ip,iv,delta_t,n):
    number_of_cpus = mp.cpu_count()



    G = 6.67408 * (10 ** -11);


    times = np.arange(n) * delta_t
    numberOfPlanets = ip.size

    coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    coordinatesAndVel[0] = np.concatenate((ip, iv))
    #


    # if __name__ == '__main__':
    times = np.arange(n) * delta_t

    coordinatesAndVel = np.empty((times.size, ip.size * 2))
    shared_pos = mp.Array('d', ip)
    n_body_for1procc = int(np.ceil((len(m)) / number_of_cpus))

    events1 = []
    events2 = []
    for body in m:
        events1.append(mp.Event())
        events2.append(mp.Event())
        events1[-1].clear()
        events2[-1].clear()

    q = mp.Queue()
    q_out = mp.Queue()
    processes = []
    for body in range(m.size):
        processes.append(
        mp.Process(target=solveForOneBodyM2, args=(q, q_out, iv, shared_pos, body, events1, events2,m,delta_t,n)))
        processes[-1].start()

    dim = ip.size // len(m)
    for i in range(len(m)):
        tmp = q_out.get()
        coordinatesAndVel[:, tmp[0] * dim:(tmp[0] + 1) * dim] = tmp[1][:, :dim]
        coordinatesAndVel[:, ip.size + tmp[0] * dim: ip.size + (tmp[0] + 1) * dim] = tmp[1][:, dim:]

    for process in processes:
        process.join()


    return

def acceleration(body, masses, positions):
    G = 6.67408 * (10 ** -11);
    dimension = positions.size // masses.size

    res = np.zeros(dimension)
    for j in range(masses.size):
        if body != j:
            displacement = positions[j*dimension:(j+1) * dimension] - positions[body*dimension:(body+1)*dimension]
            res += G*masses[j] * displacement / np.linalg.norm(displacement, 2)**3
    return res



#sync between threads
def Sync(events, controlevent, loopevent, isfreeevent):
    N = len(events)
    while(1):
        loopevent.wait()
        if(isfreeevent.is_set()):
            break
        for i in range(N):
            events[i].wait()
            events[i].clear()
            # print("Event number: ", i)
        controlevent.set()
        loopevent.clear()


def verlet_nBody_treading(m,ip,iv,delta_t,n):
    # print("Treading function ")
    G = 6.67408 * (10 ** -11);
    times = np.arange(n) * delta_t
    numberOfPlanets = ip.size
    coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    coordinatesAndVel[0] = np.concatenate((ip, iv))
    dt=delta_t
    dimension = len(ip) // len(m)


    def solveForOneBody(body, event, controlevent, loopthread):
        cur_acceleration = acceleration(body, m, coordinatesAndVel[0, :numberOfPlanets])
        for j in np.arange(n - 1) + 1:
            event.set()
            loopthread.set()
            controlevent.wait()
            controlevent.clear()
            coordinatesAndVel[j, body*dimension:(body+1)*dimension] = \
                (coordinatesAndVel[j-1, body*dimension:(body+1)*dimension]
                 + coordinatesAndVel[j-1, numberOfPlanets+body*dimension:numberOfPlanets+(body+1)*dimension] * dt
                 + 0.5 * cur_acceleration * dt**2)

            event.set()
            loopthread.set()
            controlevent.wait()
            controlevent.clear()

            next_acceleration = acceleration(body, m, coordinatesAndVel[j, :numberOfPlanets])
            coordinatesAndVel[j, numberOfPlanets + body*dimension:numberOfPlanets + (body+1)*dimension] = \
                (coordinatesAndVel[j-1, numberOfPlanets + body*dimension:numberOfPlanets + (body+1)*dimension]
                 + 0.5 * (cur_acceleration + next_acceleration) * dt)
            cur_acceleration = next_acceleration
            event.set()
            loopthread.set()
            controlevent.wait()
            controlevent.clear()
        return

    events = []
    for body in m:
        events.append(threading.Event())
        events[-1].clear()
    events[0].set()

    isfreeevent = threading.Event()
    controlevent = threading.Event()
    loopevent = threading.Event()
    controlthread = threading.Thread(target=Sync, args=(events, controlevent, loopevent, isfreeevent))
    controlthread.start()

    t=[]
    for i in range(m.size):
        t.append(threading.Thread(target=solveForOneBody, args=(i,events[i], controlevent, loopevent)))
        t[i].start()
    for i in range(m.size):
        t[i].join()
    loopevent.set()
    isfreeevent.set()
    controlthread.join()

    # print(coordinatesAndVel)
    return coordinatesAndVel



def verlet_nBody_cython_noV_noO(m,ip,iv,delta_t,n):
    # print("cython function ")
    import MyLittlePython.verletCyton_noV_noO_par as vcp
    coordinatesAndVel = vcp.verlet_nBody(m, ip, iv, delta_t, n)
    # print(coordinatesAndVel)
    return coordinatesAndVel

def verlet_nBody_cython_withV_noO(m,ip,iv,delta_t,n):
    # print("cython function ")
    import MyLittlePython.verletCyton_withV_noO_par as vcpv
    coordinatesAndVel = vcpv.verlet_nBody(m, ip, iv, delta_t, n)
    # print(coordinatesAndVel)
    return coordinatesAndVel

def verlet_nBody_cython_withV_withO(m,ip,iv,delta_t,n):
    # print("cython function ")
    import MyLittlePython.verletCyton_withV_withO_par as vcpvo
    coordinatesAndVel = vcpvo.verlet_nBody(m, ip, iv, delta_t, n)
    # print(coordinatesAndVel)
    return coordinatesAndVel

def verlet_nBody_cython_noV_withO(m,ip,iv,delta_t,n):
    # print("cython function ")
    import MyLittlePython.verletCyton_noV_withO_par as vcpo
    coordinatesAndVel = vcpo.verlet_nBody(m, ip, iv, delta_t, n)
    # print(coordinatesAndVel)
    return coordinatesAndVel


def verlet_nBody_openCL(m,ip,iv,delta_t,n):
    # print("cython function ")
    import openCL_cpu as vcl
    # coordinatesAndVel=vcl.SolveNBodiesVerletOpenCL(m, ip, iv, delta_t, n)
    coordinatesAndVel = vcl.nBodiesVerlet_OpenCL(m, ip, iv, delta_t, n)
    # print(coordinatesAndVel)
    return coordinatesAndVel


def get_many_bodies(n, dx=1e12):
    def get_newBody(cen):
        x = random.uniform(-2*dx/3, 2*dx/3)
        y = random.uniform(-2*dx/3, 2*dx/3)
        u = random.uniform(-dx**0.3, dx**0.3)
        v = random.uniform(-dx**0.3, dx**0.3)
        m = random.uniform(1, 1e5) * 1e22
        return [m, cen[0] + x, cen[1] + y, u, v]

    random.seed()
    cur_centre = np.zeros(2)
    newBodies = [get_newBody(cur_centre)]
    k = 1
    for i in range(n // 2):
        cur_centre[0] += k * dx * (-1)**k
        newBodies.append(get_newBody(cur_centre))
        cur_centre[1] += k * dx * (-1)**k
        newBodies.append(get_newBody(cur_centre))
        k += 1

    # Format for computation
    masses = []
    coord = []
    vel = []
    for p in newBodies:
        masses += p[:1]
        coord += p[1:3]
        vel += p[3:]

    return np.array(masses), np.array(coord), np.array(vel), 10 * abs((max(coord) - min(coord)) / max(vel))

def time_of_verlet_pure(n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        verlet_nBody(masses, init_pos, init_vel, dt, iterations)
        t += time.time() - t0
    return t / n_launches

def time_of_verlet_multiproc(n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        # verlet_nBody_multiprocessing2(masses, init_pos, init_vel, dt, iterations)
        mcProc(masses, init_pos, init_vel, dt, iterations)
        t += time.time() - t0
    return t / n_launches

def time_of_verlet_treading(n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        verlet_nBody_treading(masses, init_pos, init_vel, dt, iterations)
        t += time.time() - t0
    return t / n_launches

def time_of_cython_noV_noO(n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        verlet_nBody_cython_noV_noO(masses, init_pos, init_vel, dt, iterations)
        t += time.time() - t0
    return t / n_launches

def time_of_cython_withV_noO(n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        verlet_nBody_cython_withV_noO(masses, init_pos, init_vel, dt, iterations)
        t += time.time() - t0
    return t / n_launches

def time_of_cython_withV_withO(n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        verlet_nBody_cython_withV_withO(masses, init_pos, init_vel, dt, iterations)
        t += time.time() - t0
    return t / n_launches

def time_of_cython_noV_withO(n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        verlet_nBody_cython_noV_withO(masses, init_pos, init_vel, dt, iterations)
        t += time.time() - t0
    return t / n_launches

def time_of_openCL(n_launches, masses, init_pos, init_vel, dt, iterations):
    t = 0
    for i in range(n_launches):
        t0 = time.time()
        res,time_loc=verlet_nBody_openCL(masses, init_pos, init_vel, dt, iterations)
        print("local_time =",time_loc)
        t += time.time() - t0
    return t / n_launches

if __name__ == '__main__':
    m = np.array([ 1.98892e30, 7.34767309e22,5.972e24])# the sun  the moon the earth
    ip = np.array([0, 0,  (1.496e11 + 3.84467e8), 0, 1.496e11, 0,])
    iv = np.array([0, 0,   0,(2.9783e4 + 1022),        0, 2.9783e4])
    delta_t = 60 * 60 * 24
    n = 365 * 1
    # print(verlet_nBody(m,ip,iv,delta_t,n))

    #pure verlet
    masess, iPosit, iVel, t = get_many_bodies(50)
    numberOflaunches=1
    numberOf_iter=50
    # # print(masess)
    # # for i in range(0, len(masess)):
    # #     print(i)
    # # print(len(masess))
    print("time_of_verlet_pure=",time_of_verlet_pure(numberOflaunches, masess, iPosit, iVel, t / 500, numberOf_iter))
    # #treading
    # print("time_of_verlet_treading=", time_of_verlet_treading(numberOflaunches, masess, iPosit, iVel, t / 500, numberOf_iter))
    # #multiprocessing
    # print("time_of_verlet_multiproc=",time_of_verlet_multiproc(numberOflaunches, masess, iPosit, iVel, t / 500, numberOf_iter))
    # #cython
    # print("time_of_cython_noV_noO=", time_of_cython_noV_noO(numberOflaunches, masess, iPosit, iVel, t / 500, numberOf_iter))
    # #cython type
    # print("time_of_cython_withV_noO=", time_of_cython_withV_noO(numberOflaunches, masess, iPosit, iVel, t / 500, numberOf_iter))
    # #cython all
    # print("time_of_cython_withV_withO=", time_of_cython_withV_withO(numberOflaunches, masess, iPosit, iVel, t / 500, numberOf_iter))
    # # cython op
    # print("time_of_cython_noV_withO=", time_of_cython_noV_withO(numberOflaunches, masess, iPosit, iVel, t / 500, numberOf_iter))
    # # openCL
    # print("time_of_openCL=", time_of_openCL(numberOflaunches, masess, iPosit, iVel, t / 500, numberOf_iter))

    k_body=[10, 50, 100, 200, 500, 1000]
    time_openCL_local=[0.031225919723510742, 0.031228065490722656, 0.06247591972351074, 0.1093745231628418, 0.5885176658630371, 2.717256784439087]
    time_openCL_global = [3.9942848682403564, 4.13433051109314, 4.3957109451293945, 4.511442184448242, 4.658452033996582, 6.786329746246338]
    time_pure_verlet = [0.17708563804626465,1.5018691221872966, 5.898155212402344, 22.163081169128418, 138.34115624427795, 579.1826295852661]
    time_cython_noV_noO=[0.12554200490315756, 1.6305176417032878, 5.661789973576863, 21.30510155359904, 132.315310160319, 537.3494836489359]
    time_cython_withV_noO = [0.03645896911621094, 0.19271111488342285, 0.7246779600779215, 2.9870493412017822, 19.268741528193157, 78.83192118008931]
    time_cython_withV_withO = [0.010416189829508463, 0.010417540868123373, 0.04687619209289551, 0.20946931838989258,  1.218138853708903,  4.874408006668091]
    time_cython_noV_withO = [0.010416984558105469, 0.01563103993733724, 0.046881675720214844, 0.2024364471435547, 1.2184317111968994, 4.894291655222575]
    time_treading=[0.18380975723266602]
    time_multiproc=[0.9764139652252197, 1.33968186378479, 2.188474655151367, 4.140703916549683, 19.31244921684265, 66.844473361969]


    import matplotlib.pyplot as plt
    plt.plot(k_body, time_openCL_local, color='b', label="time_openCL_local")
    plt.plot(k_body, time_openCL_global, color='g', label="time_openCL_global")
    plt.plot(k_body, time_cython_noV_noO, color='r', label="time_cython_noV_noO")
    plt.plot(k_body, time_cython_withV_noO, color='k', label="time_cython_withV_noO")
    plt.plot(k_body, time_cython_withV_withO, color='y', label="time_cython_withV_withO")
    plt.plot(k_body, time_cython_noV_withO, color='m', label="time_cython_noV_withO")
    plt.plot(k_body, time_pure_verlet, color='b', label="time_pure_verlet", linestyle='--')
    plt.plot(k_body, time_multiproc, color='m', label="time_multiproc", linestyle='--')
    plt.xlabel('k_body')
    plt.ylabel("Time")
    plt.xlim([0, 1200])
    plt.grid(True)
    # plt.ylim([ymin, x_or_y_max])
    plt.legend(loc=2)
    plt.show()

    # --------------------------------------------------
    # ускорение относитеьно верле
    time_openCL_local=  np.array(time_pure_verlet)/np.array(time_openCL_local)
    time_openCL_global =  np.array(time_pure_verlet)/np.array(time_openCL_global)
    time_cython_noV_noO =  np.array(time_pure_verlet)/np.array(time_cython_noV_noO)
    time_cython_withV_noO =  np.array(time_pure_verlet)/np.array(time_cython_withV_noO)
    time_cython_withV_withO =  np.array(time_pure_verlet)/np.array(time_cython_withV_withO)
    time_cython_noV_withO =  np.array(time_pure_verlet)/np.array(time_cython_noV_withO)
    time_multiproc =  np.array(time_pure_verlet)/np.array(time_multiproc)




    plt.plot(k_body, time_openCL_local, color='b', label="time_openCL_local")
    plt.plot(k_body, time_openCL_global, color='g', label="time_openCL_global")
    plt.plot(k_body, time_cython_noV_noO, color='r', label="time_cython_noV_noO")
    plt.plot(k_body, time_cython_withV_noO, color='k', label="time_cython_withV_noO")
    plt.plot(k_body, time_cython_withV_withO, color='y', label="time_cython_withV_withO")
    plt.plot(k_body, time_cython_noV_withO, color='m', label="time_cython_noV_withO")
    # plt.plot(k_body, time_pure_verlet, color='b', label="time_pure_verlet", linestyle='--')
    plt.plot(k_body, time_multiproc, color='m', label="time_multiproc", linestyle='--')
    plt.xlabel('k_body')
    plt.ylabel("Time")
    plt.xlim([0, 1200])
    plt.grid(True)
    # plt.ylim([ymin, x_or_y_max])
    plt.legend(loc=2)
    plt.show()
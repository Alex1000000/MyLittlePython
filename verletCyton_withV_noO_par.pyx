import numpy as np

cpdef double[:] acc_get(double[:] m,double[:] coordOfplanets):
    cdef double G = 6.67408e-11
    cdef int numbOfPlanets = <int>coordOfplanets.size / <int>m.size
    # acc = np.zeros(coordOfplanets.size)
    cdef double[:] acc = np.zeros(coordOfplanets.size)
    cdef double[:] tmp = np.zeros(numbOfPlanets)
    for i in range(m.size):
        for j in range(m.size):
            if i != j:
                norm = 0
                for d in range(numbOfPlanets):
                    diff = coordOfplanets[j*numbOfPlanets + d] - coordOfplanets[i*numbOfPlanets + d]
                    norm += diff * diff
                    tmp[d] = G * m[j] * diff
                norm = pow(norm, 1.5)
                for d in range(numbOfPlanets):
                    acc[i*numbOfPlanets + d] += tmp[d] / norm
    return acc



cpdef verlet_nBody(double[:] m,double[:] ip,double[:] iv,double delta_t,int n):
    # Plots the trajectories of 3 equal masses
    cdef double G = 6.67408e-11
    # times = np.arange(n) * delta_t
    cdef double[:] times = np.arange(n) * delta_t
    # numberOfPlanets = ip.size
    cdef int numberOfPlanets = ip.size
    # coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    cdef double[:,:] pos = np.empty((2, numberOfPlanets))
    cdef double[:,:] vel = np.empty((2, numberOfPlanets))
    # coordinatesAndVel[0] = np.concatenate((ip, iv))
    # acc = acc_get(m, coordinatesAndVel[0, : numberOfPlanets])
    cdef int j, d, k
    cdef int dimension = ip.shape[0] / m.shape[0]
    # print("ip.shape[0]=", ip.shape[0])
    # print("m.shape[0]=", m.shape[0])
    # print("dimention=", dimension)
    cdef int n_bodies = m.shape[0]
    pos[0] = ip
    vel[0] = iv
    cdef double[:] cur_accelerations = acc_get(m, pos[0])
    cdef double[:] next_accelerations


    coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    coordinatesAndVel[0] = np.concatenate((ip, iv))
    acc = acc_get(m, coordinatesAndVel[0, : numberOfPlanets])

    res=np.concatenate((pos, vel), 1)
    for j in range(n - 1):
        for d in range(dimension):
            for k in range(n_bodies):
                pos[(j+1) % 2, k*dimension + d] = \
                    pos[j % 2, k*dimension + d]\
                    + vel[j % 2, k*dimension + d] * delta_t + 0.5 * cur_accelerations[k*dimension + d] * delta_t * delta_t
        next_accelerations = acc_get(m, pos[(j+1) % 2])
        for d in range(dimension):
            for k in range(n_bodies):
                vel[(j+1) % 2, k*dimension + d] = \
                    vel[j % 2, k*dimension + d] \
                    + 0.5 * (cur_accelerations[k*dimension + d] + next_accelerations[k*dimension + d]) * delta_t
        cur_accelerations = next_accelerations
        # print(np.array(pos))
        # print(np.concatenate((pos, vel), 1))
        res=np.concatenate((res, pos, vel), 1)
        # res=np.concatenate((pos, vel), 1)

    return  coordinatesAndVel


#
# cpdef verlet_nBody(double[:] m,ip,iv,double delta_t,int n):
#     # Plots the trajectories of 3 equal masses
#     cdef double G = 6.67408e-11
#     times = np.arange(n) * delta_t
#     numberOfPlanets = ip.size
#     coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
#     coordinatesAndVel[0] = np.concatenate((ip, iv))
#     acc = acc_get(m, coordinatesAndVel[0, : numberOfPlanets])
#     for j in np.arange(n - 1) + 1:
#         #r
#         coordinatesAndVel[j, :numberOfPlanets] = coordinatesAndVel[j - 1, :numberOfPlanets] + coordinatesAndVel[j - 1, numberOfPlanets:] * delta_t + 0.5 * acc * delta_t ** 2
#         acc_next = acc_get(m, coordinatesAndVel[j, :numberOfPlanets])
#         #vel
#         coordinatesAndVel[j, numberOfPlanets:] = coordinatesAndVel[j - 1, numberOfPlanets:] + 0.5 * (acc + acc_next) * delta_t
#         acc = acc_next
#     # tspan = sp.linspace(0, T, n + 1)
#     return  coordinatesAndVel





# cpdef verlet_nBody(m,ip,iv,delta_t,n):
#     # Plots the trajectories of 3 equal masses
#     cdef double G = 6.67408e-11
#     times = np.arange(n) * delta_t
#     numberOfPlanets = ip.size
#     coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
#     coordinatesAndVel[0] = np.concatenate((ip, iv))
#     acc = acc_get(m, coordinatesAndVel[0, : numberOfPlanets])
#     for j in np.arange(n - 1) + 1:
#         #r
#         coordinatesAndVel[j, :numberOfPlanets] = coordinatesAndVel[j - 1, :numberOfPlanets] + coordinatesAndVel[j - 1, numberOfPlanets:] * delta_t + 0.5 * acc * delta_t ** 2
#         acc_next = acc_get(m, coordinatesAndVel[j, :numberOfPlanets])
#         #vel
#         coordinatesAndVel[j, numberOfPlanets:] = coordinatesAndVel[j - 1, numberOfPlanets:] + 0.5 * (acc + acc_next) * delta_t
#         acc = acc_next
#     # tspan = sp.linspace(0, T, n + 1)
#     return  coordinatesAndVel



    # # Plots the trajectories of 3 equal masses
    # cdef double G = 6.67408e-11
    # # times = np.arange(n) * delta_t
    # cdef double[:] times = np.arange(n) * delta_t
    # # numberOfPlanets = ip.size
    # cdef int numberOfPlanets = ip.size
    # # coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    # cdef double[:,:] pos = np.empty((2, numberOfPlanets))
    # cdef double[:,:] vel = np.empty((2, numberOfPlanets))
    # # coordinatesAndVel[0] = np.concatenate((ip, iv))
    # # acc = acc_get(m, coordinatesAndVel[0, : numberOfPlanets])
    # cdef int j, d, k
    # cdef int dimension = ip.shape[0] / m.shape[0]
    # cdef int n_bodies = m.shape[0]
    # pos[0] = ip
    # vel[0] = iv
    # cdef double[:] cur_accelerations = acc_get(m, pos[0])
    # cdef double[:] next_accelerations
    #
    #
    # for j in range(n - 1):
    #     for d in range(dimension):
    #         for k in range(n_bodies):
    #             pos[(j+1) % 2, k*dimension + d] = \
    #                 pos[j % 2, k*dimension + d]\
    #                 + vel[j % 2, k*dimension + d] * delta_t + 0.5 * cur_accelerations[k*dimension + d] * delta_t * delta_t
    #     next_accelerations = acc_get(m, pos[(j+1) % 2])
    #     for d in range(dimension):
    #         for k in range(n_bodies):
    #             vel[(j+1) % 2, k*dimension + d] = \
    #                 vel[j % 2, k*dimension + d] \
    #                 + 0.5 * (cur_accelerations[k*dimension + d] + next_accelerations[k*dimension + d]) * delta_t
    #     cur_accelerations = next_accelerations
    #
    # # for j in np.arange(n - 1) + 1:
    # #     #r
    # #     coordinatesAndVel[j, :numberOfPlanets] = coordinatesAndVel[j - 1, :numberOfPlanets] + coordinatesAndVel[j - 1, numberOfPlanets:] * delta_t + 0.5 * acc * delta_t ** 2
    # #     acc_next = acc_get(m, coordinatesAndVel[j, :numberOfPlanets])
    # #     #vel
    # #     coordinatesAndVel[j, numberOfPlanets:] = coordinatesAndVel[j - 1, numberOfPlanets:] + 0.5 * (acc + acc_next) * delta_t
    # #     acc = acc_next
    # # # tspan = sp.linspace(0, T, n + 1)
    # # return  coordinatesAndVel
    # # return np.concatenate((pos, vel), 1)
    # return np.array(pos)

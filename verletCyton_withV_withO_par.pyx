import numpy as np
from cython.parallel cimport parallel, prange
import cython
from libc.math cimport sqrt, pow

# cpdef double[:] acc_get(double[:] m,double[:] coordOfplanets):
#     cdef double G = 6.67408e-11
#     cdef int numbOfPlanets = <int>coordOfplanets.size / <int>m.size
#     # acc = np.zeros(coordOfplanets.size)
#     cdef double[:] acc = np.zeros(coordOfplanets.size)
#     cdef double[:] tmp = np.zeros(numbOfPlanets)
#     cdef int m_size=m.size
#     cdef int i, j, d
#     cdef double diff
#     cdef norm
#     # for i in range(m.size):
#     for i in prange(m_size, nogil=True, schedule='static'):
#         for j in range(m.size):
#             if i != j:
#                 norm = 0
#                 for d in range(numbOfPlanets):
#                     diff = coordOfplanets[j*numbOfPlanets + d] - coordOfplanets[i*numbOfPlanets + d]
#                     norm += diff * diff
#                     tmp[d] = G * m[j] * diff
#                 norm = pow(norm, 1.5)
#                 for d in range(numbOfPlanets):
#                     acc[i*numbOfPlanets + d] += tmp[d] / norm
#     return acc


@cython.boundscheck(False)
cdef double[:] acc_get(double[:] m,double[:] coordOfplanets):
    cdef double G = 6.67408e-11
    cdef:
        int n_bodies = m.size
        int dimension = <int>coordOfplanets.size / <int>m.size
        double norm3
        double[:] res = np.zeros(coordOfplanets.size)
        double[:, :, :] tmp = np.zeros((n_bodies, n_bodies, dimension))
        double[:, :] norm = np.zeros((n_bodies, n_bodies))
        # np.ndarray[double, ndim=1] res = np.zeros(coordOfplanets.size)
        # np.ndarray[double, ndim=3] tmp = np.zeros((n_bodies, n_bodies, dimension))
        # np.ndarray[double, ndim=2] norm = np.zeros((n_bodies, n_bodies))
        double dx
        int i, j, d

    for i in prange(n_bodies, nogil=True, schedule='static'):
        for j in range(n_bodies):
            if i != j:
                for d in range(dimension):
                    dx = coordOfplanets[j*dimension + d] - coordOfplanets[i*dimension + d]
                    norm[i, j] += dx * dx
                    tmp[i, j, d] += G*m[j] * dx
                norm3 = pow(norm[i, j], 1.5)
                for d in range(dimension):
                    res[i*dimension + d] += tmp[i, j, d] / norm3
    return res


@cython.boundscheck(False)
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
            # for k in range(n_bodies):
            for k in prange(n_bodies, nogil=True, schedule='static'):
                pos[(j+1) % 2, k*dimension + d] = \
                    pos[j % 2, k*dimension + d]\
                    + vel[j % 2, k*dimension + d] * delta_t + 0.5 * cur_accelerations[k*dimension + d] * delta_t * delta_t
        next_accelerations = acc_get(m, pos[(j+1) % 2])
        for d in range(dimension):
            # for k in range(n_bodies):
            for k in prange(n_bodies, nogil=True, schedule='static'):
                vel[(j+1) % 2, k*dimension + d] = \
                    vel[j % 2, k*dimension + d] \
                    + 0.5 * (cur_accelerations[k*dimension + d] + next_accelerations[k*dimension + d]) * delta_t
        cur_accelerations = next_accelerations
        # print(np.array(pos))
        # print(np.concatenate((pos, vel), 1))
        res=np.concatenate((res, pos, vel), 1)
        # res=np.concatenate((pos, vel), 1)

    return  coordinatesAndVel
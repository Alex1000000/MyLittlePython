import numpy as np
from cython.parallel cimport parallel, prange
import cython
cimport numpy as np
from numpy cimport ndarray as ar
from libc.math cimport sqrt, pow




@cython.boundscheck(False)
# @cython.wraparound(False)
cdef np.ndarray[double, ndim=1] acc_get(np.ndarray[double, ndim=1] masses,
                                                      np.ndarray[double, ndim=1] positions):
    cdef double G = 6.67408e-11
    cdef:
        int n_bodies = masses.size
        int dimension = <int>positions.size / <int>masses.size
        double norm3
        np.ndarray[double, ndim=1] res = np.zeros(positions.size)
        np.ndarray[double, ndim=3] tmp = np.zeros((n_bodies, n_bodies, dimension))
        np.ndarray[double, ndim=2] norm = np.zeros((n_bodies, n_bodies))
        double dx
        int i, j, d

    for i in prange(n_bodies, nogil=True, schedule='static'):
        for j in range(n_bodies):
            if i != j:
                for d in range(dimension):
                    dx = positions[j*dimension + d] - positions[i*dimension + d]
                    norm[i, j] += dx * dx
                    tmp[i, j, d] += G*masses[j] * dx
                norm3 = pow(norm[i, j], 1.5)
                for d in range(dimension):
                    res[i*dimension + d] += tmp[i, j, d] / norm3
    return res





# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[double, ndim=1] acc_get(ar[double, ndim=1] m,ar[double, ndim=1] coordOfplanets):
#     cdef double G = 6.67408e-11
#     cdef int numbOfPlanets = <int>coordOfplanets.size / <int>m.size
#     cdef m_size=m.size
#     # acc = np.zeros(coordOfplanets.size)
#     cdef ar[double, ndim=1] acc = np.zeros(coordOfplanets.size)
#     # cdef np.ndarray[double, ndim=1] tmp = np.zeros(numbOfPlanets)
#     cdef ar[double, ndim=3] tmp = np.zeros((m_size, m_size, numbOfPlanets))
#     cdef ar[double, ndim=2] norm = np.zeros((m_size, m_size))
#     cdef i,j, d
#     cdef double norm3, dx
#
#     for i in prange(m_size, nogil=True, schedule='static'):
#         for j in range(m_size):
#             if i != j:
#                 for d in range(numbOfPlanets):
#                     dx = coordOfplanets[j*numbOfPlanets + d] - coordOfplanets[i*numbOfPlanets + d]
#                     norm[i, j] += dx * dx
#                     tmp[i, j, d] += G*m[j] * dx
#                 norm3 = pow(norm[i, j], 1.5)
#                 for d in range(numbOfPlanets):
#                     acc[i*numbOfPlanets + d] += tmp[i, j, d] / norm3
#
#     #
#     # # for i in range(m.size):
#     # for i in prange(m_size, nogil=True, schedule='static'):
#     # # with nogil, parallel(num_threads=m_size):
#     # #     for i in prange(m_size, schedule='dynamic'):
#     #     for j in range(m_size):
#     #         if i != j:
#     #             norm = 0
#     #             for d in range(numbOfPlanets):
#     #                 diff = coordOfplanets[j*numbOfPlanets + d] - coordOfplanets[i*numbOfPlanets + d]
#     #                 norm += diff * diff
#     #                 tmp[d] = G * m[j] * diff
#     #             norm = pow(norm, 1.5)
#     #             for d in range(numbOfPlanets):
#     #                 acc[i*numbOfPlanets + d] += tmp[d] / norm
#     return acc


@cython.boundscheck(False)
# @cython.wraparound(False)
cpdef  verlet_nBody(np.ndarray[double, ndim=1] m,np.ndarray[double, ndim=1] ip,np.ndarray[double, ndim=1] iv,double delta_t,int n):
    # Plots the trajectories of 3 equal masses
    cdef double G = 6.67408e-11
    # times = np.arange(n) * delta_t
    cdef np.ndarray[double, ndim=1] times = np.arange(n) * delta_t
    # numberOfPlanets = ip.size
    cdef int numberOfPlanets = ip.size
    # coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    cdef np.ndarray[double, ndim=2] pos = np.empty((2, numberOfPlanets))
    cdef np.ndarray[double, ndim=2] vel = np.empty((2, numberOfPlanets))


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
    # cdef np.ndarray[double, ndim=1] cur_accelerations = acc_get(m, pos[0])
    # cdef np.ndarray[double, ndim=1] next_accelerations
    #
    #
    # coordinatesAndVel = np.empty((times.size, numberOfPlanets * 2))
    # coordinatesAndVel[0] = np.concatenate((ip, iv))
    # acc = acc_get(m, coordinatesAndVel[0, : numberOfPlanets])
    #
    # res=np.concatenate((pos, vel), 1)
    # for j in range(n - 1):
    #     for d in range(dimension):
    #         for k in range(n_bodies):
    #         # for k in prange(n_bodies, nogil=True, schedule='static'):
    #             pos[(j+1) % 2, k*dimension + d] = \
    #                 pos[j % 2, k*dimension + d]\
    #                 + vel[j % 2, k*dimension + d] * delta_t + 0.5 * cur_accelerations[k*dimension + d] * delta_t * delta_t
    #     next_accelerations = acc_get(m, pos[(j+1) % 2])
    #     for d in range(dimension):
    #         # for k in range(n_bodies):
    #         for k in prange(n_bodies, nogil=True, schedule='static'):
    #             vel[(j+1) % 2, k*dimension + d] = \
    #                 vel[j % 2, k*dimension + d] \
    #                 + 0.5 * (cur_accelerations[k*dimension + d] + next_accelerations[k*dimension + d]) * delta_t
    #     cur_accelerations = next_accelerations
    #     # print(np.array(pos))
    #     # print(np.concatenate((pos, vel), 1))
    #     res=np.concatenate((res, pos, vel), 1)
    #     # res=np.concatenate((pos, vel), 1)
    #


    cdef np.ndarray[double, ndim=1] cur_accelerations = acc_get(m, pos[0])
    cdef np.ndarray[double, ndim=1] next_accelerations

    # res=np.array()
    res=np.concatenate((pos, vel), 1)

    for j in range(n - 1):
        # for n in prange(n_bodies, nogil=True, schedule='static'):
        #     for d in range(dimension):
        #         pos[j+1, n*dimension + d] = \
        #             pos[j, n*dimension + d]\
        #             + vel[j, n*dimension + d] * dt + 0.5 * cur_accelerations[n*dimension + d] * dt * dt
        pos[(j+1) % 2] = pos[j % 2] + vel[j % 2] * delta_t + 0.5 * cur_accelerations * delta_t * delta_t
        next_accelerations = acc_get(m, pos[(j+1) % 2])
        vel[(j+1) % 2] = vel[j % 2] + 0.5 * (cur_accelerations + next_accelerations) * delta_t
        # for n in prange(n_bodies, nogil=True, schedule='static'):
        #     for d in range(dimension):
        #         vel[j+1, n*dimension + d] = \
        #             vel[j, n*dimension + d] \
        #             + 0.5 * (cur_accelerations[n*dimension + d] + next_accelerations[n*dimension + d]) * dt
        cur_accelerations = next_accelerations
        # print(pos)
        if (j!=0):
            res=np.concatenate((res, pos, vel), 1)

    return res
    # return np.concatenate((pos, vel), 1), times
    # return  coordinatesAndVel


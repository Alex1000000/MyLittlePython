import numpy as np

cpdef acc_get(m, coordOfplanets):
    cdef double G = 6.67408e-11
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

cpdef verlet_nBody(m,ip,iv,delta_t,n):
    # Plots the trajectories of 3 equal masses
    cdef double G = 6.67408e-11
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


import numpy as np
import pyopencl as cl
from operator import attrgetter
from typing import List, Iterator


def get_default_device(use_gpu: bool = True) -> cl.Device:
    """
    Retrieves the GPU device with the most global memory if available, otherwise returns the CPU.
    :param use_gpu: Determines whether to obtain a GPU or CPU device
    """
    platforms = cl.get_platforms()
    gpu_devices = [plat.get_devices(cl.device_type.GPU) for plat in platforms]
    gpu_devices = [dev for devices in gpu_devices for dev in devices]  # Flatten to 1d if multiple GPU devices exists
    use_gpu = False
    if gpu_devices and use_gpu:
        dev = max(gpu_devices, key=attrgetter('global_mem_size'))
        print('Using GPU: {}'.format(dev.name))
        print('On platform: {} ({})\n'.format(dev.platform.name, dev.platform.version.strip()))
        return dev
    else:
        cpu_devices = [plat.get_devices(cl.device_type.CPU) for plat in platforms]
        cpu_devices = [dev for devices in cpu_devices for dev in devices]
        if cpu_devices:
            dev = max(cpu_devices, key=attrgetter('global_mem_size'))
            # print('Using CPU: {}'.format(dev.name))
            # print('On platform: {} ({})\n'.format(dev.platform.name, dev.platform.version.strip()))
            return dev
        else:
            raise RuntimeError('No suitable OpenCL GPU/CPU devices found')


def get_devices_by_name(name: str, case_sensitive: bool = False) -> List[cl.Device]:
    """
    Searches through all devices looking for a partial match for 'name' among the available devices.
    :param name: The string to search for
    :param case_sensitive: If false, different case is ignored when searching
    :return: A list of all devices that is a partial match for the specified name
    """
    if not name:
        raise RuntimeError('Device name must be specified')

    platforms = cl.get_platforms()
    devices = [plat.get_devices(cl.device_type.ALL) for plat in platforms]
    devices = [dev for devices in devices for dev in devices]

    if case_sensitive:
        name_matches = [dev for dev in devices if name in dev.name]
    else:
        name_matches = [dev for dev in devices if name.lower() in dev.name.lower()]

    return name_matches


def range_bitwise_shift(low: int, high: int, n: int) -> Iterator[int]:
    """
    Generates an upwards or downwards range through successive bitshifts according to n.
    :param low: the lower part of the range (non-inclusive)
    :param high: the higher part of the range (non-inclusive)
    :param n: the number of times to perform the bitshift, can be negative
    :return: a generator of the bitshifted range
    """
    if not n:
        raise ValueError('n cannot be zero or None')

    if low > high:
        raise ValueError('low must have a value lower than high')

    if n > 0:
        i = low
        while i < high:
            yield i
            i <<= n
    else:
        i = high
        while i > low:
            yield i
            i >>= abs(n)


def nBodiesVerlet_OpenCL(masses, init_pos, init_vel, dt, iterations):
    # ctx = cl.create_some_context()
    # queue = cl.CommandQueue(ctx)

    masses_cl = np.array(masses, dtype=np.double)
    init_pos_cl = np.array(init_pos, dtype=np.double)
    init_vel_cl = np.array(init_vel, dtype=np.double)
    dt_cl = np.array(dt, dtype=np.double)
    iterations_cl = np.array(iterations, dtype=np.int)

    pos = np.empty((2, init_pos.size), dtype=np.double)
    vel = np.empty((2, init_vel.size), dtype=np.double)
    result = np.empty((1, init_vel.size*2*iterations), dtype=np.double)
    n_bodies_cl = np.array(masses.size, dtype=np.int)
    # print("n_bodies_cl=", n_bodies_cl)
    dimension_cl = np.array(init_pos.size // masses.size, dtype=np.int)
    accelerations = np.empty((2, init_pos.size), dtype=np.double)
    tmp_dimension_arr = np.empty(init_pos.size // masses.size, dtype=np.double)

    # prg = cl.Program(ctx,
    kernel_src = """
                     #pragma OPENCL EXTENSION cl_khr_fp64: enable
                     void acceleration(__global double *positions,
                                       __global double *masses_cl, 
                                       __global double *accelerations,
                                       __global double *tmp_dimension_arr, 
                                       int n_bodies, 
                                       int dimension,
                                       int second)
                     {
                         double G = 6.67e-11;
                         /*printf(\"G constant is : %1.4e\\n\",G);*/

                         double norm, dx;
                         int shift = second * n_bodies * dimension;
                         for (int i = 0; i < n_bodies; ++i)
                         {
                             for (int d = 0; d < dimension; ++d)
                                 accelerations[shift + i*dimension + d] = 0;
                             for (int j = 0; j < n_bodies; ++j)
                                 if (i != j)
                                 {
                                    norm = 0;
                                    for (int d = 0; d < dimension; ++d)
                                    {
                                        dx = positions[shift + j*dimension + d] - positions[shift + i*dimension + d];
                                        norm += dx * dx;
                                        tmp_dimension_arr[d] = G * masses_cl[j] * dx;
                                    }
                                    norm = pow(norm, 1.5);
                                    for (int d = 0; d < dimension; ++d)
                                        accelerations[shift + i*dimension + d] += tmp_dimension_arr[d] / norm;
                                 }   
                         }
                         /*for (int i = 0; i < n_bodies; ++i)
                         {
                            printf(\"accelerations is : %1.6e\\n\",accelerations[i]);
                         } */               
                     }


                     __kernel void verlet_cl(__global double *masses_cl,
                                             __global double *init_pos_cl,
                                             __global double *init_vel_cl, 
                                             __global double *dt_cl, 
                                             __global int *iterations_cl,
                                             __global double *pos,
                                             __global double *vel,
                                             __global int *n_bodies_cl,
                                             __global int *dimension_cl,
                                             __global double *accelerations,
                                             __global double *tmp_dimension_arr,
                                             __global double *result)
                     {

                        int n_bodies = *n_bodies_cl, dimension = *dimension_cl, iterations = *iterations_cl;

                        /*printf(\"N of body is : %d\\n\",n_bodies);*/

                        double dt = *dt_cl;
                        int shift = n_bodies * dimension;
                        for (int coord = 0; coord < n_bodies * dimension; ++coord)
                        {
                            pos[coord] = init_pos_cl[coord];
                            vel[coord] = init_vel_cl[coord];
                            result[coord]=init_pos_cl[coord];
                            result[coord+n_bodies*2]=init_vel_cl[coord];
                        }
                        acceleration(pos, masses_cl, accelerations, tmp_dimension_arr, n_bodies, dimension, 0);
                        for (int t = 1; t < iterations; ++t)
                        {
                            for (int n = 0; n < n_bodies; ++n)
                                for (int d = 0; d < dimension; ++d)
                                    pos[(t%2)*shift + n*dimension + d] = pos[((t+1)%2)*shift + n*dimension + d] 
                                    + vel[((t+1)%2)*shift + n*dimension + d] * dt
                                    + 0.5 * accelerations[((t+1)%2)*shift + n*dimension + d] * dt * dt;
                            acceleration(pos, masses_cl, accelerations, tmp_dimension_arr, n_bodies, dimension, t % 2);
                            for (int n = 0; n < n_bodies; ++n)
                                for (int d = 0; d < dimension; ++d)
                                    vel[(t%2)*shift + n*dimension + d] = vel[((t+1)%2)*shift + n*dimension + d] 
                                    + 0.5 * (accelerations[n*dimension + d] + accelerations[shift + n*dimension + d]) * dt; 
                            for (int i = 0; i < n_bodies*2; ++i)
                            {
                                result[i+(n_bodies*4)*t]=pos[i+(t%2)*shift];
                                result[i+n_bodies*2+n_bodies*4*t]=vel[i+(t%2)*shift];
                            } 
                        }
                        /*printf(\"N of body2 is : %d\\n\",n_bodies);*/
                     }
                     """
    # )
    # Check for double floating point support
    dev = get_default_device()
    context = cl.Context(devices=[dev], properties=None, dev_type=None, cache_dir=None)
    queue = cl.CommandQueue(context, dev, properties=None)
    dev_extensions = dev.extensions.strip().split(' ')
    if 'cl_khr_fp64' not in dev_extensions:
        raise RuntimeError('Device does not support double precision float')
    import time
    t0 = time.time()
    # Build program in the specified context using the kernel source code
    prog = cl.Program(context, kernel_src)
    try:
        prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
        # try:
        # prg.build()

        # prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
    except:
        # print("Error:")
        # print(prog.get_build_info(context.devices[0], cl.program_build_info.LOG))
        print('Build log:')
        print(prog.get_build_info(dev, cl.program_build_info.LOG))
        raise

    mf = cl.mem_flags
    masses_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses_cl)
    init_pos_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=init_pos_cl)
    init_vel_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=init_vel_cl)
    dt_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dt_cl)
    iterations_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=iterations_cl)
    pos_buf = cl.Buffer(context, mf.WRITE_ONLY, pos.nbytes)
    vel_buf = cl.Buffer(context, mf.WRITE_ONLY, vel.nbytes)
    result_buf = cl.Buffer(context, mf.WRITE_ONLY, result.nbytes)
    n_bodies_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n_bodies_cl)
    dimension_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dimension_cl)
    accelerations_buf = cl.Buffer(context, mf.WRITE_ONLY, accelerations.nbytes)
    tmp_dimension_buf = cl.Buffer(context, mf.WRITE_ONLY, tmp_dimension_arr.nbytes)

    completeEvent = prog.verlet_cl(queue, (1,), None, masses_buf, init_pos_buf, init_vel_buf, dt_buf, iterations_buf,
                                   pos_buf, vel_buf,
                                   n_bodies_buf, dimension_buf, accelerations_buf, tmp_dimension_buf, result_buf)
    events = [completeEvent];
    completeEvent.wait()

    # cl.enqueue_read_buffer(queue, pos_buf, pos, wait_for=events).wait()
    # cl.enqueue_read_buffer(queue, vel_buf, vel, wait_for=events).wait()
    cl.enqueue_read_buffer(queue, result_buf, result, wait_for=events).wait()
    t2 = time.time() - t0
    # cl.enqueue_read_buffer(queue, pos_buf, pos, wait_for=events)
    # cl.enqueue_read_buffer(queue, vel_buf, vel, wait_for=events)
    # print("pos=",pos)
    # print("vel=", vel)
    # print("result=", result)
    return result, t2#np.concatenate((pos, vel), 1), np.arange(iterations) * dt

    #
# m = np.array([1.98892e30, 7.34767309e22, 5.972e24])  # the sun  the moon the earth
# ip = np.array([0.0, 0, 149984467000.0, 0, 149600000000.0, 0 ])
# # ip=np.array([1.0,2.0,3.0,4.0,5.0,6.0])
# iv = np.array([0, 0, 0, (2.9783e4 + 1022), 0, 2.9783e4])
# # iv = np.array([7, 8, 9, 10, 11, 12])
# delta_t = 60 * 60 * 24
# n = 10#365 * 1
#
# # print((1.496e11))
# print(nBodiesVerlet_OpenCL(m,ip,iv,delta_t,n))
# # nBodiesVerlet_OpenCL(m,ip,iv,delta_t,n)
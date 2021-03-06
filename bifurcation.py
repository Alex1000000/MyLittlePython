import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy import Symbol, solve, lambdify, Matrix

def one_p_analysis(alpha_p,k3_p):
    alpha_value=alpha_p
    ####### k3_value=k3_p
    expr = k3 / (1 - y) ** alpha
    k30_value = k3_p#########expr.subs(alpha, alpha_value).subs(k3, k3_value)
    # print k30_value
    # print eqk2.subs(k1, k1_value).subs(k_1, k_1_value).subs(k_2, k_2_value).subs(k30, k30_value).subs(alpha,
    #                                                                                                   alpha_value)

    function_k2 = lambdify(y,
                           eqk2.subs(k1, k1_value).subs(k_1, k_1_value).subs(k_2, k_2_value).subs(k30, k30_value).subs(
                               alpha, alpha_value))
    # print function_k2
    function_x_fromS = lambdify(y, x_fromSystem.subs(k1, k1_value).subs(k_1, k_1_value).subs(k30, k30_value).subs(alpha,
                                                                                                                  alpha_value))
    # print x_fromSystem.subs(k1, k1_value).subs(k_1, k_1_value).subs(k30, k30_value).subs(alpha, alpha_value)
    # print function_x_fromS

    # print detA.subs(x, x_fromSystem).subs(k2, eqk2).subs(k1, 1).subs(k_1, k_1_value).subs(k_2, k_2_value).subs(k30,
    #                                                                                                            k30_value).subs(
    #     alpha, alpha_value)

    function_detA = lambdify(y, detA.subs(x, x_fromSystem).subs(k2, eqk2).subs(k1, 1).subs(k_1, k_1_value).subs(k_2,
                                                                                                                k_2_value).subs(
        k30, k30_value).subs(alpha, alpha_value))
    function_traceA = lambdify(y, traceA.subs(x, x_fromSystem).subs(k2, eqk2).subs(k1, 1).subs(k_1, k_1_value).subs(k_2,
                                                                                                                k_2_value).subs(
        k30, k30_value).subs(alpha, alpha_value))

    detA_list = list(function_detA(y_range))
    traceA_list = list (function_traceA(y_range))
    # print detA_list
    #
    # function_forEigenVal = lambdify(y, DA.subs(x, xy).subs(k2, eqk2).subs(k1, 1).subs(k_1, 0.01).subs(k_2, 0.01).subs(k3, 10))
    #

    plt.plot(function_k2(y_range), y_range, color='b', label="$y_{k_2}$", linestyle='-')
    plt.plot(function_k2(y_range), function_x_fromS(y_range), color='g', label="$x_{k_2}$")
    # detA_list = list(function_detA(y_range))
    # detAoneparam = list(funcdetA(y_range))
    # DAarray = []
    detarray = []
    tracearray=[]
    rectification_point_trace=[]
    rectification_point_det = []
    for i in range(1, len(y_range)):
        if traceA_list[i]*traceA_list[i-1]<=0:
            #rectification
            rectification_point=y_range[i-1]-traceA_list[i-1]*(y_range[i]-y_range[i-1])/(traceA_list[i]-traceA_list[i-1])
            # print rectification_point
            rectification_point_trace.append(rectification_point)
            # print rectification_point_trace
            tracearray=tracearray+[y_range[i]]
        if detA_list[i] * detA_list[i - 1] <= 0:
            # print i, "--", y_range[i-1],"---", y_range[i], "det =",detA_list[i ] ,"===", detA_list[i-1]
            detarray = detarray + [y_range[i]]
            rectification_point_d = y_range[i-1] - detA_list[i-1] * (y_range[i ] - y_range[i-1]) / (detA_list[i] - detA_list[i-1])
            rectification_point_det.append(rectification_point_d)
    detarray = np.array(detarray)
    tracearray=np.array(tracearray)
    rectification_point_trace=np.array(rectification_point_trace)
    rectification_point_det = np.array(rectification_point_det)
    k2_b = function_k2(detarray)
    print "k2_b=", k2_b
    # print detarray
    # print rectification_point_det
    # print tracearray
    # print rectification_point_trace
    # print "detarray= ",detarray
    # print "re rect= ",rectification_point_det
    # plt.plot(function_k2(detarray), detarray, color='k', linestyle='', marker='^')
    # plt.plot(function_k2(detarray), function_x_fromS(detarray), color='k', linestyle='', marker='*')
    # plt.plot(function_k2(tracearray), tracearray, color='y', linestyle='', marker='^')
    # plt.plot(function_k2(tracearray), function_x_fromS(tracearray), color='m', linestyle='', marker='o')
    plt.plot(function_k2(rectification_point_trace), rectification_point_trace, color='r', linestyle='', marker='^')
    plt.plot(function_k2(rectification_point_trace), function_x_fromS(rectification_point_trace), color='r', linestyle='', marker='o')
    plt.plot(function_k2(rectification_point_det), rectification_point_det, color='k', linestyle='', marker='*')
    plt.plot(function_k2(rectification_point_det), function_x_fromS(rectification_point_det), color='k', linestyle='', marker='o')
    plt.title('One-parameter analysis')
    plt.xlabel('$k_2$')
    plt.ylabel("x,y")
    plt.xlim([0.0, 0.7])
    plt.grid(True)
    # # plt.ylim([ymin, x_or_y_max])
    plt.legend(loc=2)
    plt.show()
    return k2_b


def df(initial, t):
    x0 = initial[1]
    y0 = initial[0]
    k1 = 0.03
    k2 = 0.05
    # k1 = 0.2  # 0.03
    k_1 = 0.01
    # k2 = k2_value
    k_2 = 0.01
    k30 = 10.0
    alpha = 16.0
    # alpha_value = 16.0
    # k1_value = 0.03
    # k_1_value = 0.01
    # k2_value = 0.05
    # k_2_value = 0.01
    # k3_value = 10.0
    dfx = k1 * (1 - x0 - y0) - k_1 * x0 - x0 * y0 * k30 * (1 - y0) ** alpha
    dfy = k2 * (1 - x0 - y0) ** 2 - k_2 * y0 ** 2 - x0 * y0 * k30 * (1 - y0) ** alpha
    # print "k2_const=",k2
    # print "dfx=", dfx
    # print "dfy=",dfy
    return [dfy, dfx]


k1 = Symbol("k1")
k_1 = Symbol("k_1")
k2 = Symbol("k2")
k_2 = Symbol("k_2")
k3 = Symbol("k3")
k30 = Symbol("k30")
alpha = Symbol("alpha")
x = Symbol("x")
y = Symbol("y")
# eq1 = k1 * (1 - x - y)- k_1*x - k3 * x * y
# eq2 = k2 * (1 - x - y) ** 2 - k_2 * y ** 2 - k3 * x * y
eq1 = k1 * (1 - x - y)- k_1*x - x * y * k30*(1-y)**alpha
eq2 = k2 * (1 - x - y) ** 2 - k_2 * y ** 2 - x * y * k30*(1-y)**alpha
#k3= k30*(1-y)**alpha
res = solve([eq1, eq2], x, k2)
# print res
eqk2 = res[0][1]
x_fromSystem = res[0][0]
A = Matrix([eq1, eq2])
# print A
var_vector = Matrix([x, y])
jacobianA = A.jacobian(var_vector)
# print jacobianA
detA = jacobianA.det()
traceA = jacobianA.trace()
print detA
y_range = np.linspace(0, 1, 10000)
# x_range = np.linspace(0, 1, 10000)
# x_y_range = np.linspace(0, 1, 100)


alpha_value=16.0
k1_value=0.03
k_1_value=0.01
k2_value=0.05
k_2_value=0.01
k3_value=10.0
alpha_value_array = np.array([10.0,15.0,18.0,20.0,25.0])
k3_value_array = np.array([1.0,5.0,10.0,50.0,100.0])
# print alpha_value_array[0]
# alpha_value=alpha_value_array[4]
# k3_value=k3_value_array[4]
print "alpha_value=",alpha_value
print "k3_value=",k3_value
# # one_p_analysis(alpha_value_array[0],k3_value[1])
k2_points=one_p_analysis(alpha_value,k3_value)
# print "k2_points=", k2_points
# #########################################################
#
# eqprek2 = solve(detA.subs(x, x_fromSystem), k2)
# eqprek2_tr = solve(traceA.subs(x, x_fromSystem), k2)
# eqprek1 = eqprek2[0] - eqk2
# eqprek1_tr = eqprek2_tr[0] - eqk2
# eqk1 = solve(eqprek1.subs(k30, k3_value ).subs(k_1, k_1_value).subs(k_2, k_2_value).subs(alpha,alpha_value), k1)
# eqk1_tr = solve(eqprek1_tr.subs(k30, k3_value ).subs(k_1, k_1_value).subs(k_2, k_2_value).subs(alpha,alpha_value), k1)
# fk2_yk1_det = lambdify((y, k1), eqprek2[0].subs(k30, k3_value).subs(k_1, k_1_value).subs(k_2, k_2_value).subs(alpha,alpha_value), 'numpy')
# fk1_y_tr1 = lambdify(y, eqk1_tr[0], 'numpy')
# fk1_y_det2 = lambdify(y, eqk1[0], 'numpy')
# plt.plot(fk2_yk1_det(y_range, fk1_y_tr1(y_range)), fk1_y_tr1(y_range),label="neutrality line")
# plt.plot(fk2_yk1_det(y_range, fk1_y_det2(y_range)), fk1_y_det2(y_range), label="multiplicity line")
# plt.grid(True)
# plt.legend()
# axes = plt.gca()
# axes.set_xlim([0.0, 0.2])
# axes.set_ylim([0.0, 0.2])
# plt.show()

#

solution_xk2 = solve([eq1, eq2], x, k2)
x_fromSolution = solution_xk2[0][0]
k2_fromSolution = solution_xk2[0][1]

k2TraceSol = solve(traceA.subs(x,x_fromSolution),k2)[0]
k1_TraceSol = solve(k2TraceSol - k2_fromSolution,k1)[0]
k2_TraceSol = k2_fromSolution.subs(k1,k1_TraceSol)
k1Trace_y = lambdify((y,k_1,k_2,k30,alpha),k1_TraceSol,'numpy')
k2Trace_y = lambdify((y,k_1,k_2,k30,alpha),k2_TraceSol,'numpy')
plt.plot(k2Trace_y(y_range, k_1_value,k_2_value,k3_value,alpha_value), k1Trace_y(y_range, k_1_value,k_2_value,k3_value,alpha_value),color='b', linewidth=1.5, label='neutrality line')

k2DetSol = solve(detA.subs(x,x_fromSolution),k2)[0]
k1_DetSol = solve(k2DetSol - k2_fromSolution,k1)[0]
k2_DetSol = k2_fromSolution.subs(k1,k1_DetSol)
k1Det_y = lambdify((y,k_1,k_2,k30,alpha),k1_DetSol,'numpy')
k2Det_y = lambdify((y,k_1,k_2,k30,alpha),k2_DetSol,'numpy')
plt.plot(k2Det_y(y_range, k_1_value,k_2_value,k3_value,alpha_value), k1Det_y(y_range, k_1_value,k_2_value,k3_value,alpha_value),color='r',linestyle='--', linewidth=1.5, label='multiplicity line')
plt.legend()
plt.grid(True)
plt.xlabel('$k_2$')
plt.ylabel('$k_1$')
plt.xlim([-0.0, 0.2])
plt.ylim([-0.0, 0.2])
plt.show()


###############################################################################
# k1_value=0.2
# k2_value=0.8
###
# k2_value=0.03809116
# k1_value=1.07475379e-01

# k2_value=0.020
# k1_value=0.012

# k2_value=0.05
# k1_value=0.075

f1 = lambdify((x, y, k1, k_1, k30, alpha), eq1)
f2 = lambdify((x, y, k2, k_2, k30, alpha), eq2)
Y, X = np.mgrid[0:.5:1000j, 0:1:2000j]
U = f1(X, Y, k1_value, k_1_value, k3_value,alpha_value)
V = f2(X, Y, k2_value, k_2_value, k3_value,alpha_value)
print("HERE")
velocity = np.sqrt(U*U + V*V)
plt.streamplot(X, Y, U, V, density = [2.5, 0.8], color=velocity)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase portrait')
plt.show()




plt.figure("-")
t = np.linspace(0, 1000, 1000)
plt.grid(True)
plt.xlabel('t')
plt.ylabel('x,y')
init = [0.21, 0.23]
sol = odeint(df, init, t)
plt.plot(t,sol[:, 0],  color='m', label="x(t)")
plt.plot(t,sol[:, 1],  color='c', label="y(t)")
plt.legend()
plt.show()





from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from math import pi,sinh,cosh
from mshr import *

# Create mesh and define function space
box = Rectangle(Point(0, 0), Point(5,3))
mesh = generate_mesh(box, 256)
# element of degree 1
V = FunctionSpace(mesh, 'P', 1)
nearEr = 1e-14

#https://femtable.org/
#https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/poisson/python/documentation.html
#https://fenicsproject.org/olddocs/dolfin/1.3.0/python/demo/documented/subdomains-poisson/python/documentation.html
# Define boundary condition x
u_left = Expression('sin(pi*x[0]/5)', degree=1)
u_right = Expression('sin(3*pi*x[0]/5)', degree=1)
# Define boundary condition y
g_left = Expression('sin(3*pi*x[1]/3)', degree=1)
g_right = Expression('sin(2*pi*x[1]/3)', degree=1)


# Define  boundary (x = 0 or x = 5)
def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 0, nearEr)


def boundary_right(x, on_boundary):
    return on_boundary and near(x[0], 5, nearEr)

# Define  boundary (y = 0 or y = 3)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, nearEr)


class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 3, nearEr)

bc_left = DirichletBC(V, u_left, boundary_left)
bc_right = DirichletBC(V, u_right, boundary_right)
bcs = [bc_left, bc_right]



top = Top()
bottom = Bottom()
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
top.mark(boundaries, 1)
bottom.mark(boundaries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


# Define variational problem
f = Expression("sin(pi*x[0])*sin(2*pi*x[1])", degree=1)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx
L = f*v*dx - g_right * v * ds(1) - g_left * v * ds(2)

# Compute solution
fem_solution = Function(V)
solve(a == L, fem_solution, bcs)

paper_solution = Expression('sin(pi*x[0])*sin(2*pi*x[1])+sinh(2*pi*x[0]/3)*sin(2*pi*x[1]/3)/sinh(10*pi/3)+sin(pi*x[1])*(cosh(pi*x[0])-cosh(pi*5)*sinh(pi*x[0])/sinh(pi*5))+sinh(3*pi*x[1]/5)*sin(3*pi*x[0]/5)/sinh(9*pi/5)+sin(pi*x[0]/5)*(cosh(pi*x[1]/5)-cosh(3*pi/5)*sinh(pi*x[1]/5)/sinh(3*pi/5))', degree=2)
error_L2 = errornorm(paper_solution, fem_solution, 'L2')
vertex_values_fem = fem_solution.compute_vertex_values(mesh)
vertex_values_paper = paper_solution.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_fem - vertex_values_paper))
print('error_L2 =', error_L2)
print('error_max =', error_max)

# Save and Plot solution
n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

plt.figure(1)
plt.title("Analytical")
zfaces = np.asarray([paper_solution(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.colorbar()
plt.savefig('analytical_solution.pdf')

plt.figure(2)
plt.title("Numerical")
zfaces = np.asarray([fem_solution(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.colorbar()
plt.savefig('numerical_solution.pdf')

print('plot is ready')

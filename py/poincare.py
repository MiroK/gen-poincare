from dolfin import *
from scipy.sparse import diags, kron, eye, csr_matrix
from common import mat_to_csr, csr_to_petsc
import numpy as np


def fem_system(dim, n_cells, z):
    '''
    Discretize -Delta u = lmbda*(-Delta u + u) with Neumann bcs on [-1, 1]^dim
    and more importantly the we consider subspace such that (z, u)_V=0. This
    leads to a constrained saddle point problem.
    '''
    if dim == 1:
        mesh = IntervalMesh(n_cells, -1, 1)
    elif dim == 2:
        mesh = RectangleMesh(Point(-1, -1), Point(1, 1), n_cells, n_cells)
    elif dim == 3:
        mesh = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), n_cells, n_cells, n_cells)
    else:
        raise ValueError

    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'R', 0)
    W = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Get the constraint
    if isinstance(z, (Constant, Expression)):
        z = interpolate(z, V)
    else:
        z_values = z
        z = Function(V)
        z.vector().set_local(z_values)
        z.apply('insert')

    a = inner(grad(u), grad(v))*dx +\
        p*inner(grad(z), grad(v))*dx + p*inner(z, v)*dx +\
        q*inner(grad(z), grad(u))*dx + q*inner(z, u)*dx

    m = inner(grad(u), grad(v))*dx + inner(u, v)*dx + Constant(0)*inner(p, q)*dx
    L = inner(Constant(0), v)*dx + inner(Constant(0), q)*dx

    A, M = PETScMatrix(), PETScMatrix()
    b = PETScVector()
    assemble_system(a, L, A_tensor=A, b_tensor=b)
    assemble_system(m, L, A_tensor=M, b_tensor=b)

    return A, M, mesh.hmin()


def size(mat):
    try:
        return A.shape[0]
    except AttributeError:
        return A.size(0)

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from solvers import dense_solver, scipy_solver, slepc_solver

    np.set_printoptions(precision=12)

    method = slepc_solver
    system = fem_system

    dim = 1
    sizes = {1: range(8, 11)}

    z = Constant(1)
    is_hermitian = False
    for n_cells in [2**i for i in sizes[dim]]:
        A, M, h = system(dim, n_cells, z=z)
        eigw = method(A, M, is_hermitian)

        np.sort(np.abs(eigw))
        if len(eigw) > 2:
            eigw = eigw[range(2)]
        else:
            pass
        print np.r_[size(A), h, eigw].tolist()

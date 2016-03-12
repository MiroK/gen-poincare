from dolfin import *
from scipy.sparse import diags, kron, eye, csr_matrix
from common import mat_to_csr, csr_to_petsc
import numpy as np


def fem_system(dim, n_cells):
    '''
    Discretize -Delta u = lmbda*u with Neumann bcs on [-1, 1]^dim.
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
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    m = inner(u, v)*dx
    L = inner(Constant(0), v)*dx

    A, M = PETScMatrix(), PETScMatrix()
    b = PETScVector()
    assemble_system(a, L, A_tensor=A, b_tensor=b)
    assemble_system(m, L, A_tensor=M, b_tensor=b)

    return A, M, mesh.hmin()


def fourier_system(dim, n_cells):
    '''
    Discretize -Delta u = u with Neumann bcs on [-1, 1]^dim.
    '''
    A = diags(np.r_[0,                                                       
                    np.array([(k*pi/2)**2 for k in range(1, n_cells, 2)]),         
                    np.array([(k*pi/2)**2 for k in range(2, n_cells, 2)])], 0)
    M = eye(A.shape[0])

    if dim == 1:
        pass
    elif dim == 2:
        A = kron(A, M) + kron(M, A)
        M = kron(M, M)
    elif dim == 3:
        A = kron(kron(A, M), M) + kron(kron(M, A), M) + kron(M, kron(M, A))
        M = kron(kron(M, M), M)

    A, M = map(csr_matrix, (A, M))
    A, M = map(csr_to_petsc, (A, M))
    return A, M, n_cells


def size(mat):
    try:
        return A.shape[0]
    except AttributeError:
        return A.size(0)

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    from solvers import dense_solver, scipy_solver, slepc_solver

    np.set_printoptions(precision=12)

    method = dense_solver
    system = fourier_system

    dim = 1
    sizes = {1: range(8, 11),
             2: range(2, 6),
             3: range(2, 6)}

    is_hermitian = True
    for n_cells in [2**i for i in sizes[dim]]:
        A, M, h = system(dim, n_cells)
        eigw = method(A, M, is_hermitian)
        eigw = np.sort(np.abs(eigw))[range(2)]
        print np.r_[size(A), h, eigw].tolist()

from scipy.linalg import eigvalsh, eigvals
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.sparse.linalg import ArpackNoConvergence as ArpackError
from scikits.sparse.cholmod import cholesky
from dolfin import SLEPcEigenSolver
import numpy as np


def dense_solver(A, M, is_hermitian):
    '''Smallest eigenvalues'''
    if is_hermitian:
        return eigvalsh(A.array(), M.array())
    else:
        return eigvals(A.array(), M.array())


def scipy_solver(A, M, is_hermitian):
    '''Smallest eigenvalues'''
    A, M = map(mat_to_csr, (A, M))
    
    chol = cholesky(M)
    Minv = LinearOperator(matvec=lambda x, mat=chol: mat.solve_A(x),
                          shape=M.shape)

    if is_hermitian:
        try:
            eigw = eigsh(A, k=5, M=M, sigma=None, which='SM', v0=None, ncv=None,        
                         maxiter=None, tol=1E-10, return_eigenvectors=False)
        except ArpackError as e:
            print '\tDivergence'
            eigw = e.eigenvalues
    else:
        raise NotImplementedError

    return eigw


def slepc_solver(A, M, is_hermitian):
    '''Smallest eigenvalue'''
    #TODO Set this up with preconditioners!!
    solver = SLEPcEigenSolver(A, M)
    solver.parameters['spectrum'] = 'smallest magnitude'

    if is_hermitian:
        solver.parameters['problem_type'] = 'gen_hermitian'
    else:
        raise NotImplementedError

    solver.solve(3)

    eigw = [solver.get_eigenpair(i)[0]
            for i in range(solver.get_number_converged())]

    return eigw

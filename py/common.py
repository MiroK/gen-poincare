from dolfin import MPI, mpi_comm_world, PETScMatrix
from block import block_mat
from scipy.sparse import csr_matrix
from numpy import ndarray, asarray, arange
import petsc4py
petsc4py.init()
from petsc4py import PETSc


def mat_to_csr(mat):
    '''Convert any dolfin.Matrix to csr matrix in scipy.'''
    assert MPI.size(mpi_comm_world()) == 1, 'mat_to_csr assumes single process'
    # We can handle blocks
    if isinstance(mat, (list, ndarray, block_mat)):
        return [mat_to_csr(mat_) for mat_ in mat]
    # Number block can anly be zero and for bmat these are None
    elif isinstance(mat, (int, float)):
        assert abs(mat) < 1E-15
        return None
    # Actual matrix
    else:
        rows = [0]
        cols = []
        values = []
        for row in range(mat.size(0)):
            cols_, values_ = mat.getrow(row)
            rows.append(len(cols_)+rows[-1])
            cols.extend(cols_)
            values.extend(values_)

        shape = mat.size(0), mat.size(1)
        
        return csr_matrix((asarray(values, dtype='float'),
                           asarray(cols, dtype='int'),
                           asarray(rows, dtype='int')),
                           shape)

def csr_to_petsc4py(csr_matrix):
    '''Convert Scipy's csr matrix to PETSc matrix.'''
    assert MPI.size(mpi_comm_world()) == 1, 'mat_to_csr assumes single process'

    if isinstance(csr_matrix, list):
        return [csr_to_pets4py(mat) for mat in csr_matrix]
    # None is zero block
    elif csr_matrix is None:
        return None
    else:
        A = csr_matrix
        csr = (A.indptr, A.indices, A.data)
        # Convert to PETSc
        n_rows, n_cols = A.shape
        A_petsc = PETSc.Mat().createAIJ(size=A.shape, csr=csr)

        # Now set local to global mapping for indices. This is supposed to run in
        # serial only so these are identities.
        row_lgmap = PETSc.LGMap().create(list(arange(n_rows, dtype=int)))
        if not n_rows == n_cols:
            col_lgmap = PETSc.LGMap().create(list(arange(n_cols, dtype=int)))
        else:
            col_lgmap = row_lgmap

        A_petsc.setLGMap(row_lgmap, col_lgmap)
        A_petsc.assemble()

        return A_petsc


def csr_to_petsc(csr_matrix):
    '''Convert Scipy's csr matrix to DOLFIN's PETScMatrix object.'''
    if isinstance(csr_matrix, list):
        return [csr_to_petsc(mat) for mat in csr_matrix]
    # None is zero block
    elif csr_matrix is None:
        return 0
    else:
        return PETScMatrix(csr_to_petsc4py(csr_matrix))


import numpy as np
import functools

from .. import NoSolutionsException

try:
    import npycvx
except ImportError:
    raise ImportError("Please install the npycvx package to use GLPK solver module.")

def solve_lp(A: np.ndarray, b: np.ndarray, objectives: np.ndarray, int_vrs: set=set(), minimize: bool=True):
    """
    Solve the linear programming problem:
    minimize c^T x
    subject to Ax >= b

    for each c in objectives.
    """

    # Load solve-function with the now converted numpy
    # matrices/vectors into cvxopt data type...
    solve_part_fn = functools.partial(
        npycvx.solve_lp, 
        *npycvx.convert_numpy(A, b, int_vrs=int_vrs), 
        minimize
    )

    # Exectue each objective with solver function
    solutions = list(
        map(
            solve_part_fn, 
            objectives
        )
    )

    if any(map(lambda x: x[0] != 'optimal', solutions)):
        raise NoSolutionsException("Could not find solutions. Please check constraints.")
    
    return list(map(lambda x: x[1], solutions))

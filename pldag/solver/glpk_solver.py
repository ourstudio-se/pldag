import numpy as np
import npycvx
import functools

def solve_lp(A: np.ndarray, b: np.ndarray, objectives: np.ndarray, int_vrs: set=set()):
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
        False
    )

    # Exectue each objective with solver function
    solutions = list(
        map(
            solve_part_fn, 
            objectives
        )
    )

    if any(map(lambda x: x[0] != 'optimal', solutions)):
        raise Exception("Some objectives are not optimal")
    
    return list(map(lambda x: x[1], solutions))

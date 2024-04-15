import numpy as np
from hashlib import sha1
from itertools import groupby, repeat
from functools import partial
from typing import Dict, List, Set
from graphlib import TopologicalSorter

from enum import Enum
from importlib import import_module

class Solver(Enum):
    GLPK = "glpk"

# Add more solvers here as they are implemented
try:
    import_module('pldag.solver.glpk_solver')
except ImportError:
    Solver.GLPK = None

class PLDAG:

    """
        "Primitive Logic Directed Acyclic Graph" data structure, or "PL-DAG" for short, is fundamentally a Directed Acyclic Graph (DAG) where each node represents a logical relationship, and the leaf nodes correspond to literals. 
        Each node in the graph encapsulates information about how its incoming nodes or leafs are logically related. For instance, a node might represent an AND operation, meaning that if it evaluates to true, all its incoming nodes or leafs must also evaluate to true.
        
        In summary, this data structure combines elements of a DAG with a logic network, utilizing prime numbers to encode relationships and facilitate operations within the graph.
    """

    def __init__(self):
        # Adjacency matrix. Each entry is a boolean (0/1) indicating if there is a dependency
        self._amat = np.zeros((0, 0),   dtype=np.uint8)
        # Boolean vector indicating if negated
        self._nvec = np.zeros((0, ),    dtype=bool)
        # Complex vector representing bounds of complex number data type
        self._dvec = np.zeros((0, ),    dtype=complex)
        # Bias vector
        self._bvec = np.zeros((0, ),    dtype=complex)
        # Boolean vector indicating if the node is a composition
        self._cvec = np.zeros((0, ),    dtype=bool)
        # Maps id's to index
        self._imap = {}

    @property
    def bounds(self) -> np.ndarray:
        """Get the bounds of all aliases"""
        return self._dvec
    
    @property
    def ids(self) -> List[str]:
        """Get all ids"""
        return list(self._imap.keys())
    
    @property
    def _toposort(self) -> iter:
        return TopologicalSorter(
            dict(
                map(
                    lambda x: (x[0], set(map(lambda y: y[1], x[1]))), 
                    groupby(np.argwhere(self._amat), key=lambda x: x[0])
                )
            )
        ).static_order()
    
    @staticmethod
    def _composite_id(children: list, bias: int, negate: bool = False) -> str:
        """
            Create a composite ID from a list of children.
        """
        return sha1(("".join(sorted(set(children))) + str(negate) + str(bias)).encode()).hexdigest()
    
    @property
    def primitives(self) -> np.ndarray:
        return self._col_vars[~self._cvec]
    
    @property
    def composites(self) -> np.ndarray:
        return self._col_vars[self._cvec]
    
    @property
    def _revimap(self) -> dict:
        """Get the reverse map"""
        return dict(map(lambda x: (x[1], x[0]), self._imap.items()))
    
    @property
    def _col_vars(self) -> np.ndarray:
        return np.array(list(self._imap.keys()))
    
    @property
    def _row_vars(self) -> np.ndarray:
        return np.array(list(self._imap.keys()))[self._cvec]
    
    def _icol(self, i: int) -> str:
        """
            Returns the ID of the given column index.
        """
        return self._col_vars[i]
    
    def _irow(self, i: int) -> str:
        """
            Returns the ID of the given row index.
        """
        return self._row_vars[i]
    
    def _col(self, id: str) -> int:
        """
            Returns the column index of the given ID.
        """
        return self._imap[id]
    
    def _row(self, id: str) -> int:
        """
            Returns the row index of the given ID.
        """
        return self._col(id) - (~self._cvec).sum()
    
    def _set_gelineq(self, children: list, bias: int, negate: bool = False) -> str:
        """
            Add a composite constraint of at least `value`.
        """
        _id = self._composite_id(children, bias, negate)
        if not _id in self._imap:
            self._amat = np.pad(self._amat, ((0, 1), (0, 1)), mode='constant')
            self._amat[-1, [self._col(child) for child in children]] = 1
            self._dvec = np.append(self._dvec, complex(0, 1))
            self._bvec = np.append(self._bvec, complex(*repeat(bias * (1-negate) + (bias + 1) * negate * -1, 2)))
            self._nvec = np.append(self._nvec, negate)
            self._cvec = np.append(self._cvec, True)
            self._imap[_id] = self._amat.shape[1] - 1

        return _id
    
    @staticmethod
    def _prop_algo(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, F: np.ndarray, forced: np.ndarray, max_iterations: int = 100):

        """
            Propagation algorithm.

            A: the adjacency matrix
            B: the bias array
            C: which nodes are compositions
            D: the initial bounds
            F: the negation vector
            forced: boolean (won't change during propagation) (optional)
            max_iterations: the maximum number of iterations (optional)

            Returns the propagated bounds.
        """

        def _prop_once(A: np.ndarray, C: np.ndarray, F: np.ndarray, B: np.ndarray, D: np.ndarray):
            r = A.dot(D)
            # Here we negate the equation bound if the node is negated
            # E.g. -1+1j becomes -2+0j (mirrored on the line y = -x-1)
            rf = (-1j * np.conj(r) * F + (1-F) * r) + B
            d = ~C * D
            d[C] = (rf.real >= 0) + 1j*(rf.imag >= 0)
            return d

        prop = partial(_prop_once, A, C, F, B)    
        previous_D = D
        for _ in range(max_iterations):
            new_D = forced * D + ~forced * prop(previous_D)
            if (new_D == previous_D).all():
                return new_D
            previous_D = new_D
        
        raise Exception(f"Maximum iterations ({max_iterations}) reached without convergence.")
    
    def get(self, *id: str) -> np.ndarray:
        """Get the bounds of the given ID(s)"""
        return self._dvec[list(map(self._col, id))]
    
    def exists(self, id: str) -> bool:
        """Check if the given id exists"""
        return (id in self._imap)
    
    def dependencies(self, id: str) -> Set[str]:
        """
            Get the dependencies of the given ID.
        """
        return set(
            map(
                lambda x: list(self._imap)[x],
                np.argwhere(self._amat[self._row(id)] == 1).T[0]
            )
        )
    
    def negated(self, id: str) -> bool:
        """Get the negated state of the given id"""
        return bool(self._nvec[self._row(id)])

    def propagate(self):
        """
            Propagates the graph and stores the propagated bounds.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_atleast("xy", 1)
            >>> model.propagate()
            >>> model.get(a)
            1j
            
            >>> model.set_primitive("x", 1+1j)
            >>> model.propagate()
            >>> model.get(a)
            1+1j

            Returns
            -------
            None
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._bvec
        C: np.ndarray = self._cvec
        D: np.ndarray = self._dvec
        F: np.ndarray = self._nvec

        # Propagate the graph and store the result as new bounds
        self._dvec = self._prop_algo(A, B, C, D, F, np.zeros(D.shape[0], dtype=bool))

    def test(self, query: Dict[str, complex], freeze: bool = True) -> Dict[str, complex]:

        """
            Propagates the graph and returns the result.

            Parameters
            ----------
            query : Dict[str, complex]
                The query to test.

            freeze : bool = True
                If the bounds given in query should be changeable
                during propagation.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_atleast("xy", 1)
            >>> model.test({"x": 1j, "y": 1+1j}).get(a)
            1+1j

            Returns
            -------
            Dict[str, complex]
                The result of the query.
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._bvec
        C: np.ndarray = self._cvec
        D: np.ndarray = self._dvec.copy()
        F: np.ndarray = self._nvec

        # Filter query based on existing variables
        query = {k: v for k, v in query.items() if k in self._imap}

        # Query translation into primes
        qprimes = np.zeros(D.shape[0], dtype=bool)
        qprimes[[self._imap[q] for q in query]] = True and freeze

        # Replace the observed bounds
        D[[self._imap[q] for q in query]] = np.array(list(query.values()))

        return dict(zip(self._imap.keys(), self._prop_algo(A, B, C, D, F, qprimes)))
    
    def set_primitive(self, id: str, bound: complex = complex(0,1)) -> str:
        """
            Add a primitive variable.

            Parameters
            ----------
            id : str
                The ID of the primitive variable.

            bound : complex
                The bound of the primitive variable.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitive("x")
            >>> model.get("x")
            1j

            >>> model.set_primitive("x", 1+1j)
            >>> model.get("x")
            1+1j

            Returns
            -------
            str
                The ID of the primitive variable.
        """
        if id in self._imap:
            # Update value if already in map
            self._dvec[self._col(id)] = bound
        else:
            self._amat = np.hstack((self._amat, np.zeros((self._amat.shape[0], 1), dtype=np.uint8)))
            self._dvec = np.append(self._dvec, bound)
            self._cvec = np.append(self._cvec, False)
            self._imap[id] = self._amat.shape[1] - 1

        return id

    def set_primitives(self, ids: List[str], bound: complex = complex(0,1)) -> List[str]:
        """
            Add multiple primitive variables.

            Parameters
            ----------
            ids : List[str]
                The IDs of the primitive variables.

            bound : complex
                The bound of all given primitive variables.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives(["x", "y"])
            >>> model.get("x")
            1j

            >>> model.get("y")
            1j

            Returns
            -------
            List[str]
                The IDs of the primitive variables.
        """
        for id in ids:
            self.set_primitive(id, bound)
        return ids
    
    def set_atleast(self, references: List[str], value: int) -> str:
        """
            Add a composite constraint of at least `value`.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            value : int
                The minimum value to set.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_atleast(["x", "y"], 1)
            >>> model.test({"x": 1+1j}).get(a)
            1+1j

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self._set_gelineq(references, -1 * value, False)
    
    def set_atmost(self, references: List[str], value: int) -> str:
        """
            Add a composite constraint of at most `value`.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            value : int
                The maximum value to set.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_atmost(["x", "y"], 1)
            >>> model.test({"x": 1+1j, "y": 0j}).get(a)
            1+1j

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self._set_gelineq(references, -1 * (value + 1), True)
    
    def set_or(self, references: List[str]) -> str:
        """
            Add a composite constraint of an OR operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_or(["x", "y"])
            >>> model.test({"x": 1+1j}).get(a)
            1+1j

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_atleast(references, 1)
    
    def set_and(self, references: List[str]) -> str:
        """
            Add a composite constraint of an AND operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_and(["x", "y"])
            >>> model.test({"x": 1+1j, "y": 1+1j}).get(a)
            1+1j

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_atleast(references, len(references))
    
    def set_not(self, references: List[str]) -> str:
        """
            Add a composite constraint of a NOT operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_not(["x"])
            >>> model.test({"x": 1+1j}).get(a)
            0j

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_atmost(references, 0)
    
    def set_xor(self, references: List[str]) -> str:
        """
            Add a composite constraint of an XOR operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_xor(["x", "y"])
            >>> model.test({"x": 1+1j, "y": 0j}).get(a)
            1+1j

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_and([
            self.set_atleast(references, 1),
            self.set_atmost(references, 1),
        ])
    
    def set_imply(self, condition: str, consequence: str) -> str:
        """
            Add a composite constraint of an IMPLY operation.

            Parameters
            ----------
            condition : str
                The reference to the condition.

            consequence : str
                The reference to the consequence.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_imply("x", "y")
            >>> model.test({"x": 1j, "y": 1j}).get(a)
            1j

            >>> model.test({"x": 1+1j, "y": 0j}).get(a)
            0j

            >>> model.test({"x": 0j, "y": 0j}).get(a)
            1+1j

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_or([self.set_not([condition]), consequence])

    def to_polyhedron(self, fix: Dict[str, int] = {}) -> tuple:

        """
            Constructs a polyhedron of matrix A and bias vector b,
            such that A.dot(x) >= b, where x is the vector of variables.
            Every composite variable A is its own column in the matrix where
            A -> (A's composite proposition) is true. 

            Parameters
            ----------
            fix : Dict[str, int]
                A dictionary of variables to fix.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> a = model.set_atleast("xyz", 1)
            >>> A,b,vs = model.to_polyhedron()
            >>> np.array_equal(A, np.array([[1,1,1,-1]]))
            True
        """

        # Create the matrix
        A = np.zeros(self._amat.shape, dtype=np.int64)

        # Adjacent points
        adj_points = np.argwhere(self._amat == 1)

        # Fill the matrix
        A[adj_points.T[0], adj_points.T[1]] = 1

        # Flip the once that are negated
        A[self._nvec == 1] *= -1

        # Composite index in matrix
        cidx = np.array(list(self._imap.values()))[self._cvec]

        # Fetch the inner bounds
        inner_bounds = self._amat.dot(self._dvec)

        # Flip those that are negated
        inner_bounds[self._nvec == 1] = np.conj(inner_bounds[self._nvec == 1]) * -1j

        # Add onto bias so the bounds are correct
        eq_bounds = inner_bounds + self._bvec

        # Fill the composite id's with the mx value
        A[range(A.shape[0]), cidx] = eq_bounds.real

        # Create the bias vector. The linear equation should be increased
        # by the flipped minimum value
        b = eq_bounds.real - self._bvec.real

        # Fix the columns in `fix`
        for i, vs in groupby(zip(fix.values(), fix.keys()), key=lambda x: x[0]):
            v = list(map(lambda x: x[1], vs))
            a = np.zeros(A.shape[1], dtype=np.int64)
            a[np.array(list(map(self._col, v)))] = 1
            if i > 0 or i < 0:
                _b = a.sum() * i
            else:
                _b = 0

            A = np.vstack([A, a])
            b = np.append(b, _b)

        # Reverse index map
        rimap = dict(map(lambda x: (x[1], x[0]), self._imap.items()))

        # Create the variable list
        variables = np.array(
            list(
                map(
                    lambda i: (
                        rimap[i],
                        self._dvec[i],
                    ),
                    np.array(list(self._imap.values()))
                )
            )
        )

        return A, b, variables
    
    def _from_indices(self, row_idxs: np.ndarray, col_idxs: np.ndarray) -> 'PLDAG':
        """
            Create a PLDAG from the given row and column indices.

            Parameters
            ----------
            row_idxs : np.ndarray
                The row indices to include in the sub PLDAG.

            col_idxs : np.ndarray
                The column indices to include in the sub PLDAG.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> a = model.set_atleast("xyz", 1)
            >>> sub_model = model._from_indices(np.array([0, 1]), np.array([0, 1, 2]))
            >>> sub_model.get("x")
            1j

            Returns
            -------
            PLDAG
                The sub-PLDAG.
        """
        sub_model = PLDAG()
        sub_model._amat = self._amat[row_idxs][:, col_idxs]
        sub_model._nvec = self._nvec[row_idxs]
        sub_model._dvec = self._dvec[col_idxs]
        sub_model._bvec = self._bvec[row_idxs]
        sub_model._cvec = self._cvec[col_idxs]
        sub_model._imap = dict(map(lambda x: (x[1], x[0]), enumerate(self._col_vars[col_idxs])))
        return sub_model
    
    def sub(self, ids: List[str]) -> 'PLDAG':
        """
            Create a sub-PLDAG from the given IDs.

            Parameters
            ----------
            ids : List[str]
                The IDs to include in the sub PLDAG.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> a = model.set_atleast("xyz", 1)
            >>> sub_model = model.sub_pldag(["x", "y"])
            >>> sub_model.get("x")
            1j

            Returns
            -------
            PLDAG
                The sub-PLDAG.
        """

        col_vars = self._col_vars.tolist()
        row_vars = self._row_vars.tolist()

        def incoming_edges(idlist: List[int]) -> Set[int]:
            return set(self._col_vars[np.argwhere(self._amat[list(map(row_vars.index, filter(lambda v: v in row_vars, idlist)))]).T[1]].tolist())
        
        # First find all variables (columns and rows)
        # that eventually reaches any of the id's in `ids`
        max_iterations: int = 100
        matching_variables = incoming_edges(set(ids)).union(set(ids))
        for _ in range(max_iterations):
            new_matching_variables = incoming_edges(matching_variables)
            if new_matching_variables.issubset(matching_variables):
                break
            matching_variables.update(new_matching_variables)

        col_idxs = np.array(sorted(list(map(col_vars.index, matching_variables))))
        row_idxs = np.array(sorted(list(map(row_vars.index, filter(lambda v: v in self._row_vars, matching_variables)))))
        return self._from_indices(row_idxs, col_idxs)
    
    def solve(self, objectives: List[Dict[str, int]], fix: Dict[str, int], solver: Solver) -> List[Dict[str, int]]:
        """
            Solve the polyhedron with the given objectives.

            Parameters
            ----------
            objectives : List[Dict[str, int]]
                The objectives to solve for.

            fix : Dict[str, int]
                The variables to fix.

            solver : Solver
                The solver to use.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> a = model.set_atleast("xyz", 1)
            >>> A,b,vs = model.to_polyhedron()
            >>> model.solve([{"x": 0, "y": 1, "z": 0}], Solver.GLPK)
            [{'x': 0, 'y': 1, 'z': 0}]

            Returns
            -------
            List[Dict[str, int]]
                The solutions for the objectives.
        """
        A, b, variables = self.to_polyhedron(fix)
        obj_mat = np.zeros((len(objectives), len(variables)), dtype=np.int64)
        for i, obj in enumerate(objectives):
            obj_mat[i, [self._col(k) for k in obj]] = list(obj.values())

        if solver == Solver.GLPK:
            from pldag.solver.glpk_solver import solve_lp
            solutions = solve_lp(A, b, obj_mat, set(np.argwhere((self._dvec.real != 0) | (self._dvec.imag != 1)).T[0].tolist()))
        else:
            raise ValueError("Solver not implemented.")
        
        return list(
            map(
                lambda solution: dict(
                    zip(
                        variables.T[0], 
                        map(
                            lambda i: complex(i,i),
                            solution
                        )
                    )
                ),
                solutions
            )
        )
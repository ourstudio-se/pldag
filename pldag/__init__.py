import numpy as np
from hashlib import sha1
from itertools import groupby, repeat, starmap
from functools import partial, lru_cache
from typing import Dict, List, Set, Optional
from graphlib import TopologicalSorter

from enum import Enum

class NoSolutionsException(Exception):
    pass

class Solver(Enum):
    GLPK = "glpk"

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
        # Alias to id mapping
        self._amap = {}

    def __hash__(self) -> int:
        return hash(self.sha1())
    
    def sha1(self) -> str:
        return sha1(("".join(self._imap.keys()) + "".join(map(lambda c: f"{c.real}{c.imag}", self._dvec))).encode()).hexdigest()

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
        return sha1(("".join(map(lambda x: sha1(x.encode()).hexdigest(), sorted(set(children)))) + str(negate) + str(bias)).encode()).hexdigest()
    
    @property
    def primitives(self) -> np.ndarray:
        return self._col_vars[~self._cvec]
    
    @property
    def composites(self) -> np.ndarray:
        return self._col_vars[self._cvec]
    
    @property
    def columns(self) -> np.ndarray:
        return self._col_vars
    
    @property
    def rows(self) -> np.ndarray:
        return self.composites
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Get the adjacency matrix"""
        return np.vstack((np.zeros((self._amat.shape[1]-self._amat.shape[0], self._amat.shape[1])), self._amat))
    
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
    
    @property
    def _inner_bounds(self) -> np.ndarray:
        return self._amat.dot(self._dvec) + self._bvec
    
    def printable(self, id: str) -> str:
        """
            Get the printable version of the given ID.
        """
        if id in self.primitives:
            return f"{id} = {self._dvec[self._col(id)]}"
        elif id in self.composites:
            join_on = " + " if self._nvec[self._row(id)] == 0 else " - "
            return f"{id} = " + join_on.join(self.dependencies(id)) + f" + {self._bvec[self._row(id)].real} >= 0"
        else:
            raise ValueError(f"ID {id} not found.")
    
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
        return self.composites.tolist().index(id)
    
    def _set_gelineq(self, children: list, bias: int, negate: bool = False, alias: Optional[str] = None) -> str:
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

        if alias:
            self._amap[alias] = _id

        return _id
    
    @staticmethod
    def _prop_upstream_algo(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, F: np.ndarray, fixed: np.ndarray):
        """
            Propagates upstream, trying to infer values given what's been set. 
            NOTE: a composite variable may be true while the equation isn't decided yet. This can happen if a parent composite
            sets it's child, for instance a composite variable A, but the equation A = x + y >= 1 cannot be decided yet.
        """

        # There are two cases for when we can safely assume new bounds
        # for variables:
        
        # 1. When the inner upper bound is 0 AND the outer bound is true.
        #    Then we can assume each variable's bound in the composite to be their constant upper bound. I.e. set their variables lower bound to their upper bound.
        #    For instance, [1 1](A) = [0 1](x) + [0 1](y) + [0 1](z) - [3 3] >= 0 has an inner bound of [0 1]+[0 1]+[0 1]-[3 3] = [-3 0] and should result in x=1, y=1 and z=1, since A=1 is fixed
        #    Or, [1 1](A) = [-1 0](x) + [-1 0](y) + [-1 0](z) >= 0 has an inner bound of [-1 0]+[-1 0]+[-1 0] = [-3 0] and should result in x=0, y=0, z=0, since A=1 is fixed

        # 2. When the inner lower bound is -1 AND the outer bound is false.
        #    Then we can assume each variable's bound in the composite to be their constant lower bound. I.e. set their variables upper bound to their lower bound.
        #    For instance, [0 0](A) = [0 1](x) + [0 1](y) + [0 1](z) - [1 1] >= 0 has an inner bound of [0 1]+[0 1]+[0 1]-[1 1] = [-1 2] and should result in x≈0, y≈0 and z≈0, since A=0 is fixed
        #    Or, [0 0] = [-1 0](x) + [-1 0](y) + [-1 0](z) + 2 >= 0 has an inner bound of [-1 0]+[-1 0]+[-1 0]+[2 2] = [-1 2] and should result in x≈0, y≈0 and z≈0, since A=0 is fixed
        _A = np.vstack((np.zeros((A.shape[1]-A.shape[0], A.shape[1]), dtype=np.uint8), A))
        _B = np.append(np.zeros(A.shape[1]-A.shape[0], dtype=complex), B)
        _F = np.append(np.zeros(A.shape[1]-A.shape[0], dtype=bool), F)
        _fixed = fixed | (D.real == D.imag)
        M = (_A == 1) & ~_fixed
        for i in filter(
            lambda i: C[i] and M[i].any(), 
            reversed(
                list(
                    TopologicalSorter(
                        dict(
                            map(
                                lambda x: (x[0], set(map(lambda y: y[1], x[1]))), 
                                groupby(np.argwhere(_A).tolist(), key=lambda x: x[0])
                            )
                        )
                    ).static_order()
                )
            )
        ):
            r = _A[i].dot(D)
            rf = (-1j * np.conj(r) * _F[i] + (1-_F[i]) * r) + _B[i]
            re = D[M[i]].real
            im = D[M[i]].imag
            if rf.imag == 0 and D[i].real == 1:
                if _F[i]:
                    D[M[i]] = re + re * 1j
                else:
                    D[M[i]] = im + im * 1j

            elif rf.real == -1 and D[i].imag == 0:
                if _F[i]:
                    D[M[i]] = im + im * 1j
                else:
                    D[M[i]] = re + re * 1j

        return D
    
    @staticmethod
    def _prop_algo_downstream(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, F: np.ndarray, fixed: np.ndarray, max_iterations: int = 100):

        """
            Propagation algorithm.

            A: the adjacency matrix
            B: the bias array
            C: which nodes are compositions
            D: the initial bounds
            F: the negation vector
            fixed: boolean vector (optional)
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
            new_D = fixed * D + ~fixed * prop(previous_D)
            if (new_D == previous_D).all():
                return new_D
            previous_D = new_D
        
        raise Exception(f"Maximum iterations ({max_iterations}) reached without convergence.")
    
    @staticmethod
    def _prop_algo_bistream(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, F: np.ndarray, fixed: np.ndarray, max_iterations: int = 100):
        """
            Propagates the graph downstream and upstream, and returns the propagated bounds.
        """
        return PLDAG._prop_upstream_algo(A, B, C, PLDAG._prop_algo_downstream(A, B, C, D, F, fixed, max_iterations), F, fixed)
    
    def _propagate(self, method: str, query: dict, freeze: bool = True) -> dict:
        """
            Propagates the graph downstream and returns the propagated bounds.
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

        if method == "downstream":
            res = self._prop_algo_downstream(A, B, C, D, F, qprimes)
        elif method == "upstream":
            res = self._prop_upstream_algo(A, B, C, D, F, qprimes)
        elif method == "bistream":
            res = self._prop_algo_bistream(A, B, C, D, F, qprimes)
        else:
            raise ValueError(f"Method '{method}' not found.")
        
        return dict(
            zip(
                self._imap.keys(),
                res,
            )
        )
    
    def id_from_alias(self, alias: str) -> Optional[str]:
        """Get the ID of the given alias"""
        return self._amap.get(alias, None)
    
    def id_to_alias(self, id: str) -> List[str]:
        """Get the aliases of the given ID"""
        return list(
            map(
                lambda x: x[0],
                filter(
                    lambda x: x[1] == id,
                    self._amap.items(),
                )
            )
        )
    
    def get(self, *id: str) -> np.ndarray:
        """Get the bounds of the given ID(s)"""
        return self._dvec[list(map(self._col, id))]
    
    def delete(self, id: str) -> True:
        """
            Delete the given ID.
        """
        try:
            if id in self.composites:
                row_id = self._row(id)
                self._bvec = np.delete(self._bvec, row_id)
                self._nvec = np.delete(self._nvec, row_id)
                self._amap = dict(filter(lambda x: x[1] != id, self._amap.items()))
                self._amat = np.delete(self._amat, row_id, axis=0)

            col_id = self._col(id)
            self._amat = np.delete(self._amat, col_id, axis=1)
            self._dvec = np.delete(self._dvec, col_id)
            self._cvec = np.delete(self._cvec, col_id)
            del self._imap[id]
            self._imap = dict(
                zip(
                    self._imap.keys(),
                    range(len(self._imap))
                )
            )

            return True
        except:
            return False
    
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

    def propagate(self, query: dict = {}, freeze: bool = True) -> dict:
        """
            Propagates the graph downstream, towards the root node(s), and returns the propagated bounds.

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
        return self._propagate("downstream", query, freeze)
    
    def propagate_upstream(self, query: dict = {}, freeze: bool = True) -> dict:
        """
            Propagates the graph upstream, from the root node(s), and returns the propagated bounds.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_atleast("xy", 1)
            >>> model.propagate_upstream().get(a)
            1j
            
            >>> model.set_primitive("x", 1+1j)
            >>> model.propagate_upstream().get(a)
            1+1j

            Returns
            -------
            None
        """
        return self._propagate("upstream", query, freeze)

    def propagate_bistream(self, query: Dict[str, complex] = {}, freeze: bool = True) -> dict:
        """
            Propagates the graph downstream and upstream, and returns the propagated bounds.
        """
        return self._propagate("bistream", query, freeze)
    
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
    
    def set_atleast(self, references: List[str], value: int, alias: Optional[str] = None) -> str:
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
        return self._set_gelineq(references, -1 * value, False, alias)
    
    def set_atmost(self, references: List[str], value: int, alias: Optional[str] = None) -> str:
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
        return self._set_gelineq(references, -1 * (value + 1), True, alias)
    
    def set_or(self, references: List[str], alias: Optional[str] = None) -> str:
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
        return self.set_atleast(references, 1, alias)
    
    def set_and(self, references: List[str], alias: Optional[str] = None) -> str:
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
        return self.set_atleast(references, len(references), alias)
    
    def set_not(self, references: List[str], alias: Optional[str] = None) -> str:
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
        return self.set_atmost(references, 0, alias)
    
    def set_xor(self, references: List[str], alias: Optional[str] = None) -> str:
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
        ], alias)
    
    def set_xnor(self, references: List[str], alias: Optional[str] = None) -> str:
        """
            Add a composite constraint of an XNOR operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xy")
            >>> a = model.set_xnor(["x", "y"])
            >>> model.test({"x": 1+1j, "y": 0j}).get(a)
            0j

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_not([self.set_xor(references)], alias)
    
    def set_imply(self, condition: str, consequence: str, alias: Optional[str] = None) -> str:
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
        return self.set_or([self.set_not([condition]), consequence], alias)

    def set_equal(self, references: List[str], alias: Optional[str] = None) -> str:
        """
            Add a composite constraint of an EQUAL operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_or([
            self.set_and(references),
            self.set_not(references),
        ], alias)
    
    @lru_cache
    def to_polyhedron(self, double_binding: bool = True, **assume: Dict[str, complex]) -> tuple:

        """
            Constructs a polyhedron of matrix A and bias vector b,
            such that A.dot(x) >= b, where x is the vector of variables.
            Every composite variable A is its own column in the matrix where
            A -> (A's composite proposition) is true. 

            Parameters
            ----------
            assume : Dict[str, int]
                A dictionary of variables to assume tighter bounds.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> a = model.set_atleast("xyz", 1)
            >>> A, b = model.to_polyhedron()
            >>> np.array_equal(A, np.array([[1,1,1,-1]]))
            True
        """

        # From a aux variable A such that A <-> X, and the constraint X: c*x + c*y + c*z >= d, where n is the number of terms on left hand side, we have the following constraints:
        # `mx` is the maximum sum over the inner variables in a constraint X.
        # For atleast constraints:
        # 1) -dA + sum(+X) >= ±0             (-A v +X)
        # 2) +mx*A + sum(-X) >= -(d-1)       (+A v -X)

        # For atmost constraints:
        # 1) (-mx-d)A + sum(-X) >= -mx       (-A v +X)
        # 2) (-d+mn+1)A + sum(+X) >= (-d+1)     (+A v -X)

        # Create the matrix
        if double_binding:
            A = np.zeros((self._amat.shape[0] * 2, self._amat.shape[1]), dtype=np.int64)
            b = np.zeros(self._amat.shape[0] * 2, dtype=np.int64)
        else:
            A = np.zeros(self._amat.shape, dtype=np.int64)
            b = np.zeros(self._amat.shape[0], dtype=np.int64)

        # Find max of inner bounds for each composite
        ad = self._amat.dot(self._dvec)
        mn = ad.real
        mx = ad.imag

        # Extract `d` vector from bias
        d = -1 * self._bvec.real

        # Adjacent points
        adj_points = np.argwhere(self._amat == 1)

        # If no adjacent points, return empty matrix
        if adj_points.size == 0:
            return np.zeros((0, self._amat.shape[1]), dtype=np.int64), np.zeros(0, dtype=np.int64)

        # A -> X row indices
        A_X_ri = adj_points.T[0]
        A_X = np.arange(self._amat.shape[0])
        # X -> A row indices
        X_A = A_X + A_X.max() + 1
        X_A_ri = adj_points.T[0] + A_X.max() + 1

        # Fill the matrix with adjacency points
        A[A_X_ri, adj_points.T[1]] = -1
        if double_binding:
            A[X_A_ri, adj_points.T[1]] = -1

        # Assign 1 instead of -1 to the at least A -> X, and at most constraints for X -> A.
        A[A_X[~self._nvec], :] *= -1
        if double_binding:
            A[X_A[self._nvec], :] *= -1

        # Composite index in matrix
        cidx = np.array(list(self._imap.values()))[self._cvec]

        # Set coef and bias for at least aux variable A->X
        # 1) (-d+mn)A + sum(-X) >= mn            (-A v +X)
        A[A_X[~self._nvec], cidx[~self._nvec]] = -d[~self._nvec]+mn[~self._nvec]
        b[A_X[~self._nvec]] = mn[~self._nvec]

        # Set coef and bias for at least aux var X->A
        # 2) +mx*A + sum(-X) >= -(d-1)       (+A v -X)
        if double_binding:
            A[X_A[~self._nvec], cidx[~self._nvec]] = mx[~self._nvec]-(d[~self._nvec] - 1)
            b[X_A[~self._nvec]] = -(d[~self._nvec] - 1)

        # Set coef and bias for at most aux variable A->X
        # 1) (-mx-d)A + sum(-X) >= -mx       (-A v +X)
        A[A_X[self._nvec], cidx[self._nvec]] = -mx[self._nvec] - d[self._nvec]
        b[A_X[self._nvec]] = -mx[self._nvec]

        # Set coef and bias for at most aux var X->A
        # 2) (-d-mn+1)A + sum(+X) >= (-d+1)     (+A v -X)
        if double_binding:
            A[X_A[self._nvec], cidx[self._nvec]] = -d[self._nvec] + 1 - mn[self._nvec]
            b[X_A[self._nvec]] = -d[self._nvec] + 1

        # Fix constant variables
        for i, vs in groupby(
            sorted(
                map(
                    lambda x: (x[0], int(x[1].real)), 
                    filter(
                        lambda x: x[1].real == x[1].imag, 
                        assume.items()
                    )
                ),
                key=lambda x: x[1] 
            ), 
            key=lambda x: x[1]
        ):
            v = list(map(lambda x: x[0], vs))
            a = np.zeros(A.shape[1], dtype=np.int64)
            a[np.array(list(map(self._col, v)))] = 1
            if i > 0 or i < 0:
                _b = a.sum() * i
            else:
                _b = 0

            A = np.vstack([A, a])
            b = np.append(b, _b)

        # Set bounds for integer variables
        int_vars = list(set(np.argwhere((self._dvec.real != 0) | (self._dvec.imag != 1)).T[0].tolist()))

        # Declare new constraints for upper and lower bound for integer variables
        A_int = np.zeros((len(int_vars) * 2, A.shape[1]), dtype=np.int64)
        b_int = np.zeros((len(int_vars) * 2, ), dtype=np.int64)

        # Setup dvec as real and imag parts
        d_real = self._dvec.real
        d_imag = self._dvec.imag

        # Also, if fix has tighter bounds set we use them instead
        for i, bound in filter(
            lambda x: (x[1].real != x[1].imag), 
            starmap(
                lambda k,v: (self._col(k), v),
                assume.items()
            )
        ):
            if bound.real > d_real[i]:
                d_real[i] = bound.real
            elif bound.imag < d_imag[i]:
                d_imag[i] = bound.imag
        
        # Lower bound for integers..
        A_int[np.arange(len(int_vars) * 2, step=2), int_vars] = 1
        b_int[np.arange(len(int_vars) * 2, step=2)] = d_real[int_vars]
        
        # Upper bound for integers..
        A_int[np.arange(len(int_vars) * 2, step=2) + 1, int_vars] = -1
        b_int[np.arange(len(int_vars) * 2, step=2) + 1] = -1 * d_imag[int_vars]

        # Add them onto polyhedron
        A = np.vstack([A, A_int])
        b = np.append(b, b_int)

        return A, b
    
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
    
    def solve(self, objectives: List[Dict[str, int]], assume: Dict[str, complex], solver: Solver) -> List[Dict[str, complex]]:
        """
            Solves the model with the given objectives.

            Parameters
            ----------
            objectives : List[Dict[str, int]]
                The objectives to solve for.

            assume : Dict[str, complex]
                Assume new bounds for variables.

            solver : Solver
                The solver to use.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> a = model.set_atleast("xyz", 1)
            >>> model.solve([{"x": 0, "y": 1, "z": 0}], {}, Solver.GLPK)
            [{'x': 0j, 'y': 1+1j, 'z': 0j}]

            Returns
            -------
            List[Dict[str, complex]]
                The solutions for the objectives.
        """
        A, b = self.to_polyhedron(double_binding=True, **assume)
        variables = self._col_vars
        obj_mat = np.zeros((len(objectives), len(variables)), dtype=np.int64)
        for i, obj in enumerate(objectives):
            obj_mat[i, [self._col(k) for k in obj]] = list(obj.values())

        if solver == Solver.GLPK:
            from pldag.solver.glpk_solver import solve_lp
            solutions = solve_lp(A, b, obj_mat, set(np.argwhere((self._dvec.real != 0) | (self._dvec.imag != 1)).T[0].tolist()))
        else:
            raise ValueError(f"Solver `{solver}` not installed.")
        
        return list(
            map(
                lambda solution: dict(
                    zip(
                        variables, 
                        map(
                            lambda i: complex(i,i),
                            solution
                        )
                    )
                ),
                solutions
            )
        )
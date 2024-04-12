import numpy as np
from hashlib import sha1
from itertools import groupby, repeat
from functools import partial
from typing import Dict, List, Optional, Set
from graphlib import TopologicalSorter

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
    
    def _icol(self, id: str) -> int:
        """
            Returns the column index of the given ID.
        """
        return self._imap[id]
    
    def _irow(self, id: str) -> int:
        """
            Returns the row index of the given ID.
        """
        return self._icol(id) - (~self._cvec).sum()
    
    def get(self, *id: str) -> np.ndarray:
        """Get the bounds of the given ID(s)"""
        return self._dvec[list(map(self._icol, id))]
    
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
                np.argwhere(self._amat[self._irow(id)] == 1).T[0]
            )
        )
    
    def negated(self, id: str) -> bool:
        """Get the negated state of the given id"""
        return bool(self._nvec[self._irow(id)])
    
    def delete(self, id: str) -> None:
        """Delete the given id"""
        idx = self._icol(id)

        # Remove the primitive prime for all composite primes
        # May be changed later. Now we just set everything to 0
        # but keep the rows and columns
        self._amat[:, idx] = 0
    
    def set_primitive(self, id: str, bound: complex = complex(0,1)) -> str:
        """Add a primitive prime factor matrix"""
        if id in self._imap:
            # Update value if already in map
            self._dvec[self._icol(id)] = bound
        else:
            self._amat = np.hstack((self._amat, np.zeros((self._amat.shape[0], 1), dtype=np.uint8)))
            self._dvec = np.append(self._dvec, bound)
            self._cvec = np.append(self._cvec, False)
            self._imap[id] = self._amat.shape[1] - 1

        return id

    def set_primitives(self, ids: List[str], bound: complex = complex(0,1)) -> List[str]:
        """Add multiple primitive prime factor matrices"""
        for id in ids:
            self.set_primitive(id, bound)
        return ids

    def _set_gelineq(self, children: list, bias: int, negate: bool = False) -> str:
        """
            Add a composite constraint of at least `value`.
        """
        _bias = complex(*repeat(bias * (1-negate) + (bias + 1) * negate * -1, 2))
        _id = sha1(("".join(sorted(children)) + str(negate)).encode()).hexdigest()
        if _id in self._imap:
            arr = np.zeros(self._amat.shape[1], dtype=np.int8)
            arr[[self._imap[child] for child in children]] = 1
            self._amat[self._irow(_id)] = arr
            self._bvec[self._irow(_id)] = _bias
            self._nvec[self._irow(_id)] = negate
            self._dvec[self._imap[_id]] = complex(0, 1)
            self._cvec[self._imap[_id]] = True
        else:
            self._amat = np.pad(self._amat, ((0, 1), (0, 1)), mode='constant')
            arr = np.zeros(self._amat.shape[1], dtype=np.int8)
            arr[[self._icol(child) for child in children]] = 1
            self._amat[self._amat.shape[0] - 1] = arr
            self._dvec = np.append(self._dvec, complex(0, 1))
            self._bvec = np.append(self._bvec, _bias)
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

    def propagate(self):
        """
            Propagates the graph, stores and returns the propagated bounds.
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._bvec
        C: np.ndarray = self._cvec
        D: np.ndarray = self._dvec
        F: np.ndarray = self._nvec

        # Propagate the graph and store the result as new bounds
        self._dvec = self._prop_algo(A, B, C, D, F, np.zeros(D.shape[0], dtype=bool))

    def test(self, query: Dict[str, complex]) -> Dict[str, complex]:

        """
            Propagates the graph and returns the selected result.

            query:  what nodes and their bound to start propagating from

            Returns the selected propagated bounds.
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._bvec
        C: np.ndarray = self._cvec
        D: np.ndarray = self._dvec.copy()
        F: np.ndarray = self._nvec

        # Query translation into primes
        qprimes = np.zeros(D.shape[0], dtype=bool)
        qprimes[[self._imap[q] for q in query]] = True

        # Replace the observed bounds
        D[qprimes] = np.array(list(query.values()))

        return dict(zip(self._imap.keys(), self._prop_algo(A, B, C, D, F, qprimes)))
    
    def set_atleast(self, children: list, value: int) -> str:
        return self._set_gelineq(children, -1 * value, False)
    
    def set_atmost(self, children: list, value: int) -> str:
        return self._set_gelineq(children, -1 * (value + 1), True)
    
    def set_or(self, references: List[str]) -> str:
        return self.set_atleast(references, 1)
    
    def set_and(self, references: List[str]) -> str:
        return self.set_atleast(references, len(references))
    
    def set_not(self, references: List[str]) -> str:
        return self.set_atmost(references, 0)
    
    def set_xor(self, references: List[str]) -> str:
        return self.set_and([
            self.set_atleast(references, 1),
            self.set_atmost(references, 1),
        ])
    
    def set_imply(self, condition: str, consequence: str) -> str:
        return self.set_or([self.set_not([condition]), consequence])

    def to_polyhedron(self, fix: dict = {}) -> tuple:

        """
            Returns a polyhedron of a tuple (A, b, variables) representation of the PL-DAG.
            The variables list consists of tuples of (ID, bound as complex number).

            fix: dict of ids to fix in the polyhedron. E.g. {"A": 1, "B": 0} fix A=1 and B=0.
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
            a[np.array(list(map(self._icol, v)))] = 1
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
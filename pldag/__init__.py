import numpy as np
from itertools import chain
from functools import partial
from typing import Dict, List

class PLDAG:

    """
        "Primitive Logic Directed Acyclic Graph" data structure, or "PL-DAG" for short, is fundamentally a Directed Acyclic Graph (DAG) where each node represents a logical relationship, and the leaf nodes correspond to literals. 
        Each node in the graph encapsulates information about how its incoming nodes or leafs are logically related. For instance, a node might represent an AND operation, meaning that if it evaluates to true, all its incoming nodes or leafs must also evaluate to true.
        
        In summary, this data structure combines elements of a DAG with a logic network, utilizing prime numbers to encode relationships and facilitate operations within the graph.
    """

    def __init__(self):
        """
            max_deps: maximum number of dependencies allowed
        """
        # Adjacency matrix. Each entry is a boolean (0/1) indicating if there is a dependency
        self._amat = np.zeros((0, 0),               dtype=np.int64)
        # Boolean vector indicating if negated
        self._nvec = np.zeros((0, 0),               dtype=np.int64)
        # Complex vector representing bounds of complex number data type
        self._dvec = np.zeros((0, 0),               dtype=complex)
        # Maps alias to indexs
        self._amap = {}
        # Keep tracks of the aliases to ignore/avoid
        self._ign = set()

    @property
    def bounds(self) -> np.ndarray:
        """Get the bounds of all aliases"""
        return self._dvec
    
    @property
    def aliases(self) -> List[str]:
        """Get all aliases"""
        return list(filter(lambda a: a not in self._ign, self._amap.keys()))
    
    def get(self, *alias: str) -> np.ndarray:
        """Get the bounds of the given alias"""
        return self._dvec[[self._amap[a] for a in alias]]
    
    def dependencies(self, alias: str) -> List[str]:
        """Get the dependencies of the given alias"""
        return list(
            filter(
                lambda x: x != alias,
                map(
                    lambda x: list(self._amap)[x],
                    np.argwhere(self._amat[self._amap[alias]] == 1).T[0]
                )
            )
        )
    
    def negated(self, alias: str) -> bool:
        """Get the negated state of the given alias"""
        return bool(self._nvec[self._amap[alias]])
    
    def delete(self, alias: str) -> None:
        """Delete the given alias"""
        idx = self._amap[alias]

        # Remove the primitive prime for all composite primes
        # May be changed later. Now we just set everything to 0
        # but keep the rows and columns
        self._amat[idx, :] = 0
        self._amat[:, idx] = 0
        self._ign.add(alias)
    
    def set_primitive(self, alias: str, bound: complex = complex(0,1), hide: bool = False) -> None:
        """Add a primitive prime factor matrix"""
        if alias in self._amap:
            # Update value if already in map
            self._dvec[self._amap[alias]] = bound
        else:
            self._amat = np.pad(self._amat, ((0, 1), (0, 1)), mode='constant')
            self._dvec = np.append(self._dvec, bound)
            self._amap[alias] = self._amat.shape[0] - 1
            self._nvec = np.append(self._nvec, 0)

        if hide:
            self._ign.add(alias)

    def set_primitives(self, aliases: List[str], bound: complex = complex(0,1)) -> None:
        """Add multiple primitive prime factor matrices"""
        for alias in aliases:
            self.set_primitive(alias, bound)

    def set_composite(self, alias: str, children: list, bias: int, negate: bool = False) -> None:
        """
            Add a composite prime factor matrix.
            If alias already registred, only the prime factor matrix is updated.
        """
        valued_childrend = list(chain(map(lambda x: (x, complex(0,1), False), children), [(str(bias), complex(bias, bias), True)]))
        for child_alias, value, hide in valued_childrend:
            if child_alias not in self._amap:
                self.set_primitive(child_alias, value, hide)
        if alias in self._amap:
            arr = np.zeros(self._amat.shape[1], dtype=np.int64)
            arr[[self._amap[child] for child,_,_ in valued_childrend]] = 1
            self._amat[self._amap[alias]] = arr
            self._dvec[self._amap[alias]] = complex(0, 1)
            self._nvec[self._amap[alias]] = negate * 1
        else:
            self._amat = np.pad(self._amat, ((0, 1), (0, 1)), mode='constant')
            arr = np.zeros(self._amat.shape[1], dtype=np.int64)
            arr[[self._amap[child] for child,_,_ in valued_childrend]] = 1
            self._amat[self._amat.shape[0] - 1] = arr
            self._dvec = np.append(self._dvec, complex(0, 1))
            self._nvec = np.append(self._nvec, negate * 1)
            self._amap[alias] = self._amat.shape[0] - 1

    @staticmethod
    def _prop_algo(A: np.ndarray, F: np.ndarray, B: np.ndarray, max_iterations: int = 100):

        """
            Propagation algorithm.

            A: the adjacency matrix
            F: the negation array
            initial_B: the initial bounds
            max_iterations: the maximum number of iterations

            Returns the propagated bounds.
        """

        def _prop_once(A, C, F, B):
            r = A.dot(B) 
            rf = (F * -1j * np.conj(r) + (1-F) * r) * ~C
            return C*B + ((rf.real >= 0) + 1j*(rf.imag >= 0)) * ~C

        C = (A == 0).all(axis=1)
        prop = partial(_prop_once, A, C, F)
        
        previous_B = B
        for _ in range(max_iterations):
            new_B = prop(previous_B)
            if (new_B == previous_B).all():
                return new_B
            previous_B = new_B
        
        raise Exception(f"Maximum iterations ({max_iterations}) reached without convergence.")

    def propagate(self):
        """
            Propagates the graph, stores and returns the propagated bounds.
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._dvec
        F: np.ndarray = self._nvec

        # Propagate the graph and store the result as new bounds
        self._dvec = self._prop_algo(A, F, B)

    def test(self, query: Dict[str, complex], select: List[str]) -> np.ndarray:

        """
            Propagates the graph and returns the selected result.

            query:  what nodes and their bound to start propagating from
            select: what nodes to return the propagated bounds for

            Returns the selected propagated bounds.
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._dvec
        F: np.ndarray = self._nvec

        # Query translation into primes
        qprimes = np.zeros(B.shape[0], dtype=bool)
        qprimes[[self._amap[q] for q in query]] = True

        # Replace the observed bounds
        B[qprimes] = np.array(list(query.values()))

        return self._prop_algo(A, F, B)[[self._amap[s] for s in select]]

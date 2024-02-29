import numpy as np
from itertools import product, islice, chain
from typing import Dict, List

class PLDAG:

    """
        "Prime Logic Directed Acyclic Graph" data structure, or "PL-DAG" for short, is fundamentally a Directed Acyclic Graph (DAG) where each node represents a logical relationship, and the leaf nodes correspond to literals. 
        Each node in the graph encapsulates information about how its incoming nodes or leafs are logically related. For instance, a node might represent an AND operation, meaning that if it evaluates to true, all its incoming nodes or leafs must also evaluate to true.
        What sets this structure apart is that each node and leaf is associated with a prime number, and each node computes a composite prime number based on the prime numbers of its incoming nodes or leafs. This prime-based system allows for efficient manipulation and traversal of the graph while preserving logical relationships.
        In summary, this data structure combines elements of a DAG with a logic network, utilizing prime numbers to encode relationships and facilitate operations within the graph.
    """

    PRIME_HEIGHT = 15
    PRIMES = np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    ])

    def __init__(self, prime_width: int = 3):
        if not prime_width > 0:
            raise ValueError("prime_width must be a positive integer")
        
        # Width of prime combination. Restricts the possible number of variables.
        self.PRIME_WIDTH = prime_width
        # Prime combination matrix (primitive prime combination, composite prime combination)
        self._pmat = np.empty((0, 2, self.PRIME_WIDTH), dtype=np.uint64)
        # Flip bool and bound matrix (flip, complex bound)
        self._dmat = np.empty((0, 2),                   dtype=np.complex128)
        # Maps alias to prime index
        self._amap = {}

    def _next_prime_combinations(self, start: int, end: int):
        """Get the next prime combinations for the given range of indices"""
        # Note we add one to skip the first prime combination of (1, 1, 1, ...)
        return np.array(list(islice(product(self.PRIMES[:self.PRIME_HEIGHT], repeat=self.PRIME_WIDTH), start+1, end+1)), dtype=np.uint64)

    @property
    def n_max(self) -> int:
        """Maximum number of variables possible"""
        return self.PRIME_HEIGHT ** self.PRIME_WIDTH

    @property
    def n_left(self) -> int:
        """How many variables are left to be used"""
        return self.n_max - self._pmat.shape[0]
    
    @property
    def n_used(self) -> int:
        """How many variables are used"""
        return self._pmat.shape[0]
    
    @property
    def bounds(self) -> np.ndarray:
        """Get the bounds of all aliases"""
        return self._dmat.T[1]
    
    def get(self, *alias: str) -> np.ndarray:
        """Get the bounds of the given alias"""
        return self._dmat[[self._amap[a] for a in alias]].T[1]
    
    def dependencies(self, alias: str) -> List[str]:
        """Get the dependencies of the given alias"""
        return list(
            filter(
                lambda x: x != alias,
                map(
                    lambda x: list(self._amap)[x],
                    np.argwhere(
                        (np.mod(self._pmat[self._amap[alias]][1], self._pmat[:,0]) == 0).all(axis=1)
                    ).T[0]
                )
            )
        )
    
    def negated(self, alias: str) -> bool:
        """Get the negated state of the given alias"""
        return bool(self._dmat[self._amap[alias]][0])
    
    def delete(self, alias: str) -> None:
        """Delete the given alias"""
        idx = self._amap[alias]
        primitive_prime = self._pmat[idx][0]

        # Remove the primitive prime for all composite primes
        for i in np.argwhere((np.mod(self._pmat[:,1], primitive_prime) == 0).all(axis=1)).T[0]:
            self._pmat[i][1] = np.true_divide(self._pmat[i][1], primitive_prime).astype(np.uint64)

        self._pmat = np.delete(self._pmat, idx, axis=0)
        self._dmat = np.delete(self._dmat, idx, axis=0)
        self._amap = {k: v - 1 if v > idx else v for k, v in self._amap.items() if k != alias}
    
    def set_primitive(self, alias: str, bound: complex = complex(0,1)) -> None:
        """Add a primitive prime factor matrix"""
        if alias in self._amap:
            self._dmat[self._amap[alias]][1] = bound
        else:
            new_primitive_prime = self._next_prime_combinations(self._pmat.shape[0], self._pmat.shape[0] + 1)[0]
            self._pmat = np.append(self._pmat, np.array([new_primitive_prime, new_primitive_prime], dtype=np.uint64)[None], axis=0)
            self._dmat = np.append(self._dmat, np.array([0, bound])[None], axis=0)
            self._amap[alias] = len(self._pmat) - 1

    def set_primitives(self, aliases: List[str], bound: complex = complex(0,1)) -> None:
        """Add multiple primitive prime factor matrices"""
        for alias in aliases:
            self.set_primitive(alias, bound)

    def set_composite(self, alias: str, children: list, bias: int, negate: bool = False) -> None:
        """
            Add a composite prime factor matrix.
            If alias already registred, only the prime factor matrix is updated.
        """
        for child in children:
            if child not in self._amap:
                self.set_primitive(child, (0,1))
        self.set_primitive(f"{bias}", complex(bias, bias))
        composite_prime = np.lcm.reduce([self._pmat[self._amap[child]][0] for child in chain(children, [str(bias)])])
        if alias in self._amap:
            self._pmat[self._amap[alias]][1] = composite_prime
            self._dmat[self._amap[alias]] = np.array([negate * 1, complex(0, 1)])
        else:
            new_primitive_prime = self._next_prime_combinations(self._pmat.shape[0], self._pmat.shape[0] + 1)[0]
            self._pmat = np.append(self._pmat, np.array([new_primitive_prime, composite_prime], dtype=np.uint64)[None], axis=0)
            self._amap[alias] = len(self._pmat) - 1
            self._dmat = np.append(self._dmat, np.array([negate * 1, complex(0, 1)])[None], axis=0)

    @staticmethod
    def _prop_algo(D: np.ndarray, P: np.ndarray, W: np.ndarray, F: np.ndarray):

        """
            Propagation algorithm.

            D: complex bounds matrix
            P: primitive prime combination matrix
            W: composite prime combination matrix
            F: flip bool matrix

            Returns the propagated bounds.
        """

        # Initial nodes are the primitive nodes (no incoming edge)
        N = (P == W).all(axis=1)

        # Save primitive nodes for later use
        _P = N.copy()

        # Explored nodes
        L = np.zeros(P.shape[0], dtype=bool)

        # Edges matrix
        E = (W % P[:,None] == 0).all(axis=2) #& ~(P == W).all(axis=1)[None]

        while N.any():
            # Find nodes M which has all it's edges in N
            M = (E == E & N[:,None]).all(axis=0) & ~N
            if not M.any():
                break
            # Select nodes we are going to use the bounds from
            cb = D[:,None] * E[:,M]
            # Flip bounds where suppose to flip
            cbf = F[M] * -1j * np.conj(cb) + (1-F[M]) * cb
            # Sum up for each node
            cbf_sum = cbf.sum(axis=0)
            # Check each part and create a new complex number
            D[M] = ((cbf_sum.real >= 0) + 1j*(cbf_sum.imag >= 0)) * ~_P[M] + cbf_sum*_P[M]
            # Append the nodes that was connected to M to L as explored
            L |= E[:,M].any(axis=1)
            # And remove them from N while adding the new nodes to N
            N = (M | N) & ~E[:,M].any(axis=1)
            
        return D

    def propagate(self):
        """
            Propagates the graph, stores and returns the propagated bounds.
        """
        D: np.ndarray = self._dmat[:,1]
        P: np.ndarray = self._pmat[:,0]
        W: np.ndarray = self._pmat[:,1]
        F: np.ndarray = self._dmat.T[0]

        # Propagate the graph and store the result as new bounds
        self._dmat[:, 1] = self._prop_algo(D, P, W, F)

    def test(self, query: Dict[str, complex], select: List[str]) -> np.ndarray:

        """
            Propagates the graph and returns the selected result.

            query:  what nodes and their bound to start propagating from
            select: what nodes to return the propagated bounds for

            Returns the selected propagated bounds.
        """
        D: np.ndarray = self._dmat[:,1].copy()
        P: np.ndarray = self._pmat[:,0]
        W: np.ndarray = self._pmat[:,1]
        F: np.ndarray = self._dmat.T[0]

        # Query translation into primes
        qprimes = np.zeros(P.shape[0], dtype=bool)
        qprimes[[self._amap[q] for q in query]] = True

        # Replace the observed bounds
        D[qprimes] = np.array(list(query.values()))

        return self._prop_algo(D, P, W, F)[[self._amap[s] for s in select]]

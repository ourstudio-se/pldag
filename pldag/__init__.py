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

    PRIME_HEIGHT = 17
    PRIMES = np.array([
        1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
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
        # Flip bool and bound matrix (flip, lower, upper)
        self._dmat = np.empty((0, 3),                   dtype=np.int64)
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
    
    def get(self, alias: List[str]) -> np.ndarray:
        """Get the bounds of the given alias"""
        return self._dmat[self._amap[alias]][1:]
    
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
    
    def set_primitive(self, alias: str, bound: tuple = (0,1)) -> None:
        """Add a primitive prime factor matrix"""
        if len(bound) != 2:
            raise ValueError(f"bound must be of size 2")
        
        if alias in self._amap:
            self._dmat[self._amap[alias]][1:] = bound
        else:
            new_primitive_prime = self._next_prime_combinations(self._pmat.shape[0], self._pmat.shape[0] + 1)[0]
            self._pmat = np.append(self._pmat, np.array([new_primitive_prime, new_primitive_prime], dtype=np.uint64)[None], axis=0)
            self._dmat = np.append(self._dmat, np.array([0, *bound], dtype=np.int64)[None], axis=0)
            self._amap[alias] = len(self._pmat) - 1

    def set_primitives(self, aliases: List[str], bound: tuple = (0,1)) -> None:
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
        self.set_primitive(f"{bias}", (bias, bias))
        composite_prime = np.lcm.reduce([self._pmat[self._amap[child]][0] for child in chain(children, [str(bias)])])
        if alias in self._amap:
            self._pmat[self._amap[alias]][1] = composite_prime
            self._dmat[self._amap[alias]] = np.array([negate * 1, 0, 1], dtype=np.int64)[None]
        else:
            new_primitive_prime = self._next_prime_combinations(self._pmat.shape[0], self._pmat.shape[0] + 1)[0]
            self._pmat = np.append(self._pmat, np.array([new_primitive_prime, composite_prime], dtype=np.uint64)[None], axis=0)
            self._dmat = np.append(self._dmat, np.array([negate * 1, 0, 1], dtype=np.int64)[None], axis=0)
            self._amap[alias] = len(self._pmat) - 1

    @staticmethod
    def _prop_algo(D: np.ndarray, P: np.ndarray, W: np.ndarray, F: np.ndarray, M: np.ndarray):
        """
            A pure numpy implementation of the propagation algorithm

            D: Bounds matrix
            P: Primitive Prime matrix
            W: Composite Prime matrix
            F: Flip vector
            M: Mask vector
        """
        leafs = (P == W).all(axis=1)
        visited = np.zeros(D.shape[0], dtype=bool)
        while M.any() and not np.array_equal(visited, M):

            # We need to keep track of visited nodes to avoid infinite loops
            visited = M.copy()
            
            # We retreive all child bounds by checking which prime numbers are
            # factors of the composite prime numbers. We use the modulo operator.
            # Since the primes are really a combination of prime numbers we need to
            # check that all dimensions are factors of the composite prime number.
            child_bounds = (D[:,None].T * (np.mod(W[M][:, None], P) == 0).all(axis=2))
            
            # Flip those that should be flipped (meaning negated is true). Before we sum all
            # child bounds we flip them so e.g. (0,1) becomes (-1,0). And then
            # propagating by checking if the sum of all bounds is greater or equal to zero. 
            # By the end we do a product with 1 to turn the boolean into integers.
            D[M] = (((child_bounds - child_bounds.sum(axis=0) * F[M][:,None]).sum(axis=2) >= 0) * 1).T

            # Query step finding new nodes to explore. Since prime numbers are composites
            # of themselves we need to exclude them from the query.
            M = ~leafs & (np.mod(W[:,None], P[M]) == 0).all(axis=2).any(axis=1)

        return D
    
    def propagate(self) -> np.ndarray:
        """
            Propagates the graph and returns the propagated bounds.
        """
        D: np.ndarray = self._dmat[:,1:]
        P: np.ndarray = self._pmat[:,0]
        W: np.ndarray = self._pmat[:,1]
        F: np.ndarray = self._dmat.T[0]

        # Find first node level to start propagating from
        # Is done by first finding leafs..
        _msk = (W == P).all(axis=1)

        # ..and then finding the nodes connected to the leafs
        msk = ~_msk & (np.mod(W[:,None], P[_msk]) == 0).all(axis=2).any(axis=1)

        return self._prop_algo(D, P, W, F, msk)

    def test(self, query: Dict[str, tuple], select: List[str]) -> np.ndarray:

        """
            Propagates the graph and returns the selected result.

            query: what nodes and their bound to start propagating from
            select: what nodes to return the propagated bounds for

            Returns the selected propagated bounds.
        """
        D: np.ndarray = self._dmat[:,1:].copy()
        P: np.ndarray = self._pmat[:,0]
        W: np.ndarray = self._pmat[:,1]
        F: np.ndarray = self._dmat.T[0]

        # Query translation into primes
        qprimes = np.zeros(P.shape[0], dtype=bool)
        qprimes[[self._amap[q] for q in query]] = True

        # Replace the observed bounds
        D[qprimes] = np.array(list(query.values()), dtype=np.int64)

        # One step forward before starting the loop
        # We don't need to propagate leaf nodes
        msk = ~qprimes & (np.mod(W[:,None], P[qprimes]) == 0).all(axis=2).any(axis=1)

        return self._prop_algo(D, P, W, F, msk)[[self._amap[s] for s in select]]

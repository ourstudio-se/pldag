import numpy as np
from hashlib import sha256
from itertools import chain, groupby
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
        self._amat = np.zeros((0, 0),   dtype=np.int64)
        # Boolean vector indicating if negated
        self._nvec = np.zeros((0, ),    dtype=np.int64)
        # Complex vector representing bounds of complex number data type
        self._dvec = np.zeros((0, ),    dtype=complex)
        # Maps id's to index
        self._imap = {}
        # Maps alias to id's
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
        return list(filter(lambda a: a not in self._ign, self._imap.keys()))
    
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
    
    @property
    def _rev_amap(self) -> dict:
        # Reversing the dictionary
        reversed_dict = {}
        for key, value in self._amap.items():
            reversed_dict.setdefault(value, set()).add(key)
        return reversed_dict
    
    def _index_from_alias(self, alias: str) -> str:
        return self._imap[self._amap.get(alias, alias)]
    
    def get(self, *alias: str) -> np.ndarray:
        """Get the bounds of the given alias"""
        return self._dvec[list(map(self._index_from_alias, alias))]
    
    def dependencies(self, alias: str) -> Dict[str, Set[str]]:
        """
            Get the dependencies of the given alias or ID.
            A mapping of ID -> {Alias} is returned, where each ID
            is a dependency to `alias`.
        """
        return dict(
            map(
                lambda x: (x, self._rev_amap[x]),
                map(
                    lambda x: list(self._imap)[x],
                    np.argwhere(self._amat[self._index_from_alias(alias)] == 1).T[0]
                )
            )
        )
    
    def negated(self, alias: str) -> bool:
        """Get the negated state of the given alias"""
        return bool(self._nvec[self._index_from_alias(alias)])
    
    def delete(self, alias: str) -> None:
        """Delete the given alias"""
        idx = self._imap[alias]

        # Remove the primitive prime for all composite primes
        # May be changed later. Now we just set everything to 0
        # but keep the rows and columns
        self._amat[idx, :] = 0
        self._amat[:, idx] = 0
        self._ign.add(alias)
    
    def set_primitive(self, alias: str, bound: complex = complex(0,1), hide: bool = False) -> str:
        """Add a primitive prime factor matrix"""
        if alias in self._amap:
            # Update value if already in map
            self._dvec[self._index_from_alias(alias)] = bound
        else:
            self._amat = np.pad(self._amat, ((0, 1), (0, 1)), mode='constant')
            self._dvec = np.append(self._dvec, bound)
            self._imap[alias] = self._amat.shape[0] - 1
            self._amap[alias] = alias
            self._nvec = np.append(self._nvec, 0)

        if hide:
            self._ign.add(alias)

        return alias

    def set_primitives(self, aliases: List[str], bound: complex = complex(0,1)) -> List[str]:
        """Add multiple primitive prime factor matrices"""
        for alias in aliases:
            self.set_primitive(alias, bound)
        return aliases

    def _set_composite(self, valued_children: list, negate: bool = False, aliases: List[str] = []) -> str:
        for child_alias, value, hide in valued_children:
            if child_alias not in self._imap:
                self.set_primitive(child_alias, value, hide)

        _id = sha256(str(valued_children).encode()).hexdigest()
        if _id in self._imap:
            arr = np.zeros(self._amat.shape[1], dtype=np.int64)
            arr[[self._imap[child] for child,_,_ in valued_children]] = 1
            self._amat[self._imap[_id]] = arr
            self._dvec[self._imap[_id]] = complex(0, 1)
            self._nvec[self._imap[_id]] = negate * 1
        else:
            self._amat = np.pad(self._amat, ((0, 1), (0, 1)), mode='constant')
            arr = np.zeros(self._amat.shape[1], dtype=np.int64)
            arr[[self._imap[child] for child,_,_ in valued_children]] = 1
            self._amat[self._amat.shape[0] - 1] = arr
            self._dvec = np.append(self._dvec, complex(0, 1))
            self._nvec = np.append(self._nvec, negate * 1)
            self._imap[_id] = self._amat.shape[0] - 1
        
        if aliases:
            for alias in aliases:
                self._amap[alias] = _id
        else:
            self._amap[_id] = _id

        return _id
    
    def set_composite(self, children: list, bias: int, negate: bool = False, aliases: List[str] = []) -> str:
        """
            Add a composite prime factor matrix.
            If alias already registred, only the prime factor matrix is updated.
        """
        valued_children = sorted(
            chain(
                map(
                    lambda x: (
                        self._amap.get(x, x),
                        complex(0,1), 
                        False,
                    ), 
                    children
                ), 
                [
                    (
                        str(bias), 
                        complex(bias, bias), 
                        True,
                    )
                ]
            ), 
            key=lambda x: x[0]
        )
        return self._set_composite(valued_children, negate, aliases)

    @staticmethod
    def _prop_algo(A: np.ndarray, F: np.ndarray, B: np.ndarray, forced: np.ndarray, max_iterations: int = 100):

        """
            Propagation algorithm.

            A: the adjacency matrix
            F: the negation array
            B: the initial bounds
            forced: boolean (won't change during propagation) (optional)
            max_iterations: the maximum number of iterations (optional)

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
            new_B = forced * B + ~forced * prop(previous_B)
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
        self._dvec = self._prop_algo(A, F, B, np.zeros(B.shape[0], dtype=bool))
    
    def _test(self, query: Dict[str, complex]) -> np.ndarray:
        
        """
            Propagates the graph and returns the propagated bounds.

            query: what nodes and their bound to start propagating from

            Returns the propagated bounds.
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._dvec.copy()
        F: np.ndarray = self._nvec

        # Query translation into primes
        qprimes = np.zeros(B.shape[0], dtype=bool)
        qprimes[[self._imap[q] for q in query]] = True

        # Replace the observed bounds
        B[qprimes] = np.array(list(query.values()))

        return self._prop_algo(A, F, B, qprimes)

    def test(self, query: Dict[str, complex], select: Optional[List[str]] = None) -> Dict[str, complex]:

        """
            Propagates the graph and returns the selected result.

            query:  what nodes and their bound to start propagating from
            select: what nodes to return the propagated bounds for

            Returns the selected propagated bounds.
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._dvec.copy()
        F: np.ndarray = self._nvec

        # Query translation into primes
        qprimes = np.zeros(B.shape[0], dtype=bool)
        qprimes[[self._imap[q] for q in query]] = True

        # Replace the observed bounds
        B[qprimes] = np.array(list(query.values()))

        # Select all if none selected
        if select is None:
            select = set(self._amap.keys()).difference(self._ign)

        return dict(zip(select, self._prop_algo(A, F, B, qprimes)[[self._index_from_alias(s) for s in select]]))
    
    def lor(self, references: List[str], aliases: List[str] = []) -> str:
        return self.set_composite(references, -1, aliases=aliases)
    
    def land(self, references: List[str], aliases: List[str] = []) -> str:
        return self.set_composite(references, -len(references), aliases=aliases)
    
    def lnot(self, references: List[str], aliases: List[str] = []) -> str:
        return self.set_composite(references, 0, True, aliases=aliases)
    
    def lxor(self, references: List[str], aliases: List[str] = []) -> str:
        return self.land([
            self.lor(references),
            self.lnot(self.land(references)),
        ], aliases=aliases)
    
    def limply(self, condition: str, consequence: str, aliases: List[str] = []) -> str:
        return self.lor([self.lnot([condition]), consequence], aliases=aliases)

    def polyhedron(self, fix: dict = {}) -> tuple:

        """
            Returns a polyhedron of a tuple (A, b, variables) representation of the PL-DAG.
            The variables list consists of tuples of (ID, set of aliases, bound as complex number).

            fix: dict of aliases to fix in the polyhedron. E.g. {"A": 1, "B": 0} fix A=1 and B=0.
        """

        # Create the matrix (+1 for bias term)
        M = np.zeros((len(self._imap), len(self._imap)), dtype=np.int64)

        # Create the bias mask
        bias_msk = self._dvec.real == self._dvec.imag

        # Adjacent points
        adj_points = np.argwhere(self._amat * ~bias_msk == 1)

        # Fill the matrix
        M[adj_points.T[0], adj_points.T[1]] = 1

        # Flip the once that are negated
        M[self._nvec == 1] *= -1

        # Fill the diagonal
        np.fill_diagonal(M, -1*np.abs(M.sum(axis=1)))

        # Create the bias vector
        b_part_one = -1 * (self._dvec.real * bias_msk * self._amat).sum(axis=1)
        b_part_two = (self._nvec * -1) + ((1-self._nvec) * 1)
        b = b_part_one * b_part_two - (self._amat * ~bias_msk).sum(axis=1)

        # Remove all zero rows
        zero_rows_mask = ~np.all(M == 0, axis=1)
        A = M[zero_rows_mask]
        b = b[zero_rows_mask].astype(np.int64)

        # Fix the columns in `fix`
        for i, vs in groupby(zip(fix.values(), fix.keys()), key=lambda x: x[0]):
            v = list(map(lambda x: x[1], vs))
            a = np.zeros(A.shape[1], dtype=np.int64)
            a[np.array(list(map(self._index_from_alias, v)))] = 1
            if i > 0 or i < 0:
                _b = a.sum() * i
            else:
                _b = 0

            A = np.vstack([A, a])
            b = np.append(b, _b)

        # Remove all zero columns
        zero_cols_mask = ~np.all(A == 0, axis=0)
        A = A[:, zero_cols_mask]

        # Reverse index map
        rimap = dict(map(lambda x: (x[1], x[0]), self._imap.items()))

        # Create the variable list
        variables = np.array(
            list(
                map(
                    lambda i: (
                        rimap[i],
                        self._rev_amap[rimap[i]],
                        self._dvec[i],
                    ),
                    np.array(list(self._imap.values()))[zero_cols_mask]
                )
            )
        )

        return A, b, variables
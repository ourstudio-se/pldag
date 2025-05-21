import numpy as np

from enum import Enum
from hashlib import sha1
from itertools import groupby, starmap, chain, count
from functools import partial, lru_cache
from typing import Dict, List, Set, Optional, Tuple
from graphlib import TopologicalSorter
from pickle import dumps, loads, HIGHEST_PROTOCOL
from gzip import compress, decompress

class NoSolutionsException(Exception):
    pass

class MissingVariableException(Exception):
    
    def __init__(self, variable: str):
        super().__init__(f"Variable '{variable}' is missing.")

class MissingPrimitiveException(MissingVariableException):
    
    def __init__(self, primitive: str):
        super().__init__(f"Primitive '{primitive}' is missing.")

class MissingCompositeException(MissingVariableException):
    
    def __init__(self, composite: str):
        super().__init__(f"Composite '{composite}' is missing.")

class IsReferencedException(Exception):
        
    def __init__(self, variable_id: str):
        super().__init__(f"Variable '{variable_id}' is referenced by other composites.")

class FailedToCompileException(Exception):
    pass

class FailedToRebuildException(Exception):
    pass

class IsCorruptException(Exception):
    pass

class Solver(Enum):
    DEFAULT = "default"

class CompilationSetting(str, Enum):
    """
        Compilation settings for the PL-DAG.
    """
    # Compiles the model on each change.
    INSTANT = "instant"

    # Compiles the model on demand.
    ON_DEMAND = "on_demand"

class PLDAG:

    """
        "Primitive Logic Directed Acyclic Graph" data structure, or "PL-DAG" for short, is fundamentally a Directed Acyclic Graph (DAG) where each node represents a logical relationship, and the leaf nodes correspond to literals. 
        Each node in the graph encapsulates information about how its incoming nodes or leafs are logically related. For instance, a node might represent an AND operation, meaning that if it evaluates to true, all its incoming nodes or leafs must also evaluate to true.
        
        In summary, this data structure combines elements of a DAG with a logic network, utilizing prime numbers to encode relationships and facilitate operations within the graph.
    """

    def __init__(self, compilation_setting: CompilationSetting = CompilationSetting.INSTANT, dtype: np.dtype = np.int64):
        # Weighted adjacency matrix. Each entry is a coefficient indicating if there is a dependency and how strong it is.
        self._amat = np.zeros((0, 0),   dtype=dtype)
        # Complex vector representing bounds of complex number data type
        self._dvec = np.zeros((0, ),    dtype=complex)
        # Bias vector
        self._bvec = np.zeros((0, ),    dtype=complex)
        # Boolean vector indicating if the node is a composition
        self._cvec = np.zeros((0, ),    dtype=bool)
        # Keeps track of silent variables. Size is equal to the number of columns.
        self._svec = np.zeros((0, ),    dtype=bool)
        # Keeps track of variable type. Size is equal to the number of columns.
        self._tvec = np.empty((0, ),    dtype=object)
        # Maps id's to index
        self._imap = {}
        # Alias to id mapping
        self._amap = {}
        # Compilation setting
        self._compilation_setting = compilation_setting
        # Buffer for constraints
        self._buffer = {}

    def __hash__(self) -> int:
        return hash(self.sha1())
    
    def __eq__(self, other: "PLDAG") -> bool:
        return (
            self.sha1() == other.sha1()
            and np.array_equal(self._amat, other._amat)
            and np.array_equal(self._dvec, other._dvec)
            and np.array_equal(self._bvec, other._bvec)
            and np.array_equal(self._cvec, other._cvec)
            and np.array_equal(self._svec, other._svec)
            and np.array_equal(self._tvec, other._tvec)
            and self._imap == other._imap
            and self._amap == other._amap
            and self._compilation_setting == other._compilation_setting
        )

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

        """
            Topological sort of the adjacency matrix with composite and primitive names, bottoms-up.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> model.set_and("xyz")
            >>> list(model._toposort)
            ['x', 'z', 'y', 'da4ab8efc9b188b591115c3f376d0bc1ac6481ca']

            Returns
            -------
            iter
                The topological order.
        """

        return TopologicalSorter(
            dict(
                map(
                    lambda x: (
                        x[0], 
                        set(map(lambda y: y[1], x[1]))
                    ), 
                    groupby(
                        starmap(
                            lambda x,y: (
                                self._irow(x), 
                                self._icol(y),
                            ),
                            np.argwhere(self._amat), 
                        ),
                        key=lambda x: x[0]
                    )
                )
            )
        ).static_order()
    
    @property
    def _toposort_idx(self) -> iter:
        
        """
            Topological sort of the adjacency matrix with composite and primitive indices, bottoms-up.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> model.set_and("xyz")
            >>> list(model._toposort)
            [0, 1, 2, 3]

            Returns
            -------
            iter
                The topological order.
        """

        col_con_vec = np.argwhere(self._cvec).T[0]
        return TopologicalSorter(
            dict(
                map(
                    lambda x: (
                        x[0], 
                        set(map(lambda y: y[1], x[1]))
                    ), 
                    groupby(
                        starmap(
                            lambda x,y: (
                                col_con_vec[x], 
                                y,
                            ),
                            np.argwhere(self._amat), 
                        ),
                        key=lambda x: x[0]
                    )
                )
            )
        ).static_order()
    
    @staticmethod
    def _composite_id(coefficients: Dict[str,int], bias: int, unique: bool = False) -> str:
        """
            Create a composite ID from a list of children.
        """
        salt = np.random.bytes(8).hex() if unique else ""
        return sha1(("".join(map(lambda x: str(x), sorted(set(coefficients.items())))) + str(bias) + salt).encode()).hexdigest()
    
    def _corruption_middleware_runner(self, f, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            if self.is_corrupt():
                raise IsCorruptException("Model is corrupt. Run `try_rebuild`.")
            raise e
    
    @property
    def primitives(self) -> np.ndarray:
        return self._corruption_middleware_runner(lambda s: s._col_vars[~s._cvec])
    
    @property
    def integer_primitives(self) -> np.ndarray:
        return self.columns[(self._dvec.real != 0) | (self._dvec.imag != 1)]
    
    @property
    def composites(self) -> np.ndarray:
        return self._corruption_middleware_runner(lambda s: s._col_vars[s._cvec])
    
    @property
    def columns(self) -> np.ndarray:
        return self._col_vars
    
    @property
    def rows(self) -> np.ndarray:
        return self.composites
    
    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Get the adjacency matrix"""
        # Initialize a zero matrix with size len(columns) x len(columns)
        extended_matrix = np.zeros((self._amat.shape[1], self._amat.shape[1]))
        
        # Add the existing rows to the correct positions in the extended matrix
        for i,row in enumerate(self.composites):
            extended_matrix[self._col(row), :] = self._amat[i, :]
        
        return extended_matrix
    
    @property
    def _revimap(self) -> dict:
        """Get the reverse map"""
        return dict(map(lambda x: (x[1], x[0]), self._imap.items()))
    
    @property
    def _col_vars(self) -> np.ndarray:
        return np.array(list(self._imap.keys()))
    
    @property
    def _row_vars(self) -> np.ndarray:
        return self._corruption_middleware_runner(lambda s: np.array(list(s._imap.keys()))[s._cvec])
    
    @property
    def _inner_bounds(self) -> np.ndarray:
        return self._amat.dot(self._dvec) + self._bvec
    
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
        try:
            return self._imap[id]
        except KeyError:
            raise MissingVariableException(id)
    
    def _row(self, id: str) -> int:
        """
            Returns the row index of the given ID.
        """
        try:
            return self.composites.tolist().index(id)
        except ValueError:
            raise MissingCompositeException(id)
        
    def revert_from(self, model: "PLDAG") -> None:
        self._amat = model._amat
        self._dvec = model._dvec
        self._bvec = model._bvec
        self._cvec = model._cvec
        self._imap = model._imap
        self._amap = model._amap
        self._svec = model._svec
        self._tvec = model._tvec
        self._compilation_setting = model._compilation_setting

    def is_corrupt(self) -> bool:
        """
            Checks if the model is corrupt.

            Returns
            -------
            bool
                True if the model is corrupt, False otherwise.
        """
        return not (
            (len(self._imap) == self._amat.shape[1]) and
            (len(self._imap) == len(self._dvec)) and
            (self._amat.shape[1] == self._cvec.size) and
            (self._amat.shape[0] == self._bvec.size) and
            (self._amat.shape[1] == self._svec.size) and
            (self._amat.shape[1] == self._tvec.size)
        )
    
    def try_rebuild(self) -> "PLDAG":
        """
            Try to rebuild the model.

            Raises
            ------
            FailedToRebuildException
                If the model failed to rebuild.

            Returns
            -------
            PLDAG
                The rebuilt model.
        """
        if not (
            # Checks that all matrices and vectors correspond in size
            (self._amat.shape[1] == self._dvec.size) and
            (self._amat.shape[1] == self._cvec.size) and
            (self._amat.shape[0] == self._bvec.size) and
            (self._amat.shape[1] == self._svec.size) and
            (self._amat.shape[1] == self._tvec.size)
        ):
            raise FailedToRebuildException()

        # Rebuilds bottoms up, starting with the first composites (right above primitive layer)
        # and continues upwards until all composites are rebuilt
        rev_new_imap = self._revimap.copy()
        ji_corr_map = dict(zip(np.argwhere(self._cvec).T[0], count()))
        for j in filter(lambda i: self._cvec[i], self._toposort_idx):
            i = ji_corr_map[j]
            a_msk = self._amat[i] != 0
            a_idxs = np.argwhere(a_msk).T[0]
            coefficients = dict(
                zip(
                    map(rev_new_imap.get, a_idxs),
                    self._amat[i, a_idxs]
                )
            )
            id = self._composite_id(coefficients, int(self._bvec[i].real))
            rev_new_imap[j] = id

        rebuilt_model = self.copy()
        rebuilt_model._imap = dict(map(lambda x: (x[1], x[0]), rev_new_imap.items()))
        rebuilt_model._amap = dict(filter(lambda x: x[1] in rebuilt_model._imap, self._amap.items()))

        # Test if the model is corrupt
        if rebuilt_model.is_corrupt():
            raise FailedToRebuildException()

        return rebuilt_model
    
    def compile(self):
        """
            Compiles the model - sets all buffered constraints.

            Raises
            ------
            FailedToCompileException
                If the model failed to compile.
        """
        all_ids = []
        primitives = list(
            filter(
                lambda x: len(x[1][0]) == 0,
                self._buffer.items()
            )
        )

        # Set all primitives
        # Set first the existing ids
        existing_ids = list(filter(lambda x: x in self._imap, map(lambda x: x[0], primitives)))
        if existing_ids:
            self._dvec[[self._col(x) for x in existing_ids]] = list(map(lambda x: x[1][2], primitives))

        # Then add the new ids
        new_ids = list(filter(lambda x: x not in self._imap, map(lambda x: x[0], primitives)))
        if new_ids:
            self._amat = np.hstack((self._amat, np.zeros((self._amat.shape[0], len(new_ids)), dtype=self._amat.dtype)))
            self._dvec = np.append(self._dvec, np.array(list(map(lambda x: x[1][2], primitives))))
            self._cvec = np.append(self._cvec, np.array([False] * len(new_ids)))
            self._svec = np.append(self._svec, np.array([False] * len(new_ids)))
            self._tvec = np.append(self._tvec, np.array(["primitive"] * len(new_ids)))
            self._imap.update(dict(zip(new_ids, range(self._amat.shape[1] - len(new_ids), self._amat.shape[1]))))

        new_composites = list(
            starmap(
                lambda k,v: (k, ) + v,
                filter(
                    lambda x: (x[0] not in self._imap) and len(x[1][0]) > 0, 
                    self._buffer.items(),
                )
            )
        )
        if new_composites:

            # Copy the model to be able to revert if compilation fails
            copy_of_self = self.copy()

            try:

                ids, coefficients, biases, _, _, silents, types = zip(*new_composites)

                # Pad the matrix with equal many rows and columns as new composites
                self._amat = np.pad(self._amat, ((0, len(new_composites)), (0, len(new_composites))), mode='constant')

                # Update the imap with the new composites
                self._imap.update(
                    dict(
                        zip(
                            ids,
                            range(self._amat.shape[1] - len(new_composites), self._amat.shape[1])
                        )
                    )
                ) 

                # Create a rows/columns/values array from the buffer
                rcv = np.array(
                    list(
                        chain(
                            *starmap(
                                lambda irow, cvs: list(
                                    starmap(
                                        lambda v, c: (
                                            irow,
                                            self._col(v),
                                            c
                                        ),
                                        cvs.items()
                                    )
                                ), 
                                zip(
                                    range(self._amat.shape[0] - len(new_composites), self._amat.shape[0]), 
                                    coefficients,
                                )
                            )
                        )
                    )
                )
                if rcv.size:
                    self._amat[rcv.T[0], rcv.T[1]] = rcv.T[2]

                # Set d values
                self._dvec = np.append(self._dvec, np.zeros(len(new_composites), dtype=complex) + complex(0, 1))

                # Set all biases from the buffer
                self._bvec = np.append(self._bvec, np.array(list(map(lambda bias: complex(bias, bias), biases))))

                # Set equal many composite flags as new composites
                self._cvec = np.append(self._cvec, np.ones(len(new_composites), dtype=bool))

                # Set equal many silent flags as new composites
                self._svec = np.append(self._svec, np.array(silents))

                # Set equal many types as new composites
                self._tvec = np.append(self._tvec, np.array(types))

                # Update the amap with the ALL composites
                # There may be existing ones which only wants to add new aliases
                all_ids, _, _, _, aliases, _, _ = zip(*starmap(lambda k,v: (k,) + v, self._buffer.items()))
                self._amap.update(
                    dict(
                        filter(
                            lambda x: x[0] is not None,
                            zip(
                                aliases,
                                all_ids,
                            )
                        )
                    )
                )

            except MissingVariableException as e:
                self.revert_from(copy_of_self)
                raise e
            except Exception as e:
                self.revert_from(copy_of_self)
                raise FailedToCompileException(e)
            
        # Clear the buffer
        self._buffer = {}
        
        return all_ids

    def set_gelineq(self, coefficients: Dict[str, int], bias: int, alias: Optional[str] = None, silent: bool = False, ttype: str = "lineq", unique: bool = False) -> str:
        """
            Sets a linear inequality constraint, ax + by + cz + bias >= 0.

            Parameters
            ----------
            coefficients : Dict[str, int]
                The variables and their coefficients. For instance, {"x": 1, "y": 1, "z": 1}].

            bias : int
                The bias of the constraint.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="lineq")
                The type of the constraint.

            unique : bool (default=False)
                If True, the constraint is considered unique by itself and not based on its children.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> model.set_gelineq({"x": 1, "y": 1, "z": 1}, -3)
            da4ab8efc9b188b591115c3f376d0bc1ac6481ca

            Returns
            -------
            str
                The ID of the constraint.
        """
        _id = self._composite_id(
            coefficients,
            bias,
            unique=unique
        )
        self._buffer[_id] = (coefficients, bias, 1j, alias, silent, ttype)
        
        if self._compilation_setting == CompilationSetting.INSTANT:
            self.compile()
    
        return _id
    
    @staticmethod
    def negate(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
            Negate the given equations on the form Ax + b >= 0.
            Check out Polyhedron.md for more information.

            Parameters
            ----------
            A : np.ndarray (2D integer matrix)
                A weighted adjacency matrix.

            b : np.ndarray (1D integer vector)
                A bias vector.

            Examples
            --------
            >>> import numpy as np
            >>> A = np.array([[1, 1, 0], [1, 1, 1]])
            >>> b = np.array([-1, -2])
            >>> PLDAG.negate(A, b)
            (array([[-1, -1,  0],
                    [-1, -1, -1]]), array([ 0,  1]))

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                The negated matrix and bias vector.
        """
        return (-1 * A), (-1 * b) -1
    
    @staticmethod
    def _sdot(A: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
            Special dot product.
            A@d, where d is flipped if coefficient in A is negative.

            Parameters
            ----------
            A : np.ndarray (2D integer matrix)
            d : np.ndarray (1D complex vector)

            Examples
            --------
            >>> import numpy as np
            >>> A = np.array([[1],[-1]])
            >>> d = np.array([1j])
            >>> PLDAG._sdot(A, d)
            array([ 0.+1.j, -1.+0.j])

            Returns
            -------
            np.ndarray
                The vector result of the special dot product.
        """
        r = np.abs(A) * d
        return (-1j * np.conj(r * (A < 0)) + r * (A >= 0)).sum(axis=1) 
    
    @staticmethod
    def _prop_upstream_algo(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, fixed: np.ndarray):
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
        _A = np.vstack((np.zeros((A.shape[1]-A.shape[0], A.shape[1]), dtype=np.int64), A))
        _B = np.append(np.zeros(A.shape[1]-A.shape[0], dtype=complex), B)
        _fixed = fixed | (D.real == D.imag)
        M = (_A != 0) & ~_fixed
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
            rf = PLDAG._sdot(_A[i:i+1], D)[0] + _B[i]
            if rf.imag == 0 and D[i].real == 1:
                # The upper inner bound is 0 and the outer bound is true.
                # If variable's coefficient is positive, we set the variable's lower bound to its upper bound
                # If variable's coefficient is negative, we set the variable's upper bound to its lower bound
                new_value = (_A[i, M[i]] >= 0) * D[M[i]].imag + (_A[i, M[i]] < 0) * D[M[i]].real
                D[M[i]] = list(map(lambda x: complex(x, x), new_value))

            elif rf.real == -1 and D[i].imag == 0:
                # The lower inner bound is -1 and the outer bound is false.
                # If variable's coefficient is positive, we set the variable's upper bound to its lower bound
                # If variable's coefficient is negative, we set the variable's lower bound to its upper bound
                new_value = (_A[i, M[i]] >= 0) * D[M[i]].real + (_A[i, M[i]] < 0) * D[M[i]].imag
                D[M[i]] = list(map(lambda x: complex(x, x), new_value))

        return D
    
    @staticmethod
    def _prop_algo_downstream(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, fixed: np.ndarray, max_iterations: int = 100):

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

        def _prop_once(A: np.ndarray, C: np.ndarray, B: np.ndarray, D: np.ndarray):
            rf = PLDAG._sdot(A, D) + B
            d = ~C * D
            d[C] = (rf.real >= 0) + 1j*(rf.imag >= 0)
            return d

        prop = partial(_prop_once, A, C, B)    
        previous_D = D
        for _ in range(max_iterations):
            new_D = fixed * D + ~fixed * prop(previous_D)
            if (new_D == previous_D).all():
                return new_D
            previous_D = new_D
        
        raise Exception(f"Maximum iterations ({max_iterations}) reached without convergence.")
    
    @staticmethod
    def _prop_algo_bistream(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, fixed: np.ndarray, max_iterations: int = 100):
        """
            Propagates the graph downstream and upstream, and returns the propagated bounds.
        """
        return PLDAG._prop_upstream_algo(A, B, C, PLDAG._prop_algo_downstream(A, B, C, D, fixed, max_iterations), fixed)
    
    def _propagate(self, method: str, query: dict, freeze: bool = True) -> dict:
        """
            Propagates the graph downstream and returns the propagated bounds.
        """
        A: np.ndarray = self._amat
        B: np.ndarray = self._bvec
        C: np.ndarray = self._cvec
        D: np.ndarray = self._dvec.copy()

        # Filter query based on existing variables
        query = {k: v for k, v in query.items() if k in self._imap}

        # Query translation into primes
        qprimes = np.zeros(D.shape[0], dtype=bool)
        qprimes[[self._col(q) for q in query]] = True and freeze

        # Replace the observed bounds
        D[[self._col(q) for q in query]] = np.array(list(query.values()))

        if method == "downstream":
            res = self._prop_algo_downstream(A, B, C, D, qprimes)
        elif method == "upstream":
            res = self._prop_upstream_algo(A, B, C, D, qprimes)
        elif method == "bistream":
            res = self._prop_algo_bistream(A, B, C, D, qprimes)
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
    
    def id_to_alias(self, id: str) -> Optional[str]:
        """Get the aliases of the given ID"""
        return next(
            map(
                lambda x: x[0],
                filter(
                    lambda x: x[1] == id,
                    self._amap.items(),
                )
            ),
            None
        )
    
    def copy(self) -> "PLDAG":
        """Copy the model"""
        model = PLDAG()
        model._amat = self._amat.copy()
        model._dvec = self._dvec.copy()
        model._bvec = self._bvec.copy()
        model._svec = self._svec.copy()
        model._cvec = self._cvec.copy()
        model._tvec = self._tvec.copy()
        model._imap = self._imap.copy()
        model._amap = self._amap.copy()
        return model
    
    def get(self, *id: str) -> np.ndarray:
        """Get the bounds of the given ID(s)"""
        return self._dvec[list(map(self._col, id))]
    
    def exists(self, id: str) -> bool:
        """Check if the given id exists"""
        return (id in self._imap)
    
    def dependencies(self, id: str) -> Set[str]:
        """
            Get the dependencies of the given ID.

            Parameters
            ----------
            id : str
                The ID of the variable.

            NOTE: This function is recursive will ignore silent variables by default.
        """
        if not self.exists(id):
            raise MissingVariableException(id)
        
        if not self._cvec[self._col(id)]:
            return set()
        
        variables = list(self._imap)
        idxs = np.argwhere(self._amat[self._row(id)] != 0).T[0]
        return set(
            chain(
                map(
                    lambda x: variables[x],
                    filter(lambda x: not self._svec[x], idxs)
                ),
                *map(
                    lambda x: self.dependencies(variables[x]),
                    filter(lambda x: self._svec[x], idxs)
                )
            )
        )
    
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

        self._buffer[id] = ({}, None, bound, None, False, "primitive")
        if self._compilation_setting == CompilationSetting.INSTANT:
            self.compile()

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
    
    def set_atleast(self, references: List[str], value: int, alias: Optional[str] = None, silent: bool = False, ttype: str = "atleast", unique: bool = False) -> str:
        """
            Add a composite constraint of at least `value`.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            value : int
                The minimum value to set.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="atleast")
                The type of the constraint.

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
        return self.set_gelineq(dict(map(lambda r: (r, 1), references)), -1 * value, alias, silent, ttype, unique)
    
    def set_atmost(self, references: List[str], value: int, alias: Optional[str] = None, silent: bool = False, ttype: str = "atmost", unique: bool = False) -> str:
        """
            Add a composite constraint of at most `value`.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            value : int
                The maximum value to set.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttpe : str (default="lineq")
                The type of the constraint.

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
        return self.set_gelineq(dict(map(lambda r: (r, -1), references)), value, alias, silent=silent, ttype=ttype, unique=unique)
    
    def set_or(self, references: List[str], alias: Optional[str] = None, silent: bool = False, ttype: str = "or", unique: bool = False) -> str:
        """
            Add a composite constraint of an OR operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction

            ttype : str (default="or")
                The type of the constraint.

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
        return self.set_atleast(references, 1, alias, silent, ttype, unique=unique)
    
    def set_nor(self, references: List[str], alias: Optional[str] = None, silent: bool = False, ttype: str = "nor", unique: bool = False) -> str:
        """
            Add a composite constraint of an N(ot)OR operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction

            ttype : str (default="or")
                The type of the constraint.

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_not([self.set_or(references, silent=True)], alias, silent, ttype, unique=unique)
    
    def set_and(self, references: List[str], alias: Optional[str] = None, silent: bool = False, ttype: str = "and", unique: bool = False) -> str:
        """
            Add a composite constraint of an AND operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="and")
                The type of the constraint.

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
        return self.set_atleast(set(references), len(set(references)), alias, silent, ttype, unique=unique)
    
    def set_nand(self, references: List[str], alias: Optional[str] = None, silent: bool = False, ttype: str = "nand", unique: bool = False) -> str:
        """
            Add a composite constraint of an N(ot)AND operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="and")
                The type of the constraint.

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_not([self.set_and(references, silent=True)], alias, silent, ttype, unique=unique)
    
    def set_not(self, references: List[str], alias: Optional[str] = None, silent: bool = False, ttype: str = "not", unique: bool = False) -> str:
        """
            Add a composite constraint of a NOT operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="not")
                The type of the constraint.

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
        return self.set_atmost(references, 0, alias, silent, ttype, unique=unique)
    
    def set_xor(self, references: List[str], alias: Optional[str] = None, silent: bool = False, ttype: str = "xor", unique: bool = False) -> str:
        """
            Add a composite constraint of an XOR operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

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
            self.set_atleast(references, 1, silent=True),
            self.set_atmost(references, 1, silent=True),
        ], alias, silent, ttype, unique=unique)
    
    def set_xnor(self, references: List[str], alias: Optional[str] = None, silent: bool = False, ttype: str = "xnor", unique: bool = False) -> str:
        """
            Add a composite constraint of an XNOR operation.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttpe : str (default="xnor")
                The type of the constraint.

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
        return self.set_not([self.set_xor(references, silent=True)], alias, silent, ttype, unique=unique)
    
    def set_imply(self, condition: str, consequence: str, alias: Optional[str] = None, silent: bool = False, ttype: str = "imply", unique: bool = False) -> str:
        """
            Add a composite constraint of an IMPLY operation.

            Parameters
            ----------
            condition : str
                The reference to the condition.

            consequence : str
                The reference to the consequence.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="imply")
                The type of the constraint

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
        return self.set_or([self.set_not([condition], silent=True), consequence], alias, silent, ttype, unique=unique)

    def set_equal(self, references: List[str], value: int, alias: Optional[str] = None, silent: bool = False, ttype: str = "equal", unique: bool = False) -> str:
        """
            Add a composite constraint of an EQUAL operation: sum(references) == value.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            value : int
                The value to be equal.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="equal")
                The type of the constraint.

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_and([
            self.set_atleast(references, value, silent=True),
            self.set_atmost(references, value, silent=True),
        ], alias, silent, ttype, unique=unique)
    
    def set_equiv(self, lhs: str, rhs: str, alias: Optional[str] = None, silent: bool = False, ttype: str = "equiv", unique: bool = False) -> str:
        """
            Add a composite constraint of an EQUIVALENCE operation, lhs <-> rhs.
            It is equivalent to set_and([set_imply(lhs, rhs), set_imply(rhs, lhs)]).

            Parameters
            ----------
            lhs : str
                The left-hand side reference.

            rhs : str
                The right-hand side reference.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="equiv")
                The type of the constraint.

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_or([
            self.set_and([lhs, rhs], silent=True),
            self.set_not([lhs, rhs], silent=True),
        ], alias, silent, ttype, unique=unique)
    
    def set_not_equal(self, references: List[str], value: int, alias: Optional[str] = None, silent: bool = False, ttype: str = "not_equal", unique: bool = False) -> str:
        """
            Add a composite constraint of a NOT EQUAL operation: sum(references) != value.

            Parameters
            ----------
            references : List[str]
                The references to composite constraints or primitive variables.

            value : int
                The value to be not equal.

            alias : Optional[str] (default=None)
                The alias of the constraint.

            silent : bool (default=False)
                If True, the constraint is considered not be added by the user, but as a consequence of composite construction.

            ttype : str (default="not_equal")
                The type of the constraint.

            Returns
            -------
            str
                The ID of the composite constraint.
        """
        return self.set_not([self.set_equal(references, value, silent=True)], alias, silent, ttype, unique=unique)
    
    def to_polyhedron(self, double_binding: bool = True, **assume: Dict[str, complex]) -> Tuple[np.ndarray, np.ndarray]:

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

        # References to Polyhedron.md explaining the construction of the matrix A and bias vector b
        # Create the matrix
        A = self._amat.copy().astype(np.int64)
        b = self._bvec.copy().real.astype(np.int64)

        # Calculate inner bounds for each composite
        ib = self._sdot(A, self._dvec)

        # Pi -> Phi row indices
        Pi_Phi_i = np.arange(self._amat.shape[0])

        # Find coefficient to Pi (d), preparing for Pi -> Phi
        Pi_Phi_d = np.abs(np.array([ib.real, ib.imag], dtype=np.int64)).max(axis=0)

        # Step 1. Set Pi in A to be the coefficient to Pi, -dπ + ... + b >= 0
        A[Pi_Phi_i, self._cvec] = -1 * Pi_Phi_d

        # Step 2. Add onto bias: -dπ + ... + (b + d) >= 0
        b[Pi_Phi_i] += Pi_Phi_d
        
        # If double binding, we need also to set Phi -> Pi
        if double_binding:

            # Phi -> Pi row indices
            Phi_Pi_i = np.arange(self._amat.shape[0])

            # Step 1. Calculate !phi
            n_A, n_b = self.negate(self._amat, self._bvec.real.astype(np.int64))

            # Step 2. Calculate d = max( abs( !phi ) )
            n_ib = self._sdot(n_A, self._dvec)
            Phi_Pi_d = np.abs(np.array([n_ib.real, n_ib.imag], dtype=np.int64)).max(axis=0)

            # Step 3. Append (d-bias(!phi))π to A
            n_A[Phi_Pi_i, self._cvec] = Phi_Pi_d - n_b

            # Extend A, b to be able to handle Phi -> Pi
            A = np.vstack([A, n_A], dtype=np.int64)
            b = np.append(b, n_b)

        # Since we expects Ax >= b and so far assumed Ax + b >= 0, we need to flip b before continuing
        b = -1 * b

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
            v_idx = list(map(self._col, v))
            
            # Force both upper and lower bound to be the same
            for j in [-1,1]:
                a = np.zeros(A.shape[1], dtype=np.int64) * j
                a[v_idx] = 1 * j
                A = np.vstack([A, a])
                b = np.append(b, a.sum() * i)

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

        # Delete A_int and b_int
        del A_int, b_int

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
        sub_model._dvec = self._dvec[col_idxs]
        sub_model._bvec = self._bvec[row_idxs]
        sub_model._cvec = self._cvec[col_idxs]
        sub_model._svec = self._svec[row_idxs]
        sub_model._imap = dict(map(lambda x: (x[1], x[0]), enumerate(self._col_vars[col_idxs])))
        sub_model._amap = dict(filter(lambda x: x[1] in sub_model._imap, self._amap.items()))
        return sub_model
    
    def sub(self, roots: List[str], max_iterations: int = 1000) -> 'PLDAG':
        """
            Create a sub-PLDAG from the given root IDs.
            
            NOTE: This function works recursively to find all roots that eventually will be reached from any of the id's in `ids`. 
            Therefore we have a maximum number of iterations to avoid infinite loops.

            Parameters
            ----------
            roots : List[str]
                The IDs to use as roots in the new sub PLDAG.

            cuts : List[Dict]

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

        # First find all variables (columns and rows)
        # that eventually reaches any of the roots
        adj = self.adjacency_matrix
        idxs = list(map(self._col, filter(lambda id: id in self._imap, roots)))
        
        # Init by setting the given nodes
        match_vector = np.isin(range(adj.shape[1]), idxs)
        match_vector[idxs] = True

        # Find what nodes that are connected to the given nodes
        previous_match_vector = (adj[match_vector] != 0).any(axis=0)

        for _ in range(max_iterations):

            # Extend the match vector with new variables found
            match_vector |= previous_match_vector
            
            # Query for new variables
            previous_match_vector = (adj[previous_match_vector] != 0).any(axis=0)
            
            # If no new variables are found, this is the last root
            if (~previous_match_vector).all():
                break

        return self._from_indices(
            list(map(self._row, filter(lambda x: x in self.composites, self.columns[match_vector]))), 
            np.argwhere(match_vector).T[0]
        )
    
    def cut(self, cuts: Dict[str, str]) -> "PLDAG":

        """
            Cuts graph on given nodes in `cuts`. The key in cuts is what node to cut,
            the value is the new ID of the node (since it will go from a composite to a primitive).

            Note. Only connections to the given nodes in cuts will be cut. If the child nodes has
            other connections to the graph, they will still be part of the graph.

            Parameters
            ----------
            cuts : Dict[str, str]
                A mapping of nodes to cut to their new IDs.

            Returns
            -------
            PLDAG
                The cut PLDAG.
        """
        ids = list(filter(lambda id: id in self.rows, cuts.keys()))
        if len(ids) == 0:
            return self.copy()
        ex_roots = (self._amat == 0).all(axis=0)
        idxs = list(map(self._row, ids))
        a = self._amat.copy()
        a[idxs] = 0
        sub = self._from_indices(
            col_idxs=np.argwhere((a != 0).any(axis=0) | ex_roots).T[0],
            row_idxs=np.argwhere((a != 0).any(axis=1)).T[0],
        )
        sub._cvec[list(map(sub._col, ids))] = False
        sub._amap = dict(filter(lambda x: x[1] not in ids, sub._amap.items()))
        sub._imap = dict(map(lambda x: (cuts.get(x[0], x[0]), x[1]), sub._imap.items()))
        return sub
    
    def cut_sub(self, cuts: Dict[str, str], roots: List[str]) -> "PLDAG":
        """
            Cuts graph on given nodes in `cuts` and then creates a sub graph from the given root IDs.
        """
        return self.cut(cuts).sub(roots)
    
    def solve(
            self, 
            objectives: List[Dict[str, int]], 
            assume: Dict[str, complex], 
            solver: Solver, 
            double_bind_constraints: bool = True, 
            minimize: bool = False,
        ) -> List[Dict[str, complex]]:
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

            double_bind_constraints: bool = True
                If the constraints should be double binded. That is, a constraint A -> x & y & z will become two constraints:
                1 ) -dA + x + y + z >= 0
                2 ) +mA - x - y - z >= -1
                Saying that if A is 1 then x, y and z must be 1, and if A is 0 then x, y and z must be 0.

            minimize: bool = False
                If the objectives should be minimized.

            reduce: bool = True
                If the model should be reduced before solving.

            Examples
            --------
            >>> model = PLDAG()
            >>> model.set_primitives("xyz")
            >>> a = model.set_atleast("xyz", 1)
            >>> model.solve([{"x": 0, "y": 1, "z": 0}], {}, Solver.DEFAULT)
            [{'x': 0j, 'y': 1+1j, 'z': 0j}]

            Returns
            -------
            List[Dict[str, complex]]
                The solutions for the objectives.
        """
        if self._amat.shape == (0,0):
            return list(
                map(
                    lambda _: {},
                    range(len(objectives))
                )
            )
        
        else:
            A, b = self.to_polyhedron(double_binding=double_bind_constraints, **assume)

            # If no constraints, we add a dummy constraint
            if A.shape[0] == 0:
                A = np.append(A, [np.zeros(A.shape[1], dtype=np.int64)], axis=0)
                b = np.append(b, 0)

            variables = self._col_vars
            obj_mat = np.zeros((len(objectives), len(variables)), dtype=np.int64)
            for i, obj in enumerate(objectives):
                obj_mat[i, [self._col(k) for k in obj]] = list(obj.values())

            if solver == Solver.DEFAULT:
                from pldag.solver.default_solver import solve_lp
                solutions = solve_lp(A, b, obj_mat, set(np.argwhere((self._dvec.real != 0) | (self._dvec.imag != 1)).T[0].tolist()), minimize=minimize)
            else:
                raise ValueError(f"Solver `{solver}` not installed.")
            
            return list(
                map(
                    lambda solution: dict(
                        zip(
                            # Convert back to str from np.str_
                            map(str, variables), 
                            map(
                                lambda i: complex(i,i),
                                solution
                            )
                        )
                    ),
                    solutions
                )
            )
        
    def dump(self) -> bytes:
        """
            Dump the model to compressed bytes.
        """
        return compress(dumps((set(vars(self).keys()), self), protocol=HIGHEST_PROTOCOL), mtime=0)
    
    @staticmethod
    def load(data: bytes) -> 'PLDAG':
        """
            Load the model from compressed bytes.
        """
        version_attrs, model = loads(decompress(data))
        if version_attrs != set(vars(PLDAG()).keys()):
            raise ValueError(f"Version mismatch.")
        return model
    
    def to_file(self, filename: str):
        """
            Dump the model to a file.
        """
        with open(filename, "wb") as f:
            f.write(self.dump())

    @staticmethod
    def from_file(filename: str) -> 'PLDAG':
        """
            Load the model from a file.
        """
        with open(filename, "rb") as f:
            return PLDAG.load(f.read())
        
    def to_json(self) -> list:
        """
            Dump the model to JSON.
        """
        primitives = self.primitives
        composites = self.composites
        return list(
            chain(
                map(
                    lambda primitive: {
                        "id": primitive,
                        "type": "primitive",
                        "bound": {
                            "lower": int(self._dvec[self._col(primitive)].real),
                            "upper": int(self._dvec[self._col(primitive)].imag),
                        },
                    },
                    filter(
                        lambda variable: variable in primitives,
                        self._imap.keys()
                    )
                ),
                map(
                    lambda composite: {
                        "id": composite,
                        "type": "composite",
                        "bound": {
                            "lower": int(self._dvec[self._col(composite)].real),
                            "upper": int(self._dvec[self._col(composite)].imag),
                        },
                        "children": list(
                            map(
                                lambda i_child: {
                                    "id": self._icol(i_child),
                                    "coef": int(self._amat[self._row(composite), i_child]),
                                },
                                np.argwhere(self._amat[self._row(composite)] != 0).T[0]
                            )
                        ),
                    },
                    filter(
                        lambda variable: variable in composites,
                        self._imap.keys()
                    )
                )
            )
        )
    
    @staticmethod
    def from_json(data: list) -> 'PLDAG':
        """
            Load the model from JSON.
        """
        model = PLDAG(compilation_setting=CompilationSetting.ON_DEMAND)
        for item in data:
            if item["type"] == "primitive":
                model.set_primitive(item["id"], complex(item["bound"]["lower"], item["bound"]["upper"]))
            elif item["type"] == "composite":
                model.set_gelineq(
                    dict(
                        map(
                            lambda child: (child["id"], child["coef"]),
                            item["children"]
                        )
                    ),
                    item["bound"]["lower"],
                    item["id"]
                )
        model.compile()
        return model
    
    def no_outgoing_edges(self) -> np.ndarray:
        """
            Get the nodes with no outgoing edges.
        """
        return self.columns[(self._amat == 0).all(axis=0)]
    
    def no_incoming_edges(self) -> np.ndarray:
        """
            Get the nodes with no incoming edges.
        """
        return np.append(self.primitives, self.composites[(self._amat == 0).all(axis=1)])
    
    def no_edges(self) -> np.ndarray:
        """
            Get the nodes with no incoming or outgoing edges.
        """
        return np.array(set(self.no_outgoing_edges()).intersection(self.no_incoming_edges()))


class PDLite:

    def __init__(self):
        # The actual A matrix in a polyhedron (A, b)
        self._amat = np.zeros((0, 0),   dtype=np.int64)
        # The actual b vector in a polyhedron (A, b)
        self._bvec = np.zeros(0,        dtype=np.int64)
        # The bounds vector
        self._dvec = np.zeros(0,        dtype=np.complex128)
        # Keeps track of variable type. Size is equal to the number of rows in A.
        self._tvec = np.empty((0, ),    dtype=object)
        # Buffer for new variables/constraints
        self._buffer = {}
        # Index map for rows
        self._rmap = {}
        # Index map for columns
        self._imap = {}

    @property
    def columns(self) -> list:
        return list(self._imap.keys())
    
    @property
    def rows(self) -> list:
        return list(self._rmap.keys())
    
    def _col(self, k: str) -> int:
        return self._imap[k]
    
    def _row(self, k: str) -> int:
        return self._rmap[k]

    def set_primitive(self, id: str, bound: complex = complex(0,1)) -> List[str]:
        self._buffer[id] = ({}, None, bound, None, False, "primitive")
        return id

    def set_primitives(self, ids: str, bound: complex = complex(0,1)) -> list:
        for id in ids:
            self.set_primitive(id, bound)
        return ids

    def set_gelineq(self, references: Dict[str, int], value: int, alias: Optional[str] = None, ttype: str = "gelineq") -> List[str]:
        data = (references, alias, value, None, False, ttype)
        id = sha1(dumps(data)).hexdigest()
        self._buffer[id] = data
        return [id]

    def set_atleast(self, references: List[str], value: int, alias: Optional[str] = None, ttype: str = "atleast", and_condition: List[str] = []) -> List[str]:
        if and_condition:
            k = -1 * len(references)
            bias = -value -k
            crefs = dict(chain(map(lambda r: (r, 1), references), map(lambda r: (r, k), and_condition)))
            return self.set_gelineq(crefs, bias, alias, ttype)
        else:
            return self.set_gelineq(dict(map(lambda r: (r, 1), references)), -1 * value, alias, ttype)

    def set_atmost(self, references: List[str], value: int, alias: Optional[str] = None, ttype: str = "atmost", and_condition: List[str] = []) -> List[str]:
        if and_condition:
            k = -1 * len(references)
            bias = value -k
            crefs = dict(chain(map(lambda r: (r, -1), references), map(lambda r: (r, k), and_condition)))
            return self.set_gelineq(crefs, bias, alias, ttype)
        else:
            return self.set_gelineq(dict(map(lambda r: (r, -1), references)), value, alias, ttype)

    def set_and(self, references: List[str], alias: Optional[str] = None, ttype: str = "and", and_condition: List[str] = []) -> List[str]:
        return self.set_atleast(references, len(references), alias, ttype, and_condition)

    def set_or(self, references: List[str], alias: Optional[str] = None, ttype: str = "or", and_condition: List[str] = []) -> List[str]:
        return self.set_atleast(references, 1, alias, ttype, and_condition)

    def set_not(self, references: List[str], alias: Optional[str] = None, ttype: str = "not", and_condition: List[str] = []) -> List[str]:   
        return self.set_atmost(references, 0, alias, ttype, and_condition)
    
    def set_xor(self, references: List[str], alias: Optional[str] = None, ttype: str = "xor", and_condition: List[str] = []) -> List[str]:
        return [
            self.set_or(references, alias, ttype, and_condition)[0],
            self.set_atmost(references, 1, alias, ttype, and_condition)[0],
        ]

    def compile(self):
        
        # Set primitives first
        primitives = dict(filter(lambda x: len(x[1][0]) == 0, self._buffer.items()))
        if primitives:
            self._amat = np.pad(self._amat, ((0, 0), (0, len(primitives))), mode='constant', constant_values=0)
            self._dvec = np.append(self._dvec, list(map(lambda x: x[2], primitives.values())))
            self._imap = {**self._imap, **dict(zip(primitives, range(max(self._imap.values(), default=0), len(primitives))))}

        # Set composites
        composites = dict(filter(lambda x: len(x[1][0]) > 0, self._buffer.items()))
        if composites:
            _A = np.zeros((len(composites), self._amat.shape[1]))
            _b = np.zeros(len(composites))

            for i, composite in enumerate(composites):
                for ref, coef in composites[composite][0].items():
                    _A[i, self._imap[ref]] = coef
                _b[i] = composites[composite][2]

            self._amat = np.vstack([self._amat, _A])
            self._bvec = np.append(self._bvec, _b)
            self._tvec = np.append(self._tvec, list(map(lambda x: x[5], composites.values())))
            self._rmap = {**self._rmap, **dict(zip(composites, range(max(self._rmap.values(), default=0), len(composites))))}

        self._buffer = {}

    def to_polyhedron(self, **assume: Dict[str, complex]) -> Tuple[np.ndarray, np.ndarray]:

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
            >>> np.array_equal(A, np.array([[1,1,1]]))
            >>> np.array_equal(b, np.array([1]))
            True
        """

        # References to Polyhedron.md explaining the construction of the matrix A and bias vector b
        # Create the matrix and support vector (-1 * bias)
        A = self._amat.copy().astype(np.int64)
        b = -1 * self._bvec.copy().real.astype(np.int64)

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
            v_idx = list(map(self._col, v))
            
            # Force both upper and lower bound to be the same
            for j in [-1,1]:
                a = np.zeros(A.shape[1], dtype=np.int64) * j
                a[v_idx] = 1 * j
                A = np.vstack([A, a])
                b = np.append(b, a.sum() * i)

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
    
    def solve(self, objectives: List[Dict[str, int]], assume: Dict[str, complex], solver: Solver, minimize: bool = False):
        if self._amat.shape == (0,0):
            return list(
                map(
                    lambda _: {},
                    range(len(objectives))
                )
            )
        
        else:
            A, b = self.to_polyhedron(**assume)

            # If no constraints, we add a dummy constraint
            if A.shape[0] == 0:
                A = np.append(A, [np.zeros(A.shape[1], dtype=np.int64)], axis=0)
                b = np.append(b, 0)

            obj_mat = np.zeros((len(objectives), len(self.columns)), dtype=np.int64)
            for i, obj in enumerate(objectives):
                obj_mat[i, [self._col(k) for k in obj]] = list(obj.values())

            if solver == Solver.DEFAULT:
                from pldag.solver.default_solver import solve_lp
                solutions = solve_lp(A, b, obj_mat, set(np.argwhere((self._dvec.real != 0) | (self._dvec.imag != 1)).T[0].tolist()), minimize=minimize)
            else:
                raise ValueError(f"Solver `{solver}` not installed.")
            
            return list(
                map(
                    lambda solution: dict(
                        zip(
                            self.columns, 
                            map(
                                lambda i: complex(i,i),
                                solution
                            )
                        )
                    ),
                    solutions
                )
            )
        
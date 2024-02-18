# Prime Logic Directed Acyclic Graph
"Prime Logic Directed Acyclic Graph" data structure, or "PL-DAG" for short, is fundamentally a Directed Acyclic Graph (DAG) where each node represents a logical relationship, and the leaf nodes correspond to literals. 
Each node in the graph encapsulates information about how its incoming nodes or leafs are logically related. For instance, a node might represent an AND operation, meaning that if it evaluates to true, all its incoming nodes or leafs must also evaluate to true.
What sets this structure apart is that each node and leaf is associated with a prime number, and each node computes a composite prime number based on the prime numbers of its incoming nodes or leafs. This prime-based system allows for efficient manipulation and traversal of the graph while preserving logical relationships.
In summary, this data structure combines elements of a DAG with a logic network, utilizing prime numbers to encode relationships and facilitate operations within the graph.

# How it works
Each composite (node) is a linear inequality equation on the form
```A = a + b + c >= 0```. A primitive (leaf) is just a name or alias connected to a literal. A literal here is a tuple of two values `(0,1)` indicating what's the lowest value some variable could take (0) and the highest value (1). So a boolean primitive would have the literal value (0,1), since it can take on the value 0 or 1. Another primitive having (0,52) (weeks for instance) could potentially take on every discrete value in between but is expressed only with the lowest and highest value.

Under the surface, each node is connected to a prime combination number. It is used for the `propagation` algorithm to quickly find incoming nodes to the one currently being processed.

# Example
```python
import numpy as np
from pldag import PLDAG

model = PLDAG(10) # 10 here is the number of dimensions to represent a prime combination. This effects the possible number of nodes that can exist in the graph but also effects the computation complexity.
print(model.n_max) # check how many nodes could be contained
print(model.n_left) # check how many are left

# Sets x, y and z as boolean variables in model
model.set_primitives("xyz")

# Create a simple AND connected to "A"
# This is equivalent to A = x + y + z -3 >= 0
model.set_composite("and_connector", ["x","y","z"], -3)

# So if we check when all x, y and z are set to 1, then we
# expect "and_connector" also to be (1,1)
assert np.array_equal(
    model.test({"x": (1,1), "y": (1,1), "z": (1,1)}, ["and_connector"]),
    np.array([[1,1]]) # We selected "and_connector" and got that one's bounds as result
)

# Create a (x -> y & z) relationship to A
# This is equivalent to -x v y & z
# So first we create the left side of OR (-x), C = -x >= 0
# Then the right side (y & z), B = y + z >= 2
# Third we connect B and C to A: A = B + C >= 1
model.set_composite("C", ["x"], 0, True)
model.set_composite("B", ["y", "z"], -2)
model.set_composite("A", ["B", "C"], -1)
np.array_equal(
    model.test(
        {
            "x": (1,1), 
            "y": (1,1), 
            "z": (1,1)
        }, 
        select='A'
    ),
    np.array([[1,1]])
)
# Returns array([[1,1]]) since A is true when x, y and z are true.

np.array_equal(
    model.test(
        {
            "x": (1,1), 
            "y": (0,0), 
            "z": (1,1)
        }, 
        select='A'
    ),
    np.array([[0,0]])
)
# Returns array([[0,0]]) since A is false when x is true and y or z are false.

np.array_equal(
    model.test(
        {
            "x": (0,1), 
            "y": (0,0), 
            "z": (0,1)
        }, 
        select='A'
    ),
    np.array([[0,1]])
)
# Returns array([[0,1]]) since A is may be true or false.
```
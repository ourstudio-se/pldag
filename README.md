# Primitive Logic Directed Acyclic Graph
"Primitive Logic Directed Acyclic Graph" data structure, or "PL-DAG" for short, is fundamentally a Directed Acyclic Graph (DAG) where each node represents a logical relationship, and the leaf nodes correspond to literals. 
Each node in the graph encapsulates information about how its incoming nodes or leafs are logically related. For instance, a node might represent an AND operation, meaning that if it evaluates to true, all its incoming nodes or leafs must also evaluate to true.

# How it works
Each composite (node) is a linear inequality equation on the form
```A = a + b + c >= 0```. A primitive (leaf) is just a name or alias connected to a literal. A literal here is a complex number of two values `-1+3j` indicating what's the lowest value some variable could take (`-1`) and the highest value (`+3`). So a boolean primitive would have the literal value `1j`, since it can take on the value 0 or 1. Another primitive having `52j` (weeks for instance) could potentially take on every discrete value in between but is expressed only with the lowest and highest value.

# Example
```python
from pldag import PLDAG

# Init model
model = PLDAG()

# Sets x, y and z as boolean variables in model
model.set_primitives("xyz")

# Create a simple AND connected to "A"
# This is equivalent to A = x + y + z -3 >= 0
# The ID for this proposition is returned. We can also connect an alias to it, like so.
id_ref = model.set_and(["x","y","z"], alias="A")

# Later if we forget the ID, we can retrieve it like this
id_ref_again = model.id_from_alias("A")
assert id_ref == id_ref_again

# So if we check when all x, y and z are set to 1, then we
# expect `id_ref` to be 1+1j
assert model.propagate({"x": 1+1j, "y": 1+1j, "z": 1+1j}).get(id_ref) == 1+1j

# And then not all are set, we'll get just 1j (meaning the model doesn't now whether it's true or false)
assert model.propagate({"x": 1+1j, "y": 1+1j, "z": 1j}).get(id_ref) == 1j

# However, if we now that any variable is not set, being equal to 0, then the model know the composite to be false (or 0j)
assert model.propagate({"x": 1+1j, "y": 1+1j, "z": 0j}).get(id_ref) == 0j
```

There's also a quick way to use a solver. There's no built-in solver but is dependent on existing ones. Before using, reinstall the package with the solver variable set to the solver you'd want to use
```bash
pip install pldag
``` 
And then you can use it like following
```python

from pldag import Solver

# Maximize [x=1, y=0, z=0] such that rules in model holds and variable `id_ref` must be true.
solution = next(iter(model.solve(objectives=[{"x": 1}], assume={id_ref: 1+1j}, solver=Solver.DEFAULT)))

# Since x=1 and `id_ref` must be set (i.e. all(x,y,z) must be true), we could expect all variables
# be set.
assert solution.get(id_ref) == 1+1j

```

# Primitive Logic Directed Acyclic Graph

The **Primitive Logic Directed Acyclic Graph (PL-DAG)** is a data structure based on a Directed Acyclic Graph (DAG) where each node represents a logical relationship and leaf nodes correspond to literals. Each node encapsulates information about how its incoming nodes or leaves are logically related. For example, a node might represent an AND operation, meaning that if it evaluates to true, all of its incoming nodes or leaves must also evaluate to true.

## How It Works

Each composite (node) is represented as a linear inequality of the form `A = a + b + c >= 0`. A primitive (leaf) is a named alias connected to a literal. A literal is represented as a complex number with two values: `-1+3j` indicates the lowest value a variable can take (`-1`) and the highest value (`+3`). For example, a boolean primitive has the literal value `1j` since it can only be 0 or 1. A primitive with value `52j` (representing weeks, for instance) could take any discrete value between 0 and 52, though it is expressed only by its bounds.

## Example

```python
from pldag import PLDAG

# Initialize model
model = PLDAG()

# Set x, y, and z as boolean variables in the model
model.set_primitives("xyz")

# Create a simple AND operation connected to alias "A"
# This is equivalent to A = x + y + z - 3 >= 0
# The ID for this proposition is returned. We can also assign an alias to it.
id_ref = model.set_and(["x","y","z"], alias="A")

# Later, if we forget the ID, we can retrieve it like this
id_ref_again = model.id_from_alias("A")
assert id_ref == id_ref_again

# When all x, y, and z are set to 1, we expect `id_ref` to be 1+1j
assert model.propagate({"x": 1+1j, "y": 1+1j, "z": 1+1j}).get(id_ref) == 1+1j

# When not all are set, we get 1j (meaning the model doesn't know whether it's true or false)
assert model.propagate({"x": 1+1j, "y": 1+1j, "z": 1j}).get(id_ref) == 1j

# However, if we know that any variable is not set (equal to 0), the model knows the composite is false (0j)
assert model.propagate({"x": 1+1j, "y": 1+1j, "z": 0j}).get(id_ref) == 0j
```

## Using a Solver

There is no built-in solver, but PL-DAG supports integration with existing solvers. Install the package:

```bash
pip install pldag
```

Then use it as follows:

```python
from pldag import Solver

# Maximize [x=1, y=0, z=0] such that the model's rules hold and `id_ref` must be true
solution = next(iter(model.solve(objectives=[{"x": 1}], assume={id_ref: 1+1j}, solver=Solver.DEFAULT)))

# Since x=1 and `id_ref` must be true (i.e., all(x,y,z) must be true),
# we expect all variables to be set
assert solution.get(id_ref) == 1+1j
```

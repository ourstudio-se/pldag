# Assume (and Reduce) PLDAG Model

## TL;DR

This is not a complete solution to the assume-problem. It can only assume and reduce the model up to certain complexities. For instance, the following cannot be solved:
```plain
MODEL
A = x + y + z ≥ 2
B = x + y ≥ 2
C = x + z ≥ 2
D = y + z ≥ 2
E = -B -C -D ≥ 0

ASSUME
A = 1

SHOULD RESULT IN
A = 1
B = 0/1
C = 0/1
D = 0/1
E = 0
```

The information from B, C, and D together is sufficient to derive A = 1 → E = 0.

## Algorithm

The algorithm alternates between propagation from top to bottom and bottom to top until no bounds can be further tightened.

First, note that any (≥) linear inequality with variable bounds defined can be fixed if either:
1. **Inner upper bound is 0 and geq is true**: We can fix the geq variable's lower bound to its upper bound (or flipped if the coefficient is negative)
2. **Inner lower bound is -1 and geq is false**: We can fix the geq variable's upper bound to its lower bound (or flipped if the coefficient is negative)

### Examples

**Example 1:**
```plain
MODEL
A = x + y + z >= 3
x, y, z ∈ {0,1}

ASSUME
A = 1

Inner bound = x[0,1] + y[0,1] + z[0,1] - [3,3] = [-3,0]
Since geq (A) is true and inner upper bound is 0, we can safely fix x=1, y=1, and z=1
```

**Example 2:**
```plain
MODEL
A = -x + y >= 0
x, y ∈ {0,1}

ASSUME
A = 0

Inner bound = x[0,1] + y[0,1] = [-1,1]
Since geq (A) is false and inner lower bound is -1, we can safely fix x=1 and y=0
```

We use this information during top-to-bottom propagation.

## Top-to-Bottom Propagation

1. Find all root composites and add them to the queue
2. While performing breadth-first search from top to bottom, apply the following to each node:
   1. If inner upper bound is 0 and geq is true, fix the geq variable's lower bound to its upper bound (or flipped if coefficient is negative)
   2. If inner lower bound is -1 and geq is false, fix the geq variable's upper bound to its lower bound (or flipped if coefficient is negative)
   3. If the outer bound is fixed and geq is set, verify they have the same value—otherwise, we've reached a contradiction
   4. If there's only one non-fixed variable and geq is true but ib is not [1,1], tighten the non-fixed variable such that the inner upper bound is 0
   5. If there's only one non-fixed variable and geq is false but ib is not [0,0], tighten the non-fixed variable such that the inner lower bound is -1
   6. Continue until all nodes have been propagated

## Bottom-to-Top Propagation

1. Find all composites that have only primitives as inputs and add them to the queue
2. While performing breadth-first search from bottom to top, apply the following to each node `n`:
   1. Calculate the inner bound by summing all its relationship bounds (flip the bound if the coefficient is negative), including the bias. Assign the new bound `b` to the node by `B[n] = b ≥ 0`
   2. Continue until all nodes have been propagated

## Assumption-Propagation

1. Create a copy of the model's bounds to `B`
2. Perform a full run from bottom to top and then back again. Store the result in `B'`
3. Repeat step 2 until `B == B'` or the maximum allowed iterations have been reached

After assumption-propagation is complete, the bounds of all variables have been tightened to a certain degree. Note that this approach cannot use information from multiple places in the DAG to derive conclusions—only straightforward derivation is supported.
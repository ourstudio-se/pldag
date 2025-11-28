# Propagation Algorithm

## Composites and Their Evaluation

To understand the propagation algorithm, first understand how each composite is represented. A composite has links to other variables (composites or primitives) and a coefficient for each link. Ultimately, a composite is expressed as `ax + by + cz + d >= 0`, where `x`, `y`, and `z` are links to other values, and `a`, `b`, `c`, and `d` are coefficients. The coefficient `d` is a constant value with no variable link, called the **bias**. Each value of `x`, `y`, and `z` is represented by **bounds**—a two-value tuple `(lower, upper)` representing the minimum (inclusive) and maximum values a variable can take. Multiplying a bound `x` by a coefficient `a` is straightforward, but requires special attention for negative coefficients:
```
f = {a, x} => (
    a * x.lower,
    a * x.upper
) if a > 0 else (
    a * x.upper,
    a * x.lower
)
```

Notice how the lower and upper values are flipped when `a` is negative. Given all input variables **X** and all coefficients **A**, evaluating a composite is done by:

$$
( \sum_i \text{f}(A[i], X[i]) ) + d \geq 0
$$

## Propagating Bounds Using Topological Sort

Here is the algorithm in pseudocode. The input is an interpretation of the model—a mapping from an ID to a bound (e.g., `x = (1,1)`).

```
function propagate(interpretation: ID to Bound):

    # Start by defining result as a copy of the interpretation
    result = interpretation.copy()

    # Find all composite variables that only have primitives as input.
    # This is our set S from Kahn's algorithm.
    # Note: exclude composite variables already in interpretation/result,
    # since those are user-defined and should not be overridden.

    S = All composites that:
        (1) have only primitive variables as input, and
        (2) are not present in `result/interpretation`

    # Loop until S is empty, as in Kahn's algorithm
    while S is not empty:

        # Get any element from S (order doesn't matter)
        current_id = S.pop()

        # Propagate the bounds from current_id's children to current_id
        result[current_id] = Sum of all children's bounds from `result`,
                             as described in "Composites and Their Evaluation"

        # Find all composites having `current_id` as input.
        # If that composite's children are all in `result`, add it to S.
        incoming = filter all composites and select those having current_id as a child
        incoming_in_result = filter `incoming` to include only variables in result
        S.update(incoming_in_result)

    return result
```

**Note:** If the user can create a circular dependency, this algorithm will run indefinitely. Either prevent such dependencies or check if an already-computed variable appears again in the loop.

## Propagating Bounds with Metadata

This approach propagates not only bounds but also floats and sums from bottom to top.

```
function propagate_floats(interpretation: ID to Bound, data: ID to float):

    # First, propagate the bounds
    propagated_bounds = propagate(interpretation)

    # Collect all transitive dependencies (all dependencies from top to bottom).
    # For example, a model:
    # a -> b
    # a -> c
    # b -> d
    # c -> e
    # results in transitive dependencies: {a: {b, c, e}, b: {d}, c: {e}}
    # This ensures we calculate each variable's value only once.
    # NOTE: To simplify computation, ensure variables are ordered with least
    # dependent first. This avoids recomputing the topological sort.
    transitive_dependencies = ... # Use topological sort to collect these

    # Now propagate float values
    float_interpretation = create a new mapping from variable to primitive float bounds

    # Rely on the correct ordering
    for variable, dependencies in transitive_dependencies:

        if no dependencies:
            float_interpretation[variable] = propagated_bounds[variable] * data.get(variable, 0.0)
        else:
            # Sum each dependency bound multiplied by the value supplied in data
            float_interpretation[variable] = sum(
                map(
                    float_interpretation[dependency],  # default to 0.0
                    dependencies
                )
            ) + propagated_bounds[variable] * data.get(variable, 0.0)

    return float_interpretation
```
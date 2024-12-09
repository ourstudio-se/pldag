# Propagation algorithm

## Composites and its evaluation
First, understand how each composite is represented and why. A composite has links to other variables (composites/primitives) and also a coefficient for each of the links. In the end, a composite is written as `ax + by + cz + d >= 0`, where each `x,y,z` are links to other values and `a,b,c,d` are coefficients. `d` is a constant value, no variable link, and is called the bias. Each value of `x,y,z` are called `bounds`. They are a two value tuple `(lower, upper)` and represents the most (inclusive) lower and most upper value a variable can take. To multiple a bound `x` with a coefficient `a` is pretty straight forward, but needs attention to negative coefficients:
```
f = {a,x} => (
    a * x.lower,
    a * x.upper
) if a > 0 else (
    a * x.upper,
    a * x.lower
)
```
Notice how we flip lower and upper values when `a` is negative. So given all input variables X and all coefficiens A, evaluating a composite is done by
$$
    ( \sum_i \text{f}(A[i], X[i]) ) + d \geq 0
$$

## Propagating bounds using topological sort

Here's the algorithm on a pseudo code level. The input is an interpretation of the model and a mapping from an ID to a bound. For instance, x = (1,1).
```
function propagate(interpretation: ID to Bound):

    # Starting by defining result as the interpretation
    result = interpretation.copy()

    # Find all composite variables that only have primitives
    # as it's input. This is our set S from the Kahn's algorithm.
    # Note that we'll need to keep out composite variables being already
    # in interpretaion/result, since those are set from the user and we
    # don't want to override them.
    
    S = All composites that 
        (1) have only primitive variables as input and
        (2) are not being in `result/interpretation`

    # Now, just as in Kahn's algorithm, we loop until S is empty
    while S is not empty:

        # Get the first element in S (first doesn't matter, just any)
        current_id = S.pop()

        # Start now by propagating the bounds from current_id's children
        # to current_id
        result[current_id] = Sum of all childrens bounds given from `result` in the same manner as mentioned in "Composites and its evaluation".

        # Now find all composites having `current_id` as its input.
        # And if that composite's children are in `result`, then add to S.
        incoming = filter all composites and select those having current_id as child
        incoming_in_result = filter `incoming` to include only variables being in result
        S.update(incoming_in_result)

    return result
```
NOTE! If it is possible for the user to create a circular dependency, then the algorithm will continue for ever. So either make sure that the user cannot create those dependencies, or check if an already computed variable appers again in the loop.

## Propagating bounds with meta data
Here we do not only propagated bounds, but propagate floats and sums bottom up.
```
function propagate_floats(interpretation: ID to Bound, data: ID to float):

    # First we need to propagate the bounds
    propagated_bounds = propagate(interpretation)

    # Then we'll collect a list of all transitive dependencies. That is, all dependencies top to bottom. For instance, a model 
    # a -> b
    # a -> c
    # b -> d
    # c -> e
    # results in transitive dependencies {a: {b, c, e}, b: {d}, c: {e}}
    # We need those to make sure we only calculate one variable's value once
    # NOTE! To make computation much easier, make sure that the variables are in order, with least dependent first. Then we don't have to compute the topological sort again
    transitive_dependencies = ... # Use topo sort to collect these

    # Now we can propagate float values
    float_interpretation = create a new mapping from variable to primitive float bounds

    # So we rely on that the order is correct.
    for variable, dependencies in transitive_dependencies:

        if no dependencies:
            float_interpretation[variable] = propagated_bounds[variable] * data.get(variable, 0.0)
        else:
            # Just sum each dependency bound multiplied with value supplied in data
            float_interpretation[variable] = sum(
                map(
                    float_interpretation[dependency]  # default to 0.0
                    dependencies
                )
            ) + propagated_bounds[variable] * data.get(variable, 0.0)
    
    return float_interpretation

```
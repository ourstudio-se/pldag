# How to Transform $\pi \leftrightarrow \phi$

This document explains how to transform $\pi \leftrightarrow \phi$, where $\pi$ is a boolean variable and $\phi$ is a linear inequality constraint, into a new linear inequality constraint.

## A Linear Inequality Constraint

A linear inequality constraint consists of four parts:

1. **Discrete variables** ranging from any integer $i \in \mathbb{Z}$ to another integer $j \in \mathbb{Z}$, such that $i \leq j$. For instance, a variable $x \in [-2, 3]$ takes on any integer between -2 and 3 (inclusive)
2. **Coefficients** for each integer variable. A coefficient is an integer multiplied by a variable when the inequality is evaluated
3. **An operator**: $\gt$, $\geq$, $\lt$, or $\leq$. Here we always assume $\geq$
4. **A bias**: A constant integer, independent of any variable's values 

### Examples of Linear Inequality Constraints

- $x + y + z \geq 1$ where $x, y, z \in [0,1]^3$
- $x + y + z \geq 3$ where $x, y, z \in [0,1]^3$
- $-2x + y + z \geq 0$ where $x, y, z \in [0,1]^3$

## The *Inner Bound* of a Linear Inequality Constraint

An important property is the **inner bound** $\text{ib}(\phi)$ of a linear inequality constraint $\phi$. It is the sum of all variables' step bounds, excluding the bias and before evaluation with the $\geq$ operator. For instance, the inner bound of the linear inequality $-2x + y + z \geq 0$, where $x, y, z \in \{0,1\}^3$, is $[-2, 2]$, since the lowest value the sum can be is $-2$ (from the combination $x=1, y=0, z=0$) and the highest value is $2$ (from $x=0, y=1, z=1$).

## Negating an Inequality Constraint

To find the negation of an inequality constraint, follow these two steps:

1. Multiply all variables by $-1$, including the bias
2. Add $-1$ to the bias

**Example:** $\neg (x + y + z - 2 \geq 0)$ is $-x - y - z + 1 \geq 0$.

## Transforming $\pi \rightarrow \phi$

The transformation is done by following these steps:

1. Let $d = \text{max}(|\text{ib}(\phi)|)+|\text{bias}(\pi)|$
2. Append $-d\pi$ to the left side of the equation $\phi$
3. Append $-d$ to the right side of the equation $\phi$

### Example

Transform $\pi \rightarrow (x + y + z - 2 \geq 0)$, where $x, y, z \in [0,1]^3$:

1. Calculate $d = \text{max}(|\text{ib}(\phi)|)$:
   1. $d = \text{max}(|\text{ib}(x + y + z - 2 \geq 0)|)$
   2. $d = \text{max}(|[0,3]|)$
   3. $d = \text{max}([0,3])$
   4. $d = 3$
2. Use $-d$ as the coefficient for $\pi$ and append to the left side of $\phi$: $-3\pi + x + y + z - 2 \geq 0$
3. Append $-d$ to the right side of $\phi$: $-3\pi + x + y + z - 2 \geq -3$
4. Sum all constants: $-3\pi + x + y + z \geq -1$

## Transforming $\phi \rightarrow \pi$

First, note that $\phi \rightarrow \pi$ is equivalent to $\neg \phi \lor \pi$.

1. Calculate $\phi' = \neg \phi$
2. Let $d = \text{max}(|\text{ib}(\phi')|)$
3. Append $(d - \text{bias}(\phi'))\pi$ to the left side of $\phi'$
4. Keep the bias of $\phi'$ as is

### Example

Let $\phi = -a - b - c - d + 5 \geq 0$, where $a \in [-5,3]$, $b \in [0,2]$, $c \in [-4,4]$, and $d \in [-4,5]$. Create a new inequality $\theta$ from $\phi \rightarrow \pi$:

1. Start by negating $\phi$: $\neg \phi = a + b + c + d - 6 \geq 0$
2. Find the coefficient for $\pi$ by calculating $d = \text{max}(|\text{ib}(\phi')|)$:
   1. $d = \text{max}(|\text{ib}(a + b + c + d - 6 \geq 0)|)$
   2. $d = \text{max}(|[-13, 14]|)$
   3. $d = \text{max}([13, 14])$
   4. $d = 14$
3. Append $(14 - (-6))\pi = 20\pi$ to the left side of $\neg \phi$: $20\pi + a + b + c + d - 6 \geq 0$
4. Result: $\theta = 20\pi + a + b + c + d - 6 \geq 0$

## Quick Way of Fixing Variable Bounds (Boolean Variables)

We are often given a fixed variable bound, such as $x = [1,1]$ (meaning $x = 1$). Usually, solvers don't support supplying fixed bounds as an argument, so we need to explicitly set them as linear inequalities in our polyhedron.

The straightforward approach is to create two linear inequalities for each variable $x$ with fixed bound $i$: (1) $x \geq i$ and (2) $-x \geq -i$. However, this increases the number of inequalities linearly with the number of fixed variables times two.

Another approach is to fit all fixed variables sharing the same constant into one pair of linear inequalities. We call this algorithm **Cumulative Coefficient Generation (CCG)**.

Let $b(v)$ denote the bounds of a variable $v$, and $\text{span}(b(v))$ be the length (or span) of the bound (e.g., $\text{span}([-2,3]) = 3 - (-2) = 5$).

1. Group all fixed variables sharing the same constant $i$ into $V$
2. Set the first coefficient $c_0$ for the first variable $v_0 \in V$ to $c_0 = 1$, the second coefficient $c_1$ for the second variable $v_1$ to $c_1 = \text{span}(b(v_0))$, the third coefficient $c_2$ for the third variable $v_2$ to $c_2 = \text{span}(b(v_1)) c_1$, and so on until all variables are given a coefficient $C = [c_0, c_1, \dots, c_n]$. Create the lower bound as:
   $$c_0v_0 + c_1v_1 + \dots + c_nv_n \geq \sum_{c \in C} c \cdot i$$
3. Create the upper bound as:
   $$-c_0v_0 - c_1v_1 - \dots - c_nv_n \geq -\sum_{c \in C} c \cdot i$$

### Example

Fix the bounds $x = (-3,2)$, $y = (-1,3)$, and $z = (-2,0)$ to $x = -1$, $y = -1$, $z = -1$:

1. Set coefficient for $x$: $c_x = 1$
2. Set coefficient for $y$: $c_y = c_x \cdot \text{span}(b(x)) = 1 \cdot (2 - (-3)) = 5$
3. Set coefficient for $z$: $c_z = c_y \cdot \text{span}(b(y)) = 5 \cdot (3 - (-1)) = 20$

We have now computed the coefficient vector $C = [1, 5, 20]$.

4. Finally, calculate the bias: $d = c_0 \cdot i + c_1 \cdot i + c_2 \cdot i = -1 \cdot 1 + -1 \cdot 5 + -1 \cdot 20 = -26$

Construct the linear inequalities:
$$x + 5y + 20z \geq -26$$
$$-x - 5y - 20z \geq 26$$

### Limitations

There are limits to creating linear inequalities for fixing bounds this way. The last coefficient's value grows exponentially with the number of variables (except when all span values equal 1—then all coefficients are 1). If `i64` is used as your data type, the maximum coefficient value for the last coefficient is $9,223,372,036,854,775,807$, divided by the maximum absolute value of the last variable bound.

## Constraining Integer Variables

All integer variables have an explicit bound set that is not boolean. Solvers may not directly set these bounds, so they need to be set in the polyhedron. We do this simply by creating two new linear inequalities per integer variable:

1. Find all integer variables $I$
2. For each variable $x \in I$, set:
   - (1) $x - x_{lb} \geq 0$
   - (2) $-x + x_{ub} \geq 0$

where $x_{lb}$ and $x_{ub}$ denote the lower and upper bounds for variable $x$.
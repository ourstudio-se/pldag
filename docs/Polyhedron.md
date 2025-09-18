# How to transform $\pi \leftrightarrow \phi$, where $\pi$ is a boolean variable and $\phi$ is a linear inequality constraint, to a new linear inequality constraint.

## A linear inequality constraint
A linear inequality constraint consists of four parts: 
1) Discrete variables ranging from any integer $i \in \mathbb{Z}$ to another integer $j \in \mathbb{Z}$, such that $i \leq j$. For instance a variable $x \in [-2, 3]$ takes on any integer between -2 and 3, including.
2) Coefficients to each integer variable. A coefficient is an integer that is multiplied onto a variable when the inequality is evaluated.
3) An operator $\gt$, $\geq$, $\lt$ or $\leq$. Here we'll always assume $\geq$.
4) A bias. A constant integer, independent on any of the variable's values. 

Here are examples of linear inequality constraints:
- $x + y + z >= 1$ where $x,y,z \in [0,1]^3$
- $x + y + z >= 3$ where $x,y,z \in [0,1]^3$
- $-2x + y + z \geq 0$ where $x,y,z \in [0,1]^3$

## The <i>inner bound</i> of a linear inequality constraint
An important property is the inner bound $\text{ib}(\phi)$ of a linear inequality constraint $\phi$. It is the sum of all variable's step bounds, excl bias and before evaluated with the $\geq$ operator. For instance, the inner bound of the linear inequality $-2x + y + z \geq 0$, where $x,y,z \in \{0,1\}^3$, is $[-2, 2]$, since the lowest value the sum can be is $-2$ (given from the combination $x=1, y=0, z=0$) and the highest value is $2$ (from $x=0, y=1, z=1$).

## Negating an inequality constraint
To find the negation of a inequality constraint, following these two steps: v
1. Multiply $-1$ to all variables, including the bias. 
2. Add $-1$ to the bias. 

For example, $\neg (x + y + z -2 \geq 0)$ is $-x -y -z +1 \geq 0$.

## Transforming $\pi \rightarrow \phi$
Transformation is done by following these two steps: 

1. Let $d = \text{max}(|\text{ib}(\phi)|)$. 
2. Append $-d\pi$ to the left side of the equation $\phi$. 
3. Append $-d$ to the right side of the equation $\phi$ (or just sum $d$ and the bias).

For examlpe, let's transform the $\pi \rightarrow (x + y + z -2 >= 0)$, where $x,y,z \in [0,1]^3$. 
1. Calculate $d = \text{max}(|\text{ib}(\phi)|)$:
    1. $d = \text{max}(|\text{ib}(x + y + z -2 >= 0)|)$
    2. $d = \text{max}(|[0,2]|)$
    3. $d = \text{max}([0,2])$
    4. $d = 2$
2. Use $-d$ as coefficient for $\pi$ and append to the left side of $\phi$: $-2\pi + x + y + z -2 >= 0$
3. Append $-d$ to the right side of $\phi$: $-2\pi + x + y + z -2 >= -2$
4. Sum all constants: $-2\pi + x + y + z >= 0$

## Transforming $\phi \rightarrow \pi$
First note that $\phi \rightarrow \pi$ is equivilant to $\neg \phi \lor \pi$. 

1. Calculate $\phi' = \neg \phi$
2. Let $d = \text{max}(|\text{ib}(\phi')|)$ 
3. Append $(d - \text{bias}(\phi'))\pi$ to left side of $\phi'$. 
4. Keep bias of $\phi'$ as is

For example, let $\phi = -a -b -c -d +5 \geq 0$ where $a \in [-5,3]$, $b \in [0,2]$, $c \in [-4,4]$ and $d \in [-4,5]$ and create a new inequality $\theta$ from $\phi \rightarrow \pi$. 
1. We start by negating $\phi$: $\neg \phi = a + b + c + d -6 \geq 0$
2. Now we find the coefficient to $\pi$ by calculating $d = \text{max}(|\text{ib}(\phi')|)$:
    1. $d = \text{max}(|\text{ib}(a + b + c + d -6 \geq 0)|)$
    2. $d = \text{max}(|[-13, 14]|)$
    3. $d = \text{max}([13, 14])$
    4. $d = 14$
3. Append $(14-(-6))\pi = 20\pi$ to the left side of $\neg \phi: 20\pi + a + b + c + d -6 \geq 0$
4. $\theta = 20\pi + a + b + c + d -6 \geq 0$

## Fixing variable bounds
Many times we're given a fixed variable bound, such as $x=[1,1]$, meaning $x=1$. Usually, solvers doesn't support supplying fixed bounds as an argument so we need to explicitly set them as linear inequalities into our polyhedron. 

The straight forward way is to for each variable $x$ and fixed bound $i$ create two linear inequalities (1) $x \geq i$ and (2) $-x \geq -i$. However, this will increase the number of inequalities linearly with the number of fixed variables times two (since two for each variable). 

Another way is to fit all fixed variables sharing the same constant into one pair of two linear inequalities. We call this algorithm <b>Cumulative Coefficient Generation</b>, or <b>CCG</b>, for referencing purposes. 
Let $b(v)$ denote the bounds form a variable $v$, $\text{span}(b(v))$ be the lenght (or span) of the bound (e.g $\text{span}([-2,3])=3-(-2)=5$)
1. Group all fixed variables sharing the same constant $i$ into $V$.
2. Set the first coefficient $c_0$ for first variable $v_0 \in V$ to $c_0=1$, the second coefficient $c_1$ for the second variable $v_1$ to $c1=\text{span}(b(v_0))$, the third coefficient $c_2$ for the third variable $v_2$ to $c_2 = \text{span}(b(v_1))c_0$, and so on until all variables are given a coefficient $C=[c_0, c_1, \dots, c_n]$. Create the lower bound as $$c_0v_0 + c_1v_1+\dots+c_nv_n \geq \sum_{c \in C} c*i$$
3. Create the upper bound as $$-1c_0v_0 + -1c_1v_1+\dots+-1c_nv_n \geq -1*\sum_{c \in C} c*i$$

For example, fix the bounds $x=(-3,2)$, $y=(-1,3)$ and $z=(-2,0)$ to $x = -1$, $y = -1$, $z = -1$.
1. Set coefficient for $x$: $$c_x = 1$$ 
2. Set coefficient for $y$: $$c_y=c_x*\text{span(b(x))} = 1*(2-(-3))=5$$ 
3. Set coefficient for $z$: $$c_z = c_y * \text{span(b(y))} = 5 * (3-(-1))=20$$ We have now computed a coefficient vector $C=[1,5,20]$. 
4. Finally, calculate the bias $d = c_0*i + c_1*i + c_2*i = -1*1 + -1*5 + -1*20 = -26$ and can go on and construct the linear inequalities:
$$x+5y+20z \geq 26$$
$$-x-5y-20z \geq -26$$

Now there are limits to creating linear inequalities to fixing bounds this way that needs attention. The last coefficient's value grows exponentially with the number of variables (except the special case when all the span values of the bounds are 1 - then all coefficients are 1). If i64 is used as your data type, then the maximum coefficient value for the last coefficient is $9,223,372,036,854,775,807$, divided by the max abs value of the last variable bound.

## Restraining integer variables
All integer variables has an explicit bound set that is not boolean. Solvers may not directly set these bounds and so they need to be set in the polyhedron. We do this simply by creating two new linear inequalities per integer variables.
1. Find all integer variables $I$
2. For each variable $x \in I$, set (1) $x-x_{lb} \geq 0$ and (2) $-x+x_{ub}$, where $x_{lb}, x_{ub}$ subscripts denotes lower and upper bound for a variable $x$.
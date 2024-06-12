# How to transform $\pi \leftrightarrow \phi$, where $\pi$ is a boolean variable and $\phi$ is a linear inequality constraint, to a new linear inequality constraint.
First of all, there is no single linear inequality constraint that can represent both ways so we need to construct one way at the time. We'll start with $\pi \rightarrow \phi$. 

## A linear inequality constraint
Before we begin, we'll define some words. A linear inequality constraint consists of four parts: 
1) Discrete variables ranging from any integer $i \in \mathbb{Z}$ to another integer $j \in \mathbb{Z}$, such that $i \leq j$. For instance a variable $x \in [-2, 3]$ takes on any integer between -2 and 3, including.
2) Coefficients to each integer variable. A coefficient is an integer that is multiplied onto a variable when the inequality is evaluated.
3) An operator $\gt$, $\geq$, $\lt$ or $\leq$. Here we'll always assume $\geq$.
4) A bias. A constant integer, independent on any of the variable's values. 

Here are examples of linear inequality constraints:
- $x + y + z >= -1$ where $x,y,z \in [0,1]^3$
- $x + y + z >= -3$ where $x,y,z \in [0,1]^3$
- $-2x + y + z \geq 0$ where $x,y,z \in [0,1]^3$

## The <i>inner bound</i> of a linear inequality constraint
An important property is the inner bound $\text{ib}(\phi)$ of a linear inequality constraint $\phi$. It is the sum of all variable's step bounds, excl bias and before evaluated with the $\geq$ operator. For instance, the inner bound of the linear inequality $-2x + y + z \geq 0$, where $x,y,z \in \{0,1\}^3$, is $[-2, 2]$, since the lowest value the sum can be is $-2$ (given from the combination $x=1, y=0, z=0$) and the highest value is $2$ (from $x=0, y=1, z=1$).

## Negating an inequality constraint
To find the negation of a inequality constraint, following these two steps: 
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

For example, let $\phi = -a -b -c -d +5 \geq 0$ where $a \in [-5,3]$, $b \in [0,2]$, $c \in [-4,4]$ and $d \in [-4,5]$ and create a new inequality $\theta$ from $\phi \rightarrow \pi$. 
1. We start by negating $\phi$: $\neg \phi = a + b + c + d \geq 0$
2. Now we find the coefficient to $\pi$ by calculating $d = \text{max}(|\text{ib}(\phi')|)$:
    1. $d = \text{max}(|\text{ib}(a + b + c + d -6 \geq 0)|)$
    2. $d = \text{max}(|[-13, 14]|)$
    3. $d = \text{max}([13, 14])$
    4. $d = 14$
3. Append $(14-(-6))\pi = 20\pi$ to the left side of $\neg \phi: 20\pi + a + b + c + d -6 \geq 0$
4. $\theta = 20\pi + a + b + c + d -6 \geq 0$
# Physics-Informed Neural Networks

## Problem statement for 2D direct problems

We consider a partial differential equation (PDE) written in the following residual form as, 
```math
\mathcal{F}(u,x,y,u_x,u_y,...)=0,\quad(x,y)\in\Omega,
```
where $u(x,y)$ denotes the desired solution and $u_x, u_y, ...$ are the required associated partial derivatives of different orders with respect to $x$ and $y$. Specific conditions must be also imposed at the domain boundary $\partial \Omega$ depending on the problem (see below in the paper).

Note that the $(x,y)$ space variables can also include non-cartesian coordinates (see Lane-Emden equation).

## Problem statement for parametric and inverse problems

We consider a partial differential equation (PDE) written in the following residual form as, 
```math
\mathcal{F}(u,x,y,u_x,u_y,...,\theta)=0,\quad x\in\Omega,\quad\mu\in\Omega_{p},
```
where the desired solution is now $u(x,\mu)$, with $x$ being the space variable associated to the one dimensional domain $\Omega$ and $\mu$ is a scalar parameter taking different values in $\Omega_{p}$. For parametric problems, $\mu$ is treated exactly as a second variable in a 2D direct problem, but for inverse problems $\mu$ is consequently considered as an unknown. Boundary conditions (BC) are again necessary for parametric problems, but additional conditions, such as knowledge of the solution at some $x$ values must be added for inverse problems.

Note that, for the sake of simplicity, we have considered only one dimensional space variable in this work for parametric and inverse problems. The extension to higher spatial dimensions is however straightforward.
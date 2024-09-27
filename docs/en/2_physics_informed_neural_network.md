# Physics-Informed Neural Networks

## Problem statement for 2D direct problems

We consider a partial differential equation (PDE) written in the following residual form as, 
```math
\mathcal{F}(u,x,y,u_x,u_y,...)=0,\quad(x,y)\in\Omega,
```
where $u(x,y)$ denotes the desired solution and $u_x, u_y, ...$ are the required associated partial derivatives of different orders with respect to $x$ and $y$. Specific conditions must be also imposed at the domain boundary $\partial \Omega$ depending on the problem (see below in the paper).

Note that the $(x,y)$ space variables can also include non-cartesian coordinates (see Lane-Emden equation).

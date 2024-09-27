# Physics-Informed Neural Networks

## Problem statement for 2D direct problems

以下に示す残差形式で表される偏微分方程式(PDE)を、
```math
\mathcal{F}(u(x,y),u_x,u_y,...)=0, u(x,y)\in\Omega,
```
として考える。ここで、$u(x,y)$は所望の解を表し、 $u_x, u_y, ...$は $x$と $y$に関する異なる次数の必要な関連偏微分を表す。問題に応じて、領域境界 $\partial \Omega$にも特定の条件を課す必要がある(論文の後半を参照)。

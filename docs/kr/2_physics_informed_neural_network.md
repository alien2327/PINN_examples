# Physics-Informed Neural Networks

## Problem statement for 2D direct problems

다음과 같은 잔차식으로 쓰여진 편미분 방정식(PDE)을 고려해 보겠습니다. 
```math
\mathcal{F}(u,x,y,u_x,u_y,...)=0, (x,y)\in\Omega,
```
여기서 $u(x,y)$는 원하는 해를 나타내고 $u_x, u_y, ...$는 $x$와 $y$에 대해 다른 차수의 필요한 관련 편미분을 나타냅니다. 문제에 따라 도메인 경계 $\partial \Omega$에 특정 조건도 부과해야 합니다(아래 논문 참조).

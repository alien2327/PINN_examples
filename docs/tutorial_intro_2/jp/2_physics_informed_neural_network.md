# Physics-Informed Neural Networks

## Problem statement for 2D direct problems

以下の残差形式で書かれた偏微分方程式(PDE)を考える。
```math
\mathcal{F}(u,x,y,u_x,u_y,...)=0,\quad(x,y)\in\Omega,
```
ここで、$u(x,y)$ は所望の解を意味し、$u_x, u_y, ...$ は$x$と$y$に関する異なる次数の関連する部分微分である。問題に応じて、領域境界$\partial \Omega$にも特定の条件を課さなければならない(論文の下記を参照)。

また、空間変数$(x,y)$には非直交座標も含まれることに注意(レーン・エムデン方程式を参照)。

## Problem statement for parametric and inverse problems

以下の残留形式で書かれた偏微分方程式(PDE)を考える。
```math
\mathcal{F}(u,x,y,u_x,u_y,...,\theta)=0,\quad x\in\Omega,\quad\mu\in\Omega_{p},
```
ここで、目的の解は $u(x,\mu)$, となり、$x$ は1次元領域 $\Omega$ に関連する空間変数、そして $\mu$ は $\Omega_{p}$ で異なる値をとるスカラーパラメータです。パラメトリック問題では、$\mu$ は2次元直接問題における2番目の変数として正確に扱われますが、逆問題では $\mu$ は結果的に未知数とみなされます。パラメトリック問題では境界条件(BC)が再び必要ですが、逆問題では、いくつかの $x$ 値における解の知識など、追加の条件を追加する必要があります。

なお、本研究ではパラメトリック問題および逆問題について、単純化のため1次元の空間変数のみを考慮している。しかし、高次元空間への拡張は容易である。
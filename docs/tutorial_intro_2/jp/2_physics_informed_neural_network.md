# Physics-Informed Neural Networks

## Problem statement for 2D direct problems

以下の残差形式で書かれた偏微分方程式(PDE)を考える。
```math
\begin{equation}
\mathcal{F}(u,x,y,u_x,u_y,...)=0,\quad(x,y)\in\Omega,
\end{equation}
```
ここで、$u(x,y)$ は所望の解を意味し、$u_x, u_y, ...$ は$x$と$y$に関する異なる次数の関連する部分微分である。問題に応じて、領域境界$\partial \Omega$にも特定の条件を課さなければならない(論文の下記を参照)。

また、空間変数$(x,y)$には非直交座標も含まれることに注意(レーン・エムデン方程式を参照)。

## Problem statement for parametric and inverse problems

以下の残留形式で書かれた偏微分方程式(PDE)を考える。
```math
\begin{equation}
\mathcal{F}(u,x,y,u_x,u_y,...,\theta)=0,\quad x\in\Omega,\quad\mu\in\Omega_{p},
\end{equation}
```
ここで、目的の解は $u(x,\mu)$, となり、$x$ は1次元領域 $\Omega$ に関連する空間変数、そして $\mu$ は $\Omega_{p}$ で異なる値をとるスカラーパラメータです。パラメトリック問題では、$\mu$ は2次元直接問題における2番目の変数として正確に扱われますが、逆問題では $\mu$ は結果的に未知数とみなされます。パラメトリック問題では境界条件(BC)が再び必要ですが、逆問題では、いくつかの $x$ 値における解の知識など、追加の条件を追加する必要があります。

なお、本研究ではパラメトリック問題および逆問題について、単純化のため1次元の空間変数のみを考慮している。しかし、高次元空間への拡張は容易である。

## Classical deep learning approach with neural networks using training data

ニューラルネットワーク(NN)を使用する古典的なディープラーニングのアプローチでは、モデルは利用可能なトレーニングデータのみを使用してトレーニングされる。この方法では、ニューラルネットワークにインプットデータを送り、予測値と実際の出力値の差異を最小限に抑えるよう、トレーニングプロセスを通じて内部パラメータを調整する。モデルは、トレーニングデータセット内のパターンと関係性を学習し、新しい未確認データに対する予測を行う。このアプローチは、正確な予測を行うためにラベル付きのトレーニング例を活用することに重点を置く、さまざまな機械学習アプリケーションで一般的である。このように、ニューラルネットワークは非線形近似器として機能する。

|![Figure.1](../figures/figure_1.png)|
|:--|
|*Figure 1: 非線形近似問題に適用されるニューラルネットワーク(NN)の構造の概略図。入力層には、2つの空間座標変数 $x$ と $y$ に対して2つの入力変数(すなわち2つのニューロン)がある。各層に5つのニューロンを持つ3つの隠れ層が入力層と出力層に接続されており、後者には予測された解 $u_{\theta}(x,y)$ を表す1つの変数(1つのニューロン)がある。損失関数 $L_{data}(\theta)$ を用いた最小化の手順は、$u_{\theta}$ を2D領域 $\Omega$ で取得された値のトレーニングデータセット $u^{data}$ と比較することで得られる。この単純化された例では、$\theta$ は合計81個のスカラーパラメータを表す。*|

**ニューラルネットワークによる近似解。** 空間座標 $(x,y)$, 変数の組み合わせ $(x,\mu)$, または $x$ のみ、といった入力 $\boldsymbol{x}$ は、問題に応じて、解の値 $u(\boldsymbol{x})$ の近似値を計算でき、最終的にはパラメータ値 $\mu$ (逆問題の場合) を計算できるようにしたい。

このため、最も一般的なニューラルネットワークの1つである多層パーセプトロンと呼ばれるものを紹介する。他の統計モデルも代替的に使用できることに注意。目標は、パラメータ $\theta$ を調整して、$u_{\theta}$ がターゲットソリューション $u(\boldsymbol{x})$ に近似するようにすることである。 $u_{\theta}$ は非線形近似関数であり、$L+1$ 層のシーケンスに構成される。最初の層 $\mathcal{N}^0$ は入力層と呼ばれ、単純に
```math
\begin{equation}
\mathcal{N}^0(\boldsymbol{x}) = \boldsymbol{x}.
\end{equation}
```
各後続層 $l$ は、その重み行列 $W^l$ とバイアスベクトル $\boldsymbol{b}^{l}$ によってパラメータ化され、$d_{l}$ は層 $l$ の出力サイズとして定義される。$l\in[1,L-1]$は隠れ層と呼ばれ、その出力値は再帰的に定義できる。
```math
\begin{equation}
\mathcal{N}^{l}(\boldsymbol{x})=\sigma(\boldsymbol{W}^{l}\mathcal{N}^{l-1}(\boldsymbol{x})+\boldsymbol{b}^{l}),
\end{equation}
```
ここで、σは非線形関数であり、一般に活性化関数と呼ばれる。最も一般的に使用されるものは$\mathrm{ReLU}(\boldsymbol{x})=\mathrm{max}(\boldsymbol{x},0)$であるが、本研究ではPINNの構築にReLUよりも適している双曲線正接関数(tanh)を使用する。最終層は出力層であり、次式で定義される。
```math
\begin{equation}
\mathcal{N}^{L}(\boldsymbol{x})=\boldsymbol{W}^{L}\mathcal{N}^{L-1}(\boldsymbol{x})+\boldsymbol{b}^{L}.
\end{equation}
```
最後に、完全なニューラルネットワーク $u_{\theta}$ は、$u_{\theta}(\boldsymbol{x})=\mathcal{N}^{L}(\boldsymbol{x})$ と定義される。これは、非線形関数のシーケンスとして次のように表すこともできる
```math
\begin{equation}
u_{\theta}(\boldsymbol{x})=\left(\mathcal{N}^{L}\circ\mathcal{N}^{L-1}\circ\ldots\circ\mathcal{N}^{1}\circ\mathcal{N}^{0}\right)(\boldsymbol{x}).
\end{equation}
```
ここで、$\circ$ は関数の合成を表し、$\theta=\{\boldsymbol{W}^{l},\boldsymbol{b}^{l}\}_{l=1,L}$ はネットワークのパラメータを表す。

**訓練データを使用する教師あり学習アプローチ。** 古典的な教師あり学習アプローチでは、既知の入力と出力のペア $(\boldsymbol{x}, u)$ のデータセット $N_{data}$ を利用できると仮定する。
```math
\begin{equation}
    \mathcal{D}=\left\{\left(\boldsymbol{x}_{i}^{data},u_{i}^{data}\right)\right\}_{i=1}^{N_{data}},
\end{equation}
```
$i\in[1,N_{data}]$にある場合、$u_{\theta}$ は $u$ の良い近似値であると考えられる。予測値 $u_{\theta}(\boldsymbol{x}_{i})$ がすべてのデータサンプル $i$ に対してターゲット出力 $u_{i}^{data}$ に近い場合、$u_{\theta}$ は $u$ の良い近似値であると考えられる。したがって、データセットにおける予測誤差を最小化するには、以下の最適化問題の解である値 $\theta^{*}$を求めるのが自然である。
```math
\begin{equation}
    \theta^{*}=\mathrm{arg\underset{\theta}\,min}L_{data}(\theta),
\end{equation}
```
ここで
```math
\begin{equation}
    L_{data}(\theta)=\frac{1}{N_{data}}\sum_{i=1}^{N_{data}}\left|u_{\theta}(\boldsymbol{x}_{i})-u_{i}^{data}\right|^{2}.
\end{equation}
```
$L_{data}$は損失関数と呼ばれ、式(8)は学習問題である。定義された損失関数は平均二乗誤差の定式化に依存しているが、代替的な定式化も可能であることに言及しておくことは重要である。式(8)の解は、通常、(確率)勾配降下アルゴリズムによって求められる。このアルゴリズムは、ネットワークパラメータ$\theta$に対する損失$L_{data}$の勾配を計算するために、自動微分技術に依存している。このアルゴリズムは、あらかじめ設定された精度基準または指定された最大反復数に基づいて、最小値に収束するまで反復的に適用される。
```math
\begin{equation}
    \theta^{(j+1)}=\theta^{(j)}-l_{r}\nabla_{\theta}L_{data}(\theta^{(j)}),
\end{equation}
```
ここで、$L=L_{data}$、$j$番目の反復は文献ではエポックとも呼ばれ、$l_{r}$は学習率パラメータと呼ばれる。この作業では、よく知られたAdamオプティマイザを選択する。このアルゴリズムは、おそらく損失を減らすために、勾配の逆方向にネットワークパラメータ $(\theta)$ を反復的に更新する。ニューラルネットワークのパラメータ、すなわち重みとバイアスに関する微分を計算するには、標準的な自動微分技術が必要である(Baydin et al., 2018)。この手法は、複合関数の計算におけるさまざまなステップを保存し、その勾配を連鎖則を用いて計算する。実際には、特にPythonを使用する場合、TensorFlowやPyTorchなどのオープンソースソフトウェアライブラリを活用することで、学習プロセスが大幅に合理化される。これらのライブラリは、ニューラルネットワークモデルの構築、トレーニング、最適化のための事前実装された関数やツールを提供する。TensorFlowとPyTorchは、ユーザーフレンドリーなインターフェース、豊富なドキュメント、充実したコミュニティサポートを提供しており、ディープラーニングの研究者や実務者から高い人気を得ている。なお、本研究では、直接問題にはTensorFlowライブラリを、パラメトリック問題と逆問題にはPyTorchをそれぞれ使用している。

|![Figure.2](../figures/figure_2.png)|
|:--|
|Figure 2: ディリクレ型の境界条件(ソフト制約)を持つ2次元直接問題に関連する偏微分方程式の解法として適用された物理情報型ニューラルネットワークの構造の概略図。入力層には、2つの空間座標変数 $x$ と $y$ に対応する2つの入力変数(すなわち2つのニューロン)がある。 入力層と出力層には、各層に5つのニューロンを持つ3つの隠れ層が接続されており、後者には予測された解 $u_{\theta}(x,y)$ を表す1つの変数(1つのニューロン)がある。PDE 損失関数 $L_{PDE}(\theta)$ を形成するために必要な偏微分(すなわち、$u_{x,\theta}, u_{xy,\theta},...$)を評価するために、自動微分(AD)が手順で使用される。損失関数 $L_{data}(\theta)$ は、境界領域 $\partial\Omega$ にソフト制約(すなわち、トレーニングデータセットを介して)を課して取得される。|
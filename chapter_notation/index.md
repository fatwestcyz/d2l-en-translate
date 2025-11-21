# 符号
:label:`chap_notation`

在本书中，我们遵循以下符号约定。请注意，其中一些符号是占位符，而另一些则指向特定的对象。作为一条通用的经验法则，不定冠词 “a”（一个）通常表示该符号是一个占位符，以此类推，格式相似的符号可以表示同类型的其他对象。例如，“$x$：一个标量” 意味着小写字母通常代表标量值，但 “$\mathbb{Z}$：整数集” 则特指符号 $\mathbb{Z}$ 本身。


    
## 数值对象

* $x$: 一个标量
* $\mathbf{x}$: 一个向量
* $\mathbf{X}$: 一个矩阵
* $\mathsf{X}$: 一个一般张量
* $\mathbf{I}$: 单位矩阵（给定维度），即对角线元素为 $1$ 且非对角线元素为 $0$ 的方阵
* $x_i$, $[\mathbf{x}]_i$: 向量 $\mathbf{x}$ 的第 $i$ 个元素
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: 矩阵 $\mathbf{X}$ 在第 $i$ 行、第 $j$ 列的元素



## 集合论

* $\mathcal{X}$: 一个集合
* $\mathbb{Z}$: 整数集
* $\mathbb{Z}^+$: 正整数集
* $\mathbb{R}$: 实数集
* $\mathbb{R}^n$: $n$ 维实数向量的集合
* $\mathbb{R}^{a\times b}$: 包含 $a$ 行 $b$ 列的实数矩阵的集合
* $|\mathcal{X}|$: 集合 $\mathcal{X}$ 的基数（元素个数）
* $\mathcal{A}\cup\mathcal{B}$: 集合 $\mathcal{A}$ 和 $\mathcal{B}$ 的并集
* $\mathcal{A}\cap\mathcal{B}$: 集合 $\mathcal{A}$ 和 $\mathcal{B}$ 的交集
* $\mathcal{A}\setminus\mathcal{B}$: 集合 $\mathcal{A}$ 减去 $\mathcal{B}$ 的差集（仅包含属于 $\mathcal{A}$ 但不属于 $\mathcal{B}$ 的元素）



## 函数和运算符

* $f(\cdot)$：一个函数
* $\log(\cdot)$：自然对数（以 $e$ 为底）
* $\log_2(\cdot)$：以 $2$ 为底的对数
* $\exp(\cdot)$：指数函数
* $\mathbf{1}(\cdot)$：指示函数；如果布尔参数为真，则计算结果为 $1$，否则为 $0$
* $\mathbf{1}_{\mathcal{X}}(z)$：集合成员指示函数；如果元素 $z$ 属于集合 $\mathcal{X}$，则计算结果为 $1$，否则为 $0$
* $\mathbf{(\cdot)}^\top$：向量或矩阵的转置
* $\mathbf{X}^{-1}$：矩阵 $\mathbf{X}$ 的逆
* $\odot$：Hadamard（按元素）积
* $[\cdot, \cdot]$：连结（拼接）
* $\|\cdot\|_p$：$\ell_p$ 范数
* $\|\cdot\|$：$\ell_2$ 范数
* $\langle \mathbf{x}, \mathbf{y} \rangle$：向量 $\mathbf{x}$ 和 $\mathbf{y}$ 的内积（点积）
* $\sum$：对一系列元素的求和
* $\prod$：对一系列元素的连乘（求积）
* $\stackrel{\textrm{def}}{=}$：定义等号，即左侧符号的定义为右侧内容


## 微积分

* $\frac{dy}{dx}$: $y$ 关于 $x$ 的导数
* $\frac{\partial y}{\partial x}$: $y$ 关于 $x$ 的偏导数
* $\nabla_{\mathbf{x}} y$: $y$ 关于 $\mathbf{x}$ 的梯度
* $\int_a^b f(x) \;dx$: $f$ 在 $a$ 到 $b$ 区间上关于 $x$ 的定积分
* $\int f(x) \;dx$: $f$ 关于 $x$ 的不定积分



## 概率论与信息论

* $X$：一个随机变量
* $P$：一个概率分布
* $X \sim P$：随机变量 $X$ 服从分布 $P$
* $P(X=x)$：赋予随机变量 $X$ 取值 $x$ 这一事件的概率
* $P(X \mid Y)$：给定 $Y$ 时 $X$ 的条件概率分布
* $p(\cdot)$：与分布 $P$ 关联的概率密度函数 (PDF)
* ${E}[X]$：随机变量 $X$ 的期望
* $X \perp Y$：随机变量 $X$ 和 $Y$ 条件独立
* $X \perp Y \mid Z$：给定 $Z$ 时，随机变量 $X$ 和 $Y$ 条件独立
* $\sigma_X$：随机变量 $X$ 的标准差
* $\textrm{Var}(X)$：随机变量 $X$ 的方差，等于 $\sigma^2_X$
* $\textrm{Cov}(X, Y)$：随机变量 $X$ 和 $Y$ 的协方差
* $\rho(X, Y)$：$X$ 和 $Y$ 之间的皮尔逊相关系数，等于 $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$：随机变量 $X$ 的熵
* $D_{\textrm{KL}}(P\|Q)$：从分布 $Q$ 到分布 $P$ 的 KL 散度（或相对熵）



[讨论](https://discuss.d2l.ai/t/25)

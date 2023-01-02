---
title: 压缩感知初探
date: 2022-11-11 21:59:36
tags: [笔记]
---
## 关于压缩感知
原理就是一个欠定方程组
$$
    y=\phi  X
    ，其中\quad y=[y_i]_{M\times 1}，\phi=[\phi_{ij}]_{M\times N}， X=[X_i]_{N\times 1} \qquad
    M<<N
$$

$\phi$ 是观测基，前提条件是信号具有稀疏性或者可压缩性
y是（亚采样）观测到的信号，X是原信号，本质是已知y并且通过y还原X的过程
y是低频采样收集的观测数据，采样频率低于奈奎斯特采样频率，所以直接通过低频y还原**X**理论上不可行
做法是把X再进行稀疏变换变成$\alpha$（特别稀疏）

$$
\begin{array}{l}
{令}\quad X=\psi  \alpha \quad{其中，} \psi {作为稀疏基，} \alpha {仅有K个非零的} \alpha_k
\\
y=\phi\psi \alpha，其中\phi称为观测基，\psi为稀疏变换，\alpha为稀疏信号
\\
令A=\phi \psi \longrightarrow  y=A\alpha
\end{array}
$$
* 问题：可能本来$X=\psi \alpha$挺稀疏，但经过一个观测基y=$\phi$X之后，y变得不那么稀疏

<!-- more -->
![](https://gjy.pub/root/file/2022-11-13-22-31-33@123.png)
 常见的稀疏变换有傅里叶 ~，小波 ~等

 X变换成一个稀疏域$\alpha_{M\times1}$且M<<N，可以是频域，也可以是其他，变换过程叫做交换基，观测基也叫观测矩阵

 观测矩阵 $\phi$ 和稀疏变换矩阵 $\psi$ 不相关作为RIP准则，通常满足高斯随机的观测矩阵即可满足RIP准则

要想使信号完全重构，必须保证观测矩阵$\phi$不会把两个不同的K-项稀疏信号映射到同一个采样集合中，这就要求从观测矩阵中抽取的每M个列向量构成的矩阵是非奇异的

重点聚焦于重构算法：

在信号X稀疏或可压缩的前提下求解欠定方程组$y＝\phi\psi \alpha =A \alpha$的问题

找到最小L0范数解，$\alpha_{k*1}$需要确定$C_N^K$种可能的线性组合才有可能得到最优解

因此这样的计算极不稳定而且是NP困难问题
求解L1范数解相对于L0范数是简单的多的方案，可以使得问题变成一个凸优化问题，转而简单的用凸优化解决，常见有基追踪BP，线性规划LP

![](https://gjy.pub/root/file/2022-11-11-13-10-13@1.png)
![](https://gjy.pub/root/file/2022-11-11-13-10-42@2.png)
![](https://gjy.pub/root/file/2022-11-11-13-10-44@3.png)

基追踪法基于正则化，通过范数惩罚限制模型的能力，快速求得最优且稀疏的解
![](https://gjy.pub/root/file/2022-11-11-13-55-09@QQ截图20221111135458.png)

解一范数惩罚过的方程（其中 x 有稀疏要求），然后进行迭代。将上一次迭代算出的 x 带入做一个线性变换，得到 $\omega$，$\omega$ 经过一个软阈值函数（soft thresholding）后就得到这一次迭代的输出 x，重复该过程直到收敛。
$$
\begin{array}{l}
Problem: \frac{1}{2}||y-Ax||_2^2+\lambda ||x||_1 \quad y \in R^d,A \in R^{d\times n}
\\
Input:x_0 \in R^n \quad and\quad L \ge \lambda_{max}(A^T A)
\\
while\quad x_k\quad not\quad converged \qquad(k=1,2,3...)\quad do
\\
\qquad \omega_k \longleftarrow x_k +\frac{1}{L}A^T(y-Ax_k)
\\
\qquad x_{k+1} \longleftarrow  soft(\omega_k,\frac{\lambda}{L})
\\
end\quad while
\\
Output:x_* \longleftarrow x_{k}
\end{array}
$$

$$
\begin{array}{l}
\\
上上面的公式：\\
X=\psi \alpha 
\\
y=\phi\psi \alpha，其中\phi称为观测基，\psi为稀疏变换，\alpha为稀疏信号
\\
令A=\phi \psi \longrightarrow  y=A\alpha
\end{array}
$$

* 问题：稀疏字典 **$A \surd$** ，**$y \surd$** ，需要求解稀疏的 **$x ?$**
* 问题：这里的 $x$ 是不是相当于上面的 $\alpha$，在求出x之后还需要利用X=$\psi x$ 还原真实信号 $X$，最终目的是求 $X$
* 问题：这样是不是忽略了$\phi$和$\psi$，而是聚焦于利用A寻找最稀疏$x$的过程

把这个迭代过程画出来，我们得到的结构是线性算子加阈值函数，而且阈值函数的样子和激活函数 ReLU 长得很像——这就是一个神经网络。

完全从模型推导出来的最优的、收敛速度最快的算法，和深度学习通过经验找到的神经网络非常相似(软阈值迭代算法ISTA)

![](https://gjy.pub/root/file/2022-11-12-14-17-35@QQ截图20221112141728.png)
![](https://gjy.pub/root/file/2022-11-12-19-28-23@xinghaotu.png)


L2O即从其在样本问题上的表现中学习。该方法可能缺乏坚实的理论基础，但在训练过程中提高了其性能。训练过程经常离线进行，而且非常耗时。然而，该方法的在线应用是（旨在）节省时间。当遇到难以获得目标解决方案的问题时，例如非凸优化和逆问题应用，经过良好训练的 L2O 方法的解决方案可以比经典方法的解决方案具有更好的质量。让我们将优化方法（手动设计或由 L2O 训练）称为优化器，并将可通过该方法解决的优化问题称为优化器。2 图 1 比较了经典优化器和 L2O 优化器，并说明了它们如何应用于优化器（黄色框）。

应用上，L2O 优化器的收敛速度比两种流行的迭代求解器（ISTA、FISTA）快得多。

* 问题：L2O是不是可以理解为一句话：之前手工设计迭代和更新的方式，通过循环进行迭代，硬算。而L2O将迭代和循环替换成层和神经网络的概念，通过深度的网络学习进行优化？




Ada-LISTA的引入
* 引入了一种称为 Ada-LISTA 的自适应学习求解器，它接收成对的信号及其相应的字典作为输入，并学习一个通用架构来为它们提供服务。
* 该架构通过两个辅助权重矩阵包装字典。在推理时，我们的模型可以容纳信号及其相应的字典，允许处理各种模型修改而无需重复的重新训练。
$$
\begin{array}{l}
Algorithm 2 Ada-LISTA Training \\
Input: pairs of signals and dictionaries {yi, Di}N i=1 \\
Preprocessing: find xi for each pair (yi, Di) by solving \\
Equation 2 using ISTA \\
Goal: learn Θ = (W1, W2, θk, γk) \\
Init: W1, W2 = I, θk, γk = 1 \\
for\quad each\quad batch\quad {yi, Di, xi}\quad NB i=1 do\\ update\quad Θ\quad by\quad ∂Θ ∑ i∈NB ‖FK (yi, Di; Θ) − xi‖2 2 \\
end for
\end{array}
$$
![](https://gjy.pub/root/file/2022-11-12-16-04-08@jsu_ratio_0.5_T_20_lambd_0.1.png)

* 如何理解这句话？
```
字典封装的模型经常被额外的常数扰动破坏
例如压缩感知中的感知矩阵（Kulkarni 等人，2016），非盲图像去模糊中的模糊核（ Tang 等人，2014 年），以及图像修复中的空间变化掩码（Mairal 等人，2007 年）。
在所有这些情况下，LISTA 经典框架的部署都需要为每个新词典重新训练网络。

我们证明了 Ada-LISTA 对三种字典扰动的鲁棒性：
1.置换列
2.加高斯噪声
3.完全随机的字典
我们展示了我们的模型处理复杂和变化的信号模型的能力，同时仍然比学习和非学习求解器提供了令人印象深刻的优势。
```

> 需要解决的
1.如何让信号更加稀疏，关键是找到一个合适的稀疏表示矩阵
2.如何设计一个压缩观测矩阵，既要易于硬件实现，又要满足“约束等距特性”RIP
3.怎样求解一个非凸优化问题。也就是如何快速的重构。
这三个问题须从理论上突破，主要聚焦于重构算法的可能性

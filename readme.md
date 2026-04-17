# Kokkos-SDE 太阳粒子疏运模拟器 (Parker & Focused Transport)

## 1. 项目简介
本项目是一个基于 **Kokkos** 高性能计算框架开发的随机微分方程 (SDE) 求解器。其核心目标是模拟高能带电粒子（如太阳高能粒子 SEP、银河宇宙线 GCR）在日球层复杂的磁场与太阳风环境中的宏观疏运过程。

与传统的洛伦兹力轨迹追踪不同，本项目通过求解 **Parker 疏运方程** 和 **聚焦疏运方程**，利用统计学方法（随机行走）获取粒子在行星际空间中的分布、通量及能谱演化。

## 2. 核心物理模型
项目支持两种主要的疏运模型，通过 SDE (Stochastic Differential Equations) 方法实现：

*   **Parker 疏运方程 (Parker Transport Equation)**:
    *   适用于大尺度全方向同性/弱各向异性模拟。 
    * 包含项：对流 (Convection)、扩散 (Diffusion)、磁场漂移 (Drift)、绝热减速 (Adiabatic Cooling)。
*   **聚焦疏运方程 (Focused Transport Equation)**:
    *   适用于研究 SEP 的各向异性爆发。
    *   包含项：沿磁力线的流体运动、节角散射 (Pitch-angle Scattering)、磁聚焦效应 (Magnetic Focusing)。

## 3. 目前支持的湍流扩散系数计算

本节以下默认采用 Gaussian-CGS 单位制：

* 长度：cm
* 时间：s
* 速度：cm/s
* 动量：g cm/s
* 磁场：G
* 电荷：statC (esu)
* 空间扩散系数：cm$^2$/s
* 节角扩散系数 $D_{\mu\mu}$：s$^{-1}$
* 回旋频率 $\Omega_0$：s$^{-1}$

当湍流功率谱取 $\frac{5}{3}$ 时：

$$
\kappa_\parallel = 1.622 \frac{V_{sw}^{4/3} L_c^{2/3}}{\Omega_0^{1/3} \sigma^2}
$$

$$
\kappa_\perp = \frac{V_{sw}}{3} \left(\frac{3\kappa_\parallel}{V_{sw}}\right)^{1/3} \left(0.198\sigma^2_{2D}L_{2D}\right)^{2/3}
$$

With the $2/3$ exponent on the perpendicular turbulence scale, the pure-CGS
dimensional check is

$$
\left[\kappa_\perp\right]_{\rm RHS}
= \frac{\rm cm}{\rm s} \cdot {\rm cm}^{1/3} \cdot {\rm cm}^{2/3}
= {\rm cm}^{2} {\rm s}^{-1}
$$

Therefore, if $\sigma^2_{2D}$ is the usual dimensionless turbulence-amplitude ratio
and the prefactor $0.198$ is treated as dimensionless, this $\kappa_\perp$ expression
has the expected ${\rm cm}^2 {\rm s}^{-1}$ diffusion-coefficient dimension.

The two empirical diffusion formulas in this section are dimensionally closed in the
current CGS form. The remaining open issue is model provenance rather than unit
closure: the prefactor $0.198$ and the exact source convention for
$\sigma^2_{2D}L_{2D}$ still need to be traced to the original reference before this
branch is used as the default production model. Until then, the conservative production
branch is

$$
\kappa_\perp = a\kappa_\parallel
$$

with an explicitly configured constant ratio $a$.

## 4. Parker输运方程

数学形式：

$$
d\mathbf{X} = \left(\mathbf{V}_{sw} + \mathbf{V}_d + \nabla \cdot \mathbf{K}\right) dt + \sum_{\sigma} \mathbf{A}_{\sigma} \, dW_{\sigma}
$$

$$
dp = -\frac{p}{3} \left(\nabla \cdot \mathbf{V}_{sw}\right) dt
$$

这里用 $\mathbf{K}$ 表示空间扩散张量，用 $\mathbf{A}_{\sigma}$ 表示满足扩散关系的随机增量系数向量。

$$
\kappa_{ij} = \kappa_\perp \delta_{ij} - (\kappa_\perp - \kappa_\parallel) b_i b_j
$$

$$
\sum_{\sigma} \mathbf{A}_{\sigma} \mathbf{A}_{\sigma}^{T} = 2 \mathbf{K}
$$

$$
\mathbf{V}_d = \frac{pvc}{3q} \nabla \times \left(\frac{\mathbf{B}}{B^2}\right)
$$

对插值实现，更适合保存的形式是

$$
\mathbf{G} \equiv \frac{\mathbf{B}}{B^2}, \qquad \mathbf{V}_d = \frac{pvc}{3q} \nabla \times \mathbf{G}
$$

这样只需在网格上保存并插值 $\mathbf{G}$ 的三个分量，而不必分别保存 $\mathbf{B}$、$B$ 以及额外的漂移辅助场。

在 Gaussian-CGS 下，更稳妥的量纲检查方式是先引入回旋半径

$$
r_g \equiv \frac{pc}{qB}, \qquad [r_g] = {\rm cm}
$$

于是

$$
\mathbf{V}_d = \frac{v r_g}{3} \, B \, \nabla \times \left(\frac{\mathbf{B}}{B^2}\right)
$$

而

$$
\left[B \nabla \times \left(\frac{\mathbf{B}}{B^2}\right)\right] = {\rm cm}^{-1}
$$

因此

$$
\left[\mathbf{V}_d\right] = \frac{\rm cm}{\rm s} \cdot {\rm cm} \cdot {\rm cm}^{-1}
= {\rm cm}\,{\rm s}^{-1}
$$

与漂移速度的量纲一致。

### 4.1 Parker time-step control

All time-step estimates below assume that the solver has already converted physical
quantities to the internal nondimensional system. For a local coordinate spacing
$\Delta \xi_i$, the corresponding physical orthogonal distance is

$$
\Delta s_i = h_i(\mathbf{x}) \Delta \xi_i ,
$$

where $h_i$ is the Lame coefficient of the orthogonal coordinate system. The local
deterministic spatial transport velocity is

$$
U_i =
V_{sw,i} + V_{d,i} + \left(\nabla \cdot \mathbf{K}\right)_i .
$$

If the exact historical Parker-SDE limiter without explicit drift is required, use
$U_i = V_{sw,i} + (\nabla \cdot \mathbf{K})_i$ in the advective denominator. The
production default should include $\mathbf{V}_d$ because gradient and curvature drift
can dominate the local deterministic displacement in weak-flow regions.

#### 4.1.1 SDE transport-displacement limiter

This limiter is intended for the stochastic Parker solver, and it can also be used by
the deterministic-only prototype as a conservative precursor to the later stochastic
term. It combines a diffusion RMS-displacement bound and a drift-diffusion balance
bound:

$$
\Delta t_{{\rm diff},i}
= \frac{(\eta_{\rm diff}\Delta s_i)^2}{2\kappa_{{\rm eff},i}},
\qquad
\eta_{\rm diff}=0.5 ,
$$

$$
\Delta t_{{\rm bal},i}
= \frac{2\kappa_\perp}{\max(U_i^2,\epsilon_U)} .
$$

Here $\kappa_{{\rm eff},i}$ should be the tensor-projected diffusion coefficient
along the local coordinate direction,

$$
\kappa_{{\rm eff},i} = \mathbf{e}_i^T\mathbf{K}\mathbf{e}_i ,
$$

when that projection is available. Using $\kappa_\parallel$ is a conservative fallback
for strongly field-aligned diffusion because $\kappa_\parallel \ge \kappa_\perp$ in the
usual Parker transport regime. The balance bound follows from requiring the
deterministic displacement to stay below the perpendicular stochastic RMS displacement:

$$
|U_i|\Delta t \lesssim \sqrt{2\kappa_\perp\Delta t}.
$$

The local SDE-style time step is therefore

$$
\Delta t_{\rm SDE}
= C_{\rm courant}
\min_i\left(
\Delta t_{{\rm diff},i},
\Delta t_{{\rm bal},i}
\right).
$$

The current Parker solver default enables the diffusion-resolution part of this SDE
limiter and combines it with the grid-CFL and momentum-change limiters below. The
drift-diffusion balance bound is kept as an explicit optional limiter because it is an
accuracy choice rather than a strict stability requirement.

#### 4.1.2 Grid-CFL and momentum-change limiter

For a deterministic-only Parker step, or as an additional guard for the stochastic
solver, the more conventional grid-based limiter is

$$
\Delta t_{{\rm adv},i}
= \frac{\eta_{\rm adv}\Delta s_i}{\max(|U_i|,\epsilon_U)} ,
$$

which is the ordinary advective CFL condition in the local physical coordinate
distance. The momentum equation gives

$$
\frac{1}{p}\frac{dp}{dt}
= -\frac{1}{3}\nabla\cdot\mathbf{V}_{sw},
$$

so a relative momentum-change limiter is

$$
\Delta t_p
= \frac{3\eta_p}{\max(|\nabla\cdot\mathbf{V}_{sw}|,\epsilon_{\nabla V})}.
$$

The grid-CFL time step keeps the diffusion-resolution bound, the deterministic
spatial CFL bound, and the momentum-change bound separate:

$$
\Delta t_{\rm grid}
= C_{\rm courant}
\min\left[
\min_i\left(\Delta t_{{\rm diff},i},\Delta t_{{\rm adv},i}\right),
\Delta t_p
\right].
$$

In production runs, the active Parker time step should be the minimum of all enabled
limiters. The SDE transport-displacement limiter controls stochastic displacement
quality, while the grid-CFL limiter controls deterministic spatial and momentum
resolution. Keeping both forms explicit avoids hiding a numerical accuracy choice
inside a single empirical time-step expression.

## 5. 聚焦输运方程

数学形式：

$$
d\mathbf{X} = \left(v\mu \mathbf{b} + \mathbf{V}_{sw} + \mathbf{V}_d + \nabla \cdot \mathbf{K}_{\perp}\right) dt + \sum_{\sigma} \mathbf{A}_{\sigma} \, dW_{\sigma}
$$

$$
dp = -p \left[\frac{1-\mu^2}{2}\left(\nabla \cdot \mathbf{V}_{sw} - \mathbf{b}\mathbf{b} : \nabla \mathbf{V}_{sw}\right) + \mu^2 \mathbf{b}\mathbf{b} : \nabla \mathbf{V}_{sw} + \frac{\mu}{v} \mathbf{b} \cdot \frac{d\mathbf{V}_{sw}}{dt}\right] dt
$$

$$
d\mu = \left\{\frac{1-\mu^2}{2}\left[-v \mathbf{b} \cdot \nabla (\ln B) + \mu \nabla \cdot \mathbf{V}_{sw} - 3\mu \, \mathbf{b}\mathbf{b} : \nabla \mathbf{V}_{sw} - \frac{2}{v} \mathbf{b} \cdot \frac{d\mathbf{V}_{sw}}{dt}\right] + \frac{\partial D_{\mu\mu}}{\partial \mu}\right\} dt + \sqrt{2 D_{\mu\mu}} \, dW_{\mu}
$$

这里用 $\mathbf{K}_{\perp}$ 表示垂直扩散张量，用 $\mathbf{A}_{\sigma}$ 表示满足扩散关系的随机增量系数向量。

$$
\sum_{\sigma} \mathbf{A}_{\sigma} \mathbf{A}_{\sigma}^{T} = 2 \mathbf{K}_{\perp}
$$

$$
\mathbf{V}_d = \frac{pvc}{qB} \left[\frac{1-\mu^2}{2} \mathbf{b} \times \nabla \ln B + \mu^2 \mathbf{b} \times (\mathbf{b} \cdot \nabla)\mathbf{b}\right]
$$

$$
D_{\mu\mu} = \frac{\pi \Omega_0}{4} (1-\mu^2)
\frac{\sigma^2 \frac{\Omega_0 L_c}{|\mu| v}}
{1+\left(\frac{\Omega_0 L_c}{|\mu| v}\right)^{\gamma_k}}
\left[\frac{\pi}{\gamma_k \sin\left(\frac{\pi}{\gamma_k}\right)}\right]^{-1}
$$

$$
\gamma_k = \frac{5}{3}
$$

其中

$$
\left[D_{\mu\mu}\right] = \left[\Omega_0\right] = {\rm s}^{-1}
$$

因为 $(1-\mu^2)$、$\sigma^2$、$\Omega_0 L_c/(|\mu|v)$ 以及最后的谱函数因子都为无量纲量。

### 5.1 公式检查后的更正说明

上面一版公式相对于此前草稿做了以下修正：

* Wiener 增量统一写为 $dW$；
* $dp$ 方程补上了末尾的 $dt$；
* $\mathbf{b}\cdot d\mathbf{V}_{sw}/dt$ 的点乘被补齐，否则量纲不对；
* $d\mu$ 的随机项改为 $\sqrt{2D_{\mu\mu}} \, dW_{\mu}$，与前面的扩散记号
  $\sum_{\sigma} \mathbf{A}_{\sigma} \mathbf{A}_{\sigma}^{T} = 2\mathbf{K}$ 保持一致；
* $D_{\mu\mu}$ 中的 $\mu$ 改为 $|\mu|$，这样 $D_{\mu\mu}$ 关于 $\mu$ 是偶函数，并且不会在 $\mu<0$ 时出现非物理负值；
* 聚焦输运中的漂移速度保留为标准的梯度漂移 + 曲率漂移形式，去掉了原草稿中沿 $\mathbf{B}$ 方向的项，因为漂移速度应当垂直于磁场。
* 从 CGS 量纲上看，Parker 方程、Focused 方程、$D_{\mu\mu}$、$\partial D_{\mu\mu}/\partial\mu$、两种 $\mathbf{V}_d$ 写法，以及第 3 节中的两条扩散系数经验式，按当前写法都是自洽的。

### 5.2 适合插值实现的简化表达式

对 $D_{\mu\mu}$，先定义两个只和局域场值有关的插值量

$$
A(\mathbf{x},p) \equiv \frac{\pi \Omega_0(\mathbf{x},p)\sigma^2(\mathbf{x})}{4}
\left[\frac{\pi}{\gamma_k \sin\left(\frac{\pi}{\gamma_k}\right)}\right]^{-1},
\qquad
\Xi(\mathbf{x},p) \equiv \frac{\Omega_0(\mathbf{x},p)L_c(\mathbf{x})}{v}
$$

则

$$
D_{\mu\mu} = A(1-\mu^2)\frac{\Xi/|\mu|}{1+\left(\Xi/|\mu|\right)^{\gamma_k}}
= A(1-\mu^2)\frac{\Xi |\mu|^{\gamma_k-1}}{|\mu|^{\gamma_k}+\Xi^{\gamma_k}}
$$

这样在数值实现时，只需对两个标量场 $A$ 和 $\Xi$ 做插值，而不必分别插值
$\Omega_0$、$\sigma^2$、$L_c$ 三个场后再重复组合。

对 $\partial D_{\mu\mu}/\partial\mu$，不必直接对完整表达式重复求导；若记

$$
\beta \equiv \frac{\Xi}{|\mu|}, \qquad |\mu| > 0
$$

则可直接复用已经算好的 $D_{\mu\mu}$，写成

$$
\frac{\partial D_{\mu\mu}}{\partial \mu} = sgn(\mu) D_{\mu\mu} \left( -\frac{2|\mu|}{1-\mu^2} - \frac{1+(1-\gamma_k)\beta^{\gamma_k}}{|\mu|(1+\beta^{\gamma_k})} \right)
$$

这比直接对原式展开求导更适合 GPU 实现：需要插值的场更少，重复乘幂与重复组合的次数也更少。实际实现时应使用

$$
|\mu| \leftarrow \max(|\mu|,\mu_{\min})
$$

来避免 $\mu \to 0$ 的数值奇异性。

对漂移速度 $\mathbf{V}_d$，若直接使用上面的简化式

$$
\mathbf{V}_d = \frac{pvc}{qB} \left[\frac{1-\mu^2}{2} \mathbf{b} \times \nabla \ln B + \mu^2 \mathbf{b} \times (\mathbf{b} \cdot \nabla)\mathbf{b}\right]
$$

则在插值实现里，不必分别保存
$\nabla B$、$(\mathbf{B}\cdot\nabla)\mathbf{B}$、$\mathbf{B}\cdot\nabla\times\mathbf{B}$ 等多个辅助场。

更适合的做法是：

* Parker 漂移：在网格上保存 $\mathbf{G}=\mathbf{B}/B^2$；
* Focused 漂移：在网格上保存 $\mathbf{b}=\mathbf{B}/B$ 与 $\ln B$，并由插值后的场直接求
  $\nabla\ln B$ 和 $(\mathbf{b}\cdot\nabla)\mathbf{b}$。

这样能同时减少内存占用和每步粒子推进时的重复计算。
---
**维护者**: [Yihan Liu/SDU]  
**许可证**: MIT License

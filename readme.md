# Kokkos 粒子输运

基于 Kokkos 的高能粒子输运与辐射合成原型，面向太阳爆发、重联区粒子加速、SEP/GCR 输运等问题。项目使用蒙特卡罗方法求解随机微分方程，当前重点是 Parker 输运方程、Focused 输运方程，以及基于粒子快照的微波辐射合成工作流。

英文版说明见 [README.en.md](README.en.md)。

## 功能概览

- Kokkos 后端：支持 OpenMP CPU 调试/基准测试，并保留 CUDA/HIP 可移植路径。
- Parker 输运：包含太阳风对流、扩散张量、漂移项和绝热动量变化。
- Focused 输运：包含沿磁场传播、俯仰角散射、磁聚焦和动量变化。
- Athena++/Fortran MHD 输入：可将 2D MHD 背景场转换为求解器使用的紧凑场。
- 粒子诊断输出：支持 `summary` CSV、动量直方图和第 4 版粒子二进制快照。
- 辐射合成：将紧凑场、粒子快照和热背景图转换为微波图像、频谱和 ProRes 电影。
- 后处理工具：提供 HDF5/XDMF 转换、基准测试频谱/计时图和 Streamlit 辐射查看器。

## 仓库结构

```text
include/                 核心网格、场、粒子、求解器、系数和单位头文件
src/main.cpp             Parker/Focused 输运的二维重联标定驱动
apps/                    场转换、粒子转换和基准测试绘图工具
particleEmission/        微波辐射合成、验证和查看器工具
scripts/                 CPU/GPU 基准测试、验证和电影流程封装脚本
docs/code_map.md         当前代码归属和接口地图
emissionValidation/      已纳入版本管理的合成辐射验证产物
kokkos_cpu_format_example_frame180_262k/
                          已纳入版本管理的紧凑场与粒子输出示例
pygsfit_cp-main/         随仓库提供的外部陀螺同步辐射后端源码，供参考使用
```

本地生成结果写入 `benchmark_runs/`，该目录已刻意从 Git 跟踪中排除。

## 环境需求

### C++ / Kokkos

- CMake 3.24+
- C++20 编译器
- 用于 CPU 运行的带 OpenMP 支持的 Kokkos 安装
- 用于 GPU 运行的可选 CUDA Kokkos 安装
- 用于李小灿 Fortran 参考对比的可选 MPI/Fortran/HDF5 工具栈

常见的本地 CPU Kokkos 路径：

```bash
/usr/local/kokkos_cpu/lib/cmake/Kokkos
```

### Python

后处理环境需要包含：

```bash
numpy
h5py
matplotlib
streamlit
imageio-ffmpeg
```

示例环境配置：

```bash
python3 -m venv /tmp/kpt_postprocess_venv
/tmp/kpt_postprocess_venv/bin/python -m pip install -r particleEmission/requirements.txt
```

当系统未安装 `ffmpeg` 时，`imageio-ffmpeg` 会提供随包附带的 ffmpeg 可执行文件。

## 构建

CPU 构建：

```bash
cmake -S . -B cmake-build-benchmark-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native" \
  -DKokkos_DIR=/usr/local/kokkos_cpu/lib/cmake/Kokkos

cmake --build cmake-build-benchmark-cpu -j 16
```

如果已有启用 CUDA 的 Kokkos 安装，可以构建 CUDA 版本：

```bash
cmake -S . -B cmake-build-benchmark-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/local/kokkos/bin/nvcc_wrapper \
  -DKokkos_DIR=/usr/local/kokkos/lib/cmake/Kokkos

cmake --build cmake-build-benchmark-cuda -j 16
```

## 输入数据

标定驱动读取紧凑二维背景场：

```text
int32 nx
int32 ny
double x_edges[nx + 1]
double y_edges[ny + 1]
double Bx[nx * ny]
double By[nx * ny]
double Bz[nx * ny]
double Vx[nx * ny]
double Vy[nx * ny]
double Vz[nx * ny]
```

当前提供两种转换工具：

```bash
# Athena++ .athdf -> 紧凑场，可选生成辐射背景图
/tmp/kpt_postprocess_venv/bin/python apps/athena2bin.py \
  --input-glob "/path/to/reconnection.prim.*.athdf" \
  --output-dir benchmark_runs/compact_field \
  --emission-background-dir benchmark_runs/emission_background \
  --start-frame 0 \
  --end-frame 200

# 李小灿 Fortran MHD 二进制数据 -> 紧凑场
cmake-build-benchmark-cpu/fortran_mhd_to_compact_field \
  --input-dir /path/to/bin_data \
  --output-dir benchmark_runs/compact_field \
  --start-frame 0 \
  --end-frame 200
```

如果只需要热背景/背景辐射图，可以使用 `apps/athena2bin.py --skip-compact-output`。

## 运行输运

冒烟测试：

```bash
cmake-build-benchmark-cpu/kokkos_particle_transport_app \
  --profile smoke \
  --transport parker \
  --field-dir benchmark_runs/compact_field \
  --output-dir benchmark_runs/smoke \
  --frames 1
```

带逐帧粒子快照输出的 Parker CPU 运行：

```bash
OMP_NUM_THREADS=16 KOKKOS_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=threads \
cmake-build-benchmark-cpu/kokkos_particle_transport_app \
  --profile fortran-global \
  --transport parker \
  --field-dir benchmark_runs/compact_field \
  --output-dir benchmark_runs/kokkos_cpu \
  --start-frame 0 \
  --end-frame 200 \
  --particles-per-frame 1600 \
  --rank-scale 16 \
  --capacity 16000000 \
  --histogram-interval 10 \
  --particle-snapshot-interval 1
```

Focused 输运使用：

```bash
--transport focused
```

## 运行时间控制

Kokkos 驱动可以在完成某一帧后干净停止：

```bash
--walltime-hours 10 \
--walltime-reserve-minutes 30
```

Shell 封装脚本通过以下环境变量暴露同样的控制：

```bash
KPT_KOKKOS_WALLTIME_HOURS=10
KPT_KOKKOS_WALLTIME_RESERVE_MINUTES=30
```

该停止机制是帧边界上的诊断停止，不是重启/检查点系统。

## 基准测试封装脚本

常用封装脚本：

```bash
scripts/run_reconnection_three_cases.sh
scripts/run_reconnection_optimized_comparison.sh
scripts/run_reconnection_focused_comparison.sh
scripts/run_local_parker_focused_speed_accuracy_test.sh
```

这些脚本可以运行 Fortran、Kokkos CPU 和 Kokkos GPU 对比，并在 `benchmark_runs/` 下生成计时和频谱产物。

## 辐射合成与电影流程

针对单个 Parker 粒子快照：

```bash
scripts/run_parker_emission_synthesis.sh
```

针对完整 CPU Parker 重跑并随后渲染电影：

```bash
KPT_RUN_ROOT=benchmark_runs/kokkos_cpu_parker_movie_200 \
KPT_FIELD_DIR=benchmark_runs/local_parker_speed_accuracy/compact_field \
KPT_PREPARE_COMPACT_FIELD=0 \
KPT_START_FRAME=0 \
KPT_END_FRAME=200 \
KPT_MOVIE_START_FRAME=1 \
KPT_MOVIE_END_FRAME=200 \
KPT_PARTICLE_SNAPSHOT_INTERVAL=1 \
KPT_CPU_CORES=16 \
KPT_KOKKOS_CPU_THREADS=16 \
scripts/run_kokkos_cpu_parker_movie_pipeline.sh
```

电影渲染器为每一帧生成一个 2x4 面板：

1. `Bz`
2. `Jz = dBy/dx - dBx/dy`
3. 低于配置能量阈值的粒子权重
4. 高于配置能量阈值的粒子权重
5. 经过波束卷积的 1 GHz 图像
6. 经过波束卷积的 3 GHz 图像
7. 经过波束卷积的 5 GHz 图像
8. 全区域频谱

默认电影设置：

```text
图像网格：128 x 128
波束 FWHM：3 像素
编码器：通过 ffmpeg 或 imageio-ffmpeg 使用 ProRes HQ
```

逐帧粒子快照体积较大。本地默认规模下，一个 200 帧 Parker 运行可能写出 `50+ GB` 粒子数据。

## 物理模型概述

### Parker 输运

Parker SDE 可示意写为：

$$d\mathbf{X} = \mathbf{u}_X dt + \sum_\sigma \mathbf{A}_\sigma dW_\sigma .$$

$$\mathbf{u}_X = \mathbf{V}_{sw} + \mathbf{V}_d + \nabla \cdot \mathbf{K}.$$

$$dp = -\frac{p}{3}\left(\nabla \cdot \mathbf{V}_{sw}\right)dt .$$

其中

$$\kappa_{ij} = \kappa_\perp \delta_{ij} - (\kappa_\perp - \kappa_\parallel)b_i b_j .$$

$$\sum_\sigma \mathbf{A}_\sigma \mathbf{A}_\sigma^T = 2\mathbf{K}.$$

一种方便的漂移表示为：

$$\mathbf{G} = \frac{\mathbf{B}}{B^2}.$$

$$\mathbf{V}_d = \frac{pvc}{3q}\nabla \times \mathbf{G}.$$

### Focused 输运

Focused 输运分支演化位置、动量和俯仰角余弦：

$$d\mathbf{X} = \mathbf{u}_X dt + \sum_\sigma \mathbf{A}_\sigma dW_\sigma .$$

$$\mathbf{u}_X = v\mu\mathbf{b} + \mathbf{V}_{sw} + \mathbf{V}_d + \nabla \cdot \mathbf{K}_\perp .$$

这里的 Focused 漂移速度包含梯度漂移和曲率漂移。通用 `FocusTransportSolver` 预计算两个漂移基：

$$\mathbf{G}_B = \frac{\mathbf{b}\times\nabla\ln B}{B},\quad \mathbf{G}_c = \frac{\mathbf{b}\times(\mathbf{b}\cdot\nabla)\mathbf{b}}{B}.$$

$$\mathbf{V}_d = \frac{pvc}{q}\left[\frac{1-\mu^2}{2}\mathbf{G}_B + \mu^2\mathbf{G}_c\right].$$

若沿用旧 Fortran 记号，`G_mumu` 对应这里的俯仰角扩散系数 `D_mumu`。通用散射模型使用：

$$d\mu = \left(F_\mu + \frac{\partial D_{\mu\mu}}{\partial\mu}\right)dt + \sqrt{2D_{\mu\mu}}dW_\mu.$$

$$\Omega = \frac{|q|B}{\gamma mc},\quad \xi = \frac{\Omega L_c}{v},\quad \mu_h = |\mu| + h_0,\quad C_\gamma = \frac{\gamma_k\sin(\pi/\gamma_k)}{\pi}.$$

$$D_{\mu\mu} = \frac{\pi\Omega\sigma^2}{4}C_\gamma(1-\mu^2)\frac{\xi\mu_h^{\gamma_k-1}}{\mu_h^{\gamma_k}+\xi^{\gamma_k}}.$$

当前 `src/main.cpp` 的二维重联标定驱动为了匹配 Fortran local example，使用 reduced 形式：

$$D_{\mu\mu}^{2D} = d_{uu0}(1-\mu^2)(|\mu|^{\gamma_{turb}-1}+h_0)B^{2-\gamma_{turb}}\left(\frac{p}{p_0}\right)^{\gamma_{turb}-1}.$$

标定驱动默认值为 `duu0=5578.445`、`h0=0.2`、`gamma_turb=5/3`、`p0=0.1`、`mu_max=0.99`。同一标定核中，Focused 漂移的 `pvc/q` prefactor 由无量纲化的 `1/(q R_d)` 替代：

$$R_d(p)=\sqrt{\left(a_1\frac{p_0}{p}\right)^2+\left(a_2\frac{p_0^2}{p^2}\right)^2},\quad a_1=850964.408,\quad a_2=13575468.975.$$

对应实现位于 `include/LegencyModel.hpp`、`include/FocusCoefficient.hpp` 和 `src/main.cpp`。

## 当前限制

- 重联标定驱动不是通用生产级重启系统。
- Kokkos 墙钟时间停止会在完成帧后写出最终诊断，但不会序列化可恢复的 RNG/检查点状态。
- 当前化简的重联粒子快照保存标量动量大小，不保存完整动量方向。
- Parker 辐射重建将存储的 `mu` 视为不可用；Focused 输运分支才携带物理俯仰角信息。
- 辐射阈值所需的绝对粒子能量标定仍取决于具体运行选择的输运量到 CGS 单位归一化。

## 文档

- [docs/code_map.md](docs/code_map.md)：详细代码地图和接口状态。
- [particleEmission/workflow_plan.md](particleEmission/workflow_plan.md)：分阶段辐射流程设计。
- [emissionValidation/README.md](emissionValidation/README.md)：合成辐射验证产物。

## 许可证与归属

本仓库由 Yihan Liu / SDU 维护。

本仓库包含来自外部工作流的参考材料和源码，包括李小灿随机 Parker 示例以及 `pygsfit_cp-main` 陀螺同步辐射后端材料。重新分发派生工作时，请保留来源和许可证说明。

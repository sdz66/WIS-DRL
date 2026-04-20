# WIS-DRL

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Paper Appendix](https://img.shields.io/badge/Paper%20Appendix--orange.svg)

面向过驱 4WID-4WIS 移动机器人的分层混合运动控制框架。

WIS-DRL 是论文附件代码，完整实现了一个将 PPO 模式决策与受约束低层控制结合的分层混合框架，用于过驱 4WID-4WIS 移动机器人。整个仓库按 GitHub 开源项目的常见写法整理：每个工作流对应一个脚本、地图别名清晰、基准测试可复现，并配有适合论文展示的绘图工具。

**图 1. 所提出的分层混合框架总体架构。**

![图 1. 所提出的分层混合框架总体架构。](figures/fig1.png)

## 版本说明

- 中文版：本文件
- English: [README.md](README.md)

## 目录

- [项目概述](#项目概述)
- [主要特性](#主要特性)
- [框架概览](#框架概览)
- [仓库内容](#仓库内容)
- [目录结构](#目录结构)
- [支持地图](#支持地图)
- [安装](#安装)
- [快速开始](#快速开始)
- [输出目录](#输出目录)
- [说明](#说明)
- [引用](#引用)
- [许可证](#许可证)

## 项目概述

本项目将直接轮级控制拆分为两层：

- 高层观测局部轨迹几何、车辆运动状态、模式历史、转向历史和局部环境间隙。
- PPO 策略在三种离散运动模式之间进行决策：OMM、PTM、ZRM。
- 低层控制器将离散模式转换成满足约束的轮级可执行指令。
- 仓库同时提供分层策略、纯 MPC 基线、启发式切换基线以及端到端连续控制基线，便于统一比较。

该设计旨在提升：

- 训练稳定性
- 决策可解释性
- 复杂约束下的控制可行性
- 复杂地图上的执行效率

## 主要特性

- 面向 4WID-4WIS 平台的分层 DRL + MPC 混合控制
- 三种运动模式与代码模块一一对应，便于论文与代码对照
- 内置 `map_a`、`map_b`、`map_c` 的课程式训练流程
- 提供训练、评估、基准测试和绘图的一站式脚本
- 保留 AFM、APT、AZR 和 NMPC 的独立示例，方便复现论文结果

## 框架概览

上方图 1 对应论文主图的流程组织方式，也反映了本仓库的整体结构。

### 论文术语与代码对应

| 论文术语 | 代码模块 | 作用 |
| --- | --- | --- |
| OMM | `AFM` | 全向运动，对应单轨等效 NMPC |
| PTM | `APT` | 纯平移，对应几何式平移控制 |
| ZRM | `AZR` | 零半径旋转，对应原地旋转控制 |
| 高层 DRL | PPO 模式选择器 | 在 OMM / PTM / ZRM 间切换 |
| 低层控制器 | `AFM` / `APT` / `AZR` | 生成满足约束的低层指令 |
| 观测状态 | `ModeEnv` 观测 | 路径预览 + 运动状态 + 历史 + 间隙信息 |

## 仓库内容

- `train.py`：训练基于 PPO 的模式选择器。
- `test.py`：评估训练好的策略并导出详细轨迹。
- `train_end_to_end_continuous_rl.py`：训练端到端轮级连续控制基线。
- `benchmark_policy_vs_mpc.py`：对比 PPO、纯 MPC、规则切换和连续 RL。
- `benchmark_afm_module.py`：在全部论文地图上对独立 AFM 模块做基准测试。
- `main_controller.py` 和 `run_mode_switch.py`：提供模式切换演示流程。
- `nmpc_path_tracking.py`：运行 AFM 基线的独立 NMPC 跟踪。
- `draw_map.py`：绘制三模式组合地图。
- `plot_results.py`：将日志整理成论文风格图表。

## 目录结构

```text
WIS-DRL/
├── controllers/                  # AFM、APT、AZR 与鲁棒 NMPC 控制器
├── env/                          # 训练与评估环境
├── maps/                         # 地图定义与参考路径
├── scripts/                      # 常用工作流的 shell 包装脚本
├── train.py
├── test.py
├── train_end_to_end_continuous_rl.py
├── benchmark_policy_vs_mpc.py
├── benchmark_afm_module.py
├── main_controller.py
├── run_mode_switch.py
├── nmpc_path_tracking.py
├── draw_map.py
├── plot_results.py
├── README.md
├── README_zh.md
├── requirements.txt
└── LICENSE
```

## 支持地图

`MapManager` 提供以下地图类型：

| 地图名 | 说明 | 备注 |
| --- | --- | --- |
| `map_a` | AFM 开放轨迹地图 | 用于 OMM / AFM 实验 |
| `map_b` | APT 对齐地图 | 用于 PTM / APT 实验 |
| `map_c` | AZR 重新定向地图 | 用于 ZRM / AZR 实验 |
| `tri_mode_composite` | 组合基准地图 | 默认评估地图 |

如果在 `train.py` 中省略 `--map`，训练脚本会自动使用由 `map_a`、`map_b`、`map_c` 组成的内置课程。

## 安装

推荐使用 Python 3.10 或 3.11。

```bash
cd WIS-DRL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果还需要单独安装 TensorBoard：

```bash
pip install tensorboard
```

`scripts/` 目录下的 shell 包装脚本会自动加载 `scripts/env.sh`，把 matplotlib 和字体缓存放到 `.cache/` 中，并在启动 Python 前切回项目根目录。

## 快速开始

### 训练模式选择器

使用默认的内置课程：

```bash
bash scripts/train_mode_switch.sh --timesteps 500000
```

如果要训练单一地图：

```bash
python train.py --timesteps 500000 --map map_a
```

### 评估训练好的策略

```bash
bash scripts/test_mode_switch.sh \
  --model-path models/<your_model>/best_model.zip
```

如果要在指定地图上测试：

```bash
python test.py \
  --model-path models/<your_model>/best_model.zip \
  --map tri_mode_composite \
  --episodes 20
```

### 训练端到端连续基线

```bash
bash scripts/train_continuous.sh --total-timesteps 800000
```

也可以直接运行：

```bash
python train_end_to_end_continuous_rl.py
```

### 与 MPC 和规则切换进行对比

```bash
bash scripts/benchmark_policy_vs_mpc.sh \
  --model-path models/<your_model>/best_model.zip
```

如果还要把连续控制基线也纳入比较：

```bash
bash scripts/benchmark_policy_vs_mpc.sh \
  --model-path models/<your_model>/best_model.zip \
  --continuous-model-path models/<continuous_model>/best_model.zip
```

### 运行独立 AFM 基准

```bash
bash scripts/benchmark_afm_module.sh
```

### 绘制组合地图

```bash
bash scripts/draw_map.sh
```

### 绘制训练与测试结果

```bash
bash scripts/plot_results.sh --log-dir ./logs/<run_dir>
```

如果已经有测试输出：

```bash
bash scripts/plot_results.sh --test-results ./test_results/<run_dir>
```

## 输出目录

生成的结果会写入以下目录：

- `models/`：模型 checkpoint 和配置文件
- `logs/`：环境统计和评估日志
- `tb_logs/`：TensorBoard 运行日志
- `test_results/`：评估汇总与 step trace
- `benchmark_results/`：对比表格、图像和 CSV 文件
- `figures/`：地图和论文风格图表
- `outputs/`：演示轨迹与渲染结果

## 说明

- 该仓库定位为“脚本包”，不是可直接安装导入的 Python 包。
- 在 `train.py` 中省略 `--map`，或在 `train_end_to_end_continuous_rl.py` 中省略 `--map-type`，都会使用由 `map_a`、`map_b`、`map_c` 组成的内置课程；显式 CLI 地图名只有 `map_a`、`map_b`、`map_c` 和 `tri_mode_composite`。
- AFM、APT、AZR 的单独示例、NMPC 跟踪器和 benchmark 脚本都保留了下来，便于论文复现和结果检查。
- 图 1 由 `figures/fig1.svg` 渲染；如果你有论文中的原始导出图，可以直接替换这个文件。

## 引用

如果你在自己的工作中使用了这份代码，请引用对应论文。

## 许可证

MIT License

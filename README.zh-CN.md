# NMGAT

[English](README.md) | [简体中文](README.zh-CN.md)

**Graph attention network for nuclear mass predictions with interpretable attention patterns** 的官方实现。

NMGAT 将核素及其邻近关系构建为图，用图注意力网络学习 WS4、DZ 和 LDM 结合能的修正。这里提供处理好的数据、训练代码和预测结果，支持小图与全局大图两种设置。

## ⚙️ 安装

推荐使用 Python 3.12。在项目目录下运行：

```bash
pip install -r requirements.txt
```

## 📂 数据

数据已包含在仓库中，可以直接用于训练。

| 数据 | 位置 |
|---|---|
| 小图：2,456 个训练节点、23 个测试节点、1,085 个无标签节点 | `data/processed/` |
| 全局大图：WS4/LDM 各 7,879 个节点，DZ 为 7,775 个节点 | `data/global/` |
| 整理好的数据集与预测表 | `data/tables/` |
| 原始 Excel 表格 | `data/tables_original/` |

A 使用全部训练标签；B 屏蔽实验不确定度最高的 5%（123 个）标签，保留所有图节点，其余训练设置相同。全局预测表直接提供了整理好的结果，其中新增的 4,211 个核素采用冻结权重的扩图推理得到。

## 🚀 训练

以 WS4 为例：

```bash
python scripts/train.py --model WS4 --variant A --graph small --output runs/WS4-A
```

将 `--model` 改为 `DZ` 或 `LDM` 可切换基础模型；`--variant B` 使用 B 设置，`--graph global` 使用全局大图。

默认训练 200 个随机种子，每个 12,000 轮。也可以先运行一个小例子：

```bash
python scripts/train.py --model WS4 --variant B --graph global --epochs 2 --num-seeds 1 --device cpu --output runs/demo
```

权重、预测和训练记录会保存在指定的输出目录。每次实验请使用一个新的目录。

## ▶️ 预测

训练完成后，传入保存的权重即可。程序会自动读取对应的模型和图设置。

```bash
python scripts/predict.py --checkpoint runs/demo/artifacts/seed_2026013000/final.pt --output prediction.csv
```

## 📊 结果复现

预训练权重单独提供为 `nmgat-checkpoints.zip`。将其解压到项目旁边，与项目目录平级，然后运行：

```bash
python scripts/reproduce.py --checkpoints ../nmgat-checkpoints
```

默认复算三个模型的 A 结果；添加 `--variant B` 可复算 B，添加 `--model WS4` 可只运行一个模型。结果保存在 `runs/reproduced/`，程序会自动与参考结果核对。

| 模型 | 基础模型 RMSD | NMGAT RMSD |
|---|---:|---:|
| WS4 | 0.752 | 0.392 |
| DZ | 1.058 | 0.418 |
| LDM | 3.275 | 0.523 |

单位为 MeV，对应 A 设置下的 23 个测试核素。这里使用已有权重复算结果；重新训练使用统一的 A/B 配置。

## 🧩 代码结构

```text
configs/       训练配置
data/          数据集与预测表
src/nmgat/     数据处理、网络和训练实现
scripts/       训练、预测与结果复现入口
results/       参考结果
tests/         数据与代码检查
docs/          补充说明
```

想修改模型，可以从 `src/nmgat/model.py` 开始；训练参数在 `configs/aligned.json` 中。更多信息见[代码说明](docs/ARCHITECTURE.md)、[数据字段](docs/DATA_DICTIONARY.md)和[复现细节](docs/REPRODUCIBILITY.md)。

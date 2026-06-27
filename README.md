# 面向 CIFAR-100 的轻量 CNN-Transformer 混合网络设计与消融分析

本项目是《机器学习与深度学习》课程设计代码，选题对应“卷积神经网络与 ViT 结合的探索与实现”。项目以 CIFAR-100 图像分类为任务，实现轻量 ResNet baseline 和 MobileViT-style ResNet-Hybrid，并在约 2.8M 参数预算下比较纯 CNN 与 CNN-Transformer 混合结构的表现。

项目重点不是复现大型视觉模型，而是在课程设计规模内完成一组结构清楚、可复现、可分析的实验。所有正式实验使用相同的数据划分、训练轮数、优化器、数据增强和随机种子；实验结果同时报告参数量、最佳验证准确率和测试准确率，避免只比较单个准确率数字。

## 1. 实验思路

CIFAR-100 每张图片大小为 `32x32`，共有 100 个类别。小图像任务中，CNN 的局部感受野、权值共享和平移等变性很有效；Transformer 的 self-attention 则能够显式建模不同空间区域之间的关系。本项目关注的问题是：

> 在相近参数预算下，将少量 Transformer 融合块插入轻量 CNN 主干，能否带来稳定收益？这种收益是否来自 self-attention 的 token-token 交互？

为控制模型容量影响，主实验采用 `compact` 轻量组，参数量约为 2.8M。Hybrid 模型不是在 CNN 后简单接一个 Transformer 分类头，而是在 ResNet 中层特征上插入 MobileViT-style 局部-全局-局部融合块，使卷积负责局部纹理建模，Transformer 负责补充跨区域关系建模。

## 2. 实验内容

### 2.1 主实验

主实验比较三个配置：

| 实验名 | 配置文件 | 作用 |
| --- | --- | --- |
| `compact_cnn` | `configs/compact_cnn.json` | 轻量 CIFAR ResNet baseline。 |
| `compact_hybrid` | `configs/compact_hybrid.json` | 双阶段 MobileViT-style ResNet-Hybrid 主方法。 |
| `compact_hybrid_balanced` | `configs/compact_hybrid_balanced.json` | 参数量更接近 CNN baseline 的容量重分配配置。 |

### 2.2 消融实验

消融实验统一从 `compact_hybrid` 出发，每次只改变一个结构因素：

| 实验名 | 运行方式 | 目的 |
| --- | --- | --- |
| `compact_no_attention` | `--mixer no_attention` | 去掉 self-attention 的 token-token 交互。 |
| `compact_depth1` | `--depth 1` | 将 Transformer depth 从 2 降为 1。 |
| `compact_depth3` | `--depth 3` | 将 Transformer depth 从 2 增为 3。 |
| `compact_patch1` | `--patch-size 1` | 将 patch size 从 2 改为 1。 |
| `compact_patch4` | `--patch-size 4` | 将 patch size 从 2 改为 4。 |

depth 和 patch size 会自然改变参数量或 token 数量，因此结果中同时列出参数量和准确率；除被考察因素外，训练设置和主要结构保持一致。

## 3. 模型结构

CNN baseline 是适配 CIFAR-100 的轻量 ResNet。模型使用 `3x3` stem，不采用 ImageNet ResNet 的大步长卷积和 max pooling，以保留 `32x32` 小图像中的空间细节。

Hybrid 在较窄 ResNet 主干中插入两个 MobileViT-style 融合块：

```text
ResNet stem
 -> layer1
 -> layer2
 -> MobileViTBlock at 16x16
 -> layer3
 -> MobileViTBlock at 8x8
 -> layer4
 -> classifier
```

`MobileViTBlock` 先使用局部卷积提取中层特征，再通过 unfold/fold 组织 patch token，让 Transformer Encoder 建模跨区域关系，最后将全局特征还原为空间特征图并与原 CNN 特征融合。相比 stride patch embedding，这种方式不会把 patch 内像素直接压缩成单 token，更适合 CIFAR-100 小图像。

## 4. 实验结果

实验日志保存在 `outputs/runs/compact_*` 目录。每个实验目录包含：

- `config.json`: 实际训练配置。
- `metrics.csv`: 每轮训练和验证指标。
- `summary.json`: 参数量、最佳验证轮次、测试准确率和训练耗时。

训练脚本运行时会生成 `best.pt` 和 `last.pt`，但仓库中不保存 checkpoint。复查结果主要依赖 `metrics.csv`、`summary.json` 和 `outputs/figures/` 中的图表。

### 4.1 主实验结果

| 模型 | 参数量 | Best Val | Test Acc |
| --- | ---: | ---: | ---: |
| `compact_cnn` | 2.821M | 71.50% | 71.82% |
| `compact_hybrid` | 2.796M | 72.80% | 72.69% |
| `compact_hybrid_balanced` | 2.818M | 72.50% | 72.88% |

主 Hybrid 在参数量略少于 CNN baseline 的情况下，将 test accuracy 从 71.82% 提升到 72.69%。`compact_hybrid_balanced` 的 test accuracy 达到 72.88%，记录了容量重分配配置在同一训练协议下的表现。

### 4.2 消融结果

| 实验 | 参数量 | Best Val | Test Acc |
| --- | ---: | ---: | ---: |
| `compact_hybrid` | 2.796M | 72.80% | 72.69% |
| `compact_no_attention` | 2.796M | 72.34% | 70.98% |
| `compact_depth1` | 2.619M | 72.28% | 72.64% |
| `compact_depth3` | 2.973M | 74.08% | 73.04% |
| `compact_patch1` | 2.812M | 73.46% | 72.77% |
| `compact_patch4` | 2.792M | 72.94% | 72.39% |

no-attention 明显降低 test accuracy，表明 self-attention 的 token-token 交互对结果有实际贡献。depth3 表现最好，同时参数量增加到 2.973M；patch4 略低，说明在 CIFAR-100 小图像上过粗 patch 可能损失局部细节。

## 5. 文件结构

```text
configs/
  compact_cnn.json
  compact_hybrid.json
  compact_hybrid_balanced.json
src/
  train.py
  smoke_test.py
  plot_results.py
scripts/
  run_main.sh
  run_ablations.sh
outputs/runs/
  compact_cnn/
  compact_hybrid/
  compact_hybrid_balanced/
  compact_no_attention/
  compact_depth1/
  compact_depth3/
  compact_patch1/
  compact_patch4/
outputs/figures/
  main_accuracy.*
  main_training_curves.*
  ablation_accuracy.*
  ablation_training_curves.*
  main_results_table.tex
  ablation_results_table.tex
```

`configs/README.md` 对配置文件和命令行覆盖参数做了更细说明。

## 6. 环境与运行

数据集使用 CIFAR-100。首次运行训练脚本时，`torchvision` 会自动下载并解压到 `data/` 目录；如果本地已经存在该目录，训练会直接复用。`data/` 体积较大，已在 `.gitignore` 中忽略。

快速检查所有配置和消融变体能否前向传播：

```bash
python src/smoke_test.py
```

运行主实验：

```bash
./scripts/run_main.sh
```

运行消融实验：

```bash
./scripts/run_ablations.sh
```

单独运行某个实验：

```bash
python src/train.py --config configs/compact_hybrid.json --device cuda
python src/train.py --config configs/compact_hybrid.json --device cuda --experiment compact_depth3 --depth 3
```

根据已有实验日志生成图表和表格：

```bash
python src/plot_results.py --runs-dir outputs/runs --out-dir outputs/figures
```

生成内容包括：

- `main_accuracy.*` / `main_training_curves.*`
- `ablation_accuracy.*` / `ablation_training_curves.*`
- `main_results_table.tex` / `ablation_results_table.tex`

## 7. 实验协议

- 数据集：CIFAR-100。
- 训练集固定划分 5,000 张作为 validation set。
- validation set 用于选择 best checkpoint。
- test set 只在训练完成后最终评估。
- 所有正式实验使用相同数据划分、训练轮数、优化器、增强策略和随机种子。
- 修改模型结构后应重新训练，并只使用同一结构设定下产生的结果。

## 8. 实验结论

在约 2.8M 参数的轻量预算下，MobileViT-style unfold/fold CNN-Transformer Hybrid 相比轻量 CNN baseline 获得稳定小幅提升；no-attention 消融显示 self-attention 交互对性能有贡献。整体来看，CNN 的局部归纳偏置在 CIFAR-100 小图像任务中仍然强有效，Transformer 融合块更适合作为卷积主干的全局关系建模补充。

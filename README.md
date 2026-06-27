# 轻量 CNN-Transformer 混合网络在 CIFAR-100 上的探索与消融

本项目是《机器学习与深度学习》课程设计，主题为“卷积神经网络与 ViT 结合的探索与实现”。项目聚焦一条清晰主线：在 CIFAR-100 图像分类任务上，比较约 2.8M 参数预算下的轻量 CNN 与 MobileViT-style ResNet-Transformer Hybrid，并围绕 Hybrid 做结构消融。

`compact` 表示轻量参数预算组，对应约 2.8M 参数的小模型设定，用于比较相近容量下 CNN 与 Hybrid 的表现。

## 实验设计

主实验比较：

| 实验名 | 配置文件 | 作用 |
| --- | --- | --- |
| `compact_cnn` | `configs/compact_cnn.json` | 轻量 CIFAR ResNet baseline。 |
| `compact_hybrid` | `configs/compact_hybrid.json` | 双阶段 MobileViT-style ResNet-Hybrid 主方法。 |
| `compact_hybrid_balanced` | `configs/compact_hybrid_balanced.json` | 参数量更接近 CNN 的容量重分配配置。 |

消融实验统一从 `compact_hybrid` 出发，每次只改变一个因素：

| 实验名 | 运行方式 | 目的 |
| --- | --- | --- |
| `compact_no_attention` | `--mixer no_attention` | 去掉 self-attention 的 token-token 交互。 |
| `compact_depth1` | `--depth 1` | 将 Transformer depth 从 2 降为 1。 |
| `compact_depth3` | `--depth 3` | 将 Transformer depth 从 2 增为 3。 |
| `compact_patch1` | `--patch-size 1` | 将 patch size 从 2 改为 1。 |
| `compact_patch4` | `--patch-size 4` | 将 patch size 从 2 改为 4。 |

depth 和 patch size 会自然改变参数量或 token 数量，因此报告中同时列出参数量和准确率；除被考察因素外，训练设置和主要结构保持一致。

## 模型方法

CNN baseline 是适配 CIFAR-100 的轻量 ResNet：使用 `3x3` stem，不采用 ImageNet ResNet 的大步长卷积和 max pooling，以保留 `32x32` 小图像细节。

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

`MobileViTBlock` 使用局部卷积提取中层特征，再通过 unfold/fold 组织 patch token，让 Transformer Encoder 建模跨区域关系，最后将全局特征还原为空间特征图并与原 CNN 特征融合。相比 stride patch embedding，这种方式不会把 patch 内像素直接压缩成单 token，更适合 CIFAR-100 小图像。

## 实验结果

实验结果保存在 `outputs/runs/compact_*` 目录。

| 模型 | 参数量 | Best Val | Test Acc |
| --- | ---: | ---: | ---: |
| `compact_cnn` | 2.821M | 71.50% | 71.82% |
| `compact_hybrid` | 2.796M | 72.80% | 72.69% |
| `compact_hybrid_balanced` | 2.818M | 72.50% | 72.88% |

主实验显示：在轻量参数预算下，MobileViT-style unfold/fold Hybrid 相比参数量接近且略大的 CNN baseline 获得稳定小幅提升。`compact_hybrid` 参数量略少于 `compact_cnn`，validation accuracy 提升 1.30 个百分点，test accuracy 提升 0.87 个百分点。`compact_hybrid_balanced` 的 test accuracy 更高，记录了容量重分配配置在同一训练协议下的表现。

消融结果：

| 实验 | 参数量 | Best Val | Test Acc |
| --- | ---: | ---: | ---: |
| `compact_hybrid` | 2.796M | 72.80% | 72.69% |
| `compact_no_attention` | 2.796M | 72.34% | 70.98% |
| `compact_depth1` | 2.619M | 72.28% | 72.64% |
| `compact_depth3` | 2.973M | 74.08% | 73.04% |
| `compact_patch1` | 2.812M | 73.46% | 72.77% |
| `compact_patch4` | 2.792M | 72.94% | 72.39% |

no-attention 明显降低 test accuracy，表明提升不只是来自额外卷积、投影和融合结构，self-attention 的 token-token 交互有实际贡献。depth3 表现最好，同时参数量增加到 2.973M；该结果反映更深全局建模模块的潜在收益。patch4 略低，符合 CIFAR 小图像中粗 patch 可能损失局部细节的直觉。

## 目录结构

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
  make_figures_and_report.sh
report/
  main.tex
  references.bib
outputs/runs/
  compact_cnn/
  compact_hybrid/
  compact_hybrid_balanced/
  compact_no_attention/
  compact_depth1/
  compact_depth3/
  compact_patch1/
  compact_patch4/
```

`outputs/runs/` 中保留正式实验日志。每个实验目录包含：

- `config.json`: 实际训练配置。
- `metrics.csv`: 每轮训练和验证指标。
- `summary.json`: 参数量、最佳验证轮次、测试准确率和训练耗时。

训练脚本运行时会生成 `best.pt` 和 `last.pt`。为控制提交体积，项目归档不保留 checkpoint，只保留报告和复现实验分析需要的日志与汇总结果。

## 运行方式

### 数据集

数据集使用 CIFAR-100。首次运行训练脚本时，`torchvision` 会自动下载并解压到 `data/` 目录；如果本地已经存在该目录，训练会直接复用。`data/` 体积较大，不作为提交内容保留。

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

在课程 Docker 环境中可显式指定解释器：

```bash
PYTHON=/root/miniconda3/envs/mldl/bin/python ./scripts/run_main.sh
```

生成图表：

```bash
python src/plot_results.py --runs-dir outputs/runs --out-dir outputs/figures
```

图表会按报告结构分为主实验与消融实验两组：

- `main_accuracy.*` / `main_training_curves.*`
- `ablation_accuracy.*` / `ablation_training_curves.*`
- `main_results_table.tex` / `ablation_results_table.tex`

本项目保留正式实验日志和图表：`outputs/runs/` 中是各实验的 `config.json`、`metrics.csv` 和 `summary.json`，`outputs/figures/` 中是报告使用的图片和 LaTeX 表格。这些文件用于复查实验结果和重新编译报告。

编译报告：

```bash
cd report
latexmk -xelatex main.tex
```

也可以运行：

```bash
./scripts/make_figures_and_report.sh
```

## 实验协议

- 数据集：CIFAR-100。
- 训练集固定划分 5,000 张作为 validation set。
- validation set 用于选择 best checkpoint。
- test set 只在训练完成后最终评估。
- 所有正式实验使用相同数据划分、训练轮数、优化器、增强策略和随机种子。
- 修改模型结构后应重新训练，并只使用同一结构设定下产生的结果。

## 实验结论

在约 2.8M 参数的轻量预算下，实验结论集中在稳定小幅提升：

> 在约 2.8M 参数的轻量预算下，MobileViT-style unfold/fold CNN-Transformer Hybrid 在参数量略少的情况下，相比轻量 CNN baseline 获得稳定小幅提升；no-attention 消融显示 self-attention 交互对性能有贡献。

## 提交文件说明

提交包包含代码、报告和结果归档：

- `README.md`、`requirements.txt`
- `configs/`、`src/`、`scripts/`
- `outputs/runs/`、`outputs/figures/`
- `report/main.tex`、`report/main.pdf`、`report/references.bib`

提交包不包含 `data/`、模型 checkpoint、LaTeX 编译辅助文件和课程材料原件。

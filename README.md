# ResNet-Transformer 混合网络在 CIFAR-100 图像分类任务中的探索与消融分析

本项目是《机器学习与深度学习》课程设计，选题对应老师给出的“卷积神经网络与 ViT 结合的探索与实现”。项目以 CIFAR-100 图像分类为任务，实现轻量 CNN、轻量 ResNet-Transformer Hybrid、统一 MobileNetV2/MobileViT 风格轻量 Hybrid、CIFAR 版 ResNet-18 和 Small ViT，并通过严格消融实验研究 attention 机制、Transformer 层数和 patch size 的影响。

项目重点不是复现大型视觉模型，而是在课程设计可承受的训练时间内完成一套结构清楚、可复现、可分析的中等规模实验。主实验采用相同数据、相同训练流程、相同验证/测试协议，并尽量控制模型在接近参数预算下比较；报告中同时给出准确率、参数量和训练曲线，避免只看单个准确率数字。

## 1. 实验思路

CIFAR-100 是一个 100 类小图像分类数据集，每张图片大小为 32x32。相比 CIFAR-10，它类别更多、每类样本更少，因此更适合观察不同模型结构之间的差异。

本实验围绕一个核心问题展开：

> CNN 擅长提取局部纹理和形状，Transformer 擅长建模全局关系。二者结合后，是否能在 CIFAR-100 这种小数据集上取得更好的准确率和复杂度平衡？

CNN 的优势在于局部归纳偏置强，训练稳定，适合小图像任务；ViT 的优势在于 self-attention 能建模较远区域之间的关系，但在小数据集上可能更依赖正则化和训练技巧。前期探索以 ResNet-18 为主干，在中层特征上插入 MobileViT 风格的局部-全局-局部融合块。进一步的统一设计参考 MobileViT/MoCoViT：使用 MobileNetV2-style inverted residual blocks 负责浅层局部特征和降采样，在 $16\times16$、$8\times8$ 和 $4\times4$ 语义阶段插入 MobileViT-style unfold/fold block。第一版统一设计直接用 MobileViT block 替换局部 CNN stage；改进版 `mobilevit_cifar_v2` 则先保留每个 stage 的局部 MV2 blocks，再在 stage 末尾加入 MobileViT block，以避免过度削弱 CNN 局部建模能力。

为了避免“谁参数更多谁占便宜”，本项目采用参数预算匹配。原 ResNet-Hybrid 轻量组中，`compact_cnn` 使用 32 宽 ResNet，`compact_hybrid` 使用 27 宽 ResNet 主干和双阶段 MobileViT 风格融合块。新增统一轻量主线中，`mobile_cnn` 与 `mobilevit_cifar` 采用相同输入增强、训练流程、验证/测试协议和相近参数预算；正式分析时优先比较同一结构阶段内的模型，不把旧结构结果和新结构结果混作同一个主比较。

这个设计主要参考三类已有工作：MobileViT 采用局部卷积、全局 Transformer、再回到局部卷积的融合思路；CvT 把卷积归纳偏置引入 token 生成和 Transformer 结构；CoAtNet/CMT 强调卷积与 attention 在多个阶段的组合。因此本项目不采用“CNN 最后接一个 Transformer 分类头”的简单串联，而是在两个中层特征处做结构融合，并用参数预算匹配来检验这种融合是否真正有效。

## 2. 实验内容

### 2.1 主实验

主实验分为统一 MobileViT 轻量主线、原 ResNet-Hybrid 探索线和强基线组。统一轻量主线是新的主要观察对象，用来判断在同一 MobileNetV2-style 主干设计和相近参数预算下，MobileViT-style 全局建模是否比纯 CNN replacement blocks 更有效。

| 实验名 | 配置文件 | 作用 |
| --- | --- | --- |
| `mobile_cnn` | `configs/mobile_cnn.json` | 统一轻量主线中的较小纯 CNN baseline。 |
| `mobile_cnn_matched` | `configs/mobile_cnn_matched.json` | 与 `mobilevit_cifar_v2` 参数预算更接近的正式纯 CNN 对照。 |
| `mobilevit_cifar` | `configs/mobilevit_cifar.json` | 第一版统一 MobileViT-style Hybrid，用于结构探索。 |
| `mobilevit_cifar_v2` | `configs/mobilevit_cifar_v2.json` | 保留局部 MV2 blocks 后再加入 MobileViTBlock 的改进版主方法。 |
| `compact_cnn` | `configs/compact_cnn.json` | 轻量 CNN baseline，约 2.8M 参数。 |
| `compact_hybrid` | `configs/compact_hybrid.json` | 轻量双阶段 Hybrid，参数量与 `compact_cnn` 接近，是本文主要分析对象。 |
| `cnn_baseline` | `configs/cnn_baseline.json` | 强 CNN baseline，CIFAR 版 ResNet-18。 |
| `hybrid_main` | `configs/hybrid_main.json` | 强 Hybrid，参数量与 ResNet-18 接近，用于观察强基线下收益是否变小。 |
| `tiny_vit` | `configs/tiny_vit.json` | Small ViT 对照模型，用来观察纯 Transformer 的表现。 |

主实验主要比较三项指标：

- 最终测试集准确率：模型分类效果。
- 参数量：模型复杂度。
- 验证集训练曲线：模型是否稳定收敛。

### 2.2 严格消融实验

本项目的第二部分是严格消融实验。严格消融的定义是：从 `compact_hybrid` 出发，每次只改变一个因素，其他训练设置和结构超参数保持不变。参数量如果随这个因素自然变化，需要记录和分析，但这不破坏消融的严格性。

统一 MobileViT 主线对应的消融实验包括 `mobilevit_no_attention`、`mobilevit_depth1` 和 `mobilevit_patch4`。改进版主方法对应 `mobilevit_v2_no_attention`、`mobilevit_v2_depth1` 和 `mobilevit_v2_patch4`。它们分别只改变 token mixer、Transformer depth 或 patch size，用于分析新主方法的提升来源和小图像 patch 粒度影响。

本文只保留以下严格消融，不再加入退化对照或额外容量匹配实验：

- `compact_no_attention`: 保留局部卷积、token 化、FFN、token 还原和卷积融合，只把 self-attention 替换为无 token-token 交互的前馈替代模块。它回答“attention 交互本身是否有效”。
- `compact_depth1` / `compact_depth3`: 只改变两个融合块中的 Transformer 层数，其他设置不变。它回答“在这个结构中，减少或增加 Transformer 层数是否有帮助”。
- `compact_patch1` / `compact_patch4`: 只改变 patch size，其他设置不变。它回答“在这个结构中，token 粒度变细或变粗是否有帮助”。

这些实验都属于严格消融。区别只在于：`compact_no_attention` 的参数量与主模型相同；depth 和 patch size 的参数量或 token 数量可能自然变化，报告中必须把这些变化与准确率一起分析。

| 实验名 | 类型 | 目的 |
| --- | --- | --- |
| `compact_no_attention` | 严格消融 | 只去掉 self-attention 的 token-token 交互。 |
| `compact_depth1` | 严格消融 | 只把每个融合块的 Transformer 层数从 2 改为 1。 |
| `compact_depth3` | 严格消融 | 只把每个融合块的 Transformer 层数从 2 改为 3。 |
| `compact_patch1` | 严格消融 | 只把 patch size 从 2 改为 1。 |
| `compact_patch4` | 严格消融 | 只把 patch size 从 2 改为 4。 |

消融实验不单独保存为重复 JSON 文件，而是在运行命令中覆盖少量参数。训练后，程序会把实际使用的完整配置保存到 `outputs/runs/<实验名>/config.json`，方便复查。

## 3. 结果分析逻辑

实验结果不应只看“哪个准确率最高”，而应结合最终测试准确率、参数量和验证集训练曲线一起分析。如果 `compact_hybrid` 相比 `compact_cnn` 提升更明显，而 `hybrid_main` 相比 ResNet-18 只小幅提升，则可以说明 Transformer 融合在容量受限时更容易体现价值，而强 CNN 已经接近该训练设置下的性能上限。

如果 `cnn_baseline` 表现稳定，说明 CNN 的局部特征提取能力很适合 CIFAR-100 这类小图像任务。

如果 `tiny_vit` 表现弱于 CNN，说明纯 ViT 在小数据集上可能缺少局部先验，需要更多数据、训练轮数或正则化。

如果 `compact_hybrid` 优于 `compact_cnn`，说明 Transformer 在轻量 CNN 特征之上提供了有用的全局关系建模。

如果 `hybrid_main` 提升有限甚至不如某个 baseline，说明这个具体增强式融合方法没有取得预期优势。可以从数据集规模、图像分辨率、token 数量、融合方式和训练策略角度分析失败原因，而不是强行说明 hybrid 一定更好。

严格消融重点回答以下问题：

- 只去掉 attention 交互后，Hybrid 是否明显退化？
- 只改变 Transformer 层数时，准确率、参数量和训练稳定性如何变化？
- 只改变 patch size 时，token 粒度、参数量和准确率如何变化？

## 4. 文件结构

建议提交源码时保留以下内容：

| 路径 | 说明 |
| --- | --- |
| `src/train.py` | 主代码，包含数据加载、模型定义、训练和评估。 |
| `src/plot_results.py` | 根据训练日志生成准确率、参数量和训练曲线图。 |
| `src/smoke_test.py` | 快速检查主模型和所有消融变体能否正常前向传播。 |
| `configs/` | 主实验配置文件。 |
| `scripts/` | 批量运行主实验、严格消融实验和生成报告的脚本。 |
| `report/` | LaTeX 报告源码、参考文献和当前报告 PDF。 |
| `requirements.txt` | 绘图和运行所需的 Python 依赖说明。 |

不建议提交以下内容：

- `data/`: CIFAR-100 数据集，首次运行会自动下载。
- `outputs/runs/`: checkpoint 和训练日志，除非老师要求提交完整实验输出。
- `outputs/figures/`: 可由日志重新生成；最终报告需要的图可以保留。

## 5. 环境与数据

建议在课程提供的 `mldl` 容器或等价 PyTorch 环境中运行。当前环境已确认：

- Python 3.11
- PyTorch 2.11.0 + CUDA
- torchvision 0.26.0
- matplotlib 3.10.8
- seaborn 0.13.2

如果已经激活了 Python 环境，下面命令中的 `python` 可以直接使用。若在课程容器中没有激活环境，也可以显式指定解释器：

```bash
PYTHON=/root/miniconda3/envs/mldl/bin/python ./scripts/run_main.sh
```

项目脚本会优先使用当前环境中的 `python`，找不到时会自动尝试课程容器里的 mldl 解释器。

如果在新环境中缺少绘图依赖，可运行：

```bash
pip install -r requirements.txt
```

数据集默认保存到项目内相对路径：

```text
data/
```

第一次运行训练脚本时会自动下载 CIFAR-100；如果已经下载过，torchvision 会直接复用本地文件。

训练时会从 CIFAR-100 训练集中固定划分 5,000 张作为验证集，用验证集选择最佳 checkpoint；官方测试集只在训练结束后评估一次，避免用测试集挑模型。

注意：如果修改过模型结构，旧的 `outputs/runs/` 结果不能和新模型混合比较。重新跑正式实验前，建议清空或备份旧结果。

## 6. 运行方式

快速检查模型是否能正常前向传播：

```bash
python src/smoke_test.py
```

如果当前 shell 没有 `python` 命令，可使用：

```bash
/root/miniconda3/envs/mldl/bin/python src/smoke_test.py
```

运行主实验：

```bash
./scripts/run_main.sh
```

运行严格消融实验：

```bash
./scripts/run_ablations.sh
```

运行统一 MobileViT 轻量主线：

```bash
./scripts/run_mobile_experiments.sh
```

运行改进版统一 MobileViT 主线：

```bash
./scripts/run_mobile_v2_experiments.sh
```

如果只想单独运行某一个实验，可以使用：

```bash
python src/train.py --config configs/hybrid_main.json --device cuda
python src/train.py --config configs/compact_hybrid.json --device cuda --experiment compact_depth3 --depth 3
```

短跑调试可以加：

```bash
--epochs 1 --limit-train-batches 5 --limit-val-batches 5 --num-workers 0
```

训练结果会保存到 `outputs/runs/<experiment>/`，主要包括：

- `metrics.csv`: 每轮训练 loss/accuracy 和验证 loss/accuracy。
- `summary.json`: 参数量、最佳验证轮次、最终测试准确率和耗时。
- `best.pt`: 验证准确率最高的 checkpoint。
- `last.pt`: 最后一轮 checkpoint。

## 7. 生成图表和报告

训练完成后生成图表：

```bash
python src/plot_results.py --runs-dir outputs/runs --out-dir outputs/figures
```

生成内容包括：

- `accuracy_comparison.png/.pdf`: 最终测试准确率对比图。
- `parameter_comparison.png/.pdf`: 参数量对比图。
- `accuracy_params_tradeoff.png/.pdf`: 准确率与参数量权衡图。
- `training_curves.png/.pdf`: 验证集准确率曲线。
- `results_table.tex`: 可直接插入 LaTeX 报告的结果表。

编译报告：

```bash
cd report
latexmk -xelatex main.tex
```

也可以直接运行：

```bash
./scripts/make_figures_and_report.sh
```

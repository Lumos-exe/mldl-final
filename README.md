# 轻量级 CNN-Transformer 混合网络在 CIFAR-100 图像分类任务中的探索与消融分析

本项目用于《机器学习与深度学习》结课设计，选题对应老师指定方向“卷积神经网络与 ViT 结合的探索与实现”。项目实现了小型 CNN、Tiny ViT、CNN+Transformer 混合模型，并围绕 Transformer 层数、patch size 和通道/embedding 宽度进行消融分析。

## 环境

使用已创建的课程容器 `focused_bassi`：

```bash
docker start focused_bassi
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi \
  /root/miniconda3/envs/mldl/bin/python src/smoke_test.py
```

容器内已确认：

- Python 3.11
- PyTorch 2.11.0 + CUDA
- torchvision 0.26.0
- matplotlib 3.10.8

`seaborn` 是可选依赖；绘图脚本会在没有 seaborn 时自动回退到 matplotlib 样式。

## 数据集

主实验使用 CIFAR-100。本机已有数据路径：

```text
/home/lumos/Projects/mldl/CIFAR-100/data
```

容器内对应路径：

```text
/workspaces/mldl/CIFAR-100/data
```

## 快速验证

```bash
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi \
  /root/miniconda3/envs/mldl/bin/python src/smoke_test.py
```

## 训练

主实验：

```bash
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/cnn_baseline.json --device cuda
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/tiny_vit.json --device cuda
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/hybrid_main.json --device cuda
```

消融实验：

```bash
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/hybrid_no_transformer.json --device cuda
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/hybrid_depth1.json --device cuda
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/hybrid_depth4.json --device cuda
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/hybrid_patch2.json --device cuda
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/hybrid_patch8.json --device cuda
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi /root/miniconda3/envs/mldl/bin/python src/train.py --config configs/hybrid_wide.json --device cuda
```

短跑调试可以加：

```bash
--epochs 1 --limit-train-batches 5 --limit-val-batches 5 --num-workers 0
```

训练结果会保存到 `outputs/runs/<experiment>/`，包括：

- `metrics.csv`: 每轮训练/测试 loss 与 accuracy。
- `summary.json`: 参数量、最佳准确率、最佳轮次和耗时。
- `best.pt`: 测试准确率最高的 checkpoint。
- `last.pt`: 最后一轮 checkpoint。

## 绘图

训练完成后生成高质量图表：

```bash
docker exec -w /workspaces/mldl/mldl_finnal focused_bassi \
  /root/miniconda3/envs/mldl/bin/python src/plot_results.py --runs-dir outputs/runs --out-dir outputs/figures
```

输出包括：

- `outputs/figures/accuracy_comparison.png/.pdf`
- `outputs/figures/parameter_comparison.png/.pdf`
- `outputs/figures/accuracy_params_tradeoff.png/.pdf`
- `outputs/figures/training_curves.png/.pdf`
- `outputs/figures/results_table.tex`

## 报告编译

```bash
cd report
latexmk -xelatex main.tex
```

报告会优先引用 `outputs/figures` 中的实验图。如果尚未完成训练，LaTeX 会显示占位框，避免编译失败。

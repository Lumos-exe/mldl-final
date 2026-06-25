# 实验配置说明

这些 JSON 文件记录主实验的模型结构和训练超参数。实验使用相同数据划分、训练策略和评估协议，并分为轻量组与强基线组：轻量组比较 `compact_cnn` 和 `compact_hybrid`，强基线组比较 `cnn_baseline` 和 `hybrid_main`。

训练集会固定划分 5,000 张作为验证集，验证集用于选择最佳 checkpoint；官方测试集只用于最终评估。

主实验：

- `cnn_baseline.json`: CIFAR 版 ResNet-18 baseline。
- `tiny_vit.json`: Small ViT 对照模型。
- `hybrid_main.json`: 约 11.6M 参数的强 Hybrid，使用较窄 ResNet 主干，并在两个中层阶段插入局部-全局-局部融合块。
- `compact_cnn.json`: 约 2.8M 参数的轻量 CNN baseline。
- `compact_hybrid.json`: 约 2.9M 参数的轻量双阶段 Hybrid，是严格消融实验的基准。

消融实验统一基于 `compact_hybrid.json`，每组只改变一个因素，其他训练设置保持不变：

- `--mixer no_attention`: 保留局部卷积、token 化、FFN、token 还原和卷积融合，只把 self-attention 替换为无 token-token 交互的前馈替代模块。
- `--depth 1` / `--depth 3`: 只改变两个融合块中的 Transformer 层数，主模型为 `depth=2`。
- `--patch-size 1` / `--patch-size 4`: 只改变 patch size，主模型为 `patch_size=2`。

这些都属于严格消融。参数量、token 数量或训练耗时如果随被改变的因素自然变化，应作为实验结果一起报告，而不是再通过改宽度等方式强行匹配。

每次训练都会在 `outputs/runs/<实验名>/config.json` 保存实际使用的配置。

# 实验配置说明

这些 JSON 文件记录三组主实验的模型结构和训练超参数。三组主模型的参数量都控制在约 0.62M 左右，用于进行参数预算匹配下的结构对比。

训练集会固定划分 5,000 张作为验证集，验证集用于选择最佳 checkpoint；官方测试集只用于最终评估。

主实验：

- `cnn_baseline.json`: 小型 CNN baseline。
- `tiny_vit.json`: Tiny ViT 对照模型。
- `hybrid_main.json`: CNN-Transformer 混合模型主配置。

消融实验统一基于 `hybrid_main.json`，每组只改变一个因素，其他训练设置保持不变：

- `--mixer no_attention`: 只把 self-attention 替换为无 token-token 交互的前馈替代模块。
- `--depth 1` / `--depth 4`: 只改变 Transformer 层数。
- `--patch-size 2` / `--patch-size 8`: 只改变 patch size。

这些都属于严格消融。参数量、token 数量或训练耗时如果随被改变的因素自然变化，应作为实验结果一起报告，而不是再通过改宽度等方式强行匹配。

每次训练都会在 `outputs/runs/<实验名>/config.json` 保存实际使用的配置。

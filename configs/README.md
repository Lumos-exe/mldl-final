# 实验配置说明

这些 JSON 文件对应课程设计采用的 compact 轻量实验组。`compact` 表示约 2.8M 参数预算，用于在相近模型容量下比较纯 CNN 与 CNN-Transformer Hybrid。

训练集固定划分 5,000 张作为验证集，验证集用于选择最佳 checkpoint；官方测试集只用于最终评估。

主实验配置：

- `compact_cnn.json`: 约 2.82M 参数的 CIFAR 版轻量 ResNet baseline。
- `compact_hybrid.json`: 约 2.80M 参数的双阶段 MobileViT-style ResNet-Hybrid 主方法。
- `compact_hybrid_balanced.json`: 与 `compact_cnn` 参数量更接近的容量重分配配置。

消融实验统一基于 `compact_hybrid.json`，每组只改变一个因素，其他训练设置保持不变：

- `--mixer no_attention`: 保留局部卷积、token 化、FFN、token 还原和卷积融合，self-attention 使用无 token-token 交互的前馈替代模块。
- `--depth 1` / `--depth 3`: 只改变两个融合块中的 Transformer 层数，主模型为 `depth=2`。
- `--patch-size 1` / `--patch-size 4`: 只改变 patch size，主模型为 `patch_size=2`。

每次训练都会在 `outputs/runs/<实验名>/config.json` 保存实际使用的配置，便于核对命令行覆盖后的参数。

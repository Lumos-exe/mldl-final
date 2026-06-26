# 实验配置说明

这些 JSON 文件记录主实验的模型结构和训练超参数。实验使用相同数据划分、训练策略和评估协议。当前新增一组统一 MobileNetV2/MobileViT 风格轻量主线，用于在更贴近 MobileViT/MoCoViT 论文思想的主干上比较纯 CNN 与 CNN-Transformer Hybrid。

训练集会固定划分 5,000 张作为验证集，验证集用于选择最佳 checkpoint；官方测试集只用于最终评估。

统一轻量主线：

- `mobile_cnn.json`: 与 Hybrid 共享 stem、降采样 stage 和分类头的 MobileNetV2-style 纯 CNN baseline。
- `mobile_cnn_matched.json`: 与 `mobilevit_cifar_v2` 参数预算更接近的纯 CNN baseline，用额外 MV2 blocks 替代 MobileViTBlock，是 v2 主比较的正式 CNN 对照。
- `mobilevit_cifar.json`: 第一版 MobileNetV2-style 主干 + 三阶段 MobileViT-style unfold/fold block，用于验证直接用全局块替换局部 CNN stage 是否有效。
- `mobilevit_cifar_v2.json`: 改进版 MobileViT 主方法。每个语义 stage 先保留若干 MV2 local blocks，再在 stage 末尾加入 MobileViT-style global block，避免过度牺牲 CNN 局部建模能力。
- `mobilevit_no_attention.json`: 保留 Hybrid 结构容量，但去掉 token-token self-attention 交互。
- `mobilevit_depth1.json`: 只把三个 MobileViT block 的 Transformer depth 降为 1。
- `mobilevit_patch4.json`: 只把 patch size 从 2 改为 4，观察更粗 patch 对 CIFAR-100 小图像细节的影响。
- `mobilevit_v2_no_attention.json` / `mobilevit_v2_depth1.json` / `mobilevit_v2_patch4.json`: 基于 `mobilevit_cifar_v2` 的对应消融。

原 ResNet-Hybrid 探索实验：

- `cnn_baseline.json`: CIFAR 版 ResNet-18 baseline。
- `tiny_vit.json`: Small ViT 对照模型。
- `hybrid_main.json`: 约 11.6M 参数的强 Hybrid，使用较窄 ResNet 主干，并在两个中层阶段插入局部-全局-局部融合块。
- `compact_cnn.json`: 约 2.8M 参数的轻量 CNN baseline。
- `compact_hybrid.json`: 约 2.9M 参数的轻量双阶段 Hybrid，是严格消融实验的基准。

原消融实验统一基于 `compact_hybrid.json`，每组只改变一个因素，其他训练设置保持不变：

- `--mixer no_attention`: 保留局部卷积、token 化、FFN、token 还原和卷积融合，只把 self-attention 替换为无 token-token 交互的前馈替代模块。
- `--depth 1` / `--depth 3`: 只改变两个融合块中的 Transformer 层数，主模型为 `depth=2`。
- `--patch-size 1` / `--patch-size 4`: 只改变 patch size，主模型为 `patch_size=2`。

这些都属于严格消融。参数量、token 数量或训练耗时如果随被改变的因素自然变化，应作为实验结果一起报告，而不是再通过改宽度等方式强行匹配。

每次训练都会在 `outputs/runs/<实验名>/config.json` 保存实际使用的配置。

注意：统一 MobileViT 主线和原 ResNet-Hybrid 探索线属于不同结构阶段，正式表格中不要把旧结构结果和新结构结果混作同一个主比较。v2 主比较应优先使用 `mobile_cnn_matched` vs `mobilevit_cifar_v2`；`mobile_cnn` 只作为较小参数轻量 CNN 参考。第一版 `mobilevit_cifar` 如果保留，应作为结构探索中的负面结果分析，而不是最终主方法。

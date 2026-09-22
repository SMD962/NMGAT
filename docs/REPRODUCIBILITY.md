# Reproducibility / 复现说明

## Training and inference / 训练与推理

`reproduce.py` loads the original 200-seed ensembles and compares their mean
residuals with the reference CSVs (tolerance: 1e-4 MeV). `train.py` uses the shared
A/B configuration for new experiments. A/B differ only in the 123 masked training
labels; all nodes remain in the graph. `predict.py` uses the graph and settings
saved with a newly trained checkpoint.

复现命令使用已有权重，训练命令使用统一的 A/B 配置。B 只屏蔽 123 个监督标签，
保留全部节点。新训练的结果会随运行环境和随机性略有不同。

Historical DZ-A/LDM-A runs used 12,400 epochs; the others used 12,000. Historical
B used noise on training and unlabelled nodes. New A/B runs both use 12,000 epochs and training-node noise. The pretrained
ensembles share the inference settings in `configs/pretrained.json`.

## Graphs and tables / 图与结果表

The small graph contains 3,564 nodes. Global graphs contain 7,879 nodes for
WS4/LDM and 7,775 for DZ. All inputs are ready to use. Scaling is fitted to the
2,456 original training nodes. The backbone's graph LayerNorm uses graph-wide
hidden statistics; the stem's LayerNorm operates within each node.

The compiled prediction tables combine small-graph results with 4,211 common
extension nodes. The extension used
frozen weights and hidden normalization statistics captured on the small graph.
The compiled A table contains 7,775 rows; B omits the 123 masked rows. Training
directly on the global graph is a separate experiment.

大小图数据均已提供。结果表保留原小图预测，并补入 4,211 个扩展核素；
直接在大图上训练属于新的实验。

## Evaluation / 评估约定

The 23-nuclide evaluation set was used to select feature-noise strength, so its
score is not an untouched final-test estimate. Held-out labels do not enter the
input features or training loss.

23 个核素的评估集参与过噪声强度选择，因此其分数不属于完全独立的测试集估计。
这些标签不进入输入特征或训练损失。

Workbook ensemble SD uses `ddof=1`; the rounded manuscript table uses `ddof=0`.
Both measure spread across seeds, not calibrated total uncertainty.

## Tests / 测试

To check data consistency, feature calculations, and training-label isolation:

检查数据一致性、特征计算和训练标签隔离：

```bash
python -m unittest discover -s tests
python tests/verify_tables.py --output table-check.json
```

#!/bin/bash

# 数据分割索引生成和使用示例

echo "🚀 数据分割索引生成和使用示例"
echo "=================================="

# 1. 生成数据分割索引
echo "📊 步骤1: 生成数据分割索引"
python generate_splits.py \
    --data-file "path/to/your/data.parquet" \
    --split-ratios 0.8 0.1 0.1 \
    --split-seed 42 \
    --output "data_splits.npz"

echo ""

# 2. 使用生成的索引进行训练
echo "🏃 步骤2: 使用索引进行训练"
python train.py \
    --root "." \
    --data-file "path/to/your/data.parquet" \
    --target-biomarkers HE CD3 CD8 \
    --batch-size 32 \
    --lr 1e-4 \
    --eval-interval 1000 \
    --patience 5000 \
    --num-workers 4 \
    --patch-size 128 \
    --use-zarr \
    --zarr-marker HE \
    --split-file "data_splits.npz"

echo ""
echo "✅ 示例完成！"
echo ""
echo "📋 生成的文件:"
echo "   - data_splits.npz: 二进制格式的索引文件"
echo "   - data_splits.csv: CSV格式的索引文件（便于查看）"
echo ""
echo "💡 使用提示:"
echo "   - 必须先运行 generate_splits.py 生成索引分割"
echo "   - train.py 必须提供 --split-file 参数"
echo "   - 支持 train/val/test 三个数据集"
echo "   - 使用分割文件中的所有样本，不限制数量"
echo ""
echo "📁 文件结构:"
echo "   - generate_splits.py: 独立索引生成脚本"
echo "   - train.py: 训练脚本（必须使用索引分割）"

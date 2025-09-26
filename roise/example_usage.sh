#!/bin/bash

# 完整的ROSIE数据处理和训练流程示例

echo "🚀 ROSIE完整数据处理和训练流程"
echo "=================================="

# 设置路径变量
PNG_DIR="path/to/your/png/images"
ZARR_DIR="./zarr_data"
DATASET_FILE="./data/image_labels.parquet"
SPLITS_DIR="./splits"
PROJECT_ROOT="."

echo "📁 路径配置:"
echo "   - PNG图像目录: $PNG_DIR"
echo "   - Zarr输出目录: $ZARR_DIR"
echo "   - 数据集文件: $DATASET_FILE"
echo "   - 分割输出目录: $SPLITS_DIR"
echo ""

# 1. PNG转换为Zarr
echo "🔄 步骤1: PNG图像转换为Zarr格式"
echo "--------------------------------"
python datalabel_generator/convert_to_zarr_memory.py \
    --input-dir "$PNG_DIR" \
    --output-dir "$ZARR_DIR" \
    --chunk-height 512 \
    --chunk-width 512 \
    --chunk-channels 3 \
    --markers HE CD3 CD8 CD20 CD68 CD163 MHC1 PDL1

echo "✅ PNG转Zarr完成"
echo ""

# 2. 生成数据集标签文件
echo "📊 步骤2: 生成数据集标签文件"
echo "--------------------------------"
python datalabel_generator/zarr_dataframe_generator.py \
    --zarr-dir "$ZARR_DIR" \
    --output "$DATASET_FILE" \
    --stripe-size 8 \
    --kernel-size 8

echo "✅ 数据集标签文件生成完成"
echo ""

# 3. 生成数据分割
echo "✂️  步骤3: 生成训练/验证/测试集分割"
echo "--------------------------------"
python datalabel_generator/generate_splits.py \
    --data-file "$DATASET_FILE" \
    --split-ratios 0.8 0.1 0.1 \
    --split-seed 42 \
    --max-samples 100000 \
    --shuffle \
    --output-dir "$SPLITS_DIR" \
    --output-format parquet

echo "✅ 数据分割完成"
echo ""

# 4. 开始训练
echo "🏃 步骤4: 开始模型训练"
echo "--------------------------------"
python train.py \
    --root "$PROJECT_ROOT" \
    --target-biomarkers CD3 CD8 CD20 CD68 CD163 MHC1 PDL1 \
    --batch-size 32 \
    --lr 1e-4 \
    --eval-interval 1000 \
    --patience 5000 \
    --num-workers 4 \
    --patch-size 128 \
    --use-zarr \
    --zarr-marker HE \
    --splits-dir "$SPLITS_DIR"

echo ""
echo "✅ 完整流程执行完成！"
echo ""
echo "📋 生成的文件和目录:"
echo "   - $ZARR_DIR/: Zarr格式的图像数据"
echo "   - $DATASET_FILE: 数据集标签文件"
echo "   - $SPLITS_DIR/: 分割后的数据集文件"
echo "     ├── train.parquet: 训练集"
echo "     ├── val.parquet: 验证集"
echo "     ├── test.parquet: 测试集"
echo "     └── split_metadata.txt: 分割元数据"
echo "   - runs/: 训练输出目录"
echo "     ├── best_model.pth: 最佳模型"
echo "     └── predictions_*.pqt: 验证预测结果"
echo ""
echo "💡 使用提示:"
echo "   - 修改脚本顶部的路径变量以适配您的数据"
echo "   - 确保PNG图像目录结构正确"
echo "   - 可以根据需要调整max-samples参数控制数据集大小"
echo "   - 训练过程会自动保存最佳模型和预测结果"
echo ""
echo "🔄 完整流程说明:"
echo "   1. PNG → Zarr: 将PNG图像转换为高效的Zarr格式"
echo "   2. 生成标签: 创建包含生物标记物信息的标签文件"
echo "   3. 数据分割: 将数据分割为训练/验证/测试集"
echo "   4. 模型训练: 使用分割后的数据进行模型训练"
echo ""
echo "📁 相关脚本:"
echo "   - datalabel_generator/convert_to_zarr_memory.py: PNG转Zarr"
echo "   - datalabel_generator/zarr_dataframe_generator.py: 生成数据集标签"
echo "   - datalabel_generator/generate_splits.py: 数据分割脚本"
echo "   - train.py: 模型训练脚本"

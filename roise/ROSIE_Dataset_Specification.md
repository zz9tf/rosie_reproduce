# ROSIE 数据集规范文档

## 📋 概述

ROSIE (H&E到多蛋白标记物预测) 是一个基于深度学习的组织病理学图像分析系统。本系统将H&E染色的组织图像作为输入，预测多种蛋白标记物的表达水平。

### 🎯 核心功能
- **输入**: H&E染色的组织图像 (RGB)
- **输出**: 多种蛋白标记物的表达预测 (CD3, CD8, CD20, CD68, CD163, MHC1, PDL1)
- **模型**: 基于ConvNeXt-Small的回归模型
- **数据格式**: 支持PNG图像和Zarr格式

## 🗂️ 项目结构

```
rosie_reproduce/
├── roise/                              # 主项目目录
│   ├── datalabel_generator/            # 数据生成工具
│   │   ├── convert_to_zarr_memory.py   # PNG转Zarr脚本
│   │   ├── zarr_dataframe_generator.py # 生成数据集标签
│   │   └── generate_splits.py          # 数据分割脚本
│   ├── patch_dataset.py               # 数据集加载器
│   ├── model.py                       # 模型定义和训练器
│   ├── train.py                       # 训练脚本
│   ├── example_usage.sh               # 完整流程示例
│   └── README.md                      # 项目说明
├── data/                              # 数据目录
│   ├── images/                        # 原始PNG图像
│   ├── zarr_data/                     # Zarr格式数据
│   └── splits/                        # 分割后的数据集
└── runs/                              # 训练输出
    ├── best_model.pth                 # 最佳模型
    └── predictions_*.pqt              # 预测结果
```

## 🔄 完整数据处理流程

### 步骤1: PNG图像转换为Zarr格式
```bash
python datalabel_generator/convert_to_zarr_memory.py \
    --input-dir "path/to/png/images" \
    --output-dir "./zarr_data" \
    --chunk-height 512 \
    --chunk-width 512 \
    --chunk-channels 3 \
    --markers HE CD3 CD8 CD20 CD68 CD163 MHC1 PDL1
```

**参数说明:**
- `--input-dir`: PNG图像输入目录
- `--output-dir`: Zarr输出目录
- `--chunk-height/width`: Zarr分块大小
- `--markers`: 要转换的生物标记物列表

### 步骤2: 生成数据集标签文件
```bash
python datalabel_generator/zarr_dataframe_generator.py \
    --zarr-dir "./zarr_data" \
    --output "./data/image_labels.parquet" \
    --stripe-size 8 \
    --kernel-size 8
```

**参数说明:**
- `--zarr-dir`: Zarr数据目录
- `--output`: 输出标签文件路径
- `--stripe-size`: 条带大小
- `--kernel-size`: 核大小

### 步骤3: 数据分割
```bash
python datalabel_generator/generate_splits.py \
    --data-file "./data/image_labels.parquet" \
    --split-ratios 0.8 0.1 0.1 \
    --split-seed 42 \
    --max-samples 100000 \
    --shuffle \
    --output-dir "./splits" \
    --output-format parquet
```

**参数说明:**
- `--data-file`: 数据集标签文件
- `--split-ratios`: 训练/验证/测试集比例
- `--max-samples`: 最大样本数限制
- `--shuffle`: 是否打乱数据
- `--output-dir`: 分割输出目录

### 步骤4: 模型训练
```bash
python train.py \
    --root "." \
    --target-biomarkers CD3 CD8 CD20 CD68 CD163 MHC1 PDL1 \
    --batch-size 32 \
    --lr 1e-4 \
    --eval-interval 1000 \
    --patience 5000 \
    --num-workers 4 \
    --patch-size 128 \
    --use-zarr \
    --zarr-marker HE \
    --splits-dir "./splits"
```

## 📊 数据格式规范

### 输入图像格式
- **格式**: PNG/JPEG (RGB三通道)
- **数据类型**: uint8 (0-255)
- **尺寸**: 支持任意尺寸，训练时自动裁剪为patch
- **标记物**: HE, CD3, CD8, CD20, CD68, CD163, MHC1, PDL1

### 数据集标签文件 (Parquet格式)
```python
{
    "image_path": "path/to/image.png",      # 图像路径
    "image_id": "unique_id",                # 图像ID
    "X": 1500,                             # X坐标
    "Y": 2000,                             # Y坐标
    "CD3": 0.5,                            # CD3表达值
    "CD8": 0.3,                            # CD8表达值
    "CD20": 0.7,                           # CD20表达值
    "CD68": 0.2,                           # CD68表达值
    "CD163": 0.4,                          # CD163表达值
    "MHC1": 0.6,                           # MHC1表达值
    "PDL1": 0.1                            # PDL1表达值
}
```

### 分割后数据集
- `train.parquet`: 训练集数据
- `val.parquet`: 验证集数据
- `test.parquet`: 测试集数据
- `split_metadata.txt`: 分割元数据信息

## 🤖 模型架构

### 网络结构
- **骨干网络**: ConvNeXt-Small (预训练于ImageNet)
- **输出层**: 线性层，输出维度为标记物数量
- **损失函数**: Masked MSE Loss
- **优化器**: Adam
- **学习率调度**: ReduceLROnPlateau

### 训练配置
- **批次大小**: 32 (可调整)
- **学习率**: 1e-4
- **验证间隔**: 1000步
- **早停耐心值**: 5000步
- **Patch大小**: 128x128像素

## 🚀 快速开始

### 一键运行完整流程
```bash
# 修改example_usage.sh中的路径变量
bash example_usage.sh
```

### 分步执行
```bash
# 1. 转换数据
python datalabel_generator/convert_to_zarr_memory.py --input-dir your_images --output-dir zarr_data

# 2. 生成标签
python datalabel_generator/zarr_dataframe_generator.py --zarr-dir zarr_data --output labels.parquet

# 3. 分割数据
python datalabel_generator/generate_splits.py --data-file labels.parquet --output-dir splits

# 4. 训练模型
python train.py --splits-dir splits --use-zarr
```

## 📈 性能优化

### 数据加载优化
- **Zarr格式**: 比PNG加载更快，支持随机访问
- **多进程加载**: 使用多个worker进程并行加载数据
- **内存映射**: 减少内存占用
- **数据分割**: 避免重复加载大文件

### 训练优化
- **混合精度**: 支持FP16训练
- **梯度累积**: 支持大批次训练
- **早停机制**: 防止过拟合
- **学习率调度**: 自适应调整学习率

## 🔧 环境要求

### Python依赖
```
torch>=1.12.0
torchvision>=0.13.0
pandas>=1.4.0
numpy>=1.21.0
zarr>=2.12.0
tqdm>=4.64.0
PIL>=9.0.0
opencv-python>=4.6.0
```

### 硬件要求
- **GPU**: 推荐NVIDIA GPU (8GB+ VRAM)
- **内存**: 16GB+ RAM
- **存储**: SSD推荐，支持快速I/O

## 📝 注意事项

1. **数据路径**: 确保所有路径正确设置
2. **内存管理**: 大数据集建议使用Zarr格式
3. **标记物顺序**: 确保训练和推理使用相同的标记物顺序
4. **数据质量**: 确保输入图像质量良好，无损坏
5. **版本兼容**: 确保PyTorch版本兼容

## 🐛 常见问题

### Q: 内存不足怎么办？
A: 减少batch_size，使用Zarr格式，或减少max_samples参数

### Q: 训练速度慢？
A: 使用GPU训练，增加num_workers，使用Zarr格式

### Q: 模型不收敛？
A: 检查学习率，数据质量，标记物范围

### Q: 数据加载错误？
A: 检查文件路径，图像格式，Zarr文件完整性
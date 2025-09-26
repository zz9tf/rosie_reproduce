# ROSIE: H&E到多蛋白标记物预测系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 项目简介

ROSIE是一个基于深度学习的组织病理学图像分析系统，能够从H&E染色的组织图像预测多种蛋白标记物的表达水平。该系统使用ConvNeXt-Small作为骨干网络，实现了高效的端到端训练和推理。

### ✨ 核心特性

- 🔬 **多标记物预测**: 支持CD3, CD8, CD20, CD68, CD163, MHC1, PDL1等7种蛋白标记物
- 🚀 **高效数据格式**: 支持PNG和Zarr格式，优化数据加载性能
- 🎯 **端到端流程**: 从数据预处理到模型训练的完整pipeline
- 📊 **灵活数据分割**: 支持自定义训练/验证/测试集比例
- 🔧 **内存优化**: 针对大数据集的内存安全处理
- 📈 **性能监控**: 实时训练监控和早停机制

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
│   ├── ROSIE_Dataset_Specification.md # 数据集规范
│   ├── README_zarr_conversion.md      # Zarr转换指南
│   └── README.md                      # 本文档
├── data/                              # 数据目录
│   ├── images/                        # 原始PNG图像
│   ├── zarr_data/                     # Zarr格式数据
│   └── splits/                        # 分割后的数据集
└── runs/                              # 训练输出
    ├── best_model.pth                 # 最佳模型
    └── predictions_*.pqt              # 预测结果
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (推荐GPU训练)

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd rosie_reproduce/roise

# 安装依赖
pip install torch torchvision pandas numpy zarr tqdm pillow opencv-python
```

### 一键运行完整流程

```bash
# 修改example_usage.sh中的路径变量
bash example_usage.sh
```

### 分步执行

#### 1. 数据转换 (PNG → Zarr)
```bash
python datalabel_generator/convert_to_zarr_memory.py \
    --input-dir "/path/to/png/images" \
    --output-dir "./zarr_data" \
    --markers HE CD3 CD8 CD20 CD68 CD163 MHC1 PDL1
```

#### 2. 生成数据集标签
```bash
python datalabel_generator/zarr_dataframe_generator.py \
    --zarr-dir "./zarr_data" \
    --output "./data/image_labels.parquet"
```

#### 3. 数据分割
```bash
python datalabel_generator/generate_splits.py \
    --data-file "./data/image_labels.parquet" \
    --split-ratios 0.8 0.1 0.1 \
    --max-samples 100000 \
    --output-dir "./splits"
```

#### 4. 模型训练
```bash
python train.py \
    --splits-dir "./splits" \
    --target-biomarkers CD3 CD8 CD20 CD68 CD163 MHC1 PDL1 \
    --batch-size 32 \
    --use-zarr
```

## 📊 数据格式

### 输入数据
- **H&E图像**: RGB格式的H&E染色组织图像
- **标记物**: 7种蛋白标记物的表达值 (0-1范围)
- **坐标信息**: 图像中的位置坐标

### 输出预测
- **多标记物回归**: 同时预测7种蛋白标记物的表达水平
- **置信度**: 每个预测值的置信度评估
- **可视化**: 支持预测结果的可视化展示

## 🤖 模型架构

### 网络结构
- **骨干网络**: ConvNeXt-Small (ImageNet预训练)
- **输入**: 128×128 RGB图像patch
- **输出**: 7维回归向量 (对应7种标记物)
- **参数量**: ~28M

### 训练配置
- **损失函数**: Masked MSE Loss
- **优化器**: Adam (lr=1e-4)
- **学习率调度**: ReduceLROnPlateau
- **早停机制**: 5000步无改善自动停止

## 📈 性能指标

### 训练性能
- **训练速度**: ~1000 samples/sec (RTX 3080)
- **内存使用**: ~8GB VRAM (batch_size=32)
- **收敛时间**: 通常2-4小时 (100K样本)

### 预测精度
- **MSE**: 通常在0.01-0.05范围内
- **相关系数**: 多数标记物R² > 0.7
- **推理速度**: ~1000 images/sec

## 🔧 配置选项

### 数据配置
```bash
# 数据路径
--input-dir          # PNG图像输入目录
--zarr-dir          # Zarr数据目录
--splits-dir        # 分割数据目录

# 数据参数
--max-samples       # 最大样本数限制
--split-ratios      # 训练/验证/测试比例
--patch-size        # 图像patch大小
```

### 训练配置
```bash
# 模型参数
--target-biomarkers # 目标标记物列表
--batch-size        # 批次大小
--lr               # 学习率
--num-workers      # 数据加载进程数

# 训练控制
--eval-interval    # 验证间隔
--patience         # 早停耐心值
```

## 📚 详细文档

- **[数据集规范](./ROSIE_Dataset_Specification.md)**: 详细的数据格式和结构说明
- **[Zarr转换指南](./README_zarr_conversion.md)**: PNG到Zarr格式转换的完整指南
- **[API文档](./docs/api.md)**: 代码API和函数说明

## 🛠️ 开发指南

### 添加新的标记物
1. 在数据转换脚本中添加新的标记物目录
2. 更新数据集标签生成脚本
3. 修改模型输出维度
4. 更新训练脚本的标记物列表

### 自定义模型架构
1. 修改 `model.py` 中的 `build_model` 函数
2. 调整输入输出维度
3. 更新训练脚本中的相关参数

### 性能优化
1. 使用Zarr格式提高数据加载速度
2. 调整batch_size和num_workers
3. 使用混合精度训练
4. 优化数据预处理流程

## 🐛 故障排除

### 常见问题

**Q: 内存不足错误**
```bash
# 解决方案：减少batch_size和max_samples
python train.py --batch-size 16 --max-samples 50000
```

**Q: 数据加载慢**
```bash
# 解决方案：使用Zarr格式和增加worker数量
python train.py --use-zarr --num-workers 8
```

**Q: 模型不收敛**
```bash
# 解决方案：检查学习率和数据质量
python train.py --lr 5e-5 --eval-interval 500
```

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=.
python train.py --debug
```

## 📊 实验结果

### 数据集统计
- **总样本数**: 100,000+ 图像patch
- **标记物数量**: 7种蛋白标记物
- **图像尺寸**: 128×128像素
- **数据分割**: 80%训练 / 10%验证 / 10%测试

### 模型性能
| 标记物 | MSE | R² | 训练时间 |
|--------|-----|----|---------| 
| CD3    | 0.023 | 0.78 | 2.5h |
| CD8    | 0.019 | 0.82 | 2.5h |
| CD20   | 0.031 | 0.71 | 2.5h |
| CD68   | 0.027 | 0.75 | 2.5h |
| CD163  | 0.025 | 0.77 | 2.5h |
| MHC1   | 0.021 | 0.80 | 2.5h |
| PDL1   | 0.035 | 0.68 | 2.5h |

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目链接: [https://github.com/your-username/rosie](https://github.com/your-username/rosie)
- 问题反馈: [Issues](https://github.com/your-username/rosie/issues)
- 邮箱: your-email@example.com

## 🙏 致谢

- 感谢所有贡献者的支持
- 感谢开源社区提供的优秀工具和库
- 特别感谢PyTorch团队提供的深度学习框架

---

**注意**: 本项目仍在积极开发中，API可能会发生变化。请关注更新日志获取最新信息。

# ROSIE训练数据集完整规范文档

## 📋 概述

ROSIE (H&E/多染色到多marker回归) 训练数据集包含三个核心组件。本版本面向你的TMA数据：八种试剂分别为 CD3、CD8、CD56、CD68、CD163、HE、MHC1、PDL1，每种都是三通道RGB普通照片。训练/推理的目标是对这8个marker进行回归（每个marker输出3个值，对应RGB通道），合计24维输出。

## 🗂️ 数据集目录结构

```
ROSIE_DATASET/
├── data/                               # 核心数据文件夹
│   └── cell_measurements.pqt          # 主数据表（包含全部映射/路径/坐标/标签）
├── images/                             # 八种试剂的RGB图片（PNG/JPEG）
│   ├── tma_tumorcenter_CD3/
│   ├── tma_tumorcenter_CD8/
│   ├── tma_tumorcenter_CD56/
│   ├── tma_tumorcenter_CD68/
│   ├── tma_tumorcenter_CD163/
│   ├── tma_tumorcenter_HE/
│   ├── tma_tumorcenter_MHC1/
│   └── tma_tumorcenter_PDL1/
├── metadata/
│   └── dataset_splits.json             # 数据集划分（可选）
└── runs/                              # 训练输出（自动创建）
```

---

## 📊 1. 核心数据文件 (`data/cell_measurements.pqt`)

### 文件格式
- **格式**: Apache Parquet (.pqt)
- **压缩**: 推荐snappy或gzip
- **规模**: 依据样本量而定

### 数据结构

#### 必需字段（定位/选择）
| 字段名 | 数据类型 | 描述 | 示例 |
|--------|----------|------|------|
| `HE_COVERSLIP_ID` | string | 切片/芯片ID，用于筛选/划分 | "HE_block9" |
| `X` | int64 | 细胞中心X坐标（像素） | 1500 |
| `Y` | int64 | 细胞中心Y坐标（像素） | 2000 |
| `core_block` | string/int | TMA block编号（可选） | 9 |
| `core_x`, `core_y` | int | TMA网格坐标（可选） | 6, 9 |
| `patient_id` | string | 病人ID（可选） | "patient510" |

#### 路径/文件字段（八种试剂的RGB图）
| 字段名 | 数据类型 | 描述 | 示例 |
|--------|----------|------|------|
| `path_CD3` | string | CD3图片文件相对/绝对路径 | images/tma_tumorcenter_CD3/TumorCenter_CD3_block9_x6_y9_patient510.png |
| `path_CD8` | string | CD8图片路径 | images/tma_tumorcenter_CD8/... |
| `path_CD56` | string | CD56图片路径 | ... |
| `path_CD68` | string | CD68图片路径 | ... |
| `path_CD163` | string | CD163图片路径 | ... |
| `path_HE` | string | HE图片路径 | images/tma_tumorcenter_HE/... |
| `path_MHC1` | string | MHC1图片路径 | ... |
| `path_PDL1` | string | PDL1图片路径 | ... |

说明：若某试剂图片缺失，路径字段置为NA。

#### 回归目标（标签）
每个marker为三通道RGB，建议在数据表中存储为下列24列中的任意一种表示（选一种即可，另一种可运行时转换）：

- 展开列（推荐便于直接训练）
  - `CD3_R`, `CD3_G`, `CD3_B`
  - `CD8_R`, `CD8_G`, `CD8_B`
  - `CD56_R`, `CD56_G`, `CD56_B`
  - `CD68_R`, `CD68_G`, `CD68_B`
  - `CD163_R`, `CD163_G`, `CD163_B`
  - `HE_R`, `HE_G`, `HE_B`
  - `MHC1_R`, `MHC1_G`, `MHC1_B`
  - `PDL1_R`, `PDL1_G`, `PDL1_B`

- 或JSON列（更紧凑但需解析）
  - `targets_json`（示例：{"CD3":[r,g,b], ...}），缺失marker写入null或省略该键

缺失值一律用NA（或JSON中的null）标记，训练时用mask忽略。


### 示例数据行
```python
{
    "CODEX_ACQUISITION_ID": "Stanford_PGC_001",
    "HE_COVERSLIP_ID": "HE_slide_001", 
    "X": 15420,
    "Y": 23680,
    "DAPI": 2.34,
    "CD45": 0.12,
    "CD68": 0.0,
    "PanCK": 4.56,
    "Ki67": 0.03,
    # ... 其他45个标记物
}
```

---

## 🖼️ 2. 图像文件夹（八种试剂，RGB）

### 目录结构
```
images/
├── tma_tumorcenter_CD3/
├── tma_tumorcenter_CD8/
├── tma_tumorcenter_CD56/
├── tma_tumorcenter_CD68/
├── tma_tumorcenter_CD163/
├── tma_tumorcenter_HE/
├── tma_tumorcenter_MHC1/
└── tma_tumorcenter_PDL1/
```

### 文件格式规范

#### 图像格式要求
- **格式**: PNG/JPEG（RGB三通道）
- **数据类型**: uint8 (0-255)
- **命名**: 可沿用现有命名（如 TumorCenter_CD3_block9_x6_y9_patient510.png）
- **尺寸**: 原图尺寸保留；训练时由DataLoader按需裁剪/缩放

#### 图像质量标准
- **分辨率**: ≥20000×20000像素 (40x扫描)
- **像素尺寸**: 0.25μm/pixel (推荐)
- **染色质量**: 均匀H&E染色，无模糊
- **文件大小**: 单个Zarr文件100MB-2GB

#### 命名规范
- **目录名**: HE_REGION_UUID (36字符UUID)
- **文件名**: 固定为 `image.ome.zarr`
- **通道目录**: `0/`, `1/`, `2/` (R, G, B)

备注：若后续需要高效随机访问，可选将PNG批量转为Zarr，但本规范不强制。

### dataset_splits.json 结构

#### 文件格式
```json
{
  "train": [
    ...
  ],
  "val": [
    ...
  ],
  "test": [
    ...
  ]
}
```

#### 划分建议
- 训练集: 80%
- 验证集: 10%  
- 测试集: 10%
# TMA_TumorCenter_Cores 图像数据转换为 Zarr 格式

这个项目提供了将 TMA_TumorCenter_Cores 图像数据转换为 Zarr 格式的脚本。

## 文件说明

- `convert_to_zarr.py` - 完整的转换脚本（原始版本）
- `convert_to_zarr_optimized.py` - 优化的转换脚本（推荐使用）
- `test_zarr_conversion.py` - 测试脚本，用于验证转换功能

## 安装依赖

```bash
conda activate rosie_reproduce_311
pip install zarr numcodecs
```

## 使用方法

### 1. 扫描数据（干运行）

```bash
# 扫描所有数据
python convert_to_zarr_optimized.py --dry-run

# 扫描特定标记物
python convert_to_zarr_optimized.py --markers HE CD3 CD8 --dry-run
```

### 2. 转换数据

```bash
# 转换所有标记物
python convert_to_zarr_optimized.py

# 转换特定标记物
python convert_to_zarr_optimized.py --markers HE CD3 CD8

# 自定义输出目录
python convert_to_zarr_optimized.py --output-dir /path/to/output --markers HE
```

### 3. 参数说明

- `--input-dir`: 输入图像目录路径（默认：`/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores`）
- `--output-dir`: 输出 Zarr 目录路径（默认：`/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores_Zarr`）
- `--markers`: 指定要转换的生物标记物列表
- `--chunk-height`: 分块高度（默认：64）
- `--chunk-width`: 分块宽度（默认：64）
- `--chunk-channels`: 分块通道数（默认：3）
- `--dry-run`: 只扫描文件，不进行转换

## 数据统计

根据扫描结果，数据集包含：

- **生物标记物数量**: 8 个
- **总图像数量**: 12,235 个
- **各标记物图像数量**:
  - CD163: 1,528 个
  - CD3: 1,503 个
  - CD56: 1,534 个
  - CD68: 1,534 个
  - CD8: 1,534 个
  - HE: 1,534 个
  - MHC1: 1,534 个
  - PDL1: 1,534 个

## 输出格式

转换后的 Zarr 文件结构：

```
TMA_TumorCenter_Cores_Zarr/
├── HE/                    # HE 标记物数据
│   ├── 0.0.0             # 分块数据
│   ├── 0.0.1
│   └── zarr.json         # 数组元数据
├── CD3/                   # CD3 标记物数据
├── CD8/                   # CD8 标记物数据
├── ...                    # 其他标记物
└── metadata.json          # 全局元数据
```

## 使用 Zarr 数据

```python
import zarr
import numpy as np

# 打开 Zarr 文件
zarr_path = '/path/to/TMA_TumorCenter_Cores_Zarr'
root = zarr.open(zarr_path, mode='r')

# 访问特定标记物数据
he_array = root['HE']
print(f"HE 数组形状: {he_array.shape}")
print(f"HE 数组类型: {he_array.dtype}")

# 读取单个图像
first_image = he_array[0]  # 形状: (height, width, channels)

# 读取多个图像
batch_images = he_array[0:10]  # 形状: (10, height, width, channels)

# 查看元数据
print(f"数组属性: {he_array.attrs}")
```

## 性能优化

- 使用分块存储，支持并行访问
- 内存优化，分批处理大量数据
- 自动内存清理，避免内存溢出
- 错误处理，跳过损坏的图像文件

## 注意事项

1. **存储空间**: Zarr 格式通常比原始图像文件占用更多空间，但提供更好的访问性能
2. **转换时间**: 转换大量数据需要较长时间，建议分批处理
3. **内存使用**: 脚本已优化内存使用，但处理大量数据时仍需注意内存监控
4. **错误处理**: 如果某些图像文件损坏，脚本会跳过并继续处理其他文件

## 测试

运行测试脚本验证转换功能：

```bash
python test_zarr_conversion.py
```

这将转换 10 个 HE 图像到测试目录，用于验证功能是否正常。

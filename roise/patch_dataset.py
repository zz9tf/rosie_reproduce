#!/usr/bin/env python3
"""
Patch-based Image Dataset for H&E to multiplex protein prediction.

这个模块实现了基于patch的图像数据集，支持将大尺寸图像自动切分为非重叠的patches进行训练。
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from typing import Tuple, List, Dict, Optional
import pyarrow.parquet as pq
import zarr

# 默认配置
DEFAULT_PATCH_SIZE = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4

def load_image(image_path: str) -> np.ndarray:
    """
    加载图像文件并转换为numpy数组。
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        图像的numpy数组，形状为 (H, W, C)
    """
    # 支持多种图像格式
    if image_path.lower().endswith(('.tiff', '.tif')):
        # 使用cv2加载TIFF文件
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # 使用PIL加载其他格式
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
    
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    return image

def load_patch_from_zarr(zarr_path: str, image_idx: int, center_x: int, center_y: int, 
                        patch_size: int = 128) -> np.ndarray:
    """
    从zarr数组中直接提取指定位置的图像块
    
    Args:
        zarr_path: zarr文件路径
        image_idx: 图像索引
        center_x: 中心点x坐标
        center_y: 中心点y坐标
        patch_size: patch大小
        
    Returns:
        提取的图像块 (patch_size, patch_size, 3)
    """
    try:
        # 打开zarr数组
        zarr_array = zarr.open(zarr_path, mode='r')
        
        # 读取完整图像
        image = zarr_array[image_idx]  # (H, W, 3)
        H, W, C = image.shape
        
        # 计算patch边界
        half_size = patch_size // 2
        x0 = max(0, center_x - half_size)
        y0 = max(0, center_y - half_size)
        x1 = min(W, center_x + half_size)
        y1 = min(H, center_y + half_size)
        
        # 提取patch
        patch = image[y0:y1, x0:x1, :]
        
        # 如果patch小于指定大小，进行padding
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded_patch = np.zeros((patch_size, patch_size, C), dtype=patch.dtype)
            padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded_patch
        
        return patch
        
    except Exception as e:
        raise ValueError(f"从zarr提取patch失败: {e}")

def get_zarr_traceback_info(row: pd.Series, marker: str = 'HE') -> Dict[str, any]:
    """
    从DataFrame行中获取zarr追溯信息
    
    Args:
        row: DataFrame中的一行数据
        marker: 要追溯的marker名称
        
    Returns:
        包含zarr位置信息的字典
    """
    zarr_path = row.get(f'{marker}_zarr_path', '')
    image_idx = row.get(f'{marker}_image_idx', -1)
    
    if not zarr_path or image_idx == -1:
        return {
            'zarr_path': '',
            'image_idx': -1,
            'center_x': row.get('center_x', -1),
            'center_y': row.get('center_y', -1),
            'error': f'Marker {marker} 的追溯信息不完整'
        }
    
    return {
        'zarr_path': zarr_path,
        'image_idx': int(image_idx),
        'center_x': int(row.get('center_x', -1)),
        'center_y': int(row.get('center_y', -1))
    }

class PatchImageDataset(Dataset):
    """
    基于中心点的patch图像数据集类。
    
    根据DataFrame中记录的中心点坐标提取128x128的patches，每个patch对应一个训练样本。
    支持从zarr数组直接加载数据以提高速度。
    
    Args:
        parquet_path: parquet文件路径
        root_dir: 图像文件的根目录（当use_zarr=False时使用）
        patch_size: patch的大小 (默认: 128)
        transform: 图像变换
        target_biomarkers: 目标生物标记物列表
        cache_images: 是否缓存图像到内存
        max_samples: 最大样本数（用于测试，None表示使用全部数据）
        row_group_size: 每次读取的行组大小
        use_zarr: 是否使用zarr直接加载（默认: True）
        zarr_marker: 用于加载图像的marker名称（默认: 'HE'）
    """
    
    def __init__(self,
                parquet_path: str,
                root_dir: str = None,
                patch_size: int = DEFAULT_PATCH_SIZE,
                transform: Optional[Dict] = None,
                target_biomarkers: Optional[List[str]] = None,
                cache_images: bool = False,
                row_group_size: int = 10000,
                subset: Optional[List[str]] = None,
                use_zarr: bool = True,
                zarr_marker: str = 'HE'):
        
        self.parquet_path = parquet_path
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}  # 图像缓存
        self.target_biomarkers = target_biomarkers
        self.row_group_size = row_group_size
        self.subset = subset
        self.use_zarr = use_zarr
        self.zarr_marker = zarr_marker
        self.zarr_cache = {}  # zarr数组缓存
        self.subset_indices = None  # 初始化subset_indices
        
        # 初始化parquet文件
        self._init_parquet_file()
        
        # 获取生物标记物信息
        self._setup_biomarkers()
        
        # 处理subset过滤
        if self.subset is not None:
            self._setup_subset_indices()
        
        print(f"📊 流式数据集初始化完成")
        print(f"🧬 目标生物标记物: {self.target_biomarkers}")
        print(f"🧬 输出通道数: {len(self.target_biomarkers) * 3} (RGB for each biomarker)")
        print(f"📈 总行数: {self.total_rows}")
        print(f"🚀 数据加载方式: {'Zarr直接加载' if self.use_zarr else '图像文件加载'}")
        if self.use_zarr:
            print(f"🎯 Zarr marker: {self.zarr_marker}")
        if self.subset is not None:
            print(f"📋 子集过滤: {len(self.subset)} 个ID")
    
    def _init_parquet_file(self):
        """初始化parquet文件"""
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Parquet文件不存在: {self.parquet_path}")
        
        # 打开parquet文件
        self.parquet_file = pq.ParquetFile(self.parquet_path)
        self.total_rows = self.parquet_file.metadata.num_rows
        
        # 获取列信息
        self.schema = self.parquet_file.schema
        self.column_names = [field.name for field in self.schema]
        
        print(f"✅ 成功打开parquet文件: {self.parquet_path}")
        print(f"📊 总行数: {self.total_rows:,}")
        print(f"📋 列名: {self.column_names}")
    
    def _setup_biomarkers(self):
        """设置生物标记物列名"""
        # 从列名中推断所有可用的生物标记物
        exclude_cols = {'image_path', 'center_x', 'center_y', 'image_id', 'patient_id', 'block', 'core_group_x', 'core_group_y'}
        exclude_cols.update({col for col in self.column_names if col.endswith('_image_path')})
        exclude_cols.update({col for col in self.column_names if col.endswith('_zarr_path')})
        exclude_cols.update({col for col in self.column_names if col.endswith('_image_idx')})
        
        # 检查是否是RGB格式的列（如 CD3_R, CD3_G, CD3_B）
        rgb_columns = [col for col in self.column_names if col.endswith('_R') or col.endswith('_G') or col.endswith('_B')]
        
        if rgb_columns:
            # 提取所有可用的marker名称（去掉_R, _G, _B后缀）
            available_markers = set()
            for col in rgb_columns:
                marker = col.rsplit('_', 1)[0]  # 去掉最后一个下划线后的部分
                available_markers.add(marker)
            self.available_rgb_biomarkers = sorted(list(available_markers))
            print(f"🎨 检测到RGB格式的生物标记物: {self.available_rgb_biomarkers}")
        else:
            self.available_rgb_biomarkers = []
        
        # 传统格式：直接使用列名作为生物标记物
        self.available_non_rgb_biomarkers = [col for col in self.column_names if (col not in exclude_cols) and (col not in rgb_columns)]
        if self.available_non_rgb_biomarkers:
            print(f"📊 检测到传统格式的生物标记物: {self.available_non_rgb_biomarkers}")
        
        if len(self.available_rgb_biomarkers) == 0 and len(self.available_non_rgb_biomarkers) == 0:
            raise ValueError("未找到生物标记物标签列")
        
        # 设置目标生物标记物
        if self.target_biomarkers is None:
            # 如果没有指定，使用所有可用的生物标记物
            self.target_biomarkers = self.available_rgb_biomarkers + self.available_non_rgb_biomarkers
        else:
            # 验证指定的生物标记物是否可用
            missing_markers = [marker for marker in self.target_biomarkers if marker not in self.available_rgb_biomarkers and marker not in self.available_non_rgb_biomarkers]
            if missing_markers:
                raise ValueError(f"指定的生物标记物不存在: {missing_markers}")
        
        self.rgb_target = []
        self.non_rgb_target = []
        for marker in self.target_biomarkers:
            if marker in self.available_rgb_biomarkers:
                self.rgb_target.append(marker)
            else:
                self.non_rgb_target.append(marker)
        
        print(f"✅ 使用目标生物标记物: {self.target_biomarkers}")
    
    def _setup_subset_indices(self):
        """设置子集索引，用于过滤数据"""
        if self.subset is None:
            self.subset_indices = None
            return
        
        print(f"🔍 正在处理子集行索引...")
        
        # 将subset转换为整数索引
        try:
            # 尝试将subset中的字符串转换为整数索引
            self.subset_indices = [int(idx) for idx in self.subset]
        except ValueError:
            print("⚠️ 错误: subset中的值无法转换为整数索引")
            print("请确保subset包含的是行索引（整数），例如: [0, 1, 2, 100, 200]")
            self.subset_indices = None
            return
        
        # 验证索引是否在有效范围内
        valid_indices = [idx for idx in self.subset_indices if 0 <= idx < self.total_rows]
        if len(valid_indices) != len(self.subset_indices):
            invalid_count = len(self.subset_indices) - len(valid_indices)
            print(f"⚠️ 警告: {invalid_count} 个索引超出范围 [0, {self.total_rows-1}]，已过滤")
        
        self.subset_indices = valid_indices
        print(f"✅ 有效子集索引: {len(self.subset_indices)} 个")
        
    
    def __len__(self) -> int:
        """返回数据集大小"""
        if self.subset_indices is not None:
            return len(self.subset_indices)
        return self.total_rows
    
    def __getitem__(self, idx: int) -> Optional[Tuple]:
        """
        根据索引从parquet文件中读取数据并提取patch。
        
        Returns:
            Tuple: (patch_tensor, expression_values, valid_mask, center_x, center_y, image_idx)
        """
        if idx >= len(self):
            return None
        
        try:
            # 如果有子集索引，使用子集索引
            if self.subset_indices is not None:
                actual_idx = self.subset_indices[idx]
            else:
                actual_idx = idx
            
            # 计算行组索引
            row_group_idx = actual_idx // self.row_group_size
            if row_group_idx >= self.parquet_file.num_row_groups:
                return None
            
            # 使用pyarrow读取行组数据
            batch_table = self.parquet_file.read_row_group(row_group_idx)
            batch_df = batch_table.to_pandas()
            
            # 获取当前行在batch中的索引
            local_idx = actual_idx % self.row_group_size
            if local_idx >= len(batch_df):
                return None
            
            row = batch_df.iloc[local_idx]
            center_x = int(row['center_x'])
            center_y = int(row['center_y'])
            
            # 根据配置选择加载方式
            if self.use_zarr:
                # 使用zarr直接加载
                patch = self._load_patch_from_zarr(row, center_x, center_y)
            else:
                # 使用传统图像文件加载
                patch = self._load_patch_from_image_file(row, center_x, center_y)
            
            # 处理标签
            exp_vec, valid_mask = self._process_labels(row)
            
            # 应用变换
            patch_tensor = self._apply_transforms(patch)
            
            return (patch_tensor, exp_vec, valid_mask, center_x, center_y, idx)
            
        except Exception as e:
            print(f"⚠️ Error loading patch (idx: {idx}): {e}")
            return None
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像（支持缓存）"""
        if self.cache_images and image_path in self.image_cache:
            return self.image_cache[image_path]
        
        image = load_image(image_path)
        
        if self.cache_images:
            self.image_cache[image_path] = image
        
        return image
    
    def _load_patch_from_zarr(self, row: pd.Series, center_x: int, center_y: int) -> np.ndarray:
        """从zarr数组加载patch"""
        # 获取zarr追溯信息
        traceback_info = get_zarr_traceback_info(row, self.zarr_marker)
        
        if 'error' in traceback_info:
            raise ValueError(f"Zarr追溯信息不完整: {traceback_info['error']}")
        
        zarr_path = traceback_info['zarr_path']
        image_idx = traceback_info['image_idx']
        
        # 检查zarr缓存
        cache_key = f"{zarr_path}_{image_idx}"
        if cache_key in self.zarr_cache:
            zarr_array = self.zarr_cache[cache_key]
        else:
            zarr_array = zarr.open(zarr_path, mode='r')
            if self.cache_images:
                self.zarr_cache[cache_key] = zarr_array
        
        # 直接提取patch
        patch = load_patch_from_zarr(zarr_path, image_idx, center_x, center_y, self.patch_size)
        
        return patch
    
    def _load_patch_from_image_file(self, row: pd.Series, center_x: int, center_y: int) -> np.ndarray:
        """从图像文件加载patch（传统方式）"""
        # 处理图像路径
        image_path_str = row['image_path']
        image_path = os.path.join(self.root_dir, image_path_str)
        
        # 加载图像
        image = self._load_image(image_path)
        height, width = image.shape[:2]
        
        # 计算patch的边界，确保不超出图像范围
        half_patch = self.patch_size // 2
        x_start = max(0, center_x - half_patch)
        y_start = max(0, center_y - half_patch)
        x_end = min(width, center_x + half_patch)
        y_end = min(height, center_y + half_patch)
        
        # 提取patch
        patch = image[y_start:y_end, x_start:x_end]
        
        # 如果patch尺寸不足，进行padding
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            patch = self._pad_patch(patch, center_x, center_y, width, height)
        
        return patch
    
    def _get_image_path_for_marker(self, row: pd.Series, marker: str) -> str:
        """获取指定marker的图像路径"""
        # 首先尝试marker特定的路径
        marker_path_col = f'{marker}_image_path'
        if marker_path_col in row and pd.notna(row[marker_path_col]) and row[marker_path_col] != '':
            return os.path.join(self.root_dir, row[marker_path_col])
        
        # 如果marker特定路径不存在，使用通用路径
        if 'image_path' in row and pd.notna(row['image_path']) and row['image_path'] != '':
            return os.path.join(self.root_dir, row['image_path'])
        
        # 如果都没有，返回空字符串
        return ''
    
    def _pad_patch(self, patch: np.ndarray, center_x: int, center_y: int, 
                   image_width: int, image_height: int) -> np.ndarray:
        """
        对patch进行padding，确保尺寸为patch_size x patch_size。
        
        Args:
            patch: 原始patch
            center_x: 中心点x坐标
            center_y: 中心点y坐标
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            填充后的patch，尺寸为 (patch_size, patch_size, channels)
        """
        current_height, current_width = patch.shape[:2]
        
        if current_height == self.patch_size and current_width == self.patch_size:
            return patch
        
        # 计算需要的padding
        half_patch = self.patch_size // 2
        
        # 计算patch的理想边界
        x_start = center_x - half_patch
        y_start = center_y - half_patch
        x_end = center_x + half_patch
        y_end = center_y + half_patch
        
        # 计算各边的padding
        pad_left = max(0, -x_start)      # 如果x_start < 0，需要左侧padding
        pad_right = max(0, x_end - image_width)  # 如果x_end > image_width，需要右侧padding
        pad_top = max(0, -y_start)       # 如果y_start < 0，需要顶部padding
        pad_bottom = max(0, y_end - image_height)  # 如果y_end > image_height，需要底部padding
        
        # 应用padding
        if patch.ndim == 3:
            pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right))
        
        # 使用图像的最小值进行padding（通常是0或255）
        pad_value = 0 if patch.dtype == np.uint8 else patch.min()
        padded_patch = np.pad(patch, pad_shape, mode='constant', constant_values=pad_value)
        
        # 确保patch尺寸正确
        padded_patch = padded_patch[:self.patch_size, :self.patch_size]
        
        return padded_patch
    
    def _process_labels(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """处理标签数据"""
        # 计算输出维度
        rgb_output_dim = len(self.rgb_target) * 3
        non_rgb_output_dim = len(self.non_rgb_target)
        total_output_dim = rgb_output_dim + non_rgb_output_dim
        
        # 初始化输出向量
        exp_vec = np.zeros(total_output_dim, dtype=np.float32)
        valid_mask = np.zeros(total_output_dim, dtype=bool)
        
        # 处理RGB格式的markers
        if self.rgb_target:
            rgb_cols = []
            for marker in self.rgb_target:
                rgb_cols.extend([f"{marker}_R", f"{marker}_G", f"{marker}_B"])
            
            rgb_values = np.array([row[col] if not pd.isna(row[col]) else 0.0 for col in rgb_cols], dtype=np.float32)
            rgb_valid = np.array([not pd.isna(row[col]) for col in rgb_cols], dtype=bool)
            
            exp_vec[:rgb_output_dim] = rgb_values
            valid_mask[:rgb_output_dim] = rgb_valid
        
        # 处理非RGB格式的markers
        if self.non_rgb_target:
            non_rgb_values = np.array([row[col] if not pd.isna(row[col]) else 0.0 for col in self.non_rgb_target], dtype=np.float32)
            non_rgb_valid = np.array([not pd.isna(row[col]) for col in self.non_rgb_target], dtype=bool)
            
            exp_vec[rgb_output_dim:] = non_rgb_values
            valid_mask[rgb_output_dim:] = non_rgb_valid
        
        return exp_vec, valid_mask
    
    def _apply_transforms(self, patch: np.ndarray) -> torch.Tensor:
        """应用图像变换"""
        if self.transform is not None:
            # 设置随机种子
            seed = np.random.randint(2**32)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            if isinstance(self.transform, dict):
                # 分步骤变换
                patch_tensor = self.transform['all_channels'](patch)
                patch_final = self.transform['image_only'](patch_tensor)
            else:
                patch_final = self.transform(patch)
        else:
            # 默认变换
            patch_final = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
        
        return patch_final

def get_default_transforms():
    """获取默认的数据变换"""
    transform_train = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize(224, antialias=True),
            transforms.RandomRotation(degrees=(-10, 10)),
        ]),
        'image_only': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    
    transform_eval = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
        ]),
        'image_only': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    
    return transform_train, transform_eval

def collate_fn(batch):
    """自定义collate函数，过滤None值"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""
    sample_data = {
        'image_path': [
            'sample1.png',
            'sample1.png',  # 同一张图片的多个中心点
            'sample2.png', 
            'sample3.png'
        ],
        'center_x': [100, 200, 150, 300],
        'center_y': [100, 200, 150, 300],
        'patient_id': ['patient_00' + str(i) for i in range(4)], # 添加patient_id列
        'CD3': [0.5, 0.8, 0.3, 0.6],
        'CD8': [0.2, 0.6, 0.4, 0.7],
        'CD20': [0.7, 0.3, 0.9, 0.4],
        'CD68': [0.1, 0.9, 0.2, 0.8]  # 添加更多生物标记物
    }
    return pd.DataFrame(sample_data)

def create_rgb_sample_data() -> pd.DataFrame:
    """创建RGB格式的示例数据"""
    sample_data = {
        'image_path': [
            'HE/sample1.png',
            'HE/sample1.png',  # 同一张图片的多个中心点
            'CD3/sample2.png', 
            'CD8/sample3.png'
        ],
        'center_x': [100, 200, 150, 300],
        'center_y': [100, 200, 150, 300],
        'patient_id': ['patient_00' + str(i) for i in range(4)],
        # RGB格式的生物标记物
        'HE_R': [120.5, 180.8, 130.3, 160.6],
        'HE_G': [100.2, 160.6, 140.4, 170.7],
        'HE_B': [80.7, 130.3, 190.9, 140.4],
        'CD3_R': [50.1, 90.9, 20.2, 80.8],
        'CD3_G': [40.2, 80.6, 30.4, 70.7],
        'CD3_B': [30.7, 70.3, 40.9, 60.4],
        'CD8_R': [60.5, 80.8, 50.3, 70.6],
        'CD8_G': [50.2, 70.6, 60.4, 80.7],
        'CD8_B': [40.7, 60.3, 70.9, 90.4]
    }
    return pd.DataFrame(sample_data)

def test_dataset_with_real_image(image_path: str, patch_size: int = 128, 
                                use_rgb_format: bool = False, 
                                target_biomarkers: Optional[List[str]] = None):
    """使用真实图像测试数据集"""
    print("=" * 60)
    print("🧪 测试基于中心点的PatchImageDataset")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    if use_rgb_format:
        # 创建RGB格式的测试数据（模拟tma_data.parquet格式）
        test_data = pd.DataFrame({
            'image_path': [os.path.basename(image_path)] * 4,  # 同一张图片的4个中心点
            'center_x': [100, 500, 1000, 2000],  # 不同的中心点x坐标
            'center_y': [100, 500, 1000, 2000],  # 不同的中心点y坐标
            'patient_id': ['patient_00' + str(i) for i in range(4)],
            # RGB格式的生物标记物（模拟dataframe_generator.py的输出）
            'HE_R': [120.5, 180.8, 130.3, 160.6],
            'HE_G': [100.2, 160.6, 140.4, 170.7],
            'HE_B': [80.7, 130.3, 190.9, 140.4],
            'CD3_R': [50.1, 90.9, 20.2, 80.8],
            'CD3_G': [40.2, 80.6, 30.4, 70.7],
            'CD3_B': [30.7, 70.3, 40.9, 60.4],
            'CD8_R': [60.5, 80.8, 50.3, 70.6],
            'CD8_G': [50.2, 70.6, 60.4, 80.7],
            'CD8_B': [40.7, 60.3, 70.9, 90.4],
            'CD56_R': [45.1, 75.9, 35.2, 65.8],
            'CD56_G': [35.2, 65.6, 45.4, 55.7],
            'CD56_B': [25.7, 55.3, 55.9, 45.4]
        })
        print("🎨 使用RGB格式的测试数据（模拟tma_data.parquet）")
    else:
        # 创建传统格式的测试数据
        test_data = pd.DataFrame({
            'image_path': [os.path.basename(image_path)] * 4,  # 同一张图片的4个中心点
            'center_x': [100, 500, 1000, 2000],  # 不同的中心点x坐标
            'center_y': [100, 500, 1000, 2000],  # 不同的中心点y坐标
            'patient_id': ['patient_00' + str(i) for i in range(4)],
            'CD3': [0.5, 0.8, 0.3, 0.6],
            'CD8': [0.2, 0.6, 0.4, 0.7],    # 添加更多生物标记物
            'CD20': [0.7, 0.3, 0.9, 0.4],
            'CD68': [0.1, 0.9, 0.2, 0.8]
        })
        print("📊 使用传统格式的测试数据")
    
    # 设置根目录
    root_dir = os.path.dirname(image_path)
    
    # 创建数据集
    print("📦 创建数据集...")
    dataset = PatchImageDataset(
        data_df=test_data,
        root_dir=root_dir,
        patch_size=patch_size,
        transform=None,  # 不使用变换以便观察原始数据
        cache_images=True,
        target_biomarkers=target_biomarkers
    )
    
    print(f"📊 数据集大小: {len(dataset)} 个样本")
    print(f"🧬 可用生物标记物: {dataset.available_biomarkers}")
    print(f"🎯 目标生物标记物: {dataset.target_biomarkers}")
    
    # 测试数据加载
    print("\n🔍 测试数据加载...")
    if len(dataset) > 0:
        # 测试单个样本
        sample = dataset[0]
        if sample is not None:
            patch_tensor, exp_vec, valid_mask, center_x, center_y, img_idx = sample
            print(f"✅ 成功加载样本:")
            print(f"   - Patch形状: {patch_tensor.shape}")
            print(f"   - 中心点: ({center_x}, {center_y})")
            print(f"   - 图像索引: {img_idx}")
            print(f"   - 表达值: {exp_vec}")
            print(f"   - 有效掩码: {valid_mask}")
        
        # 测试DataLoader
        print("\n🔄 测试DataLoader...")
        dataloader = DataLoader(
            dataset, 
            batch_size=min(2, len(dataset)), 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0  # 避免多进程问题
        )
        
        for i, batch in enumerate(dataloader):
            if batch is not None:
                inputs, labels, masks, x_coords, y_coords, img_indices = batch
                print(f"✅ 批次 {i}:")
                print(f"   - 输入形状: {inputs.shape}")
                print(f"   - 标签形状: {labels.shape}")
                print(f"   - 掩码形状: {masks.shape}")
                print(f"   - 中心点坐标: x={x_coords.tolist()}, y={y_coords.tolist()}")
                break
            else:
                print(f"⚠️ 批次 {i} 为空")
    
    print("\n✅ 测试完成！")

def test_with_parquet_file(parquet_path: str, image_root_dir: str = None, 
                          patch_size: int = 128, 
                          target_biomarkers: Optional[List[str]] = None,
                          subset: Optional[List[str]] = None,
                          use_zarr: bool = True,
                          zarr_marker: str = 'HE'):
    """使用parquet文件测试数据集（流式读取）"""
    print("=" * 60)
    print("🧪 测试基于parquet文件的流式PatchImageDataset")
    print("=" * 60)
    
    if not os.path.exists(parquet_path):
        print(f"❌ Parquet文件不存在: {parquet_path}")
        return
    
    # 创建流式数据集
    print("📦 创建流式数据集...")
    try:
        dataset = PatchImageDataset(
            parquet_path=parquet_path,
            root_dir=image_root_dir,
            patch_size=patch_size,
            transform=None,  # 不使用变换以便观察原始数据
            cache_images=False,  # 对于大数据集，不缓存图像
            target_biomarkers=target_biomarkers,
            subset=subset,  # 子集过滤
            use_zarr=use_zarr,  # 使用zarr加载
            zarr_marker=zarr_marker  # zarr marker
        )
        
        print(f"📊 数据集大小: {len(dataset)} 个样本")
        print(f"🎯 目标生物标记物: {dataset.target_biomarkers}")
        
        # 测试数据加载
        print("\n🔍 测试数据加载...")
        if len(dataset) > 0:
            # 测试单个样本
            sample = dataset[0]
            if sample is not None:
                patch_tensor, exp_vec, valid_mask, center_x, center_y, img_idx = sample
                print(f"✅ 成功加载样本:")
                print(f"   - Patch形状: {patch_tensor.shape}")
                print(f"   - 中心点: ({center_x}, {center_y})")
                print(f"   - 图像索引: {img_idx}")
                print(f"   - 表达值形状: {exp_vec.shape}")
                print(f"   - 表达值: {exp_vec}")
                print(f"   - 有效掩码形状: {valid_mask.shape}")
                print(f"   - 有效掩码: {valid_mask}")
            
            # 测试DataLoader
            print("\n🔄 测试DataLoader...")
            dataloader = DataLoader(
                dataset, 
                batch_size=min(2, len(dataset)), 
                shuffle=False, 
                collate_fn=collate_fn,
                num_workers=0  # 避免多进程问题
            )
            
            for i, batch in enumerate(dataloader):
                if batch is not None:
                    inputs, labels, masks, x_coords, y_coords, img_indices = batch
                    print(f"✅ 批次 {i}:")
                    print(f"   - 输入形状: {inputs.shape}")
                    print(f"   - 标签形状: {labels.shape}")
                    print(f"   - 掩码形状: {masks.shape}")
                    print(f"   - 中心点坐标: x={x_coords.tolist()}, y={y_coords.tolist()}")
                    break
                else:
                    print(f"⚠️ 批次 {i} 为空")
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 创建数据集失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基于中心点的Patch数据集测试')
    parser.add_argument('--image', type=str, help='测试图像路径')
    parser.add_argument('--parquet', type=str, help='parquet文件路径')
    parser.add_argument('--image-root', type=str, help='图像根目录路径')
    parser.add_argument('--patch-size', type=int, default=128, help='patch尺寸')
    parser.add_argument('--rgb-format', action='store_true', help='使用RGB格式的测试数据')
    parser.add_argument('--target-biomarkers', nargs='+', help='目标生物标记物列表，例如: --target-biomarkers CD3 CD8')
    parser.add_argument('--subset', nargs='+', help='子集行索引列表，例如: --subset 0 1 2 100 200')
    parser.add_argument('--use-zarr', action='store_true', default=True, help='使用zarr直接加载（默认: True）')
    parser.add_argument('--use-image-files', action='store_true', help='使用图像文件加载（覆盖zarr设置）')
    parser.add_argument('--zarr-marker', type=str, default='HE', help='用于加载图像的zarr marker名称（默认: HE）')
    
    args = parser.parse_args()
    
    if args.parquet:
        # 使用parquet文件测试（流式读取）
        use_zarr = args.use_zarr and not args.use_image_files
        test_with_parquet_file(
            args.parquet, 
            args.image_root, 
            args.patch_size, 
            args.target_biomarkers, 
            args.subset,
            use_zarr,
            args.zarr_marker
        )
    elif args.image:
        # 使用单张图像测试
        test_dataset_with_real_image(args.image, args.patch_size, args.rgb_format, args.target_biomarkers)
    else:
        print("❌ 请提供测试参数")
        print("使用方法:")
        print("  # 使用单张图像测试:")
        print("  python patch_dataset.py --image /path/to/image.png")
        print("  python patch_dataset.py --image /path/to/image.png --rgb-format")
        print("  python patch_dataset.py --image /path/to/image.png --rgb-format --target-biomarkers CD3 CD8")
        print("  # 使用parquet文件测试（流式读取）:")
        print("  # Zarr直接加载（推荐，速度快）:")
        print("  python patch_dataset.py --parquet /path/to/data.parquet")
        print("  python patch_dataset.py --parquet /path/to/data.parquet --target-biomarkers CD3 CD8")
        print("  python patch_dataset.py --parquet /path/to/data.parquet --zarr-marker CD3")
        print("  # 传统图像文件加载:")
        print("  python patch_dataset.py --parquet /path/to/data.parquet --image-root /path/to/images --use-image-files")
        print("  # 其他选项:")
        print("  python patch_dataset.py --parquet /path/to/data.parquet --subset 0 1 2 100 200")

if __name__ == '__main__':
    main()

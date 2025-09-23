#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROSIE数据集生成示例脚本 📊
演示如何从SVS文件和CODEX数据生成完整的训练数据集

使用方法:
    python generate_dataset_example.py --codex-dir /path/to/codex --svs-dir /path/to/svs --output-dir /path/to/output
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ROSIE标准50个生物标记物
BIOMARKERS = [
    "DAPI", "CD45", "CD68", "CD14", "PD1", "FoxP3", "CD8", "HLA-DR", 
    "PanCK", "CD3e", "CD4", "aSMA", "CD31", "Vimentin", "CD45RO", 
    "Ki67", "CD20", "CD11c", "Podoplanin", "PDL1", "GranzymeB", 
    "CD38", "CD141", "CD21", "CD163", "BCL2", "LAG3", "EpCAM", 
    "CD44", "ICOS", "GATA3", "Gal3", "CD39", "CD34", "TIGIT", 
    "ECad", "CD40", "VISTA", "HLA-A", "MPO", "PCNA", "ATM", 
    "TP63", "IFNg", "Keratin8/18", "IDO1", "CD79a", "HLA-E", 
    "CollagenIV", "CD66"
]

class ROSIEDatasetGenerator:
    """ROSIE数据集生成器"""
    
    def __init__(self, output_dir: str):
        """
        初始化数据集生成器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """创建输出目录结构"""
        directories = [
            self.output_dir / "data",
            self.output_dir / "images", 
            self.output_dir / "metadata",
            self.output_dir / "runs"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ 创建目录结构: {self.output_dir}")
    
    def convert_svs_to_zarr(self, svs_dir: str) -> Dict[str, str]:
        """
        将SVS文件转换为Zarr格式
        
        Args:
            svs_dir: SVS文件目录
            
        Returns:
            Dict mapping SVS filename to generated UUID
        """
        print("🔄 开始SVS到Zarr转换...")
        
        svs_files = list(Path(svs_dir).glob("*.svs"))
        if not svs_files:
            print("❌ 未找到SVS文件")
            return {}
            
        svs_to_uuid = {}
        
        for svs_file in tqdm(svs_files, desc="转换SVS文件"):
            try:
                # 生成唯一UUID
                region_uuid = str(uuid.uuid4())
                
                # 创建输出目录
                zarr_dir = self.output_dir / "images" / region_uuid
                zarr_dir.mkdir(parents=True, exist_ok=True)
                
                # 模拟转换过程 (实际应调用svs_to_zarr_converter)
                print(f"  转换: {svs_file.name} → {region_uuid}")
                
                # 这里应该调用实际的转换函数:
                # from svs_to_zarr_converter import SVSToZarrConverter
                # converter = SVSToZarrConverter(str(zarr_dir))
                # converter.convert_svs_to_zarr(str(svs_file), "image")
                
                # 创建模拟的zarr结构
                self._create_mock_zarr(zarr_dir / "image.ome.zarr")
                
                svs_to_uuid[svs_file.stem] = region_uuid
                
            except Exception as e:
                print(f"❌ 转换失败 {svs_file.name}: {e}")
                
        print(f"✅ 转换完成: {len(svs_to_uuid)} 个文件")
        return svs_to_uuid
    
    def _create_mock_zarr(self, zarr_path: Path):
        """创建模拟的Zarr结构 (用于演示)"""
        zarr_path.mkdir(parents=True, exist_ok=True)
        
        # 创建基本的zarr结构
        (zarr_path / ".zgroup").touch()
        (zarr_path / ".zattrs").touch()
        
        # 创建RGB通道目录
        for channel in [0, 1, 2]:
            channel_dir = zarr_path / str(channel)
            channel_dir.mkdir(exist_ok=True)
            (channel_dir / ".zarray").touch()
            (channel_dir / "0.0.0").touch()  # 示例数据块
    
    def load_codex_data(self, codex_dir: str) -> pd.DataFrame:
        """
        加载CODEX细胞数据
        
        Args:
            codex_dir: CODEX数据目录
            
        Returns:
            包含细胞测量数据的DataFrame
        """
        print("📊 加载CODEX数据...")
        
        # 模拟加载多个CODEX实验的数据
        all_data = []
        
        # 假设有多个CSV文件或parquet文件
        data_files = list(Path(codex_dir).glob("*.csv")) + list(Path(codex_dir).glob("*.pqt"))
        
        if not data_files:
            print("⚠️ 未找到CODEX数据文件，生成模拟数据")
            return self._generate_mock_codex_data()
        
        for data_file in tqdm(data_files, desc="加载CODEX文件"):
            try:
                if data_file.suffix == '.csv':
                    df = pd.read_csv(data_file)
                elif data_file.suffix == '.pqt':
                    df = pd.read_parquet(data_file)
                else:
                    continue
                    
                # 添加数据源信息
                df['source_file'] = data_file.stem
                all_data.append(df)
                
            except Exception as e:
                print(f"❌ 加载失败 {data_file.name}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✅ 加载完成: {len(combined_df)} 个细胞")
            return combined_df
        else:
            return self._generate_mock_codex_data()
    
    def _generate_mock_codex_data(self, n_cells: int = 100000) -> pd.DataFrame:
        """生成模拟的CODEX数据用于演示"""
        print(f"🎭 生成模拟数据: {n_cells} 个细胞")
        
        np.random.seed(42)
        
        data = {
            'cell_id': range(n_cells),
            'X': np.random.randint(100, 20000, n_cells),
            'Y': np.random.randint(100, 15000, n_cells),
            'study_name': np.random.choice(['Training-1', 'Training-2', 'Stanford-PGC'], n_cells),
            'tissue_type': np.random.choice(['Pancreas', 'Liver', 'Tonsil'], n_cells),
        }
        
        # 生成生物标记物表达数据
        for marker in BIOMARKERS:
            if marker == 'DAPI':
                # DAPI通常表达较高
                data[marker] = np.random.gamma(3, 0.5, n_cells)
            elif marker in ['CD45', 'CD3e', 'CD4', 'CD8']:
                # 免疫细胞标记，部分细胞高表达
                data[marker] = np.random.beta(1, 3, n_cells) * np.random.choice([0, 1], n_cells, p=[0.7, 0.3])
            elif marker in ['Ki67', 'PCNA']:
                # 增殖标记，少数细胞表达
                data[marker] = np.random.beta(1, 5, n_cells) * np.random.choice([0, 1], n_cells, p=[0.9, 0.1])
            else:
                # 其他标记物，一般分布
                data[marker] = np.random.beta(2, 5, n_cells)
        
        return pd.DataFrame(data)
    
    def process_cell_data(self, codex_df: pd.DataFrame, svs_to_uuid: Dict[str, str]) -> pd.DataFrame:
        """
        处理细胞数据，添加必需字段
        
        Args:
            codex_df: CODEX数据DataFrame
            svs_to_uuid: SVS文件到UUID的映射
            
        Returns:
            处理后的DataFrame
        """
        print("🔧 处理细胞数据...")
        
        # 生成CODEX_ACQUISITION_ID
        unique_studies = codex_df['study_name'].unique() if 'study_name' in codex_df.columns else ['study_1']
        
        acq_ids = []
        he_coverslip_ids = []
        
        for idx, row in codex_df.iterrows():
            study = row.get('study_name', 'study_1')
            acq_id = f"{study}_{idx // 1000:03d}"  # 每1000个细胞一个获取ID
            he_id = f"HE_{study}_{idx // 2000:03d}"  # 每2000个细胞一个切片ID
            
            acq_ids.append(acq_id)
            he_coverslip_ids.append(he_id)
        
        codex_df['CODEX_ACQUISITION_ID'] = acq_ids
        codex_df['HE_COVERSLIP_ID'] = he_coverslip_ids
        
        # 确保包含所有必需的生物标记物
        for marker in BIOMARKERS:
            if marker not in codex_df.columns:
                print(f"⚠️ 缺少标记物 {marker}，设置为NaN")
                codex_df[marker] = np.nan
        
        # 重新排列列顺序
        required_cols = ['CODEX_ACQUISITION_ID', 'HE_COVERSLIP_ID', 'X', 'Y']
        other_cols = [col for col in codex_df.columns if col not in required_cols + BIOMARKERS]
        
        final_cols = required_cols + BIOMARKERS + other_cols
        codex_df = codex_df[final_cols]
        
        print(f"✅ 数据处理完成: {len(codex_df)} 行 × {len(codex_df.columns)} 列")
        return codex_df
    
    def create_metadata_dict(self, cell_df: pd.DataFrame, svs_to_uuid: Dict[str, str]) -> Dict:
        """
        创建元数据字典
        
        Args:
            cell_df: 细胞数据DataFrame
            svs_to_uuid: SVS到UUID的映射
            
        Returns:
            元数据字典
        """
        print("🗂️ 创建元数据字典...")
        
        metadata_dict = {
            "all_biomarkers": BIOMARKERS.copy()
        }
        
        # 为每个CODEX_ACQUISITION_ID创建映射
        for acq_id in cell_df['CODEX_ACQUISITION_ID'].unique():
            # 获取该获取ID的样本信息
            sample_data = cell_df[cell_df['CODEX_ACQUISITION_ID'] == acq_id].iloc[0]
            
            # 分配H&E图像UUID (简化映射)
            if svs_to_uuid:
                region_uuid = list(svs_to_uuid.values())[hash(acq_id) % len(svs_to_uuid)]
            else:
                region_uuid = str(uuid.uuid4())
            
            metadata_dict[acq_id] = {
                "HE_REGION_UUID": region_uuid,
                "study_name": sample_data.get('study_name', 'Unknown'),
                "tissue_type": sample_data.get('tissue_type', 'Unknown'),
                "biomarkers_available": [
                    marker for marker in BIOMARKERS 
                    if not pd.isna(sample_data.get(marker, np.nan))
                ]
            }
        
        print(f"✅ 元数据创建完成: {len(metadata_dict)-1} 个获取ID")
        return metadata_dict
    
    def create_dataset_splits(self, metadata_dict: Dict) -> Dict:
        """
        创建数据集划分
        
        Args:
            metadata_dict: 元数据字典
            
        Returns:
            数据集划分字典
        """
        print("📊 创建数据集划分...")
        
        acq_ids = [key for key in metadata_dict.keys() if key != "all_biomarkers"]
        np.random.shuffle(acq_ids)
        
        n_total = len(acq_ids)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        splits = {
            "train": acq_ids[:n_train],
            "val": acq_ids[n_train:n_train+n_val],
            "test": acq_ids[n_train+n_val:]
        }
        
        print(f"✅ 数据集划分: 训练{len(splits['train'])} | 验证{len(splits['val'])} | 测试{len(splits['test'])}")
        return splits
    
    def save_dataset(self, cell_df: pd.DataFrame, metadata_dict: Dict, dataset_splits: Dict):
        """
        保存完整数据集
        
        Args:
            cell_df: 细胞数据
            metadata_dict: 元数据字典
            dataset_splits: 数据集划分
        """
        print("💾 保存数据集文件...")
        
        # 保存主数据文件
        cell_data_path = self.output_dir / "data" / "cell_measurements.pqt"
        cell_df.to_parquet(cell_data_path, engine='fastparquet', compression='snappy')
        print(f"✅ 保存: {cell_data_path} ({cell_data_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # 保存元数据
        metadata_path = self.output_dir / "metadata" / "metadata_dict.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f, protocol=4)
        print(f"✅ 保存: {metadata_path} ({metadata_path.stat().st_size / 1024:.1f} KB)")
        
        # 保存数据集划分
        splits_path = self.output_dir / "metadata" / "dataset_splits.json"
        with open(splits_path, 'w') as f:
            json.dump(dataset_splits, f, indent=2)
        print(f"✅ 保存: {splits_path}")
        
        # 生成数据集摘要
        self._generate_summary(cell_df, metadata_dict, dataset_splits)
    
    def _generate_summary(self, cell_df: pd.DataFrame, metadata_dict: Dict, dataset_splits: Dict):
        """生成数据集摘要报告"""
        summary_path = self.output_dir / "dataset_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("ROSIE数据集摘要报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本统计
            f.write(f"细胞总数: {len(cell_df):,}\n")
            f.write(f"获取ID数: {len(metadata_dict) - 1}\n")
            f.write(f"生物标记物数: {len(BIOMARKERS)}\n\n")
            
            # 数据集划分
            f.write("数据集划分:\n")
            for split, ids in dataset_splits.items():
                f.write(f"  {split}: {len(ids)} 个获取ID\n")
            f.write("\n")
            
            # 组织类型分布
            if 'tissue_type' in cell_df.columns:
                tissue_counts = cell_df['tissue_type'].value_counts()
                f.write("组织类型分布:\n")
                for tissue, count in tissue_counts.items():
                    f.write(f"  {tissue}: {count:,} 个细胞\n")
                f.write("\n")
            
            # 生物标记物统计
            f.write("生物标记物表达统计:\n")
            for marker in BIOMARKERS[:5]:  # 只显示前5个
                if marker in cell_df.columns:
                    valid_count = cell_df[marker].notna().sum()
                    mean_expr = cell_df[marker].mean()
                    f.write(f"  {marker}: {valid_count:,} 个有效值, 平均表达 {mean_expr:.3f}\n")
            f.write("  ...\n")
        
        print(f"✅ 生成摘要: {summary_path}")
    
    def validate_dataset(self) -> bool:
        """验证数据集完整性"""
        print("🔍 验证数据集...")
        
        required_files = [
            self.output_dir / "data" / "cell_measurements.pqt",
            self.output_dir / "metadata" / "metadata_dict.pkl",
            self.output_dir / "metadata" / "dataset_splits.json"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                print(f"❌ 缺少文件: {file_path}")
                return False
        
        try:
            # 验证数据文件
            df = pd.read_parquet(required_files[0])
            required_cols = ['CODEX_ACQUISITION_ID', 'HE_COVERSLIP_ID', 'X', 'Y'] + BIOMARKERS
            
            for col in required_cols:
                if col not in df.columns:
                    print(f"❌ 缺少列: {col}")
                    return False
            
            # 验证元数据
            with open(required_files[1], 'rb') as f:
                metadata = pickle.load(f)
            
            if "all_biomarkers" not in metadata:
                print("❌ 元数据缺少 all_biomarkers")
                return False
            
            # 验证图像目录
            images_dir = self.output_dir / "images"
            if not images_dir.exists():
                print("❌ 缺少图像目录")
                return False
            
            print("✅ 数据集验证通过!")
            return True
            
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ROSIE数据集生成器")
    parser.add_argument("--codex-dir", required=True, help="CODEX数据目录")
    parser.add_argument("--svs-dir", required=True, help="SVS文件目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--mock-data", action="store_true", help="生成模拟数据用于测试")
    
    args = parser.parse_args()
    
    print("🚀 开始生成ROSIE数据集")
    print(f"CODEX目录: {args.codex_dir}")
    print(f"SVS目录: {args.svs_dir}")
    print(f"输出目录: {args.output_dir}")
    print("-" * 50)
    
    # 创建数据集生成器
    generator = ROSIEDatasetGenerator(args.output_dir)
    
    try:
        # 步骤1: 转换SVS到Zarr
        svs_to_uuid = generator.convert_svs_to_zarr(args.svs_dir)
        
        # 步骤2: 加载CODEX数据
        codex_df = generator.load_codex_data(args.codex_dir)
        
        # 步骤3: 处理细胞数据
        cell_df = generator.process_cell_data(codex_df, svs_to_uuid)
        
        # 步骤4: 创建元数据
        metadata_dict = generator.create_metadata_dict(cell_df, svs_to_uuid)
        
        # 步骤5: 创建数据集划分
        dataset_splits = generator.create_dataset_splits(metadata_dict)
        
        # 步骤6: 保存数据集
        generator.save_dataset(cell_df, metadata_dict, dataset_splits)
        
        # 步骤7: 验证数据集
        if generator.validate_dataset():
            print("\n🎉 数据集生成成功!")
            print(f"📁 输出目录: {args.output_dir}")
            print("💡 现在可以使用train.py开始训练了")
        else:
            print("\n❌ 数据集验证失败，请检查生成过程")
            
    except Exception as e:
        print(f"\n❌ 数据集生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

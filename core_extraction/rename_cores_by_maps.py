#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMA Core重命名脚本
根据TMA_Maps中的映射关系重命名core文件
作者: AI Assistant
日期: 2024
"""

import os
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/zheng/zheng/rosie_reproduce/core_extraction/rename_cores.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TMACoreRenamer:
    """TMA Core重命名器"""
    
    def __init__(self, maps_dir: str, cores_dir: str, backup: bool = True):
        """
        初始化重命名器
        
        Args:
            maps_dir: TMA_Maps目录路径
            cores_dir: TMA_TumorCenter_Cores目录路径
            backup: 是否创建备份
        """
        self.maps_dir = Path(maps_dir)
        self.cores_dir = Path(cores_dir)
        self.backup = backup
        self.mapping_cache = {}
        
        # 验证目录存在
        if not self.maps_dir.exists():
            raise FileNotFoundError(f"TMA_Maps目录不存在: {maps_dir}")
        if not self.cores_dir.exists():
            raise FileNotFoundError(f"TMA_TumorCenter_Cores目录不存在: {cores_dir}")
    
    def load_mapping_for_block(self, block_num: int) -> Dict[str, str]:
        """
        加载指定block的映射关系
        
        Args:
            block_num: block编号
            
        Returns:
            映射字典 {core_position: patient_id}
        """
        map_file = self.maps_dir / f"TMA_Map_block{block_num}.csv"
        
        if not map_file.exists():
            logger.warning(f"映射文件不存在: {map_file}")
            return {}
        
        mapping = {}
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    core_pos = row.get('core', '').strip()
                    patient_id = row.get('Case ID', '').strip()
                    
                    if core_pos and patient_id:
                        mapping[core_pos] = patient_id
                    elif core_pos and not patient_id:
                        logger.debug(f"Block {block_num}, Core {core_pos}: 无患者ID")
        
        except Exception as e:
            logger.error(f"读取映射文件失败 {map_file}: {e}")
            return {}
        
        logger.info(f"加载Block {block_num}映射: {len(mapping)}个有效映射")
        return mapping
    
    def parse_filename(self, filename: str) -> Optional[Tuple[str, int, int, int, str]]:
        """
        解析现有文件名
        
        Args:
            filename: 文件名
            
        Returns:
            (image_type, block_num, x, y, patient_id) 或 None
        """
        # 匹配格式: TumorCenter_HE_block1_x1_y1_patient296.png
        pattern = r'TumorCenter_(\w+)_block(\d+)_x(\d+)_y(\d+)_patient(\d+)\.png'
        match = re.match(pattern, filename)
        
        if match:
            image_type = match.group(1)
            block_num = int(match.group(2))
            x = int(match.group(3))
            y = int(match.group(4))
            patient_id = match.group(5)
            return image_type, block_num, x, y, patient_id
        
        return None
    
    def convert_xy_to_core_position(self, x: int, y: int) -> str:
        """
        将x,y坐标转换为core位置格式
        
        Args:
            x: x坐标 (1-6)
            y: y坐标 (1-12)
            
        Returns:
            core位置字符串 (如 "1-1", "2-3")
        """
        return f"{y}-{x}"
    
    def get_correct_patient_id(self, block_num: int, x: int, y: int) -> Optional[str]:
        """
        根据block和坐标获取正确的患者ID
        
        Args:
            block_num: block编号
            x: x坐标
            y: y坐标
            
        Returns:
            患者ID或None
        """
        # 加载映射（使用缓存）
        if block_num not in self.mapping_cache:
            self.mapping_cache[block_num] = self.load_mapping_for_block(block_num)
        
        core_pos = self.convert_xy_to_core_position(x, y)
        return self.mapping_cache[block_num].get(core_pos)
    
    def generate_new_filename(self, image_type: str, block_num: int, x: int, y: int, patient_id: str) -> str:
        """
        生成新的文件名
        
        Args:
            image_type: 图片类型
            block_num: block编号
            x: x坐标
            y: y坐标
            patient_id: 患者ID
            
        Returns:
            新文件名
        """
        return f"TumorCenter_{image_type}_block{block_num}_x{x}_y{y}_patient{patient_id}.png"
    
    def create_backup(self, file_path: Path) -> Path:
        """
        创建文件备份
        
        Args:
            file_path: 原文件路径
            
        Returns:
            备份文件路径
        """
        backup_path = file_path.with_suffix('.png.backup')
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def rename_file(self, old_path: Path, new_path: Path) -> bool:
        """
        重命名文件
        
        Args:
            old_path: 原文件路径
            new_path: 新文件路径
            
        Returns:
            是否成功
        """
        try:
            # 创建备份
            if self.backup:
                self.create_backup(old_path)
            
            # 重命名文件
            old_path.rename(new_path)
            logger.info(f"重命名成功: {old_path.name} -> {new_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"重命名失败 {old_path.name}: {e}")
            return False
    
    def process_directory(self, image_type: str) -> Dict[str, int]:
        """
        处理指定类型的目录
        
        Args:
            image_type: 图片类型 (HE, CD3, CD8, etc.)
            
        Returns:
            统计信息
        """
        type_dir = self.cores_dir / f"tma_tumorcenter_{image_type}"
        
        if not type_dir.exists():
            logger.warning(f"目录不存在: {type_dir}")
            return {"total": 0, "renamed": 0, "skipped": 0, "failed": 0}
        
        stats = {"total": 0, "renamed": 0, "skipped": 0, "failed": 0}
        
        logger.info(f"开始处理 {image_type} 类型文件...")
        
        # 获取所有PNG文件
        png_files = list(type_dir.glob("*.png"))
        stats["total"] = len(png_files)
        
        for png_file in png_files:
            # 解析文件名
            parsed = self.parse_filename(png_file.name)
            if not parsed:
                logger.warning(f"无法解析文件名: {png_file.name}")
                stats["skipped"] += 1
                continue
            
            image_type_parsed, block_num, x, y, current_patient_id = parsed
            
            # 获取正确的患者ID
            correct_patient_id = self.get_correct_patient_id(block_num, x, y)
            
            if not correct_patient_id:
                logger.warning(f"无法找到患者ID映射: Block {block_num}, x={x}, y={y}")
                stats["skipped"] += 1
                continue
            
            # 检查是否需要重命名
            if current_patient_id == correct_patient_id:
                logger.debug(f"患者ID已正确: {png_file.name}")
                stats["skipped"] += 1
                continue
            
            # 生成新文件名
            new_filename = self.generate_new_filename(image_type_parsed, block_num, x, y, correct_patient_id)
            new_path = type_dir / new_filename
            
            # 检查新文件是否已存在
            if new_path.exists():
                logger.warning(f"目标文件已存在: {new_filename}")
                stats["skipped"] += 1
                continue
            
            # 执行重命名
            if self.rename_file(png_file, new_path):
                stats["renamed"] += 1
            else:
                stats["failed"] += 1
        
        logger.info(f"{image_type} 处理完成: 总计{stats['total']}, 重命名{stats['renamed']}, 跳过{stats['skipped']}, 失败{stats['failed']}")
        return stats
    
    def process_all_types(self) -> Dict[str, Dict[str, int]]:
        """
        处理所有图片类型
        
        Returns:
            所有类型的统计信息
        """
        image_types = ["HE", "CD3", "CD8", "CD56", "CD68", "CD163", "MHC1", "PDL1"]
        all_stats = {}
        
        logger.info("开始批量重命名TMA Core文件...")
        
        for image_type in image_types:
            all_stats[image_type] = self.process_directory(image_type)
        
        # 输出总体统计
        total_files = sum(stats["total"] for stats in all_stats.values())
        total_renamed = sum(stats["renamed"] for stats in all_stats.values())
        total_skipped = sum(stats["skipped"] for stats in all_stats.values())
        total_failed = sum(stats["failed"] for stats in all_stats.values())
        
        logger.info("=" * 50)
        logger.info("批量重命名完成!")
        logger.info(f"总文件数: {total_files}")
        logger.info(f"成功重命名: {total_renamed}")
        logger.info(f"跳过: {total_skipped}")
        logger.info(f"失败: {total_failed}")
        logger.info(f"成功率: {total_renamed/(total_files-total_skipped)*100:.1f}%" if (total_files-total_skipped) > 0 else "0%")
        
        return all_stats

def main():
    """主函数"""
    # 设置路径
    maps_dir = "/home/zheng/zheng/mini2/hancock_data/TMA/TMA_Maps"
    cores_dir = "/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores"
    
    try:
        # 创建重命名器
        renamer = TMACoreRenamer(maps_dir, cores_dir, backup=True)
        
        # 执行批量重命名
        stats = renamer.process_all_types()
        
        # 输出详细统计
        print("\n详细统计:")
        for image_type, type_stats in stats.items():
            print(f"{image_type}: 总计{type_stats['total']}, 重命名{type_stats['renamed']}, 跳过{type_stats['skipped']}, 失败{type_stats['failed']}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

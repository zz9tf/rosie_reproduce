#!/usr/bin/env python3
"""
TMA Core Analyzer - 分析 TMA 核心完整性和生成报告

这个脚本用于：
1. 分析 TMA 核心的完整性
2. 比较不同标记的数据
3. 生成详细的统计报告
4. 检测缺失的核心

作者: AI Assistant
日期: 2025-01-26
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse
import json
from datetime import datetime


class TMACoreAnalyzer:
    """TMA 核心分析器"""
    
    def __init__(self, base_path: str = "/home/zheng/zheng/mini2/hancock_data/TMA"):
        """
        初始化 TMA 核心分析器
        
        Args:
            base_path: TMA 数据的基础路径
        """
        self.base_path = Path(base_path)
        self.original_cores_path = self.base_path / "TMA_TumorCenter_Cores"
        self.processed_cores_path = self.base_path / "TMA_TumorCenter_1024"
        
        # 定义标记类型
        self.markers = ["HE", "CD3", "CD8", "CD56", "CD68", "CD163", "MHC1", "PDL1"]
        
        # 存储分析结果
        self.analysis_results = {}
        
    def parse_original_cores(self, marker: str) -> Set[Tuple[int, int, int]]:
        """
        解析原始核心文件
        
        Returns:
            原始核心的 (block, row, col) 集合
        """
        cores = set()
        
        original_cores_path = self.original_cores_path / f"tma_tumorcenter_{marker}"
        print(original_cores_path)
        if not original_cores_path.exists():
            print(f"⚠️  原始核心目录不存在: {self.original_cores_path}")
            return cores
            
        for filename in os.listdir(original_cores_path):
            if not filename.endswith('.png'):
                continue
                
            match = re.match(f'^TumorCenter_{marker}_block(?P<block>\\d+)_x(?P<x>\\d+)_y(?P<y>\\d+)_patient(?P<pid>\\d+)\\.png$', filename)
            if match:
                block = int(match.group('block'))
                x = int(match.group('x'))
                y = int(match.group('y'))
                # 映射: row=y, col=x
                cores.add((block, y, x))
                
        return cores
    
    def parse_processed_cores(self, marker: str) -> Set[Tuple[int, int, int]]:
        """
        解析处理后的核心文件
        
        Args:
            marker: 标记类型 (HE, CD3, CD8, etc.)
            
        Returns:
            处理后核心的 (block, row, col) 集合
        """
        cores = set()
        marker_path = self.processed_cores_path / f"{marker}"
        print(marker_path)
        
        if not marker_path.exists():
            print(f"⚠️  标记目录不存在: {marker_path}")
            return cores
            
        for filename in os.listdir(marker_path):
            if not filename.endswith('.png'):
                continue
                
            match = re.match(f'^TumorCenter_{marker}_block(?P<block>\\d+)_(?P<row>\\d+)-(?P<col>\\d+)_circular\\.png$', filename)
            if match:
                block = int(match.group('block'))
                row = int(match.group('row'))
                col = int(match.group('col'))
                cores.add((block, row, col))
                
        return cores
    
    def analyze_marker_completeness(self, marker: str) -> Dict:
        """
        分析单个标记的完整性
        
        Args:
            marker: 标记类型
            
        Returns:
            分析结果字典
        """
        print(f"🔍 分析 {marker} 标记...")
        
        original_cores = self.parse_original_cores(marker)
        processed_cores = self.parse_processed_cores(marker)
        
        # 计算差异
        missing_cores = original_cores - processed_cores
        extra_cores = processed_cores - original_cores
        
        # 按 block 统计
        block_stats = {}
        for block in range(1, 24):  # block1 到 block23
            block_original = {core for core in original_cores if core[0] == block}
            block_processed = {core for core in processed_cores if core[0] == block}
            block_missing = block_original - block_processed
            block_extra = block_processed - block_original
            
            block_stats[block] = {
                'original_count': len(block_original),
                'processed_count': len(block_processed),
                'missing_count': len(block_missing),
                'extra_count': len(block_extra),
                'missing_cores': sorted(list(block_missing)),
                'extra_cores': sorted(list(block_extra))
            }
        
        result = {
            'marker': marker,
            'total_original': len(original_cores),
            'total_processed': len(processed_cores),
            'total_missing': len(missing_cores),
            'total_extra': len(extra_cores),
            'completeness_rate': len(processed_cores) / len(original_cores) * 100 if original_cores else 0,
            'missing_cores': sorted(list(missing_cores)),
            'extra_cores': sorted(list(extra_cores)),
            'block_stats': block_stats
        }
        
        return result
    
    def analyze_all_markers(self) -> Dict:
        """
        分析所有标记的完整性
        
        Returns:
            所有标记的分析结果
        """
        print("🚀 开始分析所有标记...")
        
        all_results = {}
        
        for marker in self.markers:
            try:
                result = self.analyze_marker_completeness(marker)
                all_results[marker] = result
                
                # 打印简要结果
                print(f"  {marker}: {result['total_processed']}/{result['total_original']} "
                      f"({result['completeness_rate']:.1f}%) - "
                      f"缺失: {result['total_missing']}, 额外: {result['total_extra']}")
                      
            except Exception as e:
                print(f"❌ 分析 {marker} 时出错: {e}")
                all_results[marker] = {'error': str(e)}
        
        self.analysis_results = all_results
        return all_results
    
    def generate_summary_report(self) -> str:
        """
        生成摘要报告
        
        Returns:
            报告字符串
        """
        if not self.analysis_results:
            return "❌ 没有分析结果，请先运行 analyze_all_markers()"
        
        report = []
        report.append("=" * 80)
        report.append("TMA 核心完整性分析报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据路径: {self.base_path}")
        report.append("")
        
        # 总体统计
        report.append("📊 总体统计:")
        report.append("-" * 40)
        
        total_original = 0
        total_processed = 0
        total_missing = 0
        
        for marker, result in self.analysis_results.items():
            if 'error' not in result:
                total_original += result['total_original']
                total_processed += result['total_processed']
                total_missing += result['total_missing']
        
        overall_completeness = total_processed / total_original * 100 if total_original > 0 else 0
        
        report.append(f"总原始核心数: {total_original}")
        report.append(f"总处理核心数: {total_processed}")
        report.append(f"总缺失核心数: {total_missing}")
        report.append(f"整体完整率: {overall_completeness:.1f}%")
        report.append("")
        
        # 各标记详细统计
        report.append("📋 各标记详细统计:")
        report.append("-" * 40)
        
        for marker, result in self.analysis_results.items():
            if 'error' in result:
                report.append(f"{marker}: ❌ 错误 - {result['error']}")
                continue
                
            report.append(f"{marker}:")
            report.append(f"  完整率: {result['completeness_rate']:.1f}%")
            report.append(f"  处理数: {result['total_processed']}/{result['total_original']}")
            report.append(f"  缺失数: {result['total_missing']}")
            report.append(f"  额外数: {result['total_extra']}")
            
            if result['total_missing'] > 0:
                report.append(f"  缺失样本: {result['missing_cores']}")
            report.append("")
        
        return "\n".join(report)
    
    def save_detailed_report(self, output_path: str = None):
        """
        保存详细报告到文件
        
        Args:
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = self.base_path / "tma_analysis_report.json"
        
        # 准备保存的数据
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'analysis_results': self.analysis_results
        }
        
        # 保存 JSON 报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细报告已保存到: {output_path}")
        
        # 保存文本摘要
        summary_path = str(output_path).replace('.json', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_summary_report())
        
        print(f"📄 摘要报告已保存到: {summary_path}")
    
    def find_missing_cores_by_block(self, marker: str = "HE") -> Dict[int, List[Tuple[int, int, int]]]:
        """
        按 block 查找缺失的核心
        
        Args:
            marker: 标记类型
            
        Returns:
            按 block 分组的缺失核心字典
        """
        if marker not in self.analysis_results:
            print(f"❌ 没有 {marker} 的分析结果")
            return {}
        
        result = self.analysis_results[marker]
        if 'error' in result:
            print(f"❌ {marker} 分析出错: {result['error']}")
            return {}
        
        missing_by_block = {}
        for core in result['missing_cores']:
            block = core[0]
            if block not in missing_by_block:
                missing_by_block[block] = []
            missing_by_block[block].append(core)
        
        return missing_by_block


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TMA 核心完整性分析器')
    parser.add_argument('--base-path', default='/home/zheng/zheng/mini2/hancock_data/TMA',
                       help='TMA 数据基础路径')
    parser.add_argument('--marker', choices=['HE', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'MHC1', 'PDL1', 'all'],
                       default='all', help='要分析的标记类型')
    parser.add_argument('--output', help='输出报告文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = TMACoreAnalyzer(args.base_path)
    
    if args.marker == 'all':
        # 分析所有标记
        results = analyzer.analyze_all_markers()
        
        # 生成报告
        print("\n" + analyzer.generate_summary_report())
        
        # 保存报告
        if args.output:
            analyzer.save_detailed_report(args.output)
        else:
            analyzer.save_detailed_report()
            
    else:
        # 分析单个标记
        result = analyzer.analyze_marker_completeness(args.marker)
        
        if 'error' not in result:
            print(f"\n{args.marker} 标记分析结果:")
            print(f"完整率: {result['completeness_rate']:.1f}%")
            print(f"处理数: {result['total_processed']}/{result['total_original']}")
            print(f"缺失数: {result['total_missing']}")
            print(f"额外数: {result['total_extra']}")
            
            if args.verbose and result['total_missing'] > 0:
                print(f"\n缺失的核心:")
                for core in result['missing_cores']:
                    print(f"  block{core[0]}_{core[1]}-{core[2]}")
        else:
            print(f"❌ 分析 {args.marker} 时出错: {result['error']}")


if __name__ == "__main__":
    main()

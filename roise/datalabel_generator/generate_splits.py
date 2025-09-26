#!/usr/bin/env python3
"""
独立的数据集分割脚本 - 直接生成训练/验证/测试集文件
"""

import argparse
import numpy as np
import pandas as pd
import os

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成训练/验证/测试集数据集文件")
    parser.add_argument("--data-file", required=True, help="数据parquet文件路径")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], 
                        help="训练/验证/测试集比例 [train val test]")
    parser.add_argument("--split-seed", type=int, default=42, help="数据分割随机种子")
    parser.add_argument("--max-samples", type=int, default=None, help="限制使用的最大样本数（默认使用全部样本）")
    parser.add_argument("--shuffle", action="store_true", default=True, help="是否打乱数据顺序（默认开启）")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle", help="不打乱数据顺序")
    parser.add_argument("--output-dir", type=str, default="./splits", help="输出目录")
    parser.add_argument("--output-format", choices=['parquet', 'csv'], default='parquet', help="输出文件格式")
    
    args = parser.parse_args()
    
    print("🚀 数据集分割生成器")
    print("=" * 50)
    print(f"📁 数据文件: {args.data_file}")
    print(f"📊 分割比例: {args.split_ratios}")
    print(f"🎲 随机种子: {args.split_seed}")
    print(f"📏 最大样本数: {args.max_samples if args.max_samples else '无限制'}")
    print(f"🔀 数据打乱: {'是' if args.shuffle else '否'}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📄 输出格式: {args.output_format}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成分割
    print("📊 生成数据集分割...")
    
    # 读取数据文件获取总样本数
    df = pd.read_parquet(args.data_file)
    original_total_samples = len(df)
    
    # 应用最大样本数限制
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(f"最大样本数必须大于0，当前为: {args.max_samples}")
        if args.max_samples > original_total_samples:
            print(f"⚠️  警告: 指定的最大样本数({args.max_samples})超过实际样本数({original_total_samples})，将使用全部样本")
            total_samples = original_total_samples
        else:
            total_samples = args.max_samples
    else:
        total_samples = original_total_samples
    
    print(f"📈 原始样本数: {original_total_samples}")
    print(f"📈 使用样本数: {total_samples}")
    print(f"📊 分割比例: 训练={args.split_ratios[0]:.1%}, 验证={args.split_ratios[1]:.1%}, 测试={args.split_ratios[2]:.1%}")
    
    # 验证分割比例
    if abs(sum(args.split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"分割比例之和必须为1.0，当前为: {sum(args.split_ratios)}")
    
    # 生成所有索引
    all_indices = np.arange(original_total_samples)
    
    # 设置随机种子确保可重现性
    np.random.seed(args.split_seed)
    
    # 根据shuffle参数决定是否打乱数据
    if args.shuffle:
        np.random.shuffle(all_indices)
        print("🔀 所有数据已打乱")
    else:
        print("📋 保持数据原始顺序")
    
    # 从打乱后的结果中取指定数量的样本
    if args.max_samples is not None and args.max_samples < original_total_samples:
        indices = all_indices[:args.max_samples]
        print(f"📏 从打乱后的数据中取前 {args.max_samples} 个样本")
    else:
        indices = all_indices
    
    # 计算分割点
    train_size = int(total_samples * args.split_ratios[0])
    val_size = int(total_samples * args.split_ratios[1])
    
    # 分割索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 根据索引分割数据
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    print(f"✅ 训练集: {len(train_df)} 个样本")
    print(f"✅ 验证集: {len(val_df)} 个样本")
    print(f"✅ 测试集: {len(test_df)} 个样本")
    
    # 保存数据集文件
    file_extension = f".{args.output_format}"
    
    train_file = os.path.join(args.output_dir, f"train{file_extension}")
    val_file = os.path.join(args.output_dir, f"val{file_extension}")
    test_file = os.path.join(args.output_dir, f"test{file_extension}")
    
    if args.output_format == 'parquet':
        train_df.to_parquet(train_file, index=False)
        val_df.to_parquet(val_file, index=False)
        test_df.to_parquet(test_file, index=False)
    else:  # csv
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
    
    print(f"💾 训练集已保存到: {train_file}")
    print(f"💾 验证集已保存到: {val_file}")
    print(f"💾 测试集已保存到: {test_file}")
    
    # 保存分割信息元数据
    metadata_file = os.path.join(args.output_dir, "split_metadata.txt")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("数据集分割信息\n")
        f.write("=" * 30 + "\n")
        f.write(f"原始数据文件: {args.data_file}\n")
        f.write(f"原始样本数: {original_total_samples}\n")
        f.write(f"使用样本数: {total_samples}\n")
        f.write(f"最大样本数限制: {args.max_samples if args.max_samples else '无限制'}\n")
        f.write(f"随机种子: {args.split_seed}\n")
        f.write(f"数据打乱: {'是' if args.shuffle else '否'}\n")
        f.write(f"分割比例: {args.split_ratios}\n")
        f.write(f"输出格式: {args.output_format}\n")
        f.write("\n分割结果:\n")
        f.write(f"训练集: {len(train_df)} 个样本 ({len(train_df)/total_samples:.1%})\n")
        f.write(f"验证集: {len(val_df)} 个样本 ({len(val_df)/total_samples:.1%})\n")
        f.write(f"测试集: {len(test_df)} 个样本 ({len(test_df)/total_samples:.1%})\n")
    
    print(f"📋 分割元数据已保存到: {metadata_file}")
    
    print("\n🎉 数据集分割完成！")
    print(f"📊 最终统计:")
    print(f"   - 训练集: {len(train_df)} 个样本 ({len(train_df)/total_samples:.1%})")
    print(f"   - 验证集: {len(val_df)} 个样本 ({len(val_df)/total_samples:.1%})")
    print(f"   - 测试集: {len(test_df)} 个样本 ({len(test_df)/total_samples:.1%})")
    print(f"   - 使用样本数: {total_samples}")
    if args.max_samples is not None:
        print(f"   - 原始样本数: {original_total_samples}")
        print(f"   - 样本数限制: {args.max_samples}")
    print(f"   - 随机种子: {args.split_seed}")
    print(f"   - 数据打乱: {'是' if args.shuffle else '否'}")
    print(f"   - 分割比例: {args.split_ratios}")
    print(f"   - 输出格式: {args.output_format}")
    print(f"   - 输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()

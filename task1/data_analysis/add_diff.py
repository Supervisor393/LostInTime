import os
import pandas as pd

def add_diff_column_to_csv(csv_file_path: str):
    """给单个CSV文件添加diff列并直接修改原文件"""
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 计算diff，pred_start为空时diff保持为NaN，结果保留两位小数
    df['diff'] = df.apply(lambda row: round(row['pred_start'] - row['gt_start'], 2) if pd.notnull(row['pred_start']) else None, axis=1)
    
    # 直接保存回原文件
    df.to_csv(csv_file_path, index=False)
    print(f"已更新文件: {csv_file_path}")

def process_csv_in_current_folder(folder_path: str):
    """处理当前文件夹中的所有CSV文件，不处理子文件夹"""
    # 获取当前文件夹中所有的 .csv 文件（不包括子文件夹中的文件）
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))]
    
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        
        # 处理每个 CSV 文件
        add_diff_column_to_csv(csv_file_path)

if __name__ == "__main__":
    # 输入文件夹路径，包含所有的 CSV 文件
    input_folder = '/data/user/jzt/crd/audioLLM/biaozhuncha/qwen/2'  # 你的输入文件夹路径

    # 处理当前文件夹中的所有 CSV 文件
    process_csv_in_current_folder(input_folder)

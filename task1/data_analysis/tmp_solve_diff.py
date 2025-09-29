import os
import pandas as pd

def process_csv_and_output(csv_file_path: str):
    """处理每个CSV文件并输出结果到终端"""
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 总条数（不包含表头）
    total_count = len(df)
    
    # 计算 diff 为空的行数
    diff_is_nan = df['diff'].isna().sum()
    
    # 有效数据条数 = 总条数 - diff为空的条数
    valid_count = total_count - diff_is_nan
    
    # 计算 diff 小于等于 0 的均值和数据条数
    diff_leq_0 = df[df['diff'] <= 0]['diff']
    mean_diff_leq_0 = diff_leq_0.mean() if len(diff_leq_0) > 0 else None
    count_diff_leq_0 = len(diff_leq_0)
    
    # 计算 diff 大于等于 0 的均值和数据条数
    diff_geq_0 = df[df['diff'] >= 0]['diff']
    mean_diff_geq_0 = diff_geq_0.mean() if len(diff_geq_0) > 0 else None
    count_diff_geq_0 = len(diff_geq_0)
    
    # 计算所有有效数据的 diff 的绝对值的均值
    valid_diff = df['diff'].dropna().abs()  # 只考虑有效数据并取绝对值
    mean_abs_diff = valid_diff.mean() if len(valid_diff) > 0 else None
    
    # 输出统计信息
    print(f"文件: {csv_file_path}")
    print(f"总数据条数: {total_count}")
    print(f"diff 为空的条数: {diff_is_nan}")
    print(f"有效数据条数: {valid_count}")
    print(f"diff <= 0 的均值: {mean_diff_leq_0:.2f} (条数: {count_diff_leq_0})" if mean_diff_leq_0 is not None else "diff <= 0 数据不存在")
    print(f"diff >= 0 的均值: {mean_diff_geq_0:.2f} (条数: {count_diff_geq_0})" if mean_diff_geq_0 is not None else "diff >= 0 数据不存在")
    print(f"所有有效数据的 diff 绝对值均值: {mean_abs_diff:.2f}" if mean_abs_diff is not None else "有效数据不存在")
    print("-" * 40)

def process_csv_in_current_folder(folder_path: str):
    """处理当前文件夹中的所有CSV文件，不处理子文件夹"""
    # 获取当前文件夹中所有的 .csv 文件（不包括子文件夹中的文件）
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))]
    
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        
        # 处理每个 CSV 文件并输出结果到终端
        process_csv_and_output(csv_file_path)

if __name__ == "__main__":
    # 输入文件夹路径，包含所有的 CSV 文件
    input_folder = '/data/user/jzt/crd/audioLLM/biaozhuncha/SED/2'  # 你的输入文件夹路径

    # 处理当前文件夹中的所有 CSV 文件
    process_csv_in_current_folder(input_folder)

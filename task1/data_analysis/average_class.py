import os
import pandas as pd

def process_csv(csv_file_path: str):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 统计每个文件和窗口内不同class的数量
    df_grouped = df.groupby(['file', 'win_start', 'win_end'])['class'].nunique().reset_index(name='class_count')

    # 计算所有窗口内 class 数量的平均值
    average_class_count_per_window = df_grouped['class_count'].mean()

    # 计算每个窗口内的最大 class 数量
    max_class_count_per_window = df_grouped['class_count'].max()

    # 计算最大 class 数量窗口占比
    max_class_count_percentage = (df_grouped['class_count'] == max_class_count_per_window).mean() * 100

    # 输出结果
    print(f"结果 - {csv_file_path}:")
    print(f"  平均每个窗口的类数量: {average_class_count_per_window:.2f}")
    print(f"  最大每个窗口的类数量: {max_class_count_per_window}")
    print(f"  最大类数量窗口的占比: {max_class_count_percentage:.2f}%\n")


def process_multiple_csv(csv_paths: list):
    for csv_file_path in csv_paths:
        if os.path.exists(csv_file_path):
            process_csv(csv_file_path)
        else:
            print(f"[警告] 文件不存在: {csv_file_path}")


if __name__ == "__main__":
    # 批量处理文件列表
    csv_files = [
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win05_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win10_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win20_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win30_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win40_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win50_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win60_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win90_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_sony.win120_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win05_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win10_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win20_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win30_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win40_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win50_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win60_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win90_earliest.csv',
        '/data/user/jzt/crd/audioLLM/train_data/4_limit_Qwen/Qwen_tau.win120_earliest.csv',
        # 添加其他文件路径
    ]

    process_multiple_csv(csv_files)

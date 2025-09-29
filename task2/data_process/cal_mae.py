import pandas as pd
import numpy as np

# === 修改为你的CSV路径 ===
CSV_PATH = "/data/user/jzt/crd/audioLLM/task2/qwen_preds.csv"

# 读取
df = pd.read_csv(CSV_PATH)

# 基本清洗与数值化（防止有的字段读成字符串）
for col in ["gt_onset_sec", "gt_offset_sec", "pred_onset_sec"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 计算时长与绝对误差
df["duration"] = df["gt_offset_sec"] - df["gt_onset_sec"]
df["abs_err_onset"] = (df["pred_onset_sec"] - df["gt_onset_sec"]).abs()

# 只保留我们需要的两段时长 & 丢掉关键字段缺失的行
mask_valid = df["gt_onset_sec"].notna() & df["gt_offset_sec"].notna() & df["pred_onset_sec"].notna()
mask_bucket_short = df["duration"] < 3
mask_bucket_mid = (df["duration"] >= 7) & (df["duration"] <= 10)

sub = df[mask_valid & (mask_bucket_short | mask_bucket_mid)].copy()

# 打上分桶标签
def bucket_label(d):
    if d < 3:
        return "<3s"
    elif 7 <= d <= 10:
        return "7-10s"
    else:
        return np.nan

sub["duration_bucket"] = sub["duration"].apply(bucket_label)

# 按类别与分桶聚合
group_cols = ["class_name", "duration_bucket"]
agg = (
    sub
    .groupby(group_cols, dropna=False)
    .agg(
        mae_onset=("abs_err_onset", "mean"),
        count=("abs_err_onset", "size"),
        mean_duration=("duration", "mean"),
    )
    .reset_index()
    .sort_values(group_cols)
)

# 为了可读性，保留到3位小数
for c in ["mae_onset", "mean_duration"]:
    agg[c] = agg[c].round(3)

print("=== 每个类别在两个时长区间内的统计（MAE / 条数 / 平均时长） ===")
print(agg)

# 也导出一个透视表：行=类别，列=分桶，值=MAE
pivot_mae = agg.pivot(index="class_name", columns="duration_bucket", values="mae_onset")
pivot_cnt = agg.pivot(index="class_name", columns="duration_bucket", values="count")
pivot_dur = agg.pivot(index="class_name", columns="duration_bucket", values="mean_duration")

print("\n=== MAE 透视表（行=类别，列=分桶） ===")
print(pivot_mae)

print("\n=== Count 透视表（行=类别，列=分桶） ===")
print(pivot_cnt)

print("\n=== 平均时长 透视表（行=类别，列=分桶） ===")
print(pivot_dur)

# 导出结果（可选）
agg.to_csv("per_class_bucket_stats.csv", index=False)
pivot_mae.to_csv("per_class_bucket_mae_pivot.csv")
pivot_cnt.to_csv("per_class_bucket_count_pivot.csv")
pivot_dur.to_csv("per_class_bucket_mean_duration_pivot.csv")

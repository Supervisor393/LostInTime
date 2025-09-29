#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并 Sony & TAU 两份事件清单，按窗口 {10,20,30,40,50,60}s 收集可用事件（左右都满足 4 段），
并把相同窗口的结果放在一起（每个窗口一个 CSV）。

输入：
  --sony-csv  /data/user/jzt/crd/audioLLM/train_events/sony.csv
  --tau-csv   /data/user/jzt/crd/audioLLM/train_events/tau.csv
  --sony-root /data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony
  --tau-root  /data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-tau

输出：
  out_dir/candidates_win10.csv
  out_dir/candidates_win20.csv
  ...
  列：source,file,abs_audio_path,onset,offset,class,left_extend_to,right_extend_to,
      event_duration,left_space,right_space,window_sec,segment_sec

依赖：
  pip install pandas pydub
  # 系统需安装 ffmpeg（pydub 依赖）
"""

import os
import argparse
import pandas as pd
import numpy as np
from pydub import AudioSegment
from typing import Dict, Tuple

EPS = 1e-6
WINDOWS = [10, 20, 30, 40, 50, 60]

# ---------- 音频工具 ----------

def read_audio_duration_seconds(path: str) -> float:
    """读取音频总时长（秒）。"""
    seg = AudioSegment.from_file(path)
    return len(seg) / 1000.0

# ---------- 数据加载 ----------

def load_and_tag(csv_path: str, source: str) -> pd.DataFrame:
    """读取单个清单并打上来源标签（sony/tau）。"""
    df = pd.read_csv(csv_path)
    need = {"file", "onset", "offset", "class"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列：{missing}")
    df = df.dropna(subset=["file","onset","offset","class"]).copy()
    df["onset"]  = pd.to_numeric(df["onset"], errors="coerce")
    df["offset"] = pd.to_numeric(df["offset"], errors="coerce")
    df["class"]  = pd.to_numeric(df["class"],  errors="coerce").astype("Int64")
    df = df.dropna(subset=["onset","offset","class"])
    df["source"] = source
    return df

def cache_durations(df: pd.DataFrame, roots: Dict[str, str]) -> Dict[Tuple[str,str], float]:
    """
    为 (source, file) 组合缓存时长（秒）。
    roots: {"sony": sony_root, "tau": tau_root}
    """
    durations: Dict[Tuple[str,str], float] = {}
    pairs = df[["source","file"]].drop_duplicates().itertuples(index=False, name=None)
    for source, f in pairs:
        root = roots[source]
        path = os.path.join(root, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[{source}] 音频不存在：{path}")
        durations[(source, f)] = read_audio_duration_seconds(path)
    return durations

# ---------- 同类边界与左右空间 ----------

def compute_same_class_bounds(df: pd.DataFrame,
                              durations: Dict[Tuple[str,str], float],
                              roots: Dict[str,str]) -> pd.DataFrame:
    """
    同一 (source, file, class) 组内，按 onset 排序，计算 prev/next，
    并得到 left/right 可延伸边界与左右空间。
    """
    df = df.copy()
    df["prev_off_full"] = np.nan
    df["next_on_full"]  = np.nan

    for (_, _, _), g in df.groupby(["source","file","class"], sort=False):
        g_sorted = g.sort_values(["onset", "offset"]).reset_index()
        on  = g_sorted["onset"].to_numpy()
        off = g_sorted["offset"].to_numpy()
        prev_off = np.concatenate(([np.nan], off[:-1]))
        next_on  = np.concatenate((on[1:], [np.nan]))
        df.loc[g_sorted["index"], "prev_off_full"] = prev_off
        df.loc[g_sorted["index"], "next_on_full"]  = next_on

    # 左边界：不触碰上一个同类；右边界：不触碰下一个同类且不超过文件末尾
    left_extend = np.where(
        np.isnan(df["prev_off_full"]), 0.0,
        np.maximum(0.0, df["prev_off_full"].to_numpy() + EPS)
    )

    file_durs = df.apply(lambda r: durations[(r["source"], r["file"])], axis=1).to_numpy()
    next_on_arr = df["next_on_full"].to_numpy()
    right_extend = np.where(
        np.isnan(next_on_arr),
        file_durs,
        np.minimum(next_on_arr - EPS, file_durs)
    )

    # 绝对路径（方便下游）
    abs_paths = df.apply(lambda r: os.path.join(roots[r["source"]], r["file"]), axis=1)

    df["left_extend_to"]  = np.round(left_extend, 6)
    df["right_extend_to"] = np.round(right_extend, 6)
    df["event_duration"]  = np.round(df["offset"] - df["onset"], 6)
    df["left_space"]      = np.round(df["onset"] - df["left_extend_to"], 6)
    df["right_space"]     = np.round(df["right_extend_to"] - df["offset"], 6)
    df["abs_audio_path"]  = abs_paths
    return df

# ---------- 窗口筛选（左右都满足） ----------

def collect_for_window(df_bounds: pd.DataFrame, W: float) -> pd.DataFrame:
    """
    条件（同时满足，保证能放到5段中的任意一段）：
      - duration < seg
      - right_space >= 4*seg
      - left_space  >= 4*seg
    """
    seg = W / 5.0
    cond_dur       = df_bounds["event_duration"] < seg
    cond_leftmost  = df_bounds["right_space"] >= (4.0 * seg)  # 可放最左段（右侧留4段）
    cond_rightmost = df_bounds["left_space"]  >= (4.0 * seg)  # 可放最右段（左侧留4段）
    cond_all = cond_dur & cond_leftmost & cond_rightmost

    cols = [
        "source","file","abs_audio_path","onset","offset","class",
        "left_extend_to","right_extend_to",
        "event_duration","left_space","right_space"
    ]
    out = df_bounds.loc[cond_all, cols].copy()
    out["window_sec"]  = float(W)
    out["segment_sec"] = float(seg)
    out = out.sort_values(["class","source","file","onset"]).reset_index(drop=True)
    return out

# ---------- 主流程 ----------

def process(sony_csv: str, tau_csv: str, sony_root: str, tau_root: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 读取两份清单并打标签
    df_sony = load_and_tag(sony_csv, "sony")
    df_tau  = load_and_tag(tau_csv,  "tau")
    df_all  = pd.concat([df_sony, df_tau], ignore_index=True)

    roots = {"sony": sony_root, "tau": tau_root}

    # 缓存 (source,file) 时长
    durations = cache_durations(df_all, roots)

    # 计算同类边界/左右空间/绝对路径
    df_bounds = compute_same_class_bounds(df_all, durations, roots)

    # 各窗口合并输出
    for W in WINDOWS:
        candidates = collect_for_window(df_bounds, float(W))
        out_path = os.path.join(out_dir, f"candidates_win{W}.csv")
        candidates.to_csv(out_path, index=False)
        print(f"[OK] {out_path}  (共 {len(candidates)} 条)")

def main():
    parser = argparse.ArgumentParser(description="合并 Sony & TAU：按 5 段窗口收集可用事件（左右都满足 4 段）")
    parser.add_argument("--sony-csv",  required=True, help="Sony 事件 CSV（file,onset,offset,class）")
    parser.add_argument("--tau-csv",   required=True, help="TAU  事件 CSV（file,onset,offset,class）")
    parser.add_argument("--sony-root", required=True, help="Sony 原始音频根目录")
    parser.add_argument("--tau-root",  required=True, help="TAU  原始音频根目录")
    parser.add_argument("--out-dir",   default="window_candidates_combined", help="输出目录（每个窗口一个 CSV）")
    args = parser.parse_args()

    process(args.sony_csv, args.tau_csv, args.sony_root, args.tau_root, args.out_dir)

if __name__ == "__main__":
    main()

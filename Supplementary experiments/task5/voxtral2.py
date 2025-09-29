#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Closed-set Per-Class Binary Detection
- 切窗 -> 计算窗口内真实出现的类别 -> 逐类发问（含你指定的一些“未出现类”） -> 模型回答 true/false
- 模型输出要求：严格 JSON 对象 {"present": true} 或 {"present": false}

真实标注 CSV 列：
  file,onset,offset,class  （file=音频basename; onset/offset秒; class为整型ID）
"""

import os
import re
import json
import math
import soundfile as sf
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Set, Tuple, Optional

# ====================== 配置区（按需修改） ======================

# 推理服务
VLLM_ENDPOINT    = "http://127.0.0.1:8011/v1/chat/completions"
MODEL_NAME       = "voxtral-mini-3b"
TEMPERATURE      = 0.2
TOP_P            = 0.95
MAX_TOKENS       = 1024
REQUEST_TIMEOUT  = 300  # seconds

# 窗口长度（秒）
WIN_LENS = [5, 10, 20, 30, 40, 50, 60]

# 类别映射（按你的实验定义）
CLASS_ID_TO_NAME = {
    0: "Female speech, woman speaking",
    1: "Male speech, man speaking",
    2: "Clapping",
    3: "Telephone",
    4: "Laughter",
    5: "Domestic sounds",
    6: "Walk, footsteps",
    7: "Door, open or close",
    8: "Music",
    9: "Musical instrument",
    10: "Water tap, faucet",
    11: "Bell",
    12: "Knock",
}

# 评测范围（用于计算“真实出现”的类别，只在这些类里做判断）
TARGET_CLASS_SET = sorted([1,4])
# 也可使用全部 13 类：
# TARGET_CLASS_SET = sorted(CLASS_ID_TO_NAME.keys())

# 你想“额外插入”的未出现类别（程序会在每个窗口里，从下列集合中选取确实未出现的类别来提问）
NEGATIVE_PROBE_CANDIDATES = [0,2]  # 可自定义
MAX_NEGATIVES_PER_WINDOW  = 2                # 每个窗口最多插入多少个未出现类

# 数据路径：音频 & 真实标注 CSV（可混合 TAU/SONY；按 basename 匹配）
FILES = [
    "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix002.wav",
]
EVENTS_CSVS = [
    "/data/user/jzt/crd/audioLLM/train_events/tau.csv",
    "/data/user/jzt/crd/audioLLM/train_events/sony.csv",
]

# 输出模板
DETAIL_CSV_TPL    = "{basename}.win{win:02d}.Bin.detail.csv"
SUMMARY_CSV_TPL   = "{basename}.win{win:02d}.Bin.summary.csv"
REPLIES_JSONL_TPL = "{basename}.win{win:02d}.Bin.replies.jsonl"

# ===============================================================

# ---------------- 音频与切窗 ----------------
def split_fixed_win(wav_path: str, win_len: float):
    wav, sr = sf.read(wav_path)
    if hasattr(wav, "ndim") and wav.ndim > 1:
        wav = wav.mean(axis=1)
    n = len(wav)
    samples = int(round(win_len * sr))
    if samples <= 0:
        raise ValueError(f"Invalid win_len={win_len}")
    wins = []
    i = 0
    while i + samples <= n:
        i0, i1 = i, i + samples
        t0, t1 = i0 / sr, i1 / sr
        wins.append((i0, i1, t0, t1))
        i += samples
    return wav, sr, wins

def array_to_audiochunk(arr, sr):
    import tempfile
    from mistral_common.protocol.instruct.messages import AudioChunk
    from mistral_common.audio import Audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, arr, sr, format="wav")
        audio = Audio.from_file(tmp_path, strict=False)
        return AudioChunk.from_audio(audio)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

# ---------------- 真实标签：窗口存在性集合 ----------------
def load_events_csvs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            need = {"file","onset","offset","class"}
            missing = need - set(df.columns)
            if missing:
                raise ValueError(f"Events CSV missing columns: {missing} in {p}")
            dfs.append(df[["file","onset","offset","class"]])
    if not dfs:
        raise FileNotFoundError("No valid events CSV found.")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["file"] = all_df["file"].astype(str)
    all_df["onset"] = all_df["onset"].astype(float)
    all_df["offset"] = all_df["offset"].astype(float)
    all_df["class"] = all_df["class"].astype(int)
    return all_df

def present_classes_in_window(events_df: pd.DataFrame, basename: str, t0: float, t1: float, target_set: Set[int]) -> Set[int]:
    sub = events_df[events_df["file"] == basename]
    if sub.empty:
        return set()
    mask = (sub["offset"] > t0) & (sub["onset"] < t1)
    cats = set(int(c) for c in sub.loc[mask, "class"].tolist())
    return set(c for c in cats if c in target_set)

# ---------------- 提示词 & 模型调用（逐类二分类） ----------------
def build_prompt_binary(cid: int, win_len: float) -> str:
    """要求模型严格返回 {"present": true/false}。"""
    name = CLASS_ID_TO_NAME.get(cid, "Unknown")
    return (
        f"This is a {win_len:.0f}-second audio segment.\n"
        f"Question: Does the following class occur at least once in THIS segment (any time)?\n"
        f"{cid}: {name}\n"
        'Answer with a strict JSON object: {"present": true} or {"present": false}. '
        "No extra text. No markdown. No code fences."
    )

def call_model_binary(audio_arr, sr, cid: int, win_len: float) -> str:
    from mistral_common.protocol.instruct.messages import TextChunk, UserMessage
    audio_chunk = array_to_audiochunk(audio_arr, sr)
    text_chunk  = TextChunk(text=build_prompt_binary(cid, win_len))
    user_msg    = UserMessage(content=[audio_chunk, text_chunk]).to_openai()
    payload = {
        "model": MODEL_NAME,
        "messages": [user_msg],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }
    r = requests.post(VLLM_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

# ---------------- JSON 提取/解析（对象 {"present": bool}） ----------------
def extract_top_level_json_value(text: str) -> Optional[str]:
    """优先从 ```json ...``` 或 ```...``` 中提取顶层 JSON；先找对象，再数组。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        candidate = m.group(1)
        j = _find_first_top_level(candidate, prefer_object=True)
        if j is not None:
            return j
    return _find_first_top_level(text, prefer_object=True)

def _find_first_top_level(s: str, prefer_object: bool = False) -> Optional[str]:
    pairs = [('{', '}'), ('[', ']')] if prefer_object else [('[', ']'), ('{', '}')]
    for start_char, end_char in pairs:
        start = s.find(start_char)
        while start != -1:
            i, depth, in_str, esc = start, 0, False, False
            while i < len(s):
                ch = s[i]
                if in_str:
                    if esc: esc = False
                    elif ch == '\\': esc = True
                    elif ch == '"': in_str = False
                else:
                    if ch == '"': in_str = True
                    elif ch == start_char: depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0: return s[start:i+1]
                i += 1
            start = s.find(start_char, start + 1)
    return None

def parse_binary_present(raw_reply: str) -> Optional[int]:
    """
    解析 {"present": true/false} -> 返回 1/0；解析失败返回 None。
    对于极少数模型返回的字符串 "true"/"false"，做一次宽松转换。
    """
    j = extract_top_level_json_value(raw_reply)
    if not j:
        return None
    try:
        data = json.loads(j)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "present" not in data:
        return None

    val = data["present"]
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "yes", "y", "1"):  return 1
        if v in ("false", "no", "n", "0"):  return 0
    if isinstance(val, (int, float)):
        return 1 if float(val) >= 0.5 else 0
    return None

# ---------------- 指标（按窗口长度与类别聚合） ----------------
def summarize_binary(detail_df: pd.DataFrame) -> pd.DataFrame:
    """
    输出两部分：1) overall（按 win_len 聚合）  2) per_class（按 win_len×class 聚合）
    指标：Accuracy / Precision / Recall / F1
    """
    rows = []

    def prf1(tp, fp, fn, tn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        acc  = (tp + tn) / (tp+fp+fn+tn) if (tp+fp+fn+tn) > 0 else 0.0
        return acc, prec, rec, f1

    # overall per win_len
    for win_len, g in detail_df.groupby("win_len"):
        tp = int(((g["gt_present"]==1) & (g["pred_present"]==1)).sum())
        fp = int(((g["gt_present"]==0) & (g["pred_present"]==1)).sum())
        fn = int(((g["gt_present"]==1) & (g["pred_present"]==0)).sum())
        tn = int(((g["gt_present"]==0) & (g["pred_present"]==0)).sum())
        acc, p, r, f1 = prf1(tp, fp, fn, tn)
        rows.append({
            "scope": "overall", "win_len": win_len, "class": "",
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": p, "recall": r, "f1": f1
        })

    # per-class per win_len
    for (win_len, cid), g in detail_df.groupby(["win_len","class"]):
        tp = int(((g["gt_present"]==1) & (g["pred_present"]==1)).sum())
        fp = int(((g["gt_present"]==0) & (g["pred_present"]==1)).sum())
        fn = int(((g["gt_present"]==1) & (g["pred_present"]==0)).sum())
        tn = int(((g["gt_present"]==0) & (g["pred_present"]==0)).sum())
        acc, p, r, f1 = prf1(tp, fp, fn, tn)
        rows.append({
            "scope": "per_class", "win_len": win_len, "class": int(cid),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": p, "recall": r, "f1": f1
        })

    return pd.DataFrame(rows)

# ---------------- 主流程 ----------------
def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def run_binary_for_file(wav_path: str, events_df: pd.DataFrame, target_ids: List[int]):
    base = os.path.basename(wav_path)
    base_noext = os.path.splitext(base)[0]
    target_set = set(target_ids)

    for win_len in WIN_LENS:
        wav, sr, wins = split_fixed_win(wav_path, win_len)
        detail_csv    = DETAIL_CSV_TPL.format(basename=base_noext, win=win_len)
        summary_csv   = SUMMARY_CSV_TPL.format(basename=base_noext, win=win_len)
        replies_jsonl = REPLIES_JSONL_TPL.format(basename=base_noext, win=win_len)
        ensure_dir_for(replies_jsonl)

        rows = []
        with open(replies_jsonl, "a", encoding="utf-8") as fout:
            for (i0, i1, t0, t1) in wins:
                clip = wav[i0:i1]
                # 真实出现（只在 target_set 中判定）
                gt_present_set = present_classes_in_window(events_df, base, t0, t1, target_set)

                # 本窗口要询问的类：
                # 1) 所有真实出现类（正样本）
                probe = set(gt_present_set)

                # 2) 从 NEGATIVE_PROBE_CANDIDATES 中挑选确实未出现的类（负样本）
                neg_pool = [c for c in NEGATIVE_PROBE_CANDIDATES if c not in gt_present_set]
                if MAX_NEGATIVES_PER_WINDOW is not None and MAX_NEGATIVES_PER_WINDOW >= 0:
                    neg_pool = neg_pool[:MAX_NEGATIVES_PER_WINDOW]
                probe.update(neg_pool)

                for cid in sorted(probe):
                    raw = call_model_binary(clip, sr, cid, win_len)
                    fout.write(json.dumps({
                        "file": base, "win_start": t0, "win_end": t1,
                        "class": int(cid), "reply": raw
                    }, ensure_ascii=False) + "\n")

                    pred_flag = parse_binary_present(raw)  # None/0/1
                    if pred_flag is None:
                        # 无法解析，按最保守处理：记为错误（可改策略）
                        pred_flag = -1  # 标记未解析
                    gt_flag = 1 if cid in gt_present_set else 0

                    correct = 1 if (pred_flag in (0,1) and pred_flag == gt_flag) else 0

                    rows.append({
                        "file": base,
                        "win_start": t0, "win_end": t1, "win_len": win_len,
                        "class": int(cid),
                        "class_name": CLASS_ID_TO_NAME.get(int(cid), "Unknown"),
                        "gt_present": gt_flag,
                        "pred_present": pred_flag if pred_flag in (0,1) else "",
                        "correct": correct,
                        "raw_reply": raw
                    })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(detail_csv, index=False)

            # 过滤掉解析失败的样本再汇总指标（pred_present 为空的样本不参与）
            df_eval = df[df["pred_present"].isin([0,1])].copy()
            if not df_eval.empty:
                summary = summarize_binary(df_eval)
                summary.to_csv(summary_csv, index=False)
                print(f"[detail] {len(df)} rows -> {detail_csv}")
                print(f"[summary] {len(summary)} rows -> {summary_csv}")
            else:
                print(f"[WARN] no evaluable predictions (all parse-failed) for {base} @ {win_len}s -> wrote detail only")
        else:
            print(f"[WARN] no windows for {base} @ {win_len}s")

# ---------------- 入口 ----------------
if __name__ == "__main__":
    # 载入真实事件
    events_df = load_events_csvs(EVENTS_CSVS)

    # 跑每个文件
    for wav_path in FILES:
        if not os.path.exists(wav_path):
            print(f"[ERR] not found: {wav_path}")
            continue
        run_binary_for_file(wav_path, events_df, TARGET_CLASS_SET)

    print("\nDone. Inspect *.Bin.detail.csv / *.Bin.summary.csv and replies jsonl.\n")

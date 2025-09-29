#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A-Variant: Hallucination Probe w/o "empty-array" rule
- 提示词只要求返回开始时间，强制 JSON，对不存在的类不提供“空数组”出口
- 若模型返回合法 {"start": seconds} 且 0<=seconds<win_len -> 记为胡说
- 否则（无法解析/不在范围/非 JSON 等） -> 记为未胡说

依赖：
  pip install soundfile pandas requests mistral_common
"""

import os
import re
import json
import math
import soundfile as sf
import pandas as pd
import requests
from typing import List, Optional

# ====================== 配置区（请按需修改） ======================

# 推理服务
VLLM_ENDPOINT    = "http://127.0.0.1:8011/v1/chat/completions"
MODEL_NAME       = "voxtral-mini-3b"
TEMPERATURE      = 0.2
TOP_P            = 0.95
MAX_TOKENS       = 4096
REQUEST_TIMEOUT  = 300  # seconds

# 窗口长度（秒）
WIN_LENS_ABS = [5, 10, 20, 30, 40, 50, 60]

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

# ===== 你可自定义“理应不存在”的类别清单（示例：除 1/4 外全部）=====
# ABSENT_CANDIDATES = [cid for cid in CLASS_ID_TO_NAME if cid not in {1, 4}]
# 也可手写：
ABSENT_CANDIDATES = [0,2,6,7,8,11,12]

# 要评测的音频（保证整条仅含 {1,4}）
FILES = [
    "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix001.wav",
    "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix002.wav",
]

# 输出模板
DETAIL_CSV_TPL    = "{basename}.win{win:02d}.A.detail.csv"
SUMMARY_CSV_TPL   = "{basename}.win{win:02d}.A.summary.csv"
REPLIES_JSONL_TPL = "{basename}.win{win:02d}.A.replies.jsonl"

# ===============================================================


# ---------------- Robust JSON 顶层数组/对象提取 ----------------
def extract_top_level_json(text: str) -> Optional[str]:
    """
    尝试从文本中提取一个顶层 JSON 值（对象或数组皆可）。
    优先抓取 ```json ... ``` 或 ``` ... ``` 里的内容，否则全串搜索。
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        candidate = m.group(1)
        j = _find_top_level_json(candidate)
        if j is not None:
            return j
    return _find_top_level_json(text)

def _find_top_level_json(s: str) -> Optional[str]:
    # 找对象或数组起点
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = s.find(start_char)
        while start != -1:
            i, depth, in_str, esc = start, 0, False, False
            while i < len(s):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == start_char:
                        depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0:
                            return s[start:i+1]
                i += 1
            start = s.find(start_char, start + 1)
    return None

# ---------------- 音频切窗 ----------------
def split_fixed_win(wav_path: str, win_len: float):
    """
    返回 (wav, sr, windows)
    windows: list[(i0, i1, t0, t1)]，丢弃不足一个窗口的尾巴
    """
    wav, sr = sf.read(wav_path)
    if hasattr(wav, "ndim") and wav.ndim > 1:
        import numpy as np
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


# ---------------- 音频数组 -> AudioChunk ----------------
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

# ---------------- A 版提示词（仅要求开始时间） ----------------
def build_prompt_A(target_class: int, win_len: float) -> str:
    """
    只让模型返回开始时间；不提供“空数组”规则；强制严格 JSON 对象。
    """
    meaning = CLASS_ID_TO_NAME.get(target_class, "Unknown")
    return (
        f"You will be given a {win_len:.0f}-second audio segment.\n"
        f"Detect the start time (in seconds) of the following target class within THIS segment:\n"
        f"{target_class}: {meaning}\n\n"
        "Return format (STRICT): Return ONLY a single JSON object with exactly one key:\n"
        "{\"start\": <number>}\n"
        f"Constraints: 0 <= start < {win_len:.1f}. The start time is RELATIVE to THIS segment (0 = segment start).\n"
        "No extra keys. No extra text. No markdown. No code fences."
    )

def call_model_A(audio_arr, sr, target_class: int, win_len: float) -> str:
    from mistral_common.protocol.instruct.messages import TextChunk, UserMessage
    audio_chunk = array_to_audiochunk(audio_arr, sr)
    text_chunk  = TextChunk(text=build_prompt_A(target_class, win_len))
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


# ---------------- 解析：若得到合法 start 则视为胡说 ----------------
def parse_A_reply_for_start(raw_reply: str, win_len: float) -> Optional[float]:
    """
    返回：
      - None  -> 未胡说（没解析到合法 start）
      - float -> 胡说（解析到 start 且在 [0, win_len)）
    """
    j = extract_top_level_json(raw_reply)
    if not j:
        return None
    try:
        data = json.loads(j)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "start" not in data:
        return None
    try:
        st = float(data["start"])
    except Exception:
        return None
    if math.isfinite(st) and 0.0 <= st < float(win_len) + 1e-6:
        return float(st)
    return None


# ---------------- 主流程 ----------------
def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def run_A_for_file(wav_path: str):
    base = os.path.basename(wav_path)
    base_noext = os.path.splitext(base)[0]

    for win_len in WIN_LENS_ABS:
        wav, sr, wins = split_fixed_win(wav_path, win_len)
        win_tag = f"win{int(win_len):02d}"

        detail_csv    = DETAIL_CSV_TPL.format(basename=base_noext, win=win_len)
        summary_csv   = SUMMARY_CSV_TPL.format(basename=base_noext, win=win_len)
        replies_jsonl = REPLIES_JSONL_TPL.format(basename=base_noext, win=win_len)

        ensure_dir_for(replies_jsonl)

        detail_rows = []
        with open(replies_jsonl, "a", encoding="utf-8") as fout:
            for (i0, i1, t0, t1) in wins:
                clip = wav[i0:i1]
                for tgt in ABSENT_CANDIDATES:
                    raw = call_model_A(clip, sr, tgt, win_len)

                    # 保存原始回复做审计
                    fout.write(json.dumps({
                        "file": base,
                        "win_start": t0,
                        "win_end": t1,
                        "target_class": int(tgt),
                        "reply": raw
                    }, ensure_ascii=False) + "\n")

                    rel = parse_A_reply_for_start(raw, win_len)
                    hallucinated = 1 if rel is not None else 0
                    abs_start = t0 + rel if rel is not None else ""

                    detail_rows.append({
                        "file": base,
                        "win_start": t0,
                        "win_end": t1,
                        "win_len": win_len,
                        "target_class": int(tgt),
                        "target_name": CLASS_ID_TO_NAME.get(int(tgt), "Unknown"),
                        "model_rel_start": rel if rel is not None else "",
                        "model_abs_start": abs_start,
                        "hallucinated": hallucinated,
                        "raw_reply": raw
                    })

        if detail_rows:
            df = pd.DataFrame(detail_rows)
            df.to_csv(detail_csv, index=False)
            grp = df.groupby(["win_len", "target_class", "target_name"], as_index=False).agg(
                n=("hallucinated", "size"),
                n_hallucinated=("hallucinated", "sum"),
            )
            grp["hallucination_rate"] = grp["n_hallucinated"] / grp["n"]
            grp.sort_values(["win_len", "target_class"], inplace=True)
            grp.to_csv(summary_csv, index=False)

            print(f"[detail] {len(df)} rows -> {detail_csv}")
            print(f"[summary] {len(grp)} groups -> {summary_csv}")
        else:
            print(f"[WARN] no windows produced for {base} @ {win_len}s")


# ---------------- 入口 ----------------
if __name__ == "__main__":
    for wav_path in FILES:
        if not os.path.exists(wav_path):
            print(f"[ERR] not found: {wav_path}")
            continue
        run_A_for_file(wav_path)

    print("\nDone. Inspect *.A.detail.csv / *.A.summary.csv / *.A.replies.jsonl\n")

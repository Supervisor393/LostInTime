#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Qwen2-Audio 对多个 manifest（列：file,class,onset,offset,relative_onset_event_audio,relative_offset_event_audio）
的 20 秒音频进行推理：
- 仅预测目标事件“开始时间”（秒，0~20）
- 每个音频只询问一次
- 严格 JSON 输出：{"start time": <number>}（保留 1 位小数）

依赖：
  pip install -U pandas soundfile scipy torch transformers numpy
"""

import os
import re
import json
import glob
import time
import argparse
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import soundfile as sf
from fractions import Fraction
from scipy.signal import resample_poly
import torch
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor

# ====== QWEN2-AUDIO 配置 ======
QWEN_LOCAL_DIR = "/data/user/jzt/.cache/modelscope/hub/models/Qwen/Qwen2-Audio-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 512
QWEN_DO_SAMPLE = True
TEMPERATURE = 0.2
TOP_P = 0.95

_QWEN_TARGET_SR = 16000
TARGET_TOTAL_SEC = 20.0

# 非贪婪 JSON 块，避免过度匹配
JSON_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)

# ====== CLASS ID → NAME ======
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

# ---------------- Prompt ----------------
def build_instruction(cls_id: int) -> str:
    cls_name = CLASS_ID_TO_NAME.get(int(cls_id), "Unknown")
    return f"""
You will receive a single {int(TARGET_TOTAL_SEC)}-second audio clip that contains an audio event.
And you need to give the start time of it. The audio event is ("{cls_name}").

Return ONLY the following strict JSON (no extra text, no code fences):
{{"start time": <number>}}

Constraints:
- 0 <= start time <= {int(TARGET_TOTAL_SEC)}
- Use seconds with up to 1 decimal places (e.g., 3.1)
The "start time" must be RELATIVE to THIS clip (0 means the clip start).
""".strip()

# ---------------- JSON 解析（增强版，兼容单引号） ----------------
def parse_json_from_text(text: str) -> Optional[dict]:
    """
    尝试从文本中解析出 dict：
    1) 直接当 JSON 解析（双引号）
    2) 抓第一个 {...} 块再试 JSON
    3) 用 ast.literal_eval 支持单引号风格的 Python 字面量
    4) 兜底：正则直接抽取 'start time' 的数值（单双引号、字符串数字均可）
    """
    if not text:
        return None

    # 1) 整体 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 提取第一个 {...} 块
    m = JSON_PATTERN.search(text)
    chunk = m.group(0) if m else None
    if chunk:
        try:
            return json.loads(chunk)
        except Exception:
            pass

    # 3) 单引号风格：ast.literal_eval
    if chunk:
        try:
            import ast
            val = ast.literal_eval(chunk)
            if isinstance(val, dict):
                return val
        except Exception:
            pass

    # 4) 兜底：正则抽取 "start time": number
    num_pat = re.compile(
        r"[\"']?\s*start\s*_?\s*time\s*[\"']?\s*:\s*[\"']?\s*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE
    )
    m2 = num_pat.search(chunk or text)
    if m2:
        try:
            return {"start time": float(m2.group(1))}
        except Exception:
            return None

    return None

def _to_valid_time(x) -> Optional[float]:
    try:
        f = float(x)
        if 0.0 <= f <= TARGET_TOTAL_SEC:
            return round(f, 1)
        return None
    except Exception:
        return None

def extract_start_time(data: Any) -> Optional[float]:
    if not isinstance(data, dict):
        return None
    if "start time" in data:
        return _to_valid_time(data["start time"])
    for k in ["start_time", "start", "onset", "pred_onset_sec"]:
        if k in data:
            got = _to_valid_time(data[k])
            if got is not None:
                return got
    for v in data.values():
        cand = _to_valid_time(v)
        if cand is not None:
            return cand
    return None

# ---------------- 模型加载 ----------------
_qwen_processor = Qwen2AudioProcessor.from_pretrained(
    QWEN_LOCAL_DIR,
    sampling_rate=_QWEN_TARGET_SR
)
_qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    QWEN_LOCAL_DIR, device_map={"": 1}, torch_dtype=torch.float16
).eval()

# ---------------- 音频读取（无 librosa / resampy） ----------------
def load_audio_array(audio_path: str):
    wav, sr = sf.read(audio_path, always_2d=False)
    if hasattr(wav, "ndim") and wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32, copy=False)
    if not np.isfinite(wav).all():
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    if sr != _QWEN_TARGET_SR and len(wav) > 0:
        frac = Fraction(_QWEN_TARGET_SR, sr).limit_denominator(1000)
        wav = resample_poly(wav, frac.numerator, frac.denominator).astype(np.float32, copy=False)
        sr = _QWEN_TARGET_SR
    np.clip(wav, -1.0, 1.0, out=wav)
    return wav, sr

# ---------------- 单次调用 ----------------
def call_qwen_onset_once(audio_path: str, cls_id: int) -> Tuple[Optional[float], str]:
    instruction = build_instruction(cls_id)
    wav, sr = load_audio_array(audio_path)

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": None},
            {"type": "text", "text": instruction},
        ]}
    ]
    text = _qwen_processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = _qwen_processor(text=text, audio=wav, return_tensors="pt", padding=True)
    inputs = {k: (v.to(_qwen_model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = _qwen_model.generate(
            **inputs,
            do_sample=QWEN_DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
        )
    gen_ids = gen_ids[:, inputs["input_ids"].size(1):]
    resp = _qwen_processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    data = parse_json_from_text(resp) or {}
    return extract_start_time(data), resp

# ---------------- 主流程 ----------------
def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio onset-only; single query per audio.")
    parser.add_argument("--manifests", nargs="+", required=True,
                        help="一个或多个 manifest 路径（支持通配），列为 file,class,onset,offset,relative_onset_event_audio,relative_offset_event_audio")
    parser.add_argument("--out", required=True, help="输出 CSV 路径")
    parser.add_argument("--audio-root", default="", help="若 manifest 的 file 为相对路径，指定音频根目录进行拼接")

    parser.add_argument("--col-audio", default="file")
    parser.add_argument("--col-class", default="class")
    parser.add_argument("--col-gt-onset", default="relative_onset_event_audio")
    parser.add_argument("--col-gt-offset", default="relative_offset_event_audio")

    args = parser.parse_args()

    all_paths: List[str] = []
    for pat in args.manifests:
        expanded = glob.glob(pat)
        all_paths.extend(expanded if expanded else [pat])
    if not all_paths:
        raise ValueError("没有找到任何 manifest，请检查 --manifests。")

    dfs = []
    need_cols = {args.col_audio, args.col_class, args.col_gt_onset, args.col_gt_offset}
    for p in sorted(set(all_paths)):
        dfp = pd.read_csv(p)
        miss = [c for c in need_cols if c not in dfp.columns]
        if miss:
            raise ValueError(f"清单缺少列：{miss} | 文件：{p}")
        dfs.append(dfp)
    df = pd.concat(dfs, ignore_index=True)

    results = []
    t0 = time.time()

    for i, row in df.iterrows():
        raw_path = str(row[args.col_audio])
        cls_id = int(row[args.col_class])
        gt_onset = float(row[args.col_gt_onset])
        gt_offset = float(row[args.col_gt_offset])

        audio_path = raw_path
        if not os.path.isabs(audio_path) and not os.path.exists(audio_path) and args.audio_root:
            audio_path = os.path.join(args.audio_root, raw_path)

        if not os.path.exists(audio_path):
            print(f"[SKIP {i}] 文件不存在：{audio_path}")
            results.append({
                "audio_path": audio_path,
                "class": cls_id,
                "class_name": CLASS_ID_TO_NAME.get(cls_id, "Unknown"),
                "gt_onset_sec": round(gt_onset, 1),
                "gt_offset_sec": round(gt_offset, 1),
                "pred_onset_sec": None,
                "raw_model_output": "[file_not_found]"
            })
            continue

        print(f"[{i+1}/{len(df)}] 推理: {audio_path} (class={cls_id})")
        pred, raw = call_qwen_onset_once(audio_path, cls_id)
        
        results.append({
            "audio_path": audio_path,
            "class": cls_id,
            "class_name": CLASS_ID_TO_NAME.get(cls_id, "Unknown"),
            "gt_onset_sec": round(gt_onset, 1),
            "gt_offset_sec": round(gt_offset, 1),
            "pred_onset_sec": None if pred is None else round(pred, 1),
            "raw_model_output": raw,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"[DONE] 写出结果：{args.out}，共 {len(out_df)} 条，用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

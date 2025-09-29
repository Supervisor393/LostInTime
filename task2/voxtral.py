#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对多个 manifest（统一列：file,class,onset,offset,relative_onset_event_audio,relative_offset_event_audio）
上的 20 秒音频，调用 vLLM(OpenAI 兼容)上的 Voxtral。
用简洁提示词只预测目标事件“开始时间”（秒，0~20），
对每个音频询问 N 次（默认 3 次），取平均值写入 CSV。

依赖：
  pip install pandas openai mistral-common huggingface_hub
"""

import os
import re
import json
import time
import glob
import argparse
import pandas as pd
from typing import Optional, Tuple, Any, List

from openai import OpenAI
from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage
from mistral_common.audio import Audio

# ---------- 类别到名称（可按需扩展/修改） ----------
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

JSON_PATTERN = re.compile(r'\{.*\}', re.DOTALL)
TARGET_TOTAL_SEC = 20.0

# -------------------- Prompt --------------------

def file_to_chunk(path: str) -> AudioChunk:
    audio = Audio.from_file(path, strict=False)
    return AudioChunk.from_audio(audio)

def build_user_message(audio_path: str, cls_id: int) -> dict:
    """
    极简 + 严格 JSON 的提示词：只要 "start time"，1 位小数。
    """
    audio_chunk = file_to_chunk(audio_path)
    cls_name = CLASS_ID_TO_NAME.get(int(cls_id), "Unknown")

    instruction = f"""
You will receive a single 20-second audio clip that contains an audio event.
And you need to give the start time of it. The audio event is ("{cls_name}").

Return ONLY the following strict JSON (no extra text, no code fences):
{{"start time": <number>}}

Constraints:
- 0 <= start time <= 20
- Use seconds with up to 1 decimal places (e.g., 3.1)
""".strip()

    text_chunk = TextChunk(text=instruction)
    user_msg = UserMessage(content=[audio_chunk, text_chunk]).to_openai()
    return user_msg

# -------------------- Parsing helpers --------------------

def parse_json_from_text(text: str) -> Optional[dict]:
    # 直接尝试
    try:
        return json.loads(text)
    except Exception:
        pass
    # 容错：提取首个 JSON 块
    m = JSON_PATTERN.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def extract_start_time(data: Any) -> Optional[float]:
    """
    从返回的 dict 中拿到 "start time"（兼容一些常见误差键名）。
    """
    if not isinstance(data, dict):
        return None

    # 优先严格键名
    if "start time" in data:
        return _to_valid_time(data["start time"])

    # 容错键名
    for k in ["start_time", "start", "onset", "pred_onset_sec"]:
        if k in data:
            return _to_valid_time(data[k])

    # 再宽松一些：找首个数值
    for v in data.values():
        cand = _to_valid_time(v)
        if cand is not None:
            return cand

    return None

def _to_valid_time(x) -> Optional[float]:
    try:
        f = float(x)
        if 0.0 <= f <= TARGET_TOTAL_SEC:
            # 最终统一到 1 位小数
            return round(f, 1)
        return None
    except Exception:
        return None

# -------------------- Inference (single & multiple) --------------------

def ask_voxtral_onset_once(
    client: OpenAI, model: str, audio_path: str, cls_id: int,
    max_retries: int = 3, sleep_s: float = 0.8
) -> Tuple[Optional[float], str]:
    """
    调用 Voxtral 一次，只解析 {"start time": <number>}，带简单重试。
    返回 (start_time or None, raw_text)
    """
    user_msg = build_user_message(audio_path, cls_id)
    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[user_msg],
                temperature=0.2,
                top_p=0.95,
                max_tokens=1024,
            )
            content = resp.choices[0].message.content
            last_raw = content
            data = parse_json_from_text(content) or {}
            start_time = extract_start_time(data)
            if start_time is None and attempt < max_retries:
                time.sleep(sleep_s)
                continue
            return start_time, content
        except Exception as e:
            last_raw = f"[error:{e}]"
            if attempt == max_retries:
                break
            time.sleep(sleep_s * attempt)
    return None, last_raw

def ask_voxtral_onset_avg(
    client: OpenAI, model: str, audio_path: str, cls_id: int,
    n_times: int = 3, max_retries_each: int = 3, sleep_each: float = 0.8
) -> Tuple[Optional[float], List[Optional[float]], List[str]]:
    """
    对同一音频询问 n 次，返回：
      (平均值 or None, [每次预测], [每次原始输出])
    仅对成功解析的结果求平均；若全部失败则返回 None。
    """
    preds: List[Optional[float]] = []
    raws: List[str] = []
    for _ in range(max(1, n_times)):
        p, raw = ask_voxtral_onset_once(
            client=client, model=model, audio_path=audio_path, cls_id=cls_id,
            max_retries=max_retries_each, sleep_s=sleep_each
        )
        preds.append(p)
        raws.append(raw)

    valid = [x for x in preds if x is not None]
    if not valid:
        return None, preds, raws

    mean_val = round(sum(valid) / len(valid), 1)
    # 保险裁剪到 [0, 20]
    mean_val = min(max(mean_val, 0.0), TARGET_TOTAL_SEC)
    return mean_val, preds, raws

# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Voxtral onset-only; multi-manifest & multi-queries average.")
    # 多 manifest：可用 glob；也可传入多个
    parser.add_argument("--manifests", nargs="+", required=True,
                        help="一个或多个 manifest 路径，支持通配（例如 data/**/output_manifest.csv）")
    parser.add_argument("--out", required=True, help="输出结果 CSV 路径")

    # OpenAI 兼容配置
    parser.add_argument("--api-base", default="http://127.0.0.1:8011/v1", help="OpenAI 兼容 API base")
    parser.add_argument("--api-key", default="EMPTY", help="OpenAI API key（vLLM 用占位即可）")
    parser.add_argument("--model", default="voxtral-mini-3b", help="模型名（served-model-name）")

    # 列名映射 —— 按你的 manifest 默认
    parser.add_argument("--col-audio", default="file", help="音频路径列名（默认 file）")
    parser.add_argument("--col-class", default="class", help="类别列名（默认 class）")
    parser.add_argument("--col-gt-onset", default="relative_onset_event_audio", help="GT 相对起始时间列名（默认 relative_onset_event_audio）")
    parser.add_argument("--col-gt-offset", default="relative_offset_event_audio", help="GT 相对结束时间列名（默认 relative_offset_event_audio）")

    # 询问次数与重试
    parser.add_argument("--num-queries", type=int, default=3, help="同一音频询问次数并取平均（默认 3）")
    parser.add_argument("--max-retries-each", type=int, default=3, help="单次调用的重试次数（默认 3）")
    parser.add_argument("--sleep-each", type=float, default=0.8, help="单次调用失败的重试间隔（秒，默认 0.8）")

    # 路径修复：给相对路径加根目录（可选）
    parser.add_argument("--audio-root", default="", help="若 manifest 中是相对路径，这里指定音频根目录；为空则不处理")

    args = parser.parse_args()

    # 展开/汇总多个 manifest
    all_paths: List[str] = []
    for pat in args.manifests:
        expanded = glob.glob(pat)
        if not expanded:
            # 若通配不到，按原样也加入，让后续读文件报错得更明显
            all_paths.append(pat)
        else:
            all_paths.extend(expanded)

    if not all_paths:
        raise ValueError("没有找到任何 manifest 文件，请检查 --manifests 传入的路径/通配。")

    dfs = []
    for p in sorted(set(all_paths)):
        try:
            dfp = pd.read_csv(p)
            dfs.append(dfp)
        except Exception as e:
            raise RuntimeError(f"读取 manifest 失败：{p} | {e}")

    df = pd.concat(dfs, ignore_index=True)

    # 校验列
    need_cols = {args.col_audio, args.col_class, args.col_gt_onset, args.col_gt_offset}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"清单缺少列：{missing}\n当前列：{list(df.columns)}")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    results = []
    t0 = time.time()
    total = len(df)

    for i, row in df.iterrows():
        raw_path = str(row[args.col_audio])
        cls_id = int(row[args.col_class])

        # GT 使用“相对 20 秒音频”的开始/结束时间
        gt_onset = float(row[args.col_gt_onset])
        gt_offset = float(row[args.col_gt_offset])

        # 处理可能的相对路径：优先用原路径，若不存在且 audio_root 提供，则尝试拼接
        audio_path = raw_path
        if not os.path.isabs(audio_path) and not os.path.exists(audio_path) and args.audio_root:
            audio_path = os.path.join(args.audio_root, raw_path)

        # 最终存在性检查
        if not os.path.exists(audio_path):
            print(f"[SKIP {i}] 文件不存在：{audio_path}")
            results.append({
                "audio_path": audio_path,
                "class": cls_id,
                "class_name": CLASS_ID_TO_NAME.get(cls_id, "Unknown"),
                "gt_onset_sec": round(gt_onset, 1),
                "gt_offset_sec": round(gt_offset, 1),
                "pred_onset_sec": None,  # 平均值
                "pred_onset_each": "[]", # 每次预测数组（JSON 字符串）
                "raw_model_outputs": "[file_not_found]"  # 每次原始输出数组（JSON 字符串）
            })
            continue

        print(f"[{i+1}/{total}] 推理: {audio_path} (class={cls_id})")
        mean_pred, preds_each, raws_each = ask_voxtral_onset_avg(
            client=client, model=args.model, audio_path=audio_path, cls_id=cls_id,
            n_times=args.num_queries, max_retries_each=args.max_retries_each, sleep_each=args.sleep_each
        )

        results.append({
            "audio_path": audio_path,
            "class": cls_id,
            "class_name": CLASS_ID_TO_NAME.get(cls_id, "Unknown"),
            "gt_onset_sec": round(gt_onset, 1),
            "gt_offset_sec": round(gt_offset, 1),
            "pred_onset_sec": None if mean_pred is None else round(mean_pred, 1),  # 平均值（1 位小数）
            "pred_onset_each": json.dumps([None if x is None else round(x, 1) for x in preds_each], ensure_ascii=False),
            "raw_model_outputs": json.dumps(raws_each, ensure_ascii=False)
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"[DONE] 写出结果：{args.out}，共 {len(out_df)} 条，用时 {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

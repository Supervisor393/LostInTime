import os
import random
import json
import re
import soundfile as sf
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Tuple
from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage
from mistral_common.audio import Audio

# ========== CONFIG ==========

VLLM_ENDPOINT = "http://127.0.0.1:8011/v1/chat/completions"

WIN_LENS = [5,30,60,90,120]

TAU_EVENTS_CSV  = "/data/user/jzt/crd/audioLLM/train_events/tau.csv"
SONY_EVENTS_CSV = "/data/user/jzt/crd/audioLLM/train_events/sony.csv"

TEMPERATURE = 0.2
TOP_P = 0.95
MAX_TOKENS = 4096

# ====== CLASS ID → NAME MAPPING ======
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

# ---------------- Robust JSON -> list[{"class":int, "start":float}] ----------------

def extract_json_array(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        candidate = m.group(1)
        arr = _find_top_level_array(candidate)
        if arr is not None:
            return arr
    arr = _find_top_level_array(text)
    return arr

def _find_top_level_array(s: str) -> Optional[str]:
    start = s.find('[')
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
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
            i += 1
        start = s.find('[', start + 1)
    return None

def parse_class_start_list(raw_reply: str, win_len: float) -> List[Dict[str, Any]]:
    """Parse [{"class":ID,"start":seconds}, ...] (start is RELATIVE to the window)."""
    arr_text = extract_json_array(raw_reply)
    if not arr_text:
        return []
    try:
        data = json.loads(arr_text)
    except Exception:
        return []
    out = []
    if isinstance(data, list):
        for e in data:
            if not isinstance(e, dict):
                continue
            c = e.get("class")
            st = e.get("start")
            try:
                c_int = int(c)
                st_f = float(st)
            except Exception:
                continue
            if c_int < 0 or c_int > 12:
                continue
            if not (0.0 <= st_f < win_len + 1e-6):
                continue
            out.append({"class": c_int, "start": st_f})
    return out

# ---------------- audio <-> AudioChunk (via temp file path) ----------------

def array_to_audiochunk(arr, sr) -> AudioChunk:
    import tempfile
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

# ---------------- prompt & model call ----------------

def build_prompt(present_classes: List[int], win_len: float) -> str:
    # 生成 “ID: 含义” 列表字符串
    cls_list_str = "[" + ", ".join(
        f"{cid}: {CLASS_ID_TO_NAME.get(cid, 'Unknown')}"
        for cid in sorted(set(present_classes))
    ) + "]"
    # 提示词：强调相对时间 & 禁止多余文本/代码块
    prompt = (
        f"You will be given a {win_len:.0f}-second audio segment.\n"
        f"The following target class IDs are present in THIS segment (ID: meaning): {cls_list_str}.\n"
        "Task: For EACH listed class ID, report the EARLIEST start time within THIS segment.\n"
        "Return ONLY a strict JSON array with EXACTLY one object per listed class, sorted by class id.\n"
        "{\"class\": ID, \"start\": seconds} for each object.\n"
        f"Constraints: 0 <= start < {win_len:.1f}. 'start' must be RELATIVE to THIS segment (0 means the segment start).\n"
        "Do NOT include any class not in the list. No extra text. No markdown. No code fences."
    )
    return prompt

def extract_classes_from_response(response: str) -> List[int]:
    """
    提取模型回复中的类ID，假设类ID是数字，并且在响应中以一定的格式出现。
    通过正则表达式提取所有的数字。
    """
    # 正则表达式提取所有数字
    class_ids = re.findall(r'\b\d+\b', response)  # 匹配所有的数字

    # 将提取到的数字转换为整数列表
    class_ids = [int(cid) for cid in class_ids]

    # 去除不在合法范围内的类别ID
    class_ids = [cid for cid in class_ids if 0 <= cid <= 12]  # 只保留合法的类别ID

    return class_ids

def call_model_identify_classes(audio_arr, sr, win_len: float) -> str:
    """第一次提问：让模型识别音频窗口中的事件类"""
    prompt = f"Please identify the types of audio events present in this {win_len}-second audio segment.\n"
    prompt += "Return a list of class IDs that correspond to the events detected in the segment.\n"
    prompt += "The class IDs should be from the following list:\n"
    prompt += "[0: Female speech, woman speaking, 1: Male speech, man speaking, 2: Clapping, 3: Telephone, 4: Laughter, 5: Domestic sounds, 6: Walk, footsteps, 7: Door, open or close, 8: Music, 9: Musical instrument, 10: Water tap, faucet, 11: Bell, 12: Knock]\n"
    prompt += "Return only the list of class IDs in a strict JSON array. Do not include any extra text."

    audio_chunk = array_to_audiochunk(audio_arr, sr)
    text_chunk = TextChunk(text=prompt)
    user_msg = UserMessage(content=[audio_chunk, text_chunk]).to_openai()

    payload = {
        "model": "voxtral-mini-3b",
        "messages": [user_msg],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }
    r = requests.post(VLLM_ENDPOINT, json=payload, timeout=300)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

def call_model_get_event_times(audio_arr, sr, present_classes: List[int], win_len: float, num_retries: int = 1) -> str:
    """第二次提问：获取真实事件的最早开始时间，运行多次求平均值"""
    all_responses = []

    for _ in range(num_retries):
        prompt = build_prompt(present_classes, win_len)

        audio_chunk = array_to_audiochunk(audio_arr, sr)
        text_chunk = TextChunk(text=prompt)
        user_msg = UserMessage(content=[audio_chunk, text_chunk]).to_openai()

        payload = {
            "model": "voxtral-mini-3b",
            "messages": [user_msg],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        }

        r = requests.post(VLLM_ENDPOINT, json=payload, timeout=300)
        if not r.ok:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

        raw_response = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        all_responses.append(raw_response)

    # 现在我们有了 `num_retries` 次响应，将其平均处理
    # 将所有响应合并为一个最终的响应，平均每个类别的开始时间
    return average_responses(all_responses, present_classes, win_len)

def average_responses(responses: List[str], present_classes: List[int], win_len: float) -> str:
    """
    对所有的响应进行处理，求每个类别的平均开始时间
    """
    avg_results = {}

    # 收集每次响应中的类别ID及其相对开始时间
    for response in responses:
        pred_list = parse_class_start_list(response, win_len)  # 解析出每个类别的开始时间
        pred_rel_map = {int(d["class"]): float(d["start"]) for d in pred_list}

        for cid in present_classes:
            if cid in pred_rel_map:
                if cid not in avg_results:
                    avg_results[cid] = []
                avg_results[cid].append(pred_rel_map[cid])

    # 对每个类别计算平均值
    final_avg_map = {}
    for cid, times in avg_results.items():
        final_avg_map[cid] = sum(times) / len(times)  # 计算平均值

    # 构建最终的 JSON 字符串，返回给模型
    avg_response = [{"class": cid, "start": final_avg_map[cid]} for cid in sorted(final_avg_map)]
    return json.dumps(avg_response)

# ---------------- windowing and GT computation ----------------

def split_fixed_win(wav_path: str, win_len: float):
    """
    Return (wav, sr, windows)
    windows = list of (i0, i1, t0, t1) with exact ts each; last tail (<win_len) is dropped.
    """
    wav, sr = sf.read(wav_path)
    # 若是多通道，转单通道（均值）
    if hasattr(wav, "ndim") and wav.ndim > 1:
        import numpy as np
        wav = np.mean(wav, axis=1)
    n = len(wav)
    samples = int(win_len * sr)
    wins = []
    i = 0
    while i + samples <= n:
        i0, i1 = i, i + samples
        t0, t1 = i0 / sr, i1 / sr
        wins.append((i0, i1, t0, t1))
        i += samples
    return wav, sr, wins

def load_events_csv(path: str) -> pd.DataFrame:
    """
    Expect columns: file,onset,offset,class
    'file' should match basename of wav file.
    """
    df = pd.read_csv(path)
    need = {"file", "onset", "offset", "class"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {missing} in {path}")
    return df

def earliest_class_abs_time_in_window(df_events: pd.DataFrame, basename: str, t0: float, t1: float) -> Dict[int, float]:
    """
    对窗口 [t0, t1)：获取每个类别的最早出现时间（绝对时间）
    """
    df = df_events[df_events["file"] == basename]
    if df.empty:
        return {}

    mask = (df["offset"] > t0) & (df["onset"] < t1)
    sub = df.loc[mask, ["class", "onset", "offset"]]
    if sub.empty:
        return {}

    out: Dict[int, float] = {}
    for cls_id, grp in sub.groupby("class"):
        try:
            cid = int(cls_id)
        except Exception:
            continue
        earliest_abs = min(max(float(row.onset), t0) for _, row in grp.iterrows())
        out[cid] = float(earliest_abs)
    return out

def append_rows(csv_path: str, rows: List[Dict[str, Any]]):
    """
    追加写：file, win_start, win_end, class, gt_start, pred_start
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    cols = ["file", "win_start", "win_end", "class", "gt_start", "pred_start", "is_recognized"]
    df = pd.DataFrame(rows)[cols]
    exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode=("a" if exists else "w"), header=not exists, index=False)
    print(f"[append] {len(df)} rows -> {csv_path}")

def process_file_windows(wav_path: str, events_df: pd.DataFrame, out_csv_path: str, replies_jsonl_path: str, win_len: float):
    """
    对每个窗口进行处理，分两次提问：
    - 第一次提问：识别音频窗口中出现了哪些事件类。
    - 第二次提问：基于真实标签中的事件，询问每个事件的开始时间。
    """
    wav, sr, wins = split_fixed_win(wav_path, win_len)
    base = os.path.basename(wav_path)

    os.makedirs(os.path.dirname(replies_jsonl_path) or ".", exist_ok=True)
    append_buffer: List[Dict[str, Any]] = []

    with open(replies_jsonl_path, "a", encoding="utf-8") as fout:
        for (i0, i1, t0, t1) in wins:
            gt_abs_map = earliest_class_abs_time_in_window(events_df, base, t0, t1)
            if not gt_abs_map:
                continue

            clip = wav[i0:i1]
            present_classes = sorted(gt_abs_map.keys())
            if len(present_classes) > 4:
                present_classes = random.sample(present_classes, 4)

            # 第一次提问：让模型识别音频窗口中的事件类
            raw_classes_response = call_model_identify_classes(clip, sr, win_len)
            detected_classes = extract_classes_from_response(raw_classes_response)
            print(detected_classes)
            # 第二次提问：根据真实标签中的事件类别来提问
            raw_time_response = call_model_get_event_times(clip, sr, present_classes, win_len)

            # 保存原始回复（按窗口）
            fout.write(json.dumps({
                "file": base,
                "win_start": t0,
                "win_end": t1,
                "classes": present_classes,
                "reply": raw_time_response
            }, ensure_ascii=False) + "\n")

            # 解析模型相对开始时间（带 win_len 约束）
            pred_list = parse_class_start_list(raw_time_response, win_len)  # [{"class":ID,"start":rel}, ...]
            pred_rel_map = {int(d["class"]): float(d["start"]) for d in pred_list}

            # 计算时间偏差，分别记录识别和未识别的事件
            for cid in present_classes:
                gt_abs = float(gt_abs_map[cid])  # 绝对时间
                if cid in detected_classes:  # 判断是否模型识别了该事件
                    if cid in pred_rel_map:
                        pred_abs = t0 + pred_rel_map[cid]  # 模型预测的绝对时间
                        append_buffer.append({
                            "file": base,
                            "win_start": t0,
                            "win_end": t1,
                            "class": int(cid),
                            "gt_start": gt_abs,     # 绝对时间
                            "pred_start": pred_abs,  # 绝对时间（= 窗口起点 + 相对时间）
                            "is_recognized": True   # 标记为模型识别的事件
                        })
                    else:
                        append_buffer.append({
                            "file": base,
                            "win_start": t0,
                            "win_end": t1,
                            "class": int(cid),
                            "gt_start": gt_abs,     # 绝对时间
                            "pred_start": None,  # 未识别的事件，记录为空
                            "is_recognized": True   # 标记为模型识别的事件
                        })
                else:  # 模型没有识别出该事件
                    # 如果模型给出了预测的开始时间（即在 pred_rel_map 中有该类别），记录预测时间
                    if cid in pred_rel_map:
                        pred_abs = t0 + pred_rel_map[cid]  # 模型预测的绝对时间
                        append_buffer.append({
                            "file": base,
                            "win_start": t0,
                            "win_end": t1,
                            "class": int(cid),
                            "gt_start": gt_abs,     # 绝对时间
                            "pred_start": pred_abs,  # 绝对时间（= 窗口起点 + 相对时间）
                            "is_recognized": False  # 标记为模型未识别的事件
                        })
                    else:
                        append_buffer.append({
                            "file": base,
                            "win_start": t0,
                            "win_end": t1,
                            "class": int(cid),
                            "gt_start": gt_abs,     # 绝对时间
                            "pred_start": None,  # 未识别的事件，记录为空
                            "is_recognized": False  # 标记为模型未识别的事件
                        })
    append_rows(out_csv_path, append_buffer)

# ---------------- entry ----------------

def is_sony_path(p: str) -> bool:
    return "sony" in p.lower()

def is_tau_path(p: str) -> bool:
    return "tau" in p.lower()

if __name__ == "__main__":
    files = [
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix005.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix010.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix011.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix012.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix013.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix014.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix015.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix016.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix017.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix018.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix019.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix020.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix021.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix022.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix023.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix024.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix025.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix026.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix027.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix028.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room21_mix029.wav",
    ]

    # 读取事件 CSV
    df_tau  = load_events_csv(TAU_EVENTS_CSV)  if os.path.exists(TAU_EVENTS_CSV)  else None
    df_sony = load_events_csv(SONY_EVENTS_CSV) if os.path.exists(SONY_EVENTS_CSV) else None

    for wav_path in files:
        # 针对每个窗口长度分别评测 & 产出文件
        for win_len in WIN_LENS:
            # 每个 win_len 独立的输出文件名（便于区分）
            win_tag = f"win{int(win_len):02d}"  # 5->win05, 120->win120
            replies_jsonl = f"{wav_path}.{win_tag}.4limit.replies.jsonl"

            if is_tau_path(wav_path):
                if df_tau is None:
                    raise FileNotFoundError(f"TAU events CSV not found: {TAU_EVENTS_CSV}")
                out_csv = f"tau.{win_tag}_earliest.csv"
                process_file_windows(wav_path, df_tau, out_csv, replies_jsonl, win_len)

            elif is_sony_path(wav_path):
                if df_sony is None:
                    raise FileNotFoundError(f"SONY events CSV not found: {SONY_EVENTS_CSV}")
                out_csv = f"sony.{win_tag}_earliest.csv"
                process_file_windows(wav_path, df_sony, out_csv, replies_jsonl, win_len)

            else:
                print(f"[WARN] Unknown domain (neither sony nor tau): {wav_path}")

import os
import torch
import librosa
import pandas as pd
import tempfile
import random
import json
from data_util import audioset_classes
from typing import List, Dict, Any
from models.prediction_wrapper import PredictionsWrapper
from models.beats.BEATs_wrapper import BEATsWrapper  # 你可以选择其他模型
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.frame_mn.utils import NAME_TO_WIDTH
import soundfile as sf

# ========== CONFIG ==========

WIN_LENS = [5, 30, 60, 90, 120]
TAU_EVENTS_CSV  = "/data/user/jzt/crd/audioLLM/train_events/tau.csv"
SONY_EVENTS_CSV = "/data/user/jzt/crd/audioLLM/train_events/sony.csv"

# ====== CLASS ID → NAME MAPPING ======
CLASS_ID_TO_NAME = {
    0: "Female speech, woman speaking",
    1: "Male speech, man speaking",
    2: "Clapping",
    4: "Laughter",
    6: "Walk, footsteps",
    8: "Music",
    10: "Water tap, faucet",
    12: "Knock",
}

# =================== 加载模型 ===================

def load_model(model_name: str, device: torch.device) -> PredictionsWrapper:
    """加载并返回指定的模型，只在实验开始时调用一次"""
    if model_name == "BEATs":
        beats = BEATsWrapper()
        model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1")
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    
    model.eval()
    model.to(device)
    return model

# =================== 文件窗口处理 ===================

def split_fixed_win(wav_path: str, win_len: float):
    """
    Return (wav, sr, windows)
    windows = list of (i0, i1, t0, t1) with exact ts each; last tail (<win_len) is dropped.
    """
    wav, sr = sf.read(wav_path)
    if hasattr(wav, "ndim") and wav.ndim > 1:
        wav = wav.mean(axis=1)  # Convert to mono by averaging channels
    
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

def save_temp_audio_chunk(audio_chunk, sr) -> str:
    """临时保存音频片段到磁盘"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        sf.write(tmp_path, audio_chunk, sr, format="wav")
        return tmp_path

# =================== 音频事件检测 ===================

def sound_event_detection(audio_chunk_path: str, model: PredictionsWrapper, device: torch.device):
    """
    Running Sound Event Detection on a chunk of audio.
    """
    sample_rate = 16_000  # All models are trained on 16 kHz audio
    waveform, _ = librosa.core.load(audio_chunk_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)
    waveform_len = waveform.shape[1]
    audio_len = waveform_len / sample_rate
    # print("Audio length (seconds): ", audio_len)
    # Encoder for decoding model predictions into human-readable form
    encoder = ManyHotEncoder(audioset_classes.as_strong_train_classes, audio_len=audio_len)

    # Calculate the number of chunks (each of 10 seconds)
    segment_duration = 10  # 10-second chunks
    segment_samples = segment_duration * sample_rate  # Number of samples per 10-second chunk
    
    num_chunks = waveform_len // segment_samples + (waveform_len % segment_samples != 0)

    # List to store predictions
    all_predictions = []

    # Process each 10-second chunk
    for i in range(num_chunks):
        start_idx = i * segment_samples
        end_idx = min((i + 1) * segment_samples, waveform_len)
        waveform_chunk = waveform[:, start_idx:end_idx]

        # Pad the last chunk if it's shorter than 10 seconds
        if waveform_chunk.shape[1] < segment_samples:
            pad_size = segment_samples - waveform_chunk.shape[1]
            waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, pad_size))

        # Run inference for each chunk
        with torch.no_grad():
            mel = model.mel_forward(waveform_chunk)
            y_strong, _ = model(mel)

        # Collect predictions for this chunk
        all_predictions.append(y_strong)

    # Concatenate all predictions along the time axis (for the entire audio)
    y_strong = torch.cat(all_predictions, dim=2)
    # Convert predictions into probabilities
    y_strong = torch.sigmoid(y_strong)

    # Decode predictions into human-readable format
    (scores_unprocessed, scores_postprocessed, decoded_predictions) = batched_decode_preds(
        y_strong.float(),
        [audio_chunk_path],
        encoder,
        median_filter=9,
        thresholds=(0.1,)  # Apply a threshold of 0.1
    )
    # print(decoded_predictions[0.1])
    return decoded_predictions[0.1]

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

def is_sony_path(p: str) -> bool:
    return "sony" in p.lower()

def is_tau_path(p: str) -> bool:
    return "tau" in p.lower()

def earliest_class_abs_time_in_window(df_events: pd.DataFrame, basename: str, t0: float, t1: float) -> Dict[int, float]:
    """
    对窗口 [t0, t1)：
      - 过滤出与窗口重叠的事件: offset > t0 && onset < t1
      - 对每个类别取窗口内最早出现的“绝对时间”： earliest_abs = min( max(onset, t0) )
    返回 {class_id: earliest_abs_time}（注意是绝对时间，不减 t0）
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
# =================== 文件处理和结果保存 ===================

def append_rows(csv_path: str, rows: List[Dict[str, Any]]):
    """
    追加写：file, win_start, win_end, class, gt_start, pred_start
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    cols = ["file", "win_start", "win_end", "class", "gt_start", "pred_start"]
    df = pd.DataFrame(rows)[cols]
    exists = os.path.exists(csv_path)
    df.to_csv(csv_path, mode=("a" if exists else "w"), header=not exists, index=False)
    print(f"[append] {len(df)} rows -> {csv_path}")

def parse_class_start_list(raw_reply: pd.DataFrame, win_len: float) -> List[Dict[str, Any]]:
    """
    解析模型的 DataFrame 回复，提取每个类别的最早开始时间。
    输入：raw_reply 是模型返回的 DataFrame，包含 event_label, onset, offset, filename
           win_len 是窗口长度（相对时间）
    输出：每个类别的最早开始时间，格式为 [{"class": ID, "start": seconds}, ...]
    """
    try:
        # 用于存储每个类别最早的开始时间
        earliest_events = {}
        # 遍历 DataFrame 中的每一行
        for _, event in raw_reply.iterrows():
            event_label = event.get("event_label")
            onset = float(event.get("onset"))
            
            if event_label and onset < win_len:  # 确保事件发生在窗口范围内
                # 获取类的 ID
                class_id = get_class_id_from_label(event_label)
                if class_id != -1:  # 确保找到了有效的类 ID
                    # 对于每个类，记录最早的 onset
                    if class_id not in earliest_events:
                        earliest_events[class_id] = onset
                    else:
                        # 更新最早的 onset
                        earliest_events[class_id] = min(earliest_events[class_id], onset)
        
        # 将结果格式化为 [{class: ID, start: earliest_onset}, ...]
        events = [{"class": class_id, "start": onset} for class_id, onset in earliest_events.items()]
        # print(events)
        return events
    except Exception as e:
        print(f"Error parsing class start list: {e}")
        return []

def get_class_id_from_label(label: str) -> int:
    """根据事件标签返回相应的类别ID"""
    # 提取类标签映射到数字ID
    for class_id, class_name in CLASS_ID_TO_NAME.items():
        if class_name.lower() in label.lower():
            return class_id
    return -1  # 如果未找到对应类别，返回 -1

def delete_temp_file(temp_path: str):
    """删除临时文件"""
    if os.path.exists(temp_path):
        os.remove(temp_path)

def process_file_windows(wav_path: str,
                         events_df: pd.DataFrame,
                         out_csv_path: str,
                         replies_jsonl_path: str,
                         win_len: float,
                         model: PredictionsWrapper,
                         device: torch.device):
    """
    For each window:
      - Compute the "earliest absolute time" for each class within the window (gt_start)
      - Save the chunk and process it using PretrainedSED
      - Write results to CSV and JSONL files
    """
    wav, sr, wins = split_fixed_win(wav_path, win_len)
    base = os.path.basename(wav_path)
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
            
            # Save the audio chunk to a temporary file
            temp_path = save_temp_audio_chunk(clip, sr)

            # Run inference using PretrainedSED
            raw_predictions = sound_event_detection(temp_path, model, device)

            # Process relative start times and convert them to absolute
            pred_list = parse_class_start_list(raw_predictions, win_len)
            pred_rel_map = {int(d["class"]): float(d["start"]) for d in pred_list}

            # Collect rows for CSV: pred_start = t0 + rel_start; gt_start is already absolute
            for cid in present_classes:
                gt_abs = float(gt_abs_map[cid])  # Absolute time
                pred_abs = t0 + pred_rel_map[cid] if cid in pred_rel_map else (t0 + pred_rel_map[8] if cid == 9 and 8 in pred_rel_map else "")
                append_buffer.append({
                    "file": base,
                    "win_start": t0,
                    "win_end": t1,
                    "class": int(cid),
                    "gt_start": gt_abs,  # Absolute time
                    "pred_start": pred_abs  # Absolute time (= win_start + relative time)
                })

            # Delete the temporary file after processing
            delete_temp_file(temp_path)

    append_rows(out_csv_path, append_buffer)

# =================== 主函数 ===================

if __name__ == "__main__":
    model_name = "BEATs"  # Example model, can be changed
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model only once at the beginning of the experiment
    model = load_model(model_name, device)

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
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix001.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix002.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix003.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix004.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix005.wav",   
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix006.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix007.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix008.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix009.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix010.wav",
        "/data/user/jzt/crd/audioLLM/foa_dev/foa_dev/dev-train-sony/fold3_room22_mix011.wav",
    ]

    # Read event CSVs
    df_tau  = load_events_csv(TAU_EVENTS_CSV)  if os.path.exists(TAU_EVENTS_CSV)  else None
    df_sony = load_events_csv(SONY_EVENTS_CSV) if os.path.exists(SONY_EVENTS_CSV) else None

    for wav_path in files:
        for win_len in WIN_LENS:
            win_tag = f"win{int(win_len):02d}"  # 5 -> win05, 120 -> win120
            replies_jsonl = f"{wav_path}.{win_tag}.4limit.AAsed.replies.jsonl"

            if is_tau_path(wav_path):
                if df_tau is None:
                    raise FileNotFoundError(f"TAU events CSV not found: {TAU_EVENTS_CSV}")
                out_csv = f"tau.{win_tag}_earliest.csv"
                process_file_windows(wav_path, df_tau, out_csv, replies_jsonl, win_len, model, device)

            elif is_sony_path(wav_path):
                if df_sony is None:
                    raise FileNotFoundError(f"SONY events CSV not found: {SONY_EVENTS_CSV}")
                out_csv = f"sony.{win_tag}_earliest.csv"
                process_file_windows(wav_path, df_sony, out_csv, replies_jsonl, win_len, model, device)

            else:
                print(f"[WARN] Unknown domain (neither sony nor tau): {wav_path}")

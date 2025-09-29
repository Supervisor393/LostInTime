import pandas as pd
from pathlib import Path

FRAME_SEC = 0.1  # 100ms

def frames_csv_to_events_per_class_df(csv_path: Path) -> pd.DataFrame:
    """将单个帧级CSV(无表头: frame,class,track,az,el)转为事件级DataFrame: [onset,offset,class]"""
    df = pd.read_csv(csv_path, header=None, names=["frame","class","track","az","el"])
    df["t0"] = (df["frame"] - 1) * FRAME_SEC
    df["t1"] = df["t0"] + FRAME_SEC
    rows = []
    for cls, g in df.groupby("class"):
        g = g.sort_values("t0")
        merged = []
        for _, r in g.iterrows():
            s, e = float(r.t0), float(r.t1)
            if not merged:
                merged.append([s, e])
            else:
                ps, pe = merged[-1]
                if s <= pe + 1e-9:
                    merged[-1][1] = max(pe, e)
                else:
                    merged.append([s, e])
        for s, e in merged:
            rows.append({"onset": round(s,3), "offset": round(e,3), "class": int(cls)})
    return pd.DataFrame(rows)

def main(meta_root: str, out_csv: str, audio_root_rel_anchor: str = ""):
    """
    meta_root:   元数据(帧级CSV)根目录，例如 /data/.../metadata_dev/metadata_dev
    out_csv:     汇总事件级CSV输出路径，例如 /data/events/events.csv
    audio_root_rel_anchor: 【可选】从 meta_root 下的哪一层开始作为相对路径前缀。
        - 默认 "" 表示直接使用 meta_root 下的相对子路径，例如 dev-test-sony/fold4_room23_mix001.csv
        - 如果你的 foa_dev 的目录结构与 meta_root 的子结构完全一致，则保持默认即可
    """
    meta_root = Path(meta_root)
    all_rows = []
    for csv_path in meta_root.rglob("*.csv"):
        # 计算相对路径（不含扩展名），用作 file 列（.csv -> .wav）
        rel = csv_path.relative_to(meta_root)  # e.g. dev-test-sony/fold4_room23_mix001.csv
        rel_no_ext = rel.with_suffix("")       # dev-test-sony/fold4_room23_mix001
        file_rel_wav = (Path(audio_root_rel_anchor) / rel_no_ext).with_suffix(".wav").as_posix()

        # 单文件转换
        ev_df = frames_csv_to_events_per_class_df(csv_path)
        if ev_df.empty:
            continue
        ev_df.insert(0, "file", file_rel_wav)  # 在首列插入 file
        all_rows.append(ev_df)

    if not all_rows:
        print("[WARN] 未在元数据目录下找到任何CSV或转换结果为空。")
        return

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(by=["file","class","onset"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] 写出事件级汇总: {out_csv}  共 {len(out)} 行")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_root", required=True, help="帧级CSV根目录（递归遍历）")
    ap.add_argument("--out_csv", required=True, help="事件级汇总CSV输出路径")
    ap.add_argument("--audio_root_rel_anchor", default="", help="相对前缀（一般留空）")
    args = ap.parse_args()
    main(args.meta_root, args.out_csv, args.audio_root_rel_anchor)

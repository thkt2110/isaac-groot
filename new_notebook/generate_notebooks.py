import json
from pathlib import Path
from textwrap import dedent


OUT = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(s).strip().splitlines(True)}


def code(s):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(s).strip().splitlines(True),
    }


def write(name, cells):
    stem = Path(name).stem.replace("_", "-")
    for i, cell in enumerate(cells, 1):
        # Them cell id de notebook dung chuan nbformat moi va tranh warning khi upload/chay.
        cell.setdefault("id", f"{stem[:40]}-{i:02d}")
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


COMMON_MODEL = r'''
class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim, buckets=1000):
        super().__init__()
        self.buckets = buckets
        self.embed = nn.Embedding(buckets, hidden_dim)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
    def forward(self, t):
        idx = torch.clamp((t * self.buckets).long(), 0, self.buckets - 1)
        return self.mlp(self.embed(idx))

class CrossBlock(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(hidden_dim)
        self.sa = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(hidden_dim)
        self.ca = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout)
        )
    def forward(self, x, mem):
        y = self.n1(x)
        x = x + self.sa(y, y, y, need_weights=False)[0]
        y = self.n2(x)
        x = x + self.ca(y, mem, mem, need_weights=False)[0]
        return x + self.ff(self.n3(x))

class OfficialMiniVLDiT(nn.Module):
    def __init__(self, state_dim, action_dim, state_horizon, action_horizon, vl_dim,
                 hidden_dim=512, layers=8, heads=8, dropout=0.1, buckets=1000):
        super().__init__()
        self.state_horizon = state_horizon
        self.action_horizon = action_horizon
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.vl_proj = nn.Linear(vl_dim, hidden_dim)
        self.time = TimeEmbedding(hidden_dim, buckets)
        self.state_pos = nn.Embedding(state_horizon, hidden_dim)
        self.action_pos = nn.Embedding(action_horizon, hidden_dim)
        self.blocks = nn.ModuleList([CrossBlock(hidden_dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
    def forward(self, noisy_action, state_history, vl_features, t):
        s_pos = self.state_pos(torch.arange(self.state_horizon, device=noisy_action.device))[None]
        a_pos = self.action_pos(torch.arange(self.action_horizon, device=noisy_action.device))[None]
        s = self.state_proj(state_history) + s_pos
        a = self.action_proj(noisy_action) + a_pos
        x = torch.cat([s, a], dim=1) + self.time(t)[:, None, :]
        mem = self.vl_proj(vl_features)
        for block in self.blocks:
            x = block(x, mem)
        return self.out(self.norm(x[:, self.state_horizon:]))
'''


NOTEBOOK_02 = [
md("""
# 02 - Prepare Official Sequence Data

Chuẩn bị dữ liệu chính thức cho GR00T Strong Mini:

- `states_[split].npy`: `[N, S, 44]`, mặc định `S=1`.
- `actions_[split]_chunk.npy`: `[N, H, 44]`, mặc định `H=16`.
- `action_mask_[split].npy`: `[N, H, 44]`.
- `video_frame_manifest_[split].parquet` cho frozen VLM encode.
- `normalization_stats.json` để Notebook 03 normalize và Notebook 04 denormalize metrics.

Notebook chạy CPU. Bật `SMOKE_TEST=True` để debug nhanh.
"""),
code("""!pip install -q pandas pyarrow numpy tqdm"""),
code(r'''
from pathlib import Path
import json, random, re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SMOKE_TEST = False
MAX_EPISODES_PER_SUBSET = 5 if SMOKE_TEST else None
MAX_SAMPLES_PER_SPLIT = None

DEFAULT_SUBSETS = ["gr1_arms_only.CanSort", "gr1_arms_waist.CanToDrawer", "gr1_arms_waist.CupToDrawer"]

def load_subset_plan(default):
    # Neu nhieu Notebook 01 download rieng tung subset duoc add lam Kaggle Input,
    # Notebook 02 se merge tat ca selected_subsets_for_notebook02.json.
    merged = []
    seen = set()
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if not base.exists():
            continue
        for p in base.rglob("selected_subsets_for_notebook02.json"):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                subsets = obj.get("pipeline_safe_subsets") or obj.get("subsets") or obj
                subsets = [str(x) for x in subsets if str(x).startswith("gr1_")]
                if subsets:
                    print("Loaded subset plan:", p)
                    for subset in subsets:
                        if subset not in seen:
                            merged.append(subset)
                            seen.add(subset)
            except Exception as exc:
                print("WARN cannot read subset plan", p, exc)
    if merged:
        return merged
    return list(default)

SUBSETS = load_subset_plan(DEFAULT_SUBSETS)
TRAIN_RATIO = 0.8
STATE_HORIZON = 1
ACTION_HORIZON = 16
STATE_DIM = ACTION_DIM = 44
CAMERA_KEY = "observation.images.ego_view"
OUTPUT_ROOT = Path(f"/kaggle/working/gr00t_prepared_official_H{ACTION_HORIZON}")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
'''),
code(r'''
def parse_episode_id(path: Path):
    m = re.search(r"episode[_-](\d+)", path.stem)
    return int(m.group(1)) if m else None

def roots():
    out = []
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if base.exists():
            for p in base.rglob("gr00t_x_embodiment_sim*"):
                if p.is_dir() and any((p / s).exists() for s in SUBSETS):
                    out.append(p)
    out = sorted(set(out), key=str)
    if not out:
        raise FileNotFoundError("Không tìm thấy gr00t_x_embodiment_sim*. Add Input các output download.")
    return out

def collect_sources():
    src = {s: {"data": None, "videos": []} for s in SUBSETS}
    for r in roots():
        for s in SUBSETS:
            p = r / s
            if not p.exists():
                continue
            if (p / "data").exists() and (p / "meta").exists() and src[s]["data"] is None:
                src[s]["data"] = p
            if (p / "videos").exists():
                src[s]["videos"].append(p)
    missing = [s for s, v in src.items() if v["data"] is None]
    if missing:
        raise FileNotFoundError(f"Thiếu data/meta: {missing}")
    return src

def video_index(video_roots):
    idx = {}
    for root in video_roots:
        vids = list((root / "videos").rglob(f"{CAMERA_KEY}/*.mp4")) or list((root / "videos").rglob("*.mp4"))
        for v in vids:
            ep = parse_episode_id(v)
            if ep is not None and ep not in idx:
                idx[ep] = str(v)
    return idx

def load_tasks(meta):
    tasks = {}
    for name in ["tasks.jsonl", "tasks.json"]:
        p = meta / name
        if not p.exists():
            continue
        try:
            if p.suffix == ".jsonl":
                rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            else:
                rows = json.loads(p.read_text(encoding="utf-8"))
            for obj in rows:
                idx = obj.get("task_index", obj.get("index"))
                txt = obj.get("task", obj.get("text", obj.get("description")))
                if idx is not None and txt is not None:
                    tasks[int(idx)] = str(txt)
        except Exception as exc:
            print("WARN tasks:", p, exc)
    return tasks

def vec(x, dim, col):
    if isinstance(x, np.ndarray):
        a = x.astype(np.float32, copy=False)
    elif isinstance(x, (list, tuple)):
        a = np.asarray(x, np.float32)
    elif isinstance(x, str):
        try:
            a = np.asarray(json.loads(x), np.float32)
        except Exception:
            a = np.fromstring(x.strip("[]"), sep=",", dtype=np.float32)
    else:
        a = np.asarray(x, np.float32)
    a = a.reshape(-1)
    if a.size != dim:
        raise ValueError(f"{col} dim mismatch: expected={dim}, got={a.size}")
    return a

def order_col(df):
    for c in ["frame_index", "timestamp", "index"]:
        if c in df.columns:
            return c
    return df.columns[0]

def ep_id(df, path):
    if "episode_index" in df.columns and len(df):
        return int(df["episode_index"].iloc[0])
    ep = parse_episode_id(path)
    return int(ep if ep is not None else -1)

def task_text(row, tasks):
    for c in ["annotation.human.action.task_description", "task_index"]:
        if c in row.index:
            v = row[c]
            if isinstance(v, str):
                return v
            try:
                return tasks.get(int(v), str(v))
            except Exception:
                return str(v)
    return ""

sources = collect_sources()
for s, v in sources.items():
    print(s, "data=", v["data"], "video_roots=", len(v["videos"]))
'''),
code(r'''
# Pass 1: scan episode length và split theo episode.
episodes, scan_rows = [], []
for subset in SUBSETS:
    root = sources[subset]["data"]
    vids = video_index(sources[subset]["videos"])
    files = sorted((root / "data").rglob("*.parquet"))
    if MAX_EPISODES_PER_SUBSET is not None:
        files = files[:MAX_EPISODES_PER_SUBSET]
    for path in tqdm(files, desc=f"scan {subset}"):
        try:
            df = pd.read_parquet(path)
            if "observation.state" not in df.columns or "action" not in df.columns:
                scan_rows.append({"subset": subset, "path": str(path), "status": "missing_cols"})
                continue
            df = df.sort_values(order_col(df)).reset_index(drop=True)
            ep = ep_id(df, path)
            valid = max(0, len(df) - (STATE_HORIZON - 1) - ACTION_HORIZON + 1)
            episodes.append({"subset": subset, "path": str(path), "episode_index": ep, "length": len(df),
                             "valid_samples": valid, "video_path": vids.get(ep, ""), "has_video": ep in vids})
            scan_rows.append({"subset": subset, "path": str(path), "status": "ok", "episode_index": ep,
                              "length": len(df), "valid_samples": valid})
        except Exception as exc:
            scan_rows.append({"subset": subset, "path": str(path), "status": "error", "error": str(exc)})

episodes_df = pd.DataFrame(episodes)
if episodes_df.empty:
    raise RuntimeError("Không có episode nào hợp lệ.")

train_keys, test_keys = set(), set()
for subset, g in episodes_df[episodes_df.valid_samples > 0].groupby("subset"):
    eps = list(g.episode_index)
    # Dung seed on dinh theo ten subset, khong dung hash() vi Python randomize hash moi session.
    subset_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(str(subset)))
    rng = random.Random(SEED + subset_seed % 10000)
    rng.shuffle(eps)
    n = max(1, int(len(eps) * TRAIN_RATIO)) if len(eps) > 1 else len(eps)
    train_keys.update((subset, int(e)) for e in eps[:n])
    test_keys.update((subset, int(e)) for e in eps[n:])

counts = {"train": 0, "test": 0}
for _, r in episodes_df.iterrows():
    k = (r.subset, int(r.episode_index))
    if r.valid_samples <= 0:
        continue
    if k in train_keys:
        counts["train"] += int(r.valid_samples)
    elif k in test_keys:
        counts["test"] += int(r.valid_samples)
if MAX_SAMPLES_PER_SPLIT is not None:
    counts = {k: min(v, int(MAX_SAMPLES_PER_SPLIT)) for k, v in counts.items()}
print("counts:", counts)
display(episodes_df.groupby("subset").agg(episodes=("episode_index", "count"), valid_samples=("valid_samples", "sum"), video_episodes=("has_video", "sum")))
'''),
code(r'''
# Pass 2: ghi memmap .npy và parquet theo batch để tiết kiệm RAM.
arrays = {
    "states_train": np.lib.format.open_memmap(OUTPUT_ROOT / "states_train.npy", "w+", np.float32, (counts["train"], STATE_HORIZON, STATE_DIM)),
    "states_test": np.lib.format.open_memmap(OUTPUT_ROOT / "states_test.npy", "w+", np.float32, (counts["test"], STATE_HORIZON, STATE_DIM)),
    "actions_train": np.lib.format.open_memmap(OUTPUT_ROOT / "actions_train_chunk.npy", "w+", np.float32, (counts["train"], ACTION_HORIZON, ACTION_DIM)),
    "actions_test": np.lib.format.open_memmap(OUTPUT_ROOT / "actions_test_chunk.npy", "w+", np.float32, (counts["test"], ACTION_HORIZON, ACTION_DIM)),
    "mask_train": np.lib.format.open_memmap(OUTPUT_ROOT / "action_mask_train.npy", "w+", np.float32, (counts["train"], ACTION_HORIZON, ACTION_DIM)),
    "mask_test": np.lib.format.open_memmap(OUTPUT_ROOT / "action_mask_test.npy", "w+", np.float32, (counts["test"], ACTION_HORIZON, ACTION_DIM)),
}
writers, batch_rows = {}, {}

def write_batch(name, rows):
    if not rows:
        return
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    path = OUTPUT_ROOT / f"{name}.parquet"
    if name not in writers:
        writers[name] = pq.ParquetWriter(path, table.schema, compression="zstd")
    writers[name].write_table(table)

stat = {"ss": np.zeros(STATE_DIM), "ss2": np.zeros(STATE_DIM), "sn": 0,
        "as": np.zeros(ACTION_DIM), "as2": np.zeros(ACTION_DIM), "an": 0}
idx = {"train": 0, "test": 0}
BATCH = 100000

for _, er in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc="fill"):
    if er.valid_samples <= 0:
        continue
    split = "train" if (er.subset, int(er.episode_index)) in train_keys else "test"
    if idx[split] >= counts[split]:
        continue
    root = sources[er.subset]["data"]
    tasks = load_tasks(root / "meta")
    # Doc parquet mot lan roi sort de giam I/O tren Kaggle.
    df = pd.read_parquet(er.path)
    df = df.sort_values(order_col(df)).reset_index(drop=True)
    states = np.stack([vec(x, STATE_DIM, "state") for x in df["observation.state"].to_list()]).astype(np.float32)
    actions = np.stack([vec(x, ACTION_DIM, "action") for x in df["action"].to_list()]).astype(np.float32)
    oc = order_col(df)
    n = min(int(er.valid_samples), counts[split] - idx[split])
    for local in range(n):
        base = local + STATE_HORIZON - 1
        sid = idx[split]
        sh = states[base - STATE_HORIZON + 1:base + 1]
        ac = actions[base:base + ACTION_HORIZON]
        arrays[f"states_{split}"][sid] = sh
        arrays[f"actions_{split}"][sid] = ac
        arrays[f"mask_{split}"][sid] = 1.0
        if split == "train":
            stat["ss"] += sh.reshape(-1, STATE_DIM).sum(0); stat["ss2"] += (sh.reshape(-1, STATE_DIM) ** 2).sum(0); stat["sn"] += sh.shape[0]
            stat["as"] += ac.sum(0); stat["as2"] += (ac ** 2).sum(0); stat["an"] += ac.shape[0]
        frame = df[oc].iloc[base] if oc in df.columns else base
        rec = {"sample_id": int(sid), "subset": er.subset, "episode_index": int(er.episode_index),
               "frame_index": int(frame) if pd.notna(frame) else int(base),
               "timestamp": float(df["timestamp"].iloc[base]) if "timestamp" in df.columns else float(base),
               "state_horizon": STATE_HORIZON, "action_horizon": ACTION_HORIZON,
               "action_start": int(base), "action_end_exclusive": int(base + ACTION_HORIZON),
               "has_video": bool(er.has_video), "task_text": task_text(df.iloc[base], tasks)}
        man = {**rec, "video_path": str(er.video_path) if er.has_video else "", "camera_key": CAMERA_KEY, "split": split}
        for name, row in [(f"{split}_samples", rec), (f"video_frame_manifest_{split}", man)]:
            batch_rows.setdefault(name, []).append(row)
            if len(batch_rows[name]) >= BATCH:
                write_batch(name, batch_rows[name]); batch_rows[name] = []
        idx[split] += 1

for name, rows in batch_rows.items():
    write_batch(name, rows)
for w in writers.values():
    w.close()
for a in arrays.values():
    a.flush()
print("filled:", idx)
'''),
code(r'''
def mean_std(s, s2, n):
    mean = s / max(n, 1)
    var = s2 / max(n, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.tolist(), std.tolist()

sm, ss = mean_std(stat["ss"], stat["ss2"], stat["sn"])
am, ast = mean_std(stat["as"], stat["as2"], stat["an"])
norm = {"state_mean": sm, "state_std": ss, "action_mean": am, "action_std": ast,
        "state_n": int(stat["sn"]), "action_n": int(stat["an"]), "space": "raw_train_split"}
(OUTPUT_ROOT / "normalization_stats.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
(OUTPUT_ROOT / "dataset_scan_report.json").write_text(json.dumps(scan_rows, indent=2), encoding="utf-8")
pd.DataFrame(scan_rows).to_parquet(OUTPUT_ROOT / "dataset_scan_report.parquet", index=False)
episodes_df.to_parquet(OUTPUT_ROOT / "episodes_report.parquet", index=False)
report = {"status": "completed", "state_horizon": STATE_HORIZON, "action_horizon": ACTION_HORIZON,
          "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "train_samples": int(idx["train"]),
          "test_samples": int(idx["test"]), "split_by_episode": True, "output_root": str(OUTPUT_ROOT),
          "smoke_test": SMOKE_TEST, "files": sorted(p.name for p in OUTPUT_ROOT.iterdir())}
(OUTPUT_ROOT / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
for p in sorted(OUTPUT_ROOT.iterdir()):
    print("-", p.name, round(p.stat().st_size / (1024 ** 2), 3), "MB")
'''),
md("""
## Acceptance

- `states_train.npy`: `[N, S, 44]`.
- `actions_train_chunk.npy`: `[N, H, 44]`.
- `action_mask_train.npy`: `[N, H, 44]`.
- Split theo episode.
- Có `video_frame_manifest_train.parquet`, `video_frame_manifest_test.parquet` và `normalization_stats.json`.
"""),
]


NOTEBOOK_03 = [
md("""
# 03 - Frozen VLM Encode + Train OfficialMiniVLDiT

Phase 1 encode video frame + language bằng frozen encoder. Primary là NVIDIA/Eagle; fallback SigLIP.
Phase 2 train DiT/action head từ scratch với cross-attention, action chunk H=16, Beta timestep, masked flow-matching loss.
"""),
code("""!pip install -q pandas pyarrow numpy tqdm matplotlib pillow transformers accelerate av opencv-python"""),
code(r'''
from pathlib import Path
import json, random, time
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Beta

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0))

SMOKE_TEST = False
ACTION_HORIZON = 16
STATE_HORIZON = 1
MAX_ENCODE_TRAIN_SAMPLES = 512 if SMOKE_TEST else None
MAX_ENCODE_TEST_SAMPLES = 256 if SMOKE_TEST else None
MAX_TRAIN_SAMPLES = 2048 if SMOKE_TEST else None
MAX_STEPS_PER_EPOCH = 20 if SMOKE_TEST else None
EPOCHS = 1; BATCH_SIZE = 256; NUM_WORKERS = 2
LR = 1e-4; WEIGHT_DECAY = 1e-5; USE_AMP = True
HIDDEN_DIM = 512; NUM_LAYERS = 8; NUM_HEADS = 8; DROPOUT = 0.1
PRIMARY_ENCODER = "nvidia/Eagle-Block2A-2B-v2"
FALLBACK_ENCODER = "google/siglip-base-patch16-224"
FORCE_FALLBACK = False
OUTPUT_ROOT = Path("/kaggle/working/gr00t_official_mini_runs"); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_NAME = f"official_mini_vldit_H{ACTION_HORIZON}"
'''),
code(r'''
def find_dir(name, required):
    cand = []
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if base.exists(): cand.extend(base.rglob(name))
    for c in sorted(cand, key=str):
        if c.is_dir() and all((c / r).exists() for r in required): return c
    raise FileNotFoundError(f"Không tìm thấy {name}")

PREPARED_ROOT = find_dir(f"gr00t_prepared_official_H{ACTION_HORIZON}", ["states_train.npy", "actions_train_chunk.npy", "action_mask_train.npy", "video_frame_manifest_train.parquet", "normalization_stats.json", "prepare_report.json"])
stats = json.loads((PREPARED_ROOT / "normalization_stats.json").read_text())
prepare_report = json.loads((PREPARED_ROOT / "prepare_report.json").read_text())
print("PREPARED_ROOT:", PREPARED_ROOT)
print(json.dumps({k: prepare_report[k] for k in ["state_horizon", "action_horizon", "train_samples", "test_samples"]}, indent=2))
'''),
code(r'''
def decode_frame_pyav(video_path, frame_index):
    import av
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        last = None
        for i, frame in enumerate(container.decode(stream)):
            last = frame
            if i >= frame_index:
                return frame.to_image().convert("RGB")
        if last is None: raise RuntimeError("empty video")
        return last.to_image().convert("RGB")
    finally:
        container.close()

def decode_frame_cv2(video_path, frame_index):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError("cv2 cannot open video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1)); ok, frame = cap.read()
    cap.release()
    if not ok: raise RuntimeError("cv2 cannot decode frame")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def decode_frame(video_path, frame_index):
    # Uu tien PyAV vi doc video on dinh; neu loi thi fallback sang OpenCV.
    try: return decode_frame_pyav(video_path, int(frame_index))
    except Exception: return decode_frame_cv2(video_path, int(frame_index))
'''),
code(r'''
from transformers import AutoModel, AutoProcessor

class FrozenVLEncoder:
    def __init__(self, primary_id, fallback_id, force_fallback=False):
        self.primary_id = primary_id; self.fallback_id = fallback_id; self.force_fallback = force_fallback
        self.encoder_name = None; self.fallback_used = False; self.processor = None; self.model = None
        self.feature_dim = None; self.num_tokens = 2
        self.load()
    def try_load(self, model_id):
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, low_cpu_mem_usage=True,
                                           torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        model.eval().to(device)
        for p in model.parameters(): p.requires_grad_(False)
        return proc, model
    def load(self):
        if not self.force_fallback:
            try:
                self.processor, self.model = self.try_load(self.primary_id)
                self.encoder_name = self.primary_id
                print("Loaded primary encoder:", self.encoder_name)
                return
            except Exception as exc:
                print("WARN primary failed, fallback:", repr(exc))
        self.processor, self.model = self.try_load(self.fallback_id)
        self.encoder_name = self.fallback_id; self.fallback_used = True
        print("Loaded fallback encoder:", self.encoder_name)
    @torch.no_grad()
    def encode_batch(self, images, texts):
        # Encoder duoc freeze hoan toan; Notebook 03 chi train DiT/action head tu scratch.
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):
            if hasattr(self.model, "get_image_features") and hasattr(self.model, "get_text_features"):
                image_feat = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                text_keys = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
                text_feat = self.model.get_text_features(**text_keys) if text_keys else torch.zeros_like(image_feat)
                feats = torch.stack([F.normalize(image_feat.float(), dim=-1), F.normalize(text_feat.float(), dim=-1)], 1)
            else:
                out = self.model(**inputs, output_hidden_states=True, return_dict=True)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    hidden = out.hidden_states[-1]
                pooled = hidden.float().mean(1)
                feats = torch.stack([pooled, pooled], 1)
        self.feature_dim = int(feats.shape[-1]); self.num_tokens = int(feats.shape[1])
        return feats.cpu().numpy().astype(np.float16)
'''),
code(r'''
def select_manifest(split, max_samples):
    df = pd.read_parquet(PREPARED_ROOT / f"video_frame_manifest_{split}.parquet")
    df = df[(df.has_video == True) & (df.video_path.astype(str).str.len() > 0)].copy()
    if df.empty: raise RuntimeError(f"No video-backed samples for {split}")
    df = df.sort_values(["subset", "episode_index", "frame_index"]).reset_index(drop=True)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=int(max_samples), random_state=SEED).sort_values("sample_id").reset_index(drop=True)
    return df

def encode_split(split, max_samples, encoder, batch_size=16):
    feat_path = OUTPUT_ROOT / f"vl_features_{split}.npy"
    idx_path = OUTPUT_ROOT / f"vl_feature_index_{split}.parquet"
    if feat_path.exists() and idx_path.exists():
        return feat_path, idx_path, {"cached": True, "shape": list(np.load(feat_path, mmap_mode="r").shape)}
    manifest = select_manifest(split, max_samples)
    chunks, rows, failed = [], [], []
    for start in tqdm(range(0, len(manifest), batch_size), desc=f"encode {split}"):
        batch = manifest.iloc[start:start+batch_size]
        images, texts, ok = [], [], []
        for _, r in batch.iterrows():
            try:
                images.append(decode_frame(r.video_path, int(r.frame_index)))
                texts.append(str(r.task_text) if str(r.task_text) else "perform the manipulation task")
                ok.append(r)
            except Exception as exc:
                failed.append({"sample_id": int(r.sample_id), "error": str(exc)})
        if not images: continue
        feats = encoder.encode_batch(images, texts); base = sum(c.shape[0] for c in chunks); chunks.append(feats)
        for j, r in enumerate(ok):
            rows.append({"feature_index": int(base + j), "sample_id": int(r.sample_id), "subset": r.subset, "episode_index": int(r.episode_index), "frame_index": int(r.frame_index), "task_text": str(r.task_text)})
    features = np.concatenate(chunks, 0)
    np.save(feat_path, features); pd.DataFrame(rows).to_parquet(idx_path, index=False)
    report = {"split": split, "requested": int(len(manifest)), "encoded": int(features.shape[0]), "failed": int(len(failed)),
              "feature_shape": list(features.shape), "encoder_name": encoder.encoder_name, "fallback_used": encoder.fallback_used, "failed_examples": failed[:20]}
    (OUTPUT_ROOT / f"video_decode_report_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return feat_path, idx_path, report

encoder = FrozenVLEncoder(PRIMARY_ENCODER, FALLBACK_ENCODER, FORCE_FALLBACK)
train_feat, train_idx, train_enc_report = encode_split("train", MAX_ENCODE_TRAIN_SAMPLES, encoder)
test_feat, test_idx, test_enc_report = encode_split("test", MAX_ENCODE_TEST_SAMPLES, encoder)
print(train_enc_report); print(test_enc_report)
'''),
code(r'''
class OfficialDataset(Dataset):
    def __init__(self, split, feat_path, idx_path, max_samples=None):
        self.split = split
        self.states = np.load(PREPARED_ROOT / f"states_{split}.npy", mmap_mode="r")
        self.actions = np.load(PREPARED_ROOT / f"actions_{split}_chunk.npy", mmap_mode="r")
        self.masks = np.load(PREPARED_ROOT / f"action_mask_{split}.npy", mmap_mode="r")
        self.features = np.load(feat_path, mmap_mode="r")
        self.index = pd.read_parquet(idx_path).sort_values("feature_index").reset_index(drop=True)
        if max_samples is not None and len(self.index) > max_samples:
            self.index = self.index.sample(n=int(max_samples), random_state=SEED).reset_index(drop=True)
        self.sm = np.asarray(stats["state_mean"], np.float32); self.ss = np.asarray(stats["state_std"], np.float32)
        self.am = np.asarray(stats["action_mean"], np.float32); self.asd = np.asarray(stats["action_std"], np.float32)
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        r = self.index.iloc[i]; sid = int(r.sample_id); fid = int(r.feature_index)
        # Normalize bang thong ke train split; Notebook 04 se denormalize lai khi tinh metric raw.
        state = (np.asarray(self.states[sid], np.float32) - self.sm) / self.ss
        action = (np.asarray(self.actions[sid], np.float32) - self.am) / self.asd
        return {"state": torch.from_numpy(state), "action": torch.from_numpy(action),
                "mask": torch.from_numpy(np.asarray(self.masks[sid], np.float32)),
                "vl": torch.from_numpy(np.asarray(self.features[fid], np.float32)), "sample_id": sid}
def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) if k != "sample_id" else torch.tensor([b[k] for b in batch]) for k in batch[0]}

train_ds = OfficialDataset("train", train_feat, train_idx, MAX_TRAIN_SAMPLES)
test_ds = OfficialDataset("test", test_feat, test_idx, None)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=True, collate_fn=collate)
FEATURE_DIM = int(train_ds.features.shape[-1]); NUM_TOKENS = int(train_ds.features.shape[1])
print("train_samples:", len(train_ds), "FEATURE_DIM:", FEATURE_DIM, "tokens:", NUM_TOKENS)
'''),
code(COMMON_MODEL + r'''
model = OfficialMiniVLDiT(44, 44, STATE_HORIZON, ACTION_HORIZON, FEATURE_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT).to(device)
num_params = sum(p.numel() for p in model.parameters())
print("params:", num_params)
'''),
code(r'''
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.95, 0.999), eps=1e-8)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available())
beta = Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
logs, global_step, start = [], 0, time.time()
model.train()
for epoch in range(1, EPOCHS + 1):
    max_steps = len(train_loader) if MAX_STEPS_PER_EPOCH is None else min(MAX_STEPS_PER_EPOCH, len(train_loader))
    for step, b in enumerate(tqdm(train_loader, total=max_steps, desc=f"epoch {epoch}/{EPOCHS}"), 1):
        if step > max_steps: break
        state = b["state"].to(device); action = b["action"].to(device); mask = b["mask"].to(device); vl = b["vl"].to(device)
        t = beta.sample((action.size(0),)).to(device=device, dtype=action.dtype)
        noise = torch.randn_like(action)
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action
        # Flow matching dung huong official: velocity = action - noise.
        target = action - noise
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):
            pred = model(noisy, state, vl, t)
            loss = ((pred - target).pow(2) * mask).sum() / (mask.sum() + 1e-6)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        global_step += 1
        if global_step == 1 or global_step % 100 == 0:
            logs.append({"epoch": epoch, "step": step, "global_step": global_step, "loss": float(loss.detach().cpu()), "elapsed_sec": round(time.time() - start, 2)})

run_dir = OUTPUT_ROOT / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
cfg = {"model_name": "OfficialMiniVLDiT", "state_horizon": STATE_HORIZON, "action_horizon": ACTION_HORIZON, "state_dim": 44, "action_dim": 44,
       "vl_feature_dim": FEATURE_DIM, "vl_num_tokens": NUM_TOKENS, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
       "dropout": DROPOUT, "num_timestep_buckets": 1000, "timestep_sampling": "Beta(1.5,1.0)", "target_velocity_sign": "action_minus_noise",
       "masked_loss": True, "learning_rate": LR, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "num_params": int(num_params),
       "encoder_name": encoder.encoder_name, "fallback_used": encoder.fallback_used, "prepared_root": str(PREPARED_ROOT)}
pd.DataFrame(logs).to_csv(run_dir / "train_log.csv", index=False)
(run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
torch.save({"model_state_dict": model.state_dict(), "config": cfg, "normalization": stats, "global_step": global_step}, run_dir / "checkpoint_last.pt")
summary = {"status": "completed", "run_dir": str(run_dir), "full_epoch_completed": MAX_TRAIN_SAMPLES is None and MAX_STEPS_PER_EPOCH is None,
           "global_steps": int(global_step), "train_samples": int(len(train_ds)), "last_loss": logs[-1]["loss"] if logs else None,
           "elapsed_sec": round(time.time() - start, 2), "acceptance": {"action_horizon": ACTION_HORIZON, "state_horizon": STATE_HORIZON,
           "uses_vl_features": True, "cross_attention_dim": FEATURE_DIM, "masked_loss": True, "target_velocity_sign": "action_minus_noise"}}
(run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(run_dir / "encode_report.json").write_text(json.dumps({"train": train_enc_report, "test": test_enc_report, "encoder_name": encoder.encoder_name, "fallback_used": encoder.fallback_used}, indent=2), encoding="utf-8")
if logs:
    df = pd.DataFrame(logs); plt.figure(figsize=(8,4)); plt.plot(df.global_step, df.loss); plt.grid(True, alpha=.3); plt.xlabel("step"); plt.ylabel("masked flow loss"); plt.tight_layout(); plt.savefig(run_dir / "loss_curve.png", dpi=160); plt.show()
print(json.dumps(summary, indent=2))
for p in sorted(run_dir.iterdir()): print("-", p.name, round(p.stat().st_size / (1024 ** 2), 3), "MB")
'''),
]


NOTEBOOK_04 = [
md("# 04 - Official Denormalized Open-Loop Evaluation\n\nK-step Euler ODE inference. MSE/MAE/RMSE tính trên **raw denormalized action space**."),
code("!pip install -q pandas pyarrow numpy tqdm matplotlib"),
code(r'''
from pathlib import Path
import json, math, time
import numpy as np, pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE_TEST = False; ACTION_HORIZON = 16; STATE_HORIZON = 1; K = 4
MAX_EVAL_SAMPLES = 2048 if SMOKE_TEST else None
BATCH_SIZE = 512; NUM_WORKERS = 2; USE_AMP = True
OUTPUT_ROOT = Path("/kaggle/working/gr00t_official_eval"); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
'''),
code(r'''
def find_dir(name, required):
    cand = []
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if base.exists(): cand.extend(base.rglob(name))
    for c in sorted(cand, key=str):
        if c.is_dir() and all((c / r).exists() for r in required): return c
    raise FileNotFoundError(name)
PREPARED_ROOT = find_dir(f"gr00t_prepared_official_H{ACTION_HORIZON}", ["states_test.npy", "actions_test_chunk.npy", "action_mask_test.npy", "normalization_stats.json"])
RUN_ROOT = find_dir("official_mini_vldit_H16", ["checkpoint_last.pt", "config.json", "train_summary.json"])
FEATURE_ROOT = RUN_ROOT.parent
ckpt = torch.load(RUN_ROOT / "checkpoint_last.pt", map_location=device); cfg = ckpt["config"]; stats = ckpt["normalization"]
print("PREPARED_ROOT:", PREPARED_ROOT); print("RUN_ROOT:", RUN_ROOT)
'''),
code(COMMON_MODEL + r'''
model = OfficialMiniVLDiT(44, 44, cfg["state_horizon"], cfg["action_horizon"], cfg["vl_feature_dim"], cfg["hidden_dim"], cfg["num_layers"], cfg["num_heads"], cfg["dropout"]).to(device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
'''),
code(r'''
class EvalDS(Dataset):
    def __init__(self):
        self.states = np.load(PREPARED_ROOT / "states_test.npy", mmap_mode="r")
        self.actions = np.load(PREPARED_ROOT / "actions_test_chunk.npy", mmap_mode="r")
        self.masks = np.load(PREPARED_ROOT / "action_mask_test.npy", mmap_mode="r")
        self.features = np.load(FEATURE_ROOT / "vl_features_test.npy", mmap_mode="r")
        self.index = pd.read_parquet(FEATURE_ROOT / "vl_feature_index_test.parquet").sort_values("feature_index").reset_index(drop=True)
        if MAX_EVAL_SAMPLES is not None and len(self.index) > MAX_EVAL_SAMPLES:
            self.index = self.index.sample(n=int(MAX_EVAL_SAMPLES), random_state=42).reset_index(drop=True)
        self.sm = np.asarray(stats["state_mean"], np.float32); self.ss = np.asarray(stats["state_std"], np.float32)
        self.am = np.asarray(stats["action_mean"], np.float32); self.asd = np.asarray(stats["action_std"], np.float32)
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        r = self.index.iloc[i]; sid = int(r.sample_id); fid = int(r.feature_index)
        state = (np.asarray(self.states[sid], np.float32) - self.sm) / self.ss
        action = (np.asarray(self.actions[sid], np.float32) - self.am) / self.asd
        return {"state": torch.from_numpy(state), "action": torch.from_numpy(action), "mask": torch.from_numpy(np.asarray(self.masks[sid], np.float32)),
                "vl": torch.from_numpy(np.asarray(self.features[fid], np.float32)), "subset": r.subset}
def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in ["state","action","mask","vl"]} | {"subset": [b["subset"] for b in batch]}
ds = EvalDS(); loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), collate_fn=collate)
am = torch.tensor(ds.am, device=device)[None,None,:]; ast = torch.tensor(ds.asd, device=device)[None,None,:]
sm = torch.tensor(ds.sm, device=device)[None,None,:]; ss = torch.tensor(ds.ss, device=device)[None,None,:]
print("eval samples:", len(ds))
'''),
code(r'''
@torch.no_grad()
def sample(state, vl, steps):
    x = torch.randn(state.size(0), ACTION_HORIZON, 44, device=state.device, dtype=state.dtype)
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((state.size(0),), k / steps, device=state.device, dtype=state.dtype)
        x = x + dt * model(x, state, vl, t)
    return x

def bucket(): return {"sq":0.0, "ab":0.0, "count":0, "samples":0}
def add(b, sq, ab, m, samples): b["sq"] += float(sq.sum()); b["ab"] += float(ab.sum()); b["count"] += int(m.sum()); b["samples"] += int(samples)
def fin(b):
    mse = b["sq"] / max(b["count"], 1); mae = b["ab"] / max(b["count"], 1)
    return {"samples": int(b["samples"]), "values": int(b["count"]), "raw_mse": mse, "raw_mae": mae, "raw_rmse": math.sqrt(mse)}

overall = bucket(); baseline_mean = bucket(); baseline_last_state = bucket()
per_subset = {}; per_h = [bucket() for _ in range(ACTION_HORIZON)]; per_dim = [bucket() for _ in range(44)]
flow_sum = 0.0; flow_batches = 0; preview_p = []; preview_t = []; start = time.time()
for b in tqdm(loader, desc="eval"):
    state = b["state"].to(device); action = b["action"].to(device); mask = b["mask"].to(device); vl = b["vl"].to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):
        pred = sample(state, vl, K)
        t = torch.rand(action.size(0), device=device, dtype=action.dtype); noise = torch.randn_like(action); target = action - noise
        flow = ((model((1-t[:,None,None])*noise+t[:,None,None]*action, state, vl, t) - target).pow(2) * mask).sum() / (mask.sum()+1e-6)
    pred_raw = pred.float() * ast + am; target_raw = action.float() * ast + am
    diff = pred_raw - target_raw; sq = (diff.square() * mask).cpu().numpy(); ab = (diff.abs() * mask).cpu().numpy(); m = mask.cpu().numpy()
    add(overall, sq, ab, m, action.size(0))
    # Baseline 1: action trung binh train split trong raw space.
    mean_diff = am.expand_as(target_raw) - target_raw
    add(baseline_mean, (mean_diff.square() * mask).cpu().numpy(), (mean_diff.abs() * mask).cpu().numpy(), m, action.size(0))
    # Baseline 2: lap lai state cuoi cung lam action du doan. Chi dung de so sanh tham khao vi state/action cung 44 dim trong GR-1 subset.
    state_raw = state.float() * ss + sm
    last_state_pred = state_raw[:, -1:, :].expand_as(target_raw)
    state_diff = last_state_pred - target_raw
    add(baseline_last_state, (state_diff.square() * mask).cpu().numpy(), (state_diff.abs() * mask).cpu().numpy(), m, action.size(0))
    for subset in set(b["subset"]):
        ids = [i for i, s in enumerate(b["subset"]) if s == subset]
        add(per_subset.setdefault(subset, bucket()), sq[ids], ab[ids], m[ids], len(ids))
    for h in range(ACTION_HORIZON): add(per_h[h], sq[:,h,:], ab[:,h,:], m[:,h,:], action.size(0))
    for d in range(44): add(per_dim[d], sq[:,:,d], ab[:,:,d], m[:,:,d], action.size(0))
    flow_sum += float(flow.cpu()); flow_batches += 1
    if len(preview_p) < 512:
        take = min(512-len(preview_p), pred_raw.size(0)); preview_p += pred_raw[:take,0,0].cpu().tolist(); preview_t += target_raw[:take,0,0].cpu().tolist()

metrics = fin(overall); metrics["flow_matching_test_loss_normalized"] = flow_sum / max(flow_batches, 1); metrics["denoising_steps"] = K
compare_df = pd.DataFrame([
    {"model":"baseline_train_action_mean", "input":"none", "S":STATE_HORIZON, "H":ACTION_HORIZON, "K":0, **fin(baseline_mean)},
    {"model":"baseline_last_state_repeat", "input":"state", "S":STATE_HORIZON, "H":ACTION_HORIZON, "K":0, **fin(baseline_last_state)},
    {"model":"OfficialMiniVLDiT", "input":"state+video_language", "S":STATE_HORIZON, "H":ACTION_HORIZON, "K":K, **metrics},
])
subset_df = pd.DataFrame([{"subset": k, **fin(v)} for k, v in sorted(per_subset.items())])
h_df = pd.DataFrame([{"horizon_step": i, **fin(v)} for i, v in enumerate(per_h)])
dim_df = pd.DataFrame([{"action_dim": i, **fin(v)} for i, v in enumerate(per_dim)])
display(compare_df); display(subset_df); display(h_df.head())
'''),
code(r'''
compare_df.to_csv(OUTPUT_ROOT/"eval_metrics.csv", index=False)
compare_df.to_csv(OUTPUT_ROOT/"baseline_vs_official_mini.csv", index=False)
subset_df.to_csv(OUTPUT_ROOT/"per_subset_metrics.csv", index=False); h_df.to_csv(OUTPUT_ROOT/"per_horizon_metrics.csv", index=False); dim_df.to_csv(OUTPUT_ROOT/"per_action_dim_metrics.csv", index=False)
plt.figure(figsize=(5,5)); plt.scatter(preview_t, preview_p, s=7, alpha=.35); mn=min(preview_t+preview_p); mx=max(preview_t+preview_p); plt.plot([mn,mx],[mn,mx],color="black"); plt.xlabel("GT raw action dim0"); plt.ylabel("Pred raw action dim0"); plt.tight_layout(); plt.savefig(OUTPUT_ROOT/"prediction_vs_ground_truth_raw.png", dpi=160); plt.show()
plt.figure(figsize=(7,4)); plt.plot(h_df.horizon_step, h_df.raw_mse, marker="o"); plt.xlabel("horizon step"); plt.ylabel("raw MSE"); plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig(OUTPUT_ROOT/"horizon_error_curve_raw.png", dpi=160); plt.show()
summary = {"status":"completed", "eval_samples": int(metrics["samples"]), "metrics": metrics, "comparison_table": compare_df.to_dict("records"), "denormalized_raw_action_metrics": True, "uses_action_mask": True, "elapsed_sec": round(time.time()-start, 2)}
(OUTPUT_ROOT/"eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
'''),
]


NOTEBOOK_05 = [
md("# 05 - Strong Ablation Official Mini\n\nRequired: component, model, K inference, LR, and H=8 vs H=16 if both prepared roots are available."),
code("!pip install -q pandas pyarrow numpy tqdm matplotlib"),
code(r'''
from pathlib import Path
import json, math, time, random
import numpy as np, pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Beta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
SMOKE_TEST=False; BATCH_SIZE=256; EVAL_BATCH_SIZE=512; EPOCHS=1; NUM_WORKERS=2; USE_AMP=True
MAX_TRAIN_SAMPLES=2048 if SMOKE_TEST else None; MAX_EVAL_SAMPLES=1024 if SMOKE_TEST else None; MAX_STEPS_PER_EPOCH=20 if SMOKE_TEST else None
OUTPUT_ROOT=Path("/kaggle/working/gr00t_official_ablation"); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RUNS=[
 {"run":"state_vl_layers8_lr1e4","type":"component/model/hyperparam","use_vl":True,"layers":8,"hidden":512,"lr":1e-4,"H":16},
 {"run":"state_only_layers8_lr1e4","type":"component","use_vl":False,"layers":8,"hidden":512,"lr":1e-4,"H":16},
 {"run":"state_vl_layers4_lr1e4","type":"model","use_vl":True,"layers":4,"hidden":512,"lr":1e-4,"H":16},
 {"run":"state_vl_layers8_lr5e5","type":"hyperparam","use_vl":True,"layers":8,"hidden":512,"lr":5e-5,"H":16},
 {"run":"state_vl_H8_lr1e4","type":"horizon","use_vl":True,"layers":8,"hidden":512,"lr":1e-4,"H":8},
]
K_VALUES=[1,4,8,16]
'''),
code(COMMON_MODEL),
code(r'''
def find_dir(name, required):
    cand=[]
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if base.exists(): cand.extend(base.rglob(name))
    for c in sorted(cand, key=str):
        if c.is_dir() and all((c/r).exists() for r in required): return c
    return None
def root(H): return find_dir(f"gr00t_prepared_official_H{H}", ["states_train.npy","actions_train_chunk.npy","action_mask_train.npy","normalization_stats.json"])
def feature_root(H): return find_dir(f"official_mini_vldit_H{H}", ["config.json"])
print("roots:", {h: root(h) for h in [8,16]})
'''),
code(r'''
class DS(Dataset):
    def __init__(self, prepared, fr, split, H, use_vl, max_samples):
        self.H=H; self.use_vl=use_vl; self.prepared=prepared
        self.states=np.load(prepared/f"states_{split}.npy", mmap_mode="r"); self.actions=np.load(prepared/f"actions_{split}_chunk.npy", mmap_mode="r"); self.masks=np.load(prepared/f"action_mask_{split}.npy", mmap_mode="r")
        self.stats=json.loads((prepared/"normalization_stats.json").read_text()); self.sm=np.asarray(self.stats["state_mean"],np.float32); self.ss=np.asarray(self.stats["state_std"],np.float32); self.am=np.asarray(self.stats["action_mean"],np.float32); self.asd=np.asarray(self.stats["action_std"],np.float32)
        if use_vl:
            if fr is None or not (fr.parent/f"vl_features_{split}.npy").exists(): raise FileNotFoundError(f"Missing VL features for H={H}. Run Notebook 03 for this H.")
            self.features=np.load(fr.parent/f"vl_features_{split}.npy",mmap_mode="r"); self.index=pd.read_parquet(fr.parent/f"vl_feature_index_{split}.parquet").sort_values("feature_index").reset_index(drop=True); self.vl_dim=int(self.features.shape[-1])
        else:
            samples=pd.read_parquet(prepared/f"{split}_samples.parquet"); self.index=pd.DataFrame({"sample_id":samples.sample_id.values}); self.vl_dim=1
        if max_samples is not None and len(self.index)>max_samples: self.index=self.index.sample(n=int(max_samples),random_state=SEED).reset_index(drop=True)
    def __len__(self): return len(self.index)
    def __getitem__(self,i):
        r=self.index.iloc[i]; sid=int(r.sample_id); state=(np.asarray(self.states[sid],np.float32)-self.sm)/self.ss; action=(np.asarray(self.actions[sid],np.float32)-self.am)/self.asd; mask=np.asarray(self.masks[sid],np.float32)
        vl=np.asarray(self.features[int(r.feature_index)],np.float32) if self.use_vl else np.zeros((1,1),np.float32)
        return {"state":torch.from_numpy(state),"action":torch.from_numpy(action),"mask":torch.from_numpy(mask),"vl":torch.from_numpy(vl)}
def collate(batch): return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
'''),
code(r'''
def train_eval(run):
    H=run["H"]; pr=root(H); fr=feature_root(H)
    if pr is None: return {"run":run["run"],"status":"skipped","reason":f"missing prepared H{H}"}
    try:
        tr=DS(pr,fr,"train",H,run["use_vl"],MAX_TRAIN_SAMPLES); ev=DS(pr,fr,"test",H,run["use_vl"],MAX_EVAL_SAMPLES)
    except Exception as exc:
        return {"run":run["run"],"status":"skipped","reason":str(exc)}
    tl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available(),drop_last=True,collate_fn=collate); el=DataLoader(ev,batch_size=EVAL_BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available(),collate_fn=collate)
    # Moi ablation run khoi tao model moi tu dau de so sanh cong bang.
    model=OfficialMiniVLDiT(44,44,1,H,tr.vl_dim,run["hidden"],run["layers"],8,.1).to(device); opt=torch.optim.AdamW(model.parameters(),lr=run["lr"],weight_decay=1e-5,betas=(.95,.999)); scaler=torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available()); beta=Beta(torch.tensor(1.5,device=device),torch.tensor(1.0,device=device))
    losses=[]; start=time.time(); steps=0; model.train(); max_steps=len(tl) if MAX_STEPS_PER_EPOCH is None else min(MAX_STEPS_PER_EPOCH,len(tl))
    for i,b in enumerate(tqdm(tl,total=max_steps,desc=run["run"])):
        if i>=max_steps: break
        state=b["state"].to(device); action=b["action"].to(device); mask=b["mask"].to(device); vl=b["vl"].to(device); t=beta.sample((action.size(0),)).to(device=device,dtype=action.dtype); noise=torch.randn_like(action); noisy=(1-t[:,None,None])*noise+t[:,None,None]*action; target=action-noise
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):
            pred=model(noisy,state,vl,t); loss=((pred-target).pow(2)*mask).sum()/(mask.sum()+1e-6)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu())); steps+=1
    def sample(state,vl,K):
        x=torch.randn(state.size(0),H,44,device=state.device,dtype=state.dtype); dt=1.0/K
        for k in range(K): x=x+dt*model(x,state,vl,torch.full((state.size(0),),k/K,device=state.device,dtype=state.dtype))
        return x
    am=torch.tensor(ev.am,device=device)[None,None,:]; ast=torch.tensor(ev.asd,device=device)[None,None,:]; k_rows=[]; model.eval()
    with torch.no_grad():
        # K-step Euler inference: doi K nhung giu nguyen checkpoint de do trade-off toc do/chat luong.
        for K in K_VALUES:
            sq=ab=count=samples=0
            for b in tqdm(el,desc=f"eval {run['run']} K={K}"):
                state=b["state"].to(device); action=b["action"].to(device); mask=b["mask"].to(device); vl=b["vl"].to(device)
                pred=sample(state,vl,K); diff=(pred.float()*ast+am)-(action.float()*ast+am)
                sq+=float((diff.square()*mask).sum().cpu()); ab+=float((diff.abs()*mask).sum().cpu()); count+=int(mask.sum().cpu()); samples+=int(action.size(0))
            mse=sq/max(count,1); k_rows.append({"K":K,"raw_mse":mse,"raw_mae":ab/max(count,1),"raw_rmse":math.sqrt(mse),"eval_samples":samples})
    best=next(x for x in k_rows if x["K"]==4)
    return {"run":run["run"],"status":"completed","type":run["type"],"use_vl":run["use_vl"],"H":H,"layers":run["layers"],"hidden_dim":run["hidden"],"lr":run["lr"],"global_steps":steps,"train_last_loss":losses[-1] if losses else None,"train_elapsed_sec":round(time.time()-start,2),**best,"k_results":k_rows}

results=[]
for run in RUNS:
    res=train_eval(run); print(res); results.append(res)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
'''),
code(r'''
summary_df=pd.DataFrame([{k:v for k,v in r.items() if k!="k_results"} for r in results])
summary_df.to_csv(OUTPUT_ROOT/"ablation_summary.csv",index=False)
k_rows=[]
for r in results:
    if r.get("status")=="completed":
        for kr in r["k_results"]: k_rows.append({"run":r["run"],**kr})
pd.DataFrame(k_rows).to_csv(OUTPUT_ROOT/"k_step_tradeoff.csv",index=False)
display(summary_df)
completed=summary_df[summary_df.status=="completed"] if "status" in summary_df else pd.DataFrame()
if len(completed):
    plt.figure(figsize=(9,4)); plt.bar(completed.run, completed.raw_mse); plt.xticks(rotation=30, ha="right"); plt.ylabel("Raw MSE @K=4"); plt.tight_layout(); plt.savefig(OUTPUT_ROOT/"ablation_bar_chart.png",dpi=160); plt.show()
summary={"status":"completed","num_runs":len(results),"num_completed":int(len(completed)),"required_ablation_types":["component","model","inference_K","hyperparam","horizon_H8_vs_H16"],"metrics_space":"denormalized_raw_action_space","uses_action_mask":True,"results":results}
(OUTPUT_ROOT/"ablation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps({"num_completed":len(completed),"output_root":str(OUTPUT_ROOT)}, indent=2))
'''),
]


write("02_prepare/02_prepare_official_sequence_data.ipynb", NOTEBOOK_02)
write("03_train/03_encode_vlm_and_train_official_mini_vldit.ipynb", NOTEBOOK_03)
write("04_eval/04_evaluate_official_mini_denormalized.ipynb", NOTEBOOK_04)
write("05_ablation/05_strong_ablation_official_mini.ipynb", NOTEBOOK_05)
print("Generated notebooks:")
for p in sorted(OUT.rglob("*.ipynb")):
    print("-", p.name, p.stat().st_size)

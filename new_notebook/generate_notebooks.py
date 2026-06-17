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


NOTEBOOK_03 = json.loads('[{"cell_type": "markdown", "id": "03-encode-vlm-and-train-official-mini-vl-01", "metadata": {}, "source": "# 03 - Frozen VLM Encode + Pretrain/Posttrain OfficialMiniVLDiT\n\nTwo-stage official-mini training:\n\n1. Frozen VLM/Eagle/SigLIP encoding to cached `vl_features`.\n2. Pretrain a random-init DiT/action head on all prepared training samples.\n3. Posttrain the same DiT/action head on a task-specific subset, matching the teacher\'s requested pretrain + posttrain flow.\n", "outputs": [], "execution_count": null}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-02", "metadata": {}, "outputs": [], "source": "!pip install -q pandas pyarrow numpy tqdm matplotlib pillow transformers accelerate av opencv-python"}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-03", "metadata": {}, "outputs": [], "source": "from pathlib import Path\nimport json, random, time\nimport numpy as np, pandas as pd\nfrom PIL import Image\nfrom tqdm.auto import tqdm\nimport matplotlib.pyplot as plt\nimport torch\nfrom torch import nn\nfrom torch.utils.data import Dataset, DataLoader\nimport torch.nn.functional as F\nfrom torch.distributions import Beta\n\nSEED = 42\nrandom.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)\ndevice = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\nprint(\"device:\", device)\nif torch.cuda.is_available(): print(\"gpu:\", torch.cuda.get_device_name(0))\n\nSMOKE_TEST = False\nACTION_HORIZON = 16\nSTATE_HORIZON = 1\nMAX_ENCODE_TRAIN_SAMPLES = 512 if SMOKE_TEST else None\nMAX_ENCODE_TEST_SAMPLES = 256 if SMOKE_TEST else None\nMAX_TRAIN_SAMPLES = 2048 if SMOKE_TEST else None\nPOSTTRAIN_MAX_SAMPLES = 512 if SMOKE_TEST else None\nMAX_STEPS_PER_EPOCH = 20 if SMOKE_TEST else None\nEPOCHS_PRETRAIN = 1\nEPOCHS_POSTTRAIN = 1\nRUN_PRETRAIN = True\nRUN_POSTTRAIN = True\nPOSTTRAIN_SUBSETS = [\"gr1_arms_only.CanSort\"]\nBATCH_SIZE = 256; NUM_WORKERS = 2\nLR = 1e-4; POSTTRAIN_LR = 5e-5; WEIGHT_DECAY = 1e-5; USE_AMP = True\nHIDDEN_DIM = 512; NUM_LAYERS = 8; NUM_HEADS = 8; DROPOUT = 0.1\nPRIMARY_ENCODER = \"nvidia/Eagle-Block2A-2B-v2\"\nFALLBACK_ENCODER = \"google/siglip-base-patch16-224\"\nFORCE_FALLBACK = False\nOUTPUT_ROOT = Path(\"/kaggle/working/gr00t_official_mini_runs\"); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\nRUN_NAME = f\"official_mini_vldit_H{ACTION_HORIZON}\"\n"}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-04", "metadata": {}, "outputs": [], "source": "def find_dir(name, required):\n    cand = []\n    for base in [Path(\"/kaggle/input\"), Path(\"/kaggle/working\")]:\n        if base.exists(): cand.extend(base.rglob(name))\n    for c in sorted(cand, key=str):\n        if c.is_dir() and all((c / r).exists() for r in required): return c\n    raise FileNotFoundError(f\"Không tìm thấy {name}\")\n\nPREPARED_ROOT = find_dir(f\"gr00t_prepared_official_H{ACTION_HORIZON}\", [\"states_train.npy\", \"actions_train_chunk.npy\", \"action_mask_train.npy\", \"video_frame_manifest_train.parquet\", \"normalization_stats.json\", \"prepare_report.json\"])\nstats = json.loads((PREPARED_ROOT / \"normalization_stats.json\").read_text())\nprepare_report = json.loads((PREPARED_ROOT / \"prepare_report.json\").read_text())\nprint(\"PREPARED_ROOT:\", PREPARED_ROOT)\nprint(json.dumps({k: prepare_report[k] for k in [\"state_horizon\", \"action_horizon\", \"train_samples\", \"test_samples\"]}, indent=2))"}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-05", "metadata": {}, "outputs": [], "source": "def decode_frame_pyav(video_path, frame_index):\n    import av\n    container = av.open(video_path)\n    try:\n        stream = container.streams.video[0]\n        last = None\n        for i, frame in enumerate(container.decode(stream)):\n            last = frame\n            if i >= frame_index:\n                return frame.to_image().convert(\"RGB\")\n        if last is None: raise RuntimeError(\"empty video\")\n        return last.to_image().convert(\"RGB\")\n    finally:\n        container.close()\n\ndef decode_frame_cv2(video_path, frame_index):\n    import cv2\n    cap = cv2.VideoCapture(video_path)\n    if not cap.isOpened(): raise RuntimeError(\"cv2 cannot open video\")\n    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))\n    ok, frame = cap.read()\n    if not ok:\n        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1)); ok, frame = cap.read()\n    cap.release()\n    if not ok: raise RuntimeError(\"cv2 cannot decode frame\")\n    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\n\ndef decode_frame(video_path, frame_index):\n    # Uu tien PyAV vi doc video on dinh; neu loi thi fallback sang OpenCV.\n    try: return decode_frame_pyav(video_path, int(frame_index))\n    except Exception: return decode_frame_cv2(video_path, int(frame_index))"}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-06", "metadata": {}, "outputs": [], "source": "from transformers import AutoModel, AutoProcessor\n\nclass FrozenVLEncoder:\n    def __init__(self, primary_id, fallback_id, force_fallback=False):\n        self.primary_id = primary_id; self.fallback_id = fallback_id; self.force_fallback = force_fallback\n        self.encoder_name = None; self.fallback_used = False; self.processor = None; self.model = None\n        self.feature_dim = None; self.num_tokens = 2\n        self.load()\n    def try_load(self, model_id):\n        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)\n        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, low_cpu_mem_usage=True,\n                                           torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)\n        model.eval().to(device)\n        for p in model.parameters(): p.requires_grad_(False)\n        return proc, model\n    def load(self):\n        if not self.force_fallback:\n            try:\n                self.processor, self.model = self.try_load(self.primary_id)\n                self.encoder_name = self.primary_id\n                print(\"Loaded primary encoder:\", self.encoder_name)\n                return\n            except Exception as exc:\n                print(\"WARN primary failed, fallback:\", repr(exc))\n        self.processor, self.model = self.try_load(self.fallback_id)\n        self.encoder_name = self.fallback_id; self.fallback_used = True\n        print(\"Loaded fallback encoder:\", self.encoder_name)\n    @torch.no_grad()\n    def encode_batch(self, images, texts):\n        # Encoder duoc freeze hoan toan; Notebook 03 chi train DiT/action head tu scratch.\n        inputs = self.processor(text=texts, images=images, padding=True, return_tensors=\"pt\")\n        inputs = {k: v.to(device) if hasattr(v, \"to\") else v for k, v in inputs.items()}\n        with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):\n            if hasattr(self.model, \"get_image_features\") and hasattr(self.model, \"get_text_features\"):\n                image_feat = self.model.get_image_features(pixel_values=inputs[\"pixel_values\"])\n                text_keys = {k: v for k, v in inputs.items() if k in [\"input_ids\", \"attention_mask\"]}\n                text_feat = self.model.get_text_features(**text_keys) if text_keys else torch.zeros_like(image_feat)\n                feats = torch.stack([F.normalize(image_feat.float(), dim=-1), F.normalize(text_feat.float(), dim=-1)], 1)\n            else:\n                out = self.model(**inputs, output_hidden_states=True, return_dict=True)\n                hidden = getattr(out, \"last_hidden_state\", None)\n                if hidden is None:\n                    hidden = out.hidden_states[-1]\n                pooled = hidden.float().mean(1)\n                feats = torch.stack([pooled, pooled], 1)\n        self.feature_dim = int(feats.shape[-1]); self.num_tokens = int(feats.shape[1])\n        return feats.cpu().numpy().astype(np.float16)"}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-07", "metadata": {}, "outputs": [], "source": "def select_manifest(split, max_samples):\n    df = pd.read_parquet(PREPARED_ROOT / f\"video_frame_manifest_{split}.parquet\")\n    df = df[(df.has_video == True) & (df.video_path.astype(str).str.len() > 0)].copy()\n    if df.empty: raise RuntimeError(f\"No video-backed samples for {split}\")\n    df = df.sort_values([\"subset\", \"episode_index\", \"frame_index\"]).reset_index(drop=True)\n    if max_samples is not None and len(df) > max_samples:\n        df = df.sample(n=int(max_samples), random_state=SEED).sort_values(\"sample_id\").reset_index(drop=True)\n    return df\n\ndef encode_split(split, max_samples, encoder, batch_size=16):\n    feat_path = OUTPUT_ROOT / f\"vl_features_{split}.npy\"\n    idx_path = OUTPUT_ROOT / f\"vl_feature_index_{split}.parquet\"\n    if feat_path.exists() and idx_path.exists():\n        return feat_path, idx_path, {\"cached\": True, \"shape\": list(np.load(feat_path, mmap_mode=\"r\").shape)}\n    manifest = select_manifest(split, max_samples)\n    chunks, rows, failed = [], [], []\n    for start in tqdm(range(0, len(manifest), batch_size), desc=f\"encode {split}\"):\n        batch = manifest.iloc[start:start+batch_size]\n        images, texts, ok = [], [], []\n        for _, r in batch.iterrows():\n            try:\n                images.append(decode_frame(r.video_path, int(r.frame_index)))\n                texts.append(str(r.task_text) if str(r.task_text) else \"perform the manipulation task\")\n                ok.append(r)\n            except Exception as exc:\n                failed.append({\"sample_id\": int(r.sample_id), \"error\": str(exc)})\n        if not images: continue\n        feats = encoder.encode_batch(images, texts); base = sum(c.shape[0] for c in chunks); chunks.append(feats)\n        for j, r in enumerate(ok):\n            rows.append({\"feature_index\": int(base + j), \"sample_id\": int(r.sample_id), \"subset\": r.subset, \"episode_index\": int(r.episode_index), \"frame_index\": int(r.frame_index), \"task_text\": str(r.task_text)})\n    features = np.concatenate(chunks, 0)\n    np.save(feat_path, features); pd.DataFrame(rows).to_parquet(idx_path, index=False)\n    report = {\"split\": split, \"requested\": int(len(manifest)), \"encoded\": int(features.shape[0]), \"failed\": int(len(failed)),\n              \"feature_shape\": list(features.shape), \"encoder_name\": encoder.encoder_name, \"fallback_used\": encoder.fallback_used, \"failed_examples\": failed[:20]}\n    (OUTPUT_ROOT / f\"video_decode_report_{split}.json\").write_text(json.dumps(report, indent=2), encoding=\"utf-8\")\n    return feat_path, idx_path, report\n\nencoder = FrozenVLEncoder(PRIMARY_ENCODER, FALLBACK_ENCODER, FORCE_FALLBACK)\ntrain_feat, train_idx, train_enc_report = encode_split(\"train\", MAX_ENCODE_TRAIN_SAMPLES, encoder)\ntest_feat, test_idx, test_enc_report = encode_split(\"test\", MAX_ENCODE_TEST_SAMPLES, encoder)\nprint(train_enc_report); print(test_enc_report)"}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-08", "metadata": {}, "outputs": [], "source": "class OfficialDataset(Dataset):\n    def __init__(self, split, feat_path, idx_path, max_samples=None, subset_filter=None):\n        self.split = split\n        self.states = np.load(PREPARED_ROOT / f\"states_{split}.npy\", mmap_mode=\"r\")\n        self.actions = np.load(PREPARED_ROOT / f\"actions_{split}_chunk.npy\", mmap_mode=\"r\")\n        self.masks = np.load(PREPARED_ROOT / f\"action_mask_{split}.npy\", mmap_mode=\"r\")\n        self.features = np.load(feat_path, mmap_mode=\"r\")\n        self.index = pd.read_parquet(idx_path).sort_values(\"feature_index\").reset_index(drop=True)\n        self.requested_subset_filter = list(subset_filter or [])\n        self.actual_subset_filter = []\n        self.subset_fallback_used = False\n        if self.requested_subset_filter:\n            requested_df = self.index[self.index.subset.isin(self.requested_subset_filter)].copy()\n            if requested_df.empty:\n                counts = self.index.subset.value_counts()\n                fallback_subset = str(counts.index[0])\n                requested_df = self.index[self.index.subset == fallback_subset].copy()\n                self.actual_subset_filter = [fallback_subset]\n                self.subset_fallback_used = True\n                print(f\"WARN: requested posttrain subsets {self.requested_subset_filter} not found; fallback to {fallback_subset}\")\n            else:\n                self.actual_subset_filter = sorted(requested_df.subset.astype(str).unique().tolist())\n            self.index = requested_df.reset_index(drop=True)\n        if max_samples is not None and len(self.index) > max_samples:\n            self.index = self.index.sample(n=int(max_samples), random_state=SEED).reset_index(drop=True)\n        self.sm = np.asarray(stats[\"state_mean\"], np.float32); self.ss = np.asarray(stats[\"state_std\"], np.float32)\n        self.am = np.asarray(stats[\"action_mean\"], np.float32); self.asd = np.asarray(stats[\"action_std\"], np.float32)\n    def __len__(self): return len(self.index)\n    def __getitem__(self, i):\n        r = self.index.iloc[i]; sid = int(r.sample_id); fid = int(r.feature_index)\n        # Normalize bang thong ke train split; Notebook 04 se denormalize lai khi tinh metric raw.\n        state = (np.asarray(self.states[sid], np.float32) - self.sm) / self.ss\n        action = (np.asarray(self.actions[sid], np.float32) - self.am) / self.asd\n        return {\"state\": torch.from_numpy(state), \"action\": torch.from_numpy(action),\n                \"mask\": torch.from_numpy(np.asarray(self.masks[sid], np.float32)),\n                \"vl\": torch.from_numpy(np.asarray(self.features[fid], np.float32)), \"sample_id\": sid,\n                \"subset\": str(r.subset)}\n\ndef collate(batch):\n    out = {k: torch.stack([b[k] for b in batch]) for k in [\"state\", \"action\", \"mask\", \"vl\"]}\n    out[\"sample_id\"] = torch.tensor([b[\"sample_id\"] for b in batch])\n    out[\"subset\"] = [b[\"subset\"] for b in batch]\n    return out\n\ndef make_loader(ds, shuffle, batch_size=BATCH_SIZE):\n    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS,\n                      pin_memory=torch.cuda.is_available(), drop_last=False, collate_fn=collate)\n\npretrain_ds = OfficialDataset(\"train\", train_feat, train_idx, MAX_TRAIN_SAMPLES)\ntest_ds = OfficialDataset(\"test\", test_feat, test_idx, None)\nposttrain_ds = OfficialDataset(\"train\", train_feat, train_idx, POSTTRAIN_MAX_SAMPLES, subset_filter=POSTTRAIN_SUBSETS)\nFEATURE_DIM = int(pretrain_ds.features.shape[-1]); NUM_TOKENS = int(pretrain_ds.features.shape[1])\nprint(\"pretrain_samples:\", len(pretrain_ds), \"posttrain_samples:\", len(posttrain_ds), \"FEATURE_DIM:\", FEATURE_DIM, \"tokens:\", NUM_TOKENS)\nprint(\"posttrain_subsets:\", posttrain_ds.actual_subset_filter, \"fallback:\", posttrain_ds.subset_fallback_used)\n"}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-09", "metadata": {}, "outputs": [], "source": "class TimeEmbedding(nn.Module):\n    def __init__(self, hidden_dim, buckets=1000):\n        super().__init__()\n        self.buckets = buckets\n        self.embed = nn.Embedding(buckets, hidden_dim)\n        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))\n    def forward(self, t):\n        idx = torch.clamp((t * self.buckets).long(), 0, self.buckets - 1)\n        return self.mlp(self.embed(idx))\n\nclass CrossBlock(nn.Module):\n    def __init__(self, hidden_dim, heads, dropout):\n        super().__init__()\n        self.n1 = nn.LayerNorm(hidden_dim)\n        self.sa = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)\n        self.n2 = nn.LayerNorm(hidden_dim)\n        self.ca = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)\n        self.n3 = nn.LayerNorm(hidden_dim)\n        self.ff = nn.Sequential(\n            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),\n            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout)\n        )\n    def forward(self, x, mem):\n        y = self.n1(x)\n        x = x + self.sa(y, y, y, need_weights=False)[0]\n        y = self.n2(x)\n        x = x + self.ca(y, mem, mem, need_weights=False)[0]\n        return x + self.ff(self.n3(x))\n\nclass OfficialMiniVLDiT(nn.Module):\n    def __init__(self, state_dim, action_dim, state_horizon, action_horizon, vl_dim,\n                 hidden_dim=512, layers=8, heads=8, dropout=0.1, buckets=1000):\n        super().__init__()\n        self.state_horizon = state_horizon\n        self.action_horizon = action_horizon\n        self.state_proj = nn.Linear(state_dim, hidden_dim)\n        self.action_proj = nn.Linear(action_dim, hidden_dim)\n        self.vl_proj = nn.Linear(vl_dim, hidden_dim)\n        self.time = TimeEmbedding(hidden_dim, buckets)\n        self.state_pos = nn.Embedding(state_horizon, hidden_dim)\n        self.action_pos = nn.Embedding(action_horizon, hidden_dim)\n        self.blocks = nn.ModuleList([CrossBlock(hidden_dim, heads, dropout) for _ in range(layers)])\n        self.norm = nn.LayerNorm(hidden_dim)\n        self.out = nn.Linear(hidden_dim, action_dim)\n    def forward(self, noisy_action, state_history, vl_features, t):\n        s_pos = self.state_pos(torch.arange(self.state_horizon, device=noisy_action.device))[None]\n        a_pos = self.action_pos(torch.arange(self.action_horizon, device=noisy_action.device))[None]\n        s = self.state_proj(state_history) + s_pos\n        a = self.action_proj(noisy_action) + a_pos\n        x = torch.cat([s, a], dim=1) + self.time(t)[:, None, :]\n        mem = self.vl_proj(vl_features)\n        for block in self.blocks:\n            x = block(x, mem)\n        return self.out(self.norm(x[:, self.state_horizon:]))\n\nmodel = OfficialMiniVLDiT(44, 44, STATE_HORIZON, ACTION_HORIZON, FEATURE_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT).to(device)\nnum_params = sum(p.numel() for p in model.parameters())\nprint(\"params:\", num_params)"}, {"cell_type": "code", "execution_count": null, "id": "03-encode-vlm-and-train-official-mini-vl-10", "metadata": {}, "outputs": [], "source": "run_dir = OUTPUT_ROOT / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)\nbase_cfg = {\"model_name\": \"OfficialMiniVLDiT\", \"state_horizon\": STATE_HORIZON, \"action_horizon\": ACTION_HORIZON,\n       \"state_dim\": 44, \"action_dim\": 44, \"vl_feature_dim\": FEATURE_DIM, \"vl_num_tokens\": NUM_TOKENS,\n       \"hidden_dim\": HIDDEN_DIM, \"num_layers\": NUM_LAYERS, \"num_heads\": NUM_HEADS,\n       \"dropout\": DROPOUT, \"num_timestep_buckets\": 1000, \"timestep_sampling\": \"Beta(1.5,1.0)\",\n       \"target_velocity_sign\": \"action_minus_noise\", \"masked_loss\": True, \"learning_rate\": LR,\n       \"posttrain_learning_rate\": POSTTRAIN_LR, \"batch_size\": BATCH_SIZE, \"epochs_pretrain\": EPOCHS_PRETRAIN,\n       \"epochs_posttrain\": EPOCHS_POSTTRAIN, \"encoder_name\": encoder.encoder_name, \"fallback_used\": encoder.fallback_used,\n       \"prepared_root\": str(PREPARED_ROOT), \"teacher_requirement_alignment\": \"pretrain_plus_posttrain_dit_head\"}\n\nmodel = OfficialMiniVLDiT(44, 44, STATE_HORIZON, ACTION_HORIZON, FEATURE_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT).to(device)\nnum_params = sum(p.numel() for p in model.parameters())\nbase_cfg[\"num_params\"] = int(num_params)\nprint(\"params:\", num_params)\n\nbeta = Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))\n\ndef run_training_stage(stage_name, dataset, epochs, lr, model, checkpoint_name, log_name):\n    if epochs <= 0 or len(dataset) == 0:\n        return {\"stage\": stage_name, \"completed\": False, \"reason\": \"empty_dataset_or_zero_epochs\", \"samples\": int(len(dataset)), \"global_steps\": 0}, []\n    loader = make_loader(dataset, shuffle=True)\n    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, betas=(0.95, 0.999), eps=1e-8)\n    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available())\n    logs, global_step, start = [], 0, time.time()\n    model.train()\n    for epoch in range(1, epochs + 1):\n        max_steps = len(loader) if MAX_STEPS_PER_EPOCH is None else min(MAX_STEPS_PER_EPOCH, len(loader))\n        for step, b in enumerate(tqdm(loader, total=max_steps, desc=f\"{stage_name} epoch {epoch}/{epochs}\"), 1):\n            if step > max_steps: break\n            state = b[\"state\"].to(device); action = b[\"action\"].to(device); mask = b[\"mask\"].to(device); vl = b[\"vl\"].to(device)\n            t = beta.sample((action.size(0),)).to(device=device, dtype=action.dtype)\n            noise = torch.randn_like(action)\n            noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action\n            # Flow matching dung huong official: velocity = action - noise.\n            target = action - noise\n            opt.zero_grad(set_to_none=True)\n            with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):\n                pred = model(noisy, state, vl, t)\n                loss = ((pred - target).pow(2) * mask).sum() / (mask.sum() + 1e-6)\n            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()\n            global_step += 1\n            if global_step == 1 or global_step % 100 == 0:\n                logs.append({\"stage\": stage_name, \"epoch\": epoch, \"step\": step, \"global_step\": global_step,\n                             \"loss\": float(loss.detach().cpu()), \"elapsed_sec\": round(time.time() - start, 2)})\n    pd.DataFrame(logs).to_csv(run_dir / log_name, index=False)\n    stage_cfg = dict(base_cfg); stage_cfg.update({\"stage\": stage_name, \"learning_rate\": lr})\n    torch.save({\"model_state_dict\": model.state_dict(), \"config\": stage_cfg, \"normalization\": stats,\n                \"global_step\": global_step, \"stage\": stage_name}, run_dir / checkpoint_name)\n    summary = {\"stage\": stage_name, \"completed\": global_step > 0, \"samples\": int(len(dataset)), \"global_steps\": int(global_step),\n               \"last_loss\": logs[-1][\"loss\"] if logs else None, \"elapsed_sec\": round(time.time() - start, 2),\n               \"checkpoint\": checkpoint_name, \"log\": log_name}\n    return summary, logs\n\nall_logs = []\npretrain_summary = {\"stage\": \"pretrain\", \"completed\": False, \"reason\": \"RUN_PRETRAIN_FALSE\", \"samples\": int(len(pretrain_ds)), \"global_steps\": 0}\nposttrain_summary = {\"stage\": \"posttrain\", \"completed\": False, \"reason\": \"RUN_POSTTRAIN_FALSE\", \"samples\": int(len(posttrain_ds)), \"global_steps\": 0}\n\nif RUN_PRETRAIN:\n    pretrain_summary, logs = run_training_stage(\"pretrain\", pretrain_ds, EPOCHS_PRETRAIN, LR, model, \"pretrain_checkpoint.pt\", \"pretrain_log.csv\")\n    all_logs.extend(logs)\n\nif RUN_POSTTRAIN and pretrain_summary.get(\"completed\"):\n    posttrain_summary, logs = run_training_stage(\"posttrain\", posttrain_ds, EPOCHS_POSTTRAIN, POSTTRAIN_LR, model, \"posttrain_checkpoint.pt\", \"posttrain_log.csv\")\n    all_logs.extend(logs)\nelif RUN_POSTTRAIN:\n    posttrain_summary.update({\"reason\": \"pretrain_not_completed\"})\n\nbest_ckpt_name = \"posttrain_checkpoint.pt\" if posttrain_summary.get(\"completed\") else \"pretrain_checkpoint.pt\"\nbest_stage = \"posttrain\" if posttrain_summary.get(\"completed\") else \"pretrain\"\nbest_ckpt = torch.load(run_dir / best_ckpt_name, map_location=\"cpu\") if (run_dir / best_ckpt_name).exists() else None\nif best_ckpt is not None:\n    torch.save(best_ckpt, run_dir / \"checkpoint_last.pt\")\n\ncfg = dict(base_cfg)\ncfg.update({\"best_stage\": best_stage, \"posttrain_requested_subsets\": POSTTRAIN_SUBSETS,\n            \"posttrain_actual_subsets\": posttrain_ds.actual_subset_filter,\n            \"posttrain_subset_fallback_used\": posttrain_ds.subset_fallback_used})\n(run_dir / \"config.json\").write_text(json.dumps(cfg, indent=2), encoding=\"utf-8\")\npd.DataFrame(all_logs).to_csv(run_dir / \"train_log.csv\", index=False)\nsummary = {\"status\": \"completed\" if pretrain_summary.get(\"completed\") else \"failed\", \"run_dir\": str(run_dir),\n           \"best_stage\": best_stage, \"best_checkpoint\": best_ckpt_name,\n           \"full_epoch_completed\": MAX_TRAIN_SAMPLES is None and MAX_STEPS_PER_EPOCH is None,\n           \"pretrain_completed\": bool(pretrain_summary.get(\"completed\")), \"posttrain_completed\": bool(posttrain_summary.get(\"completed\")),\n           \"pretrain_samples\": int(len(pretrain_ds)), \"posttrain_samples\": int(len(posttrain_ds)),\n           \"posttrain_subsets\": posttrain_ds.actual_subset_filter, \"posttrain_requested_subsets\": POSTTRAIN_SUBSETS,\n           \"posttrain_subset_fallback_used\": posttrain_ds.subset_fallback_used,\n           \"teacher_requirement_alignment\": \"pretrain_plus_posttrain_dit_head\",\n           \"pretrain\": pretrain_summary, \"posttrain\": posttrain_summary,\n           \"elapsed_sec_total\": round(sum(x.get(\"elapsed_sec\", 0) for x in [pretrain_summary, posttrain_summary]), 2),\n           \"acceptance\": {\"action_horizon\": ACTION_HORIZON, \"state_horizon\": STATE_HORIZON,\n           \"uses_vl_features\": True, \"cross_attention_dim\": FEATURE_DIM, \"masked_loss\": True,\n           \"target_velocity_sign\": \"action_minus_noise\", \"pretrain_plus_posttrain\": bool(pretrain_summary.get(\"completed\") and posttrain_summary.get(\"completed\"))}}\n(run_dir / \"train_summary.json\").write_text(json.dumps(summary, indent=2), encoding=\"utf-8\")\n(run_dir / \"encode_report.json\").write_text(json.dumps({\"train\": train_enc_report, \"test\": test_enc_report, \"encoder_name\": encoder.encoder_name, \"fallback_used\": encoder.fallback_used}, indent=2), encoding=\"utf-8\")\nif all_logs:\n    df = pd.DataFrame(all_logs); plt.figure(figsize=(8,4))\n    for stage, sdf in df.groupby(\"stage\"):\n        plt.plot(sdf.global_step, sdf.loss, label=stage)\n    plt.grid(True, alpha=.3); plt.xlabel(\"stage step\"); plt.ylabel(\"masked flow loss\"); plt.legend(); plt.tight_layout(); plt.savefig(run_dir / \"loss_curve.png\", dpi=160); plt.show()\nprint(json.dumps(summary, indent=2))\nfor p in sorted(run_dir.iterdir()): print(\"-\", p.name, round(p.stat().st_size / (1024 ** 2), 3), \"MB\")\n"}]')

NOTEBOOK_04 = json.loads('[{"cell_type": "markdown", "id": "04-evaluate-official-mini-denormalized-01", "metadata": {}, "source": "# 04 - Official Denormalized Open-Loop Evaluation\n\nK-step Euler ODE inference. MSE/MAE/RMSE t?nh tr?n **raw denormalized action space**.\n\nThis version evaluates pretrain and posttrain checkpoints, and attempts an optional-safe NVIDIA zero-shot baseline.\n", "outputs": [], "execution_count": null}, {"cell_type": "code", "execution_count": null, "id": "04-evaluate-official-mini-denormalized-02", "metadata": {}, "outputs": [], "source": "!pip install -q pandas pyarrow numpy tqdm matplotlib transformers accelerate"}, {"cell_type": "code", "execution_count": null, "id": "04-evaluate-official-mini-denormalized-03", "metadata": {}, "outputs": [], "source": "from pathlib import Path\nimport json, math, time\nimport numpy as np, pandas as pd\nfrom tqdm.auto import tqdm\nimport matplotlib.pyplot as plt\nimport torch\nfrom torch import nn\nfrom torch.utils.data import Dataset, DataLoader\n\ntry:\n    from transformers import AutoModel, AutoProcessor\nexcept Exception:\n    AutoModel = None; AutoProcessor = None\n\ndevice = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\nSMOKE_TEST = False; ACTION_HORIZON = 16; STATE_HORIZON = 1; K = 4\nMAX_EVAL_SAMPLES = 2048 if SMOKE_TEST else None\nBATCH_SIZE = 512; NUM_WORKERS = 2; USE_AMP = True\nRUN_NVIDIA_ZEROSHOT = True\nNVIDIA_MODEL_ID = \"nvidia/GR00T-N1.6-3B\"\nOUTPUT_ROOT = Path(\"/kaggle/working/gr00t_official_eval\"); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\n"}, {"cell_type": "code", "execution_count": null, "id": "04-evaluate-official-mini-denormalized-04", "metadata": {}, "outputs": [], "source": "def find_dir(name, required):\n    cand = []\n    for base in [Path(\"/kaggle/input\"), Path(\"/kaggle/working\")]:\n        if base.exists(): cand.extend(base.rglob(name))\n    for c in sorted(cand, key=str):\n        if c.is_dir() and all((c / r).exists() for r in required): return c\n    raise FileNotFoundError(name)\n\nPREPARED_ROOT = find_dir(f\"gr00t_prepared_official_H{ACTION_HORIZON}\", [\"states_test.npy\", \"actions_test_chunk.npy\", \"action_mask_test.npy\", \"normalization_stats.json\"])\nRUN_ROOT = find_dir(\"official_mini_vldit_H16\", [\"checkpoint_last.pt\", \"config.json\", \"train_summary.json\"])\nFEATURE_ROOT = RUN_ROOT.parent\ntrain_summary = json.loads((RUN_ROOT / \"train_summary.json\").read_text(encoding=\"utf-8\"))\nckpt_last = torch.load(RUN_ROOT / \"checkpoint_last.pt\", map_location=device)\nstats = ckpt_last[\"normalization\"]\nCHECKPOINTS = [\n    {\"model\": \"OfficialMiniVLDiT_pretrain\", \"path\": RUN_ROOT / \"pretrain_checkpoint.pt\", \"stage\": \"pretrain\"},\n    {\"model\": \"OfficialMiniVLDiT_posttrain\", \"path\": RUN_ROOT / \"posttrain_checkpoint.pt\", \"stage\": \"posttrain\"},\n]\nBEST_STAGE = train_summary.get(\"best_stage\", \"posttrain\" if (RUN_ROOT / \"posttrain_checkpoint.pt\").exists() else \"pretrain\")\nprint(\"PREPARED_ROOT:\", PREPARED_ROOT); print(\"RUN_ROOT:\", RUN_ROOT); print(\"BEST_STAGE:\", BEST_STAGE)\n"}, {"cell_type": "code", "execution_count": null, "id": "04-evaluate-official-mini-denormalized-05", "metadata": {}, "outputs": [], "source": "class TimeEmbedding(nn.Module):\n    def __init__(self, hidden_dim, buckets=1000):\n        super().__init__()\n        self.buckets = buckets\n        self.embed = nn.Embedding(buckets, hidden_dim)\n        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))\n    def forward(self, t):\n        idx = torch.clamp((t * self.buckets).long(), 0, self.buckets - 1)\n        return self.mlp(self.embed(idx))\n\nclass CrossBlock(nn.Module):\n    def __init__(self, hidden_dim, heads, dropout):\n        super().__init__()\n        self.n1 = nn.LayerNorm(hidden_dim)\n        self.sa = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)\n        self.n2 = nn.LayerNorm(hidden_dim)\n        self.ca = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)\n        self.n3 = nn.LayerNorm(hidden_dim)\n        self.ff = nn.Sequential(\n            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),\n            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout)\n        )\n    def forward(self, x, mem):\n        y = self.n1(x)\n        x = x + self.sa(y, y, y, need_weights=False)[0]\n        y = self.n2(x)\n        x = x + self.ca(y, mem, mem, need_weights=False)[0]\n        return x + self.ff(self.n3(x))\n\nclass OfficialMiniVLDiT(nn.Module):\n    def __init__(self, state_dim, action_dim, state_horizon, action_horizon, vl_dim,\n                 hidden_dim=512, layers=8, heads=8, dropout=0.1, buckets=1000):\n        super().__init__()\n        self.state_horizon = state_horizon\n        self.action_horizon = action_horizon\n        self.state_proj = nn.Linear(state_dim, hidden_dim)\n        self.action_proj = nn.Linear(action_dim, hidden_dim)\n        self.vl_proj = nn.Linear(vl_dim, hidden_dim)\n        self.time = TimeEmbedding(hidden_dim, buckets)\n        self.state_pos = nn.Embedding(state_horizon, hidden_dim)\n        self.action_pos = nn.Embedding(action_horizon, hidden_dim)\n        self.blocks = nn.ModuleList([CrossBlock(hidden_dim, heads, dropout) for _ in range(layers)])\n        self.norm = nn.LayerNorm(hidden_dim)\n        self.out = nn.Linear(hidden_dim, action_dim)\n    def forward(self, noisy_action, state_history, vl_features, t):\n        s_pos = self.state_pos(torch.arange(self.state_horizon, device=noisy_action.device))[None]\n        a_pos = self.action_pos(torch.arange(self.action_horizon, device=noisy_action.device))[None]\n        s = self.state_proj(state_history) + s_pos\n        a = self.action_proj(noisy_action) + a_pos\n        x = torch.cat([s, a], dim=1) + self.time(t)[:, None, :]\n        mem = self.vl_proj(vl_features)\n        for block in self.blocks:\n            x = block(x, mem)\n        return self.out(self.norm(x[:, self.state_horizon:]))\n\ndef load_official_mini_checkpoint(path):\n    ckpt = torch.load(path, map_location=device)\n    cfg = ckpt[\"config\"]\n    model = OfficialMiniVLDiT(44, 44, cfg[\"state_horizon\"], cfg[\"action_horizon\"], cfg[\"vl_feature_dim\"],\n                              cfg[\"hidden_dim\"], cfg[\"num_layers\"], cfg[\"num_heads\"], cfg[\"dropout\"]).to(device)\n    model.load_state_dict(ckpt[\"model_state_dict\"]); model.eval()\n    return model, cfg\n"}, {"cell_type": "code", "execution_count": null, "id": "04-evaluate-official-mini-denormalized-06", "metadata": {}, "outputs": [], "source": "class EvalDS(Dataset):\n    def __init__(self):\n        self.states = np.load(PREPARED_ROOT / \"states_test.npy\", mmap_mode=\"r\")\n        self.actions = np.load(PREPARED_ROOT / \"actions_test_chunk.npy\", mmap_mode=\"r\")\n        self.masks = np.load(PREPARED_ROOT / \"action_mask_test.npy\", mmap_mode=\"r\")\n        self.features = np.load(FEATURE_ROOT / \"vl_features_test.npy\", mmap_mode=\"r\")\n        self.index = pd.read_parquet(FEATURE_ROOT / \"vl_feature_index_test.parquet\").sort_values(\"feature_index\").reset_index(drop=True)\n        if MAX_EVAL_SAMPLES is not None and len(self.index) > MAX_EVAL_SAMPLES:\n            self.index = self.index.sample(n=int(MAX_EVAL_SAMPLES), random_state=42).reset_index(drop=True)\n        self.sm = np.asarray(stats[\"state_mean\"], np.float32); self.ss = np.asarray(stats[\"state_std\"], np.float32)\n        self.am = np.asarray(stats[\"action_mean\"], np.float32); self.asd = np.asarray(stats[\"action_std\"], np.float32)\n    def __len__(self): return len(self.index)\n    def __getitem__(self, i):\n        r = self.index.iloc[i]; sid = int(r.sample_id); fid = int(r.feature_index)\n        state = (np.asarray(self.states[sid], np.float32) - self.sm) / self.ss\n        action = (np.asarray(self.actions[sid], np.float32) - self.am) / self.asd\n        return {\"state\": torch.from_numpy(state), \"action\": torch.from_numpy(action), \"mask\": torch.from_numpy(np.asarray(self.masks[sid], np.float32)),\n                \"vl\": torch.from_numpy(np.asarray(self.features[fid], np.float32)), \"subset\": r.subset}\ndef collate(batch):\n    return {k: torch.stack([b[k] for b in batch]) for k in [\"state\",\"action\",\"mask\",\"vl\"]} | {\"subset\": [b[\"subset\"] for b in batch]}\nds = EvalDS(); loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), collate_fn=collate)\nam = torch.tensor(ds.am, device=device)[None,None,:]; ast = torch.tensor(ds.asd, device=device)[None,None,:]\nsm = torch.tensor(ds.sm, device=device)[None,None,:]; ss = torch.tensor(ds.ss, device=device)[None,None,:]\nprint(\"eval samples:\", len(ds))"}, {"cell_type": "code", "execution_count": null, "id": "04-evaluate-official-mini-denormalized-07", "metadata": {}, "outputs": [], "source": "@torch.no_grad()\ndef sample(model, state, vl, steps, action_horizon=ACTION_HORIZON):\n    x = torch.randn(state.size(0), action_horizon, 44, device=state.device, dtype=state.dtype)\n    dt = 1.0 / steps\n    for k in range(steps):\n        t = torch.full((state.size(0),), k / steps, device=state.device, dtype=state.dtype)\n        x = x + dt * model(x, state, vl, t)\n    return x\n\ndef bucket(): return {\"sq\":0.0, \"ab\":0.0, \"count\":0, \"samples\":0}\ndef add(b, sq, ab, m, samples): b[\"sq\"] += float(sq.sum()); b[\"ab\"] += float(ab.sum()); b[\"count\"] += int(m.sum()); b[\"samples\"] += int(samples)\ndef fin(b):\n    mse = b[\"sq\"] / max(b[\"count\"], 1); mae = b[\"ab\"] / max(b[\"count\"], 1)\n    return {\"samples\": int(b[\"samples\"]), \"values\": int(b[\"count\"]), \"raw_mse\": mse, \"raw_mae\": mae,\n            \"raw_rmse\": math.sqrt(mse), \"masked_raw_mse\": mse, \"masked_raw_mae\": mae, \"masked_raw_rmse\": math.sqrt(mse)}\n\ndef evaluate_simple_baselines():\n    baseline_mean = bucket(); baseline_last_state = bucket()\n    for b in tqdm(loader, desc=\"eval simple baselines\"):\n        state = b[\"state\"].to(device); action = b[\"action\"].to(device); mask = b[\"mask\"].to(device)\n        target_raw = action.float() * ast + am\n        mean_diff = am.expand_as(target_raw) - target_raw\n        add(baseline_mean, (mean_diff.square() * mask).cpu().numpy(), (mean_diff.abs() * mask).cpu().numpy(), mask.cpu().numpy(), action.size(0))\n        state_raw = state.float() * ss + sm\n        last_state_pred = state_raw[:, -1:, :].expand_as(target_raw)\n        state_diff = last_state_pred - target_raw\n        add(baseline_last_state, (state_diff.square() * mask).cpu().numpy(), (state_diff.abs() * mask).cpu().numpy(), mask.cpu().numpy(), action.size(0))\n    return [\n        {\"model\":\"baseline_train_action_mean\", \"status\":\"completed\", \"input\":\"none\", \"stage\":\"baseline\", \"S\":STATE_HORIZON, \"H\":ACTION_HORIZON, \"K\":0, \"flow_matching_test_loss_normalized\": np.nan, **fin(baseline_mean)},\n        {\"model\":\"baseline_last_state_repeat\", \"status\":\"completed\", \"input\":\"state\", \"stage\":\"baseline\", \"S\":STATE_HORIZON, \"H\":ACTION_HORIZON, \"K\":0, \"flow_matching_test_loss_normalized\": np.nan, **fin(baseline_last_state)},\n    ]\n\ndef evaluate_official_checkpoint(label, ckpt_path, stage, collect_breakdowns=False):\n    if not ckpt_path.exists():\n        return {\"model\": label, \"status\": \"skipped\", \"skip_reason\": f\"missing checkpoint: {ckpt_path.name}\", \"input\":\"state+video_language\", \"stage\":stage, \"S\":STATE_HORIZON, \"H\":ACTION_HORIZON, \"K\":K}, None\n    model, cfg = load_official_mini_checkpoint(ckpt_path)\n    overall = bucket(); per_subset = {}; per_h = [bucket() for _ in range(ACTION_HORIZON)]; per_dim = [bucket() for _ in range(44)]\n    flow_sum = 0.0; flow_batches = 0; preview_p = []; preview_t = []; start = time.time()\n    for b in tqdm(loader, desc=f\"eval {label}\"):\n        state = b[\"state\"].to(device); action = b[\"action\"].to(device); mask = b[\"mask\"].to(device); vl = b[\"vl\"].to(device)\n        with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):\n            pred = sample(model, state, vl, K, cfg[\"action_horizon\"])\n            t = torch.rand(action.size(0), device=device, dtype=action.dtype); noise = torch.randn_like(action); target = action - noise\n            flow = ((model((1-t[:,None,None])*noise+t[:,None,None]*action, state, vl, t) - target).pow(2) * mask).sum() / (mask.sum()+1e-6)\n        pred_raw = pred.float() * ast + am; target_raw = action.float() * ast + am\n        diff = pred_raw - target_raw; sq = (diff.square() * mask).cpu().numpy(); ab = (diff.abs() * mask).cpu().numpy(); m = mask.cpu().numpy()\n        add(overall, sq, ab, m, action.size(0))\n        if collect_breakdowns:\n            for subset in set(b[\"subset\"]):\n                ids = [i for i, s in enumerate(b[\"subset\"]) if s == subset]\n                add(per_subset.setdefault(subset, bucket()), sq[ids], ab[ids], m[ids], len(ids))\n            for h in range(ACTION_HORIZON): add(per_h[h], sq[:,h,:], ab[:,h,:], m[:,h,:], action.size(0))\n            for d in range(44): add(per_dim[d], sq[:,:,d], ab[:,:,d], m[:,:,d], action.size(0))\n            if len(preview_p) < 512:\n                take = min(512-len(preview_p), pred_raw.size(0)); preview_p += pred_raw[:take,0,0].cpu().tolist(); preview_t += target_raw[:take,0,0].cpu().tolist()\n        flow_sum += float(flow.cpu()); flow_batches += 1\n    row = {\"model\": label, \"status\": \"completed\", \"input\": \"state+video_language\", \"stage\": stage, \"S\":STATE_HORIZON, \"H\":ACTION_HORIZON, \"K\":K,\n           \"flow_matching_test_loss_normalized\": flow_sum / max(flow_batches, 1), \"elapsed_sec\": round(time.time()-start, 2), **fin(overall)}\n    breakdowns = None\n    if collect_breakdowns:\n        breakdowns = {\"subset_df\": pd.DataFrame([{\"subset\": k, **fin(v)} for k, v in sorted(per_subset.items())]),\n                      \"h_df\": pd.DataFrame([{\"horizon_step\": i, **fin(v)} for i, v in enumerate(per_h)]),\n                      \"dim_df\": pd.DataFrame([{\"action_dim\": i, **fin(v)} for i, v in enumerate(per_dim)]),\n                      \"preview_p\": preview_p, \"preview_t\": preview_t}\n    del model\n    if torch.cuda.is_available(): torch.cuda.empty_cache()\n    return row, breakdowns\n\ndef evaluate_nvidia_zeroshot():\n    row = {\"model\":\"NVIDIA_GR00T_zero_shot\", \"status\":\"skipped\", \"input\":\"state+video_language\", \"stage\":\"pretrained_zero_shot\", \"S\":STATE_HORIZON, \"H\":ACTION_HORIZON, \"K\":0,\n           \"samples\":0, \"values\":0, \"raw_mse\":np.nan, \"raw_mae\":np.nan, \"raw_rmse\":np.nan, \"masked_raw_mse\":np.nan,\n           \"masked_raw_mae\":np.nan, \"masked_raw_rmse\":np.nan, \"flow_matching_test_loss_normalized\":np.nan,\n           \"zeroshot_model_id\": NVIDIA_MODEL_ID}\n    if not RUN_NVIDIA_ZEROSHOT:\n        row[\"skip_reason\"] = \"RUN_NVIDIA_ZEROSHOT_FALSE\"; return row\n    if AutoModel is None or AutoProcessor is None:\n        row[\"skip_reason\"] = \"transformers AutoModel/AutoProcessor unavailable\"; return row\n    try:\n        _processor = AutoProcessor.from_pretrained(NVIDIA_MODEL_ID, trust_remote_code=True)\n        _model = AutoModel.from_pretrained(NVIDIA_MODEL_ID, trust_remote_code=True, low_cpu_mem_usage=True,\n                                           torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)\n        _model.eval().to(device)\n        supported = any(hasattr(_model, name) for name in [\"predict_action\", \"get_action\", \"act\"])\n        if not supported:\n            row[\"skip_reason\"] = \"model loaded but no compatible action prediction API was found\"\n            return row\n        row[\"skip_reason\"] = \"model loaded, but this notebook intentionally skips unknown official action API adapter for safety\"\n        return row\n    except Exception as exc:\n        row[\"skip_reason\"] = repr(exc)\n        return row\n\nstart = time.time()\nrows = evaluate_simple_baselines()\nzeroshot_row = evaluate_nvidia_zeroshot(); rows.append(zeroshot_row)\nbreakdowns = None\nfor ck in CHECKPOINTS:\n    collect = ck[\"stage\"] == BEST_STAGE\n    row, bd = evaluate_official_checkpoint(ck[\"model\"], ck[\"path\"], ck[\"stage\"], collect_breakdowns=collect)\n    rows.append(row)\n    if bd is not None: breakdowns = bd\ncompare_df = pd.DataFrame(rows)\nif breakdowns is None:\n    subset_df = pd.DataFrame(); h_df = pd.DataFrame(); dim_df = pd.DataFrame(); preview_p = []; preview_t = []\nelse:\n    subset_df = breakdowns[\"subset_df\"]; h_df = breakdowns[\"h_df\"]; dim_df = breakdowns[\"dim_df\"]; preview_p = breakdowns[\"preview_p\"]; preview_t = breakdowns[\"preview_t\"]\ndisplay(compare_df); display(subset_df); display(h_df.head() if len(h_df) else h_df)\n"}, {"cell_type": "code", "execution_count": null, "id": "04-evaluate-official-mini-denormalized-08", "metadata": {}, "outputs": [], "source": "compare_df.to_csv(OUTPUT_ROOT/\"eval_metrics.csv\", index=False)\ncompare_df.to_csv(OUTPUT_ROOT/\"baseline_vs_official_mini.csv\", index=False)\nsubset_df.to_csv(OUTPUT_ROOT/\"per_subset_metrics.csv\", index=False); h_df.to_csv(OUTPUT_ROOT/\"per_horizon_metrics.csv\", index=False); dim_df.to_csv(OUTPUT_ROOT/\"per_action_dim_metrics.csv\", index=False)\nif preview_t and preview_p:\n    plt.figure(figsize=(5,5)); plt.scatter(preview_t, preview_p, s=7, alpha=.35); mn=min(preview_t+preview_p); mx=max(preview_t+preview_p); plt.plot([mn,mx],[mn,mx],color=\"black\"); plt.xlabel(\"GT raw action dim0\"); plt.ylabel(\"Pred raw action dim0\"); plt.tight_layout(); plt.savefig(OUTPUT_ROOT/\"prediction_vs_ground_truth_raw.png\", dpi=160); plt.show()\nif len(h_df):\n    plt.figure(figsize=(7,4)); plt.plot(h_df.horizon_step, h_df.raw_mse, marker=\"o\"); plt.xlabel(\"horizon step\"); plt.ylabel(\"raw MSE\"); plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig(OUTPUT_ROOT/\"horizon_error_curve_raw.png\", dpi=160); plt.show()\nsummary = {\"status\":\"completed\", \"eval_samples\": int(compare_df.samples.fillna(0).max()) if \"samples\" in compare_df else 0,\n           \"comparison_table\": compare_df.where(pd.notnull(compare_df), None).to_dict(\"records\"),\n           \"denormalized_raw_action_metrics\": True, \"uses_action_mask\": True,\n           \"zeroshot_attempted\": bool(RUN_NVIDIA_ZEROSHOT), \"zeroshot_status\": str(zeroshot_row.get(\"status\")),\n           \"zeroshot_model_id\": NVIDIA_MODEL_ID, \"zeroshot_skip_reason\": zeroshot_row.get(\"skip_reason\"),\n           \"pretrain_eval_completed\": bool(((compare_df.model == \"OfficialMiniVLDiT_pretrain\") & (compare_df.status == \"completed\")).any()),\n           \"posttrain_eval_completed\": bool(((compare_df.model == \"OfficialMiniVLDiT_posttrain\") & (compare_df.status == \"completed\")).any()),\n           \"elapsed_sec\": round(time.time()-start, 2)}\n(OUTPUT_ROOT/\"eval_summary.json\").write_text(json.dumps(summary, indent=2), encoding=\"utf-8\")\nprint(json.dumps(summary, indent=2))\n"}]')

NOTEBOOK_05 = json.loads('[{"cell_type": "markdown", "id": "05-strong-ablation-official-mini-01", "metadata": {}, "source": "# 05 - Strong Ablation Official Mini\n\nRequired: component, model, K inference, LR, and H=8 vs H=16 if both prepared roots are available.\n\nThis version also adds a lightweight checkpoint-stage analysis: pretrain-only vs posttrain-task-specific, if Notebook 03 produced both checkpoints.\n", "outputs": [], "execution_count": null}, {"cell_type": "code", "execution_count": null, "id": "05-strong-ablation-official-mini-02", "metadata": {}, "outputs": [], "source": "!pip install -q pandas pyarrow numpy tqdm matplotlib"}, {"cell_type": "code", "execution_count": null, "id": "05-strong-ablation-official-mini-03", "metadata": {}, "outputs": [], "source": "from pathlib import Path\nimport json, math, time, random\nimport numpy as np, pandas as pd\nfrom tqdm.auto import tqdm\nimport matplotlib.pyplot as plt\nimport torch\nfrom torch import nn\nfrom torch.utils.data import Dataset, DataLoader\nfrom torch.distributions import Beta\n\ndevice = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\nSEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)\nSMOKE_TEST=False; BATCH_SIZE=256; EVAL_BATCH_SIZE=512; EPOCHS=1; NUM_WORKERS=2; USE_AMP=True\nMAX_TRAIN_SAMPLES=2048 if SMOKE_TEST else None; MAX_EVAL_SAMPLES=1024 if SMOKE_TEST else None; MAX_STEPS_PER_EPOCH=20 if SMOKE_TEST else None\nOUTPUT_ROOT=Path(\"/kaggle/working/gr00t_official_ablation\"); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)\nRUNS=[\n {\"run\":\"state_vl_layers8_lr1e4\",\"type\":\"component/model/hyperparam\",\"use_vl\":True,\"layers\":8,\"hidden\":512,\"lr\":1e-4,\"H\":16},\n {\"run\":\"state_only_layers8_lr1e4\",\"type\":\"component\",\"use_vl\":False,\"layers\":8,\"hidden\":512,\"lr\":1e-4,\"H\":16},\n {\"run\":\"state_vl_layers4_lr1e4\",\"type\":\"model\",\"use_vl\":True,\"layers\":4,\"hidden\":512,\"lr\":1e-4,\"H\":16},\n {\"run\":\"state_vl_layers8_lr5e5\",\"type\":\"hyperparam\",\"use_vl\":True,\"layers\":8,\"hidden\":512,\"lr\":5e-5,\"H\":16},\n {\"run\":\"state_vl_H8_lr1e4\",\"type\":\"horizon\",\"use_vl\":True,\"layers\":8,\"hidden\":512,\"lr\":1e-4,\"H\":8},\n]\nK_VALUES=[1,4,8,16]"}, {"cell_type": "code", "execution_count": null, "id": "05-strong-ablation-official-mini-04", "metadata": {}, "outputs": [], "source": "class TimeEmbedding(nn.Module):\n    def __init__(self, hidden_dim, buckets=1000):\n        super().__init__()\n        self.buckets = buckets\n        self.embed = nn.Embedding(buckets, hidden_dim)\n        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))\n    def forward(self, t):\n        idx = torch.clamp((t * self.buckets).long(), 0, self.buckets - 1)\n        return self.mlp(self.embed(idx))\n\nclass CrossBlock(nn.Module):\n    def __init__(self, hidden_dim, heads, dropout):\n        super().__init__()\n        self.n1 = nn.LayerNorm(hidden_dim)\n        self.sa = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)\n        self.n2 = nn.LayerNorm(hidden_dim)\n        self.ca = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)\n        self.n3 = nn.LayerNorm(hidden_dim)\n        self.ff = nn.Sequential(\n            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),\n            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout)\n        )\n    def forward(self, x, mem):\n        y = self.n1(x)\n        x = x + self.sa(y, y, y, need_weights=False)[0]\n        y = self.n2(x)\n        x = x + self.ca(y, mem, mem, need_weights=False)[0]\n        return x + self.ff(self.n3(x))\n\nclass OfficialMiniVLDiT(nn.Module):\n    def __init__(self, state_dim, action_dim, state_horizon, action_horizon, vl_dim,\n                 hidden_dim=512, layers=8, heads=8, dropout=0.1, buckets=1000):\n        super().__init__()\n        self.state_horizon = state_horizon\n        self.action_horizon = action_horizon\n        self.state_proj = nn.Linear(state_dim, hidden_dim)\n        self.action_proj = nn.Linear(action_dim, hidden_dim)\n        self.vl_proj = nn.Linear(vl_dim, hidden_dim)\n        self.time = TimeEmbedding(hidden_dim, buckets)\n        self.state_pos = nn.Embedding(state_horizon, hidden_dim)\n        self.action_pos = nn.Embedding(action_horizon, hidden_dim)\n        self.blocks = nn.ModuleList([CrossBlock(hidden_dim, heads, dropout) for _ in range(layers)])\n        self.norm = nn.LayerNorm(hidden_dim)\n        self.out = nn.Linear(hidden_dim, action_dim)\n    def forward(self, noisy_action, state_history, vl_features, t):\n        s_pos = self.state_pos(torch.arange(self.state_horizon, device=noisy_action.device))[None]\n        a_pos = self.action_pos(torch.arange(self.action_horizon, device=noisy_action.device))[None]\n        s = self.state_proj(state_history) + s_pos\n        a = self.action_proj(noisy_action) + a_pos\n        x = torch.cat([s, a], dim=1) + self.time(t)[:, None, :]\n        mem = self.vl_proj(vl_features)\n        for block in self.blocks:\n            x = block(x, mem)\n        return self.out(self.norm(x[:, self.state_horizon:]))"}, {"cell_type": "code", "execution_count": null, "id": "05-strong-ablation-official-mini-05", "metadata": {}, "outputs": [], "source": "def find_dir(name, required):\n    cand=[]\n    for base in [Path(\"/kaggle/input\"), Path(\"/kaggle/working\")]:\n        if base.exists(): cand.extend(base.rglob(name))\n    for c in sorted(cand, key=str):\n        if c.is_dir() and all((c/r).exists() for r in required): return c\n    return None\ndef root(H): return find_dir(f\"gr00t_prepared_official_H{H}\", [\"states_train.npy\",\"actions_train_chunk.npy\",\"action_mask_train.npy\",\"normalization_stats.json\"])\ndef feature_root(H): return find_dir(f\"official_mini_vldit_H{H}\", [\"config.json\"])\nprint(\"roots:\", {h: root(h) for h in [8,16]})"}, {"cell_type": "code", "execution_count": null, "id": "05-strong-ablation-official-mini-06", "metadata": {}, "outputs": [], "source": "class DS(Dataset):\n    def __init__(self, prepared, fr, split, H, use_vl, max_samples):\n        self.H=H; self.use_vl=use_vl; self.prepared=prepared\n        self.states=np.load(prepared/f\"states_{split}.npy\", mmap_mode=\"r\"); self.actions=np.load(prepared/f\"actions_{split}_chunk.npy\", mmap_mode=\"r\"); self.masks=np.load(prepared/f\"action_mask_{split}.npy\", mmap_mode=\"r\")\n        self.stats=json.loads((prepared/\"normalization_stats.json\").read_text()); self.sm=np.asarray(self.stats[\"state_mean\"],np.float32); self.ss=np.asarray(self.stats[\"state_std\"],np.float32); self.am=np.asarray(self.stats[\"action_mean\"],np.float32); self.asd=np.asarray(self.stats[\"action_std\"],np.float32)\n        if use_vl:\n            if fr is None or not (fr.parent/f\"vl_features_{split}.npy\").exists(): raise FileNotFoundError(f\"Missing VL features for H={H}. Run Notebook 03 for this H.\")\n            self.features=np.load(fr.parent/f\"vl_features_{split}.npy\",mmap_mode=\"r\"); self.index=pd.read_parquet(fr.parent/f\"vl_feature_index_{split}.parquet\").sort_values(\"feature_index\").reset_index(drop=True); self.vl_dim=int(self.features.shape[-1])\n        else:\n            samples=pd.read_parquet(prepared/f\"{split}_samples.parquet\"); self.index=pd.DataFrame({\"sample_id\":samples.sample_id.values}); self.vl_dim=1\n        if max_samples is not None and len(self.index)>max_samples: self.index=self.index.sample(n=int(max_samples),random_state=SEED).reset_index(drop=True)\n    def __len__(self): return len(self.index)\n    def __getitem__(self,i):\n        r=self.index.iloc[i]; sid=int(r.sample_id); state=(np.asarray(self.states[sid],np.float32)-self.sm)/self.ss; action=(np.asarray(self.actions[sid],np.float32)-self.am)/self.asd; mask=np.asarray(self.masks[sid],np.float32)\n        vl=np.asarray(self.features[int(r.feature_index)],np.float32) if self.use_vl else np.zeros((1,1),np.float32)\n        return {\"state\":torch.from_numpy(state),\"action\":torch.from_numpy(action),\"mask\":torch.from_numpy(mask),\"vl\":torch.from_numpy(vl)}\ndef collate(batch): return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}"}, {"cell_type": "code", "execution_count": null, "id": "05-strong-ablation-official-mini-07", "metadata": {}, "outputs": [], "source": "def train_eval(run):\n    H=run[\"H\"]; pr=root(H); fr=feature_root(H)\n    if pr is None: return {\"run\":run[\"run\"],\"status\":\"skipped\",\"reason\":f\"missing prepared H{H}\"}\n    try:\n        tr=DS(pr,fr,\"train\",H,run[\"use_vl\"],MAX_TRAIN_SAMPLES); ev=DS(pr,fr,\"test\",H,run[\"use_vl\"],MAX_EVAL_SAMPLES)\n    except Exception as exc:\n        return {\"run\":run[\"run\"],\"status\":\"skipped\",\"reason\":str(exc)}\n    tl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available(),drop_last=True,collate_fn=collate); el=DataLoader(ev,batch_size=EVAL_BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available(),collate_fn=collate)\n    # Moi ablation run khoi tao model moi tu dau de so sanh cong bang.\n    model=OfficialMiniVLDiT(44,44,1,H,tr.vl_dim,run[\"hidden\"],run[\"layers\"],8,.1).to(device); opt=torch.optim.AdamW(model.parameters(),lr=run[\"lr\"],weight_decay=1e-5,betas=(.95,.999)); scaler=torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available()); beta=Beta(torch.tensor(1.5,device=device),torch.tensor(1.0,device=device))\n    losses=[]; start=time.time(); steps=0; model.train(); max_steps=len(tl) if MAX_STEPS_PER_EPOCH is None else min(MAX_STEPS_PER_EPOCH,len(tl))\n    for i,b in enumerate(tqdm(tl,total=max_steps,desc=run[\"run\"])):\n        if i>=max_steps: break\n        state=b[\"state\"].to(device); action=b[\"action\"].to(device); mask=b[\"mask\"].to(device); vl=b[\"vl\"].to(device); t=beta.sample((action.size(0),)).to(device=device,dtype=action.dtype); noise=torch.randn_like(action); noisy=(1-t[:,None,None])*noise+t[:,None,None]*action; target=action-noise\n        opt.zero_grad(set_to_none=True)\n        with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):\n            pred=model(noisy,state,vl,t); loss=((pred-target).pow(2)*mask).sum()/(mask.sum()+1e-6)\n        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu())); steps+=1\n    def sample(state,vl,K):\n        x=torch.randn(state.size(0),H,44,device=state.device,dtype=state.dtype); dt=1.0/K\n        for k in range(K): x=x+dt*model(x,state,vl,torch.full((state.size(0),),k/K,device=state.device,dtype=state.dtype))\n        return x\n    am=torch.tensor(ev.am,device=device)[None,None,:]; ast=torch.tensor(ev.asd,device=device)[None,None,:]; k_rows=[]; model.eval()\n    with torch.no_grad():\n        # K-step Euler inference: doi K nhung giu nguyen checkpoint de do trade-off toc do/chat luong.\n        for K in K_VALUES:\n            sq=ab=count=samples=0\n            for b in tqdm(el,desc=f\"eval {run[\'run\']} K={K}\"):\n                state=b[\"state\"].to(device); action=b[\"action\"].to(device); mask=b[\"mask\"].to(device); vl=b[\"vl\"].to(device)\n                pred=sample(state,vl,K); diff=(pred.float()*ast+am)-(action.float()*ast+am)\n                sq+=float((diff.square()*mask).sum().cpu()); ab+=float((diff.abs()*mask).sum().cpu()); count+=int(mask.sum().cpu()); samples+=int(action.size(0))\n            mse=sq/max(count,1); k_rows.append({\"K\":K,\"raw_mse\":mse,\"raw_mae\":ab/max(count,1),\"raw_rmse\":math.sqrt(mse),\"eval_samples\":samples})\n    best=next(x for x in k_rows if x[\"K\"]==4)\n    return {\"run\":run[\"run\"],\"status\":\"completed\",\"type\":run[\"type\"],\"use_vl\":run[\"use_vl\"],\"H\":H,\"layers\":run[\"layers\"],\"hidden_dim\":run[\"hidden\"],\"lr\":run[\"lr\"],\"global_steps\":steps,\"train_last_loss\":losses[-1] if losses else None,\"train_elapsed_sec\":round(time.time()-start,2),**best,\"k_results\":k_rows}\n\nresults=[]\nfor run in RUNS:\n    res=train_eval(run); print(res); results.append(res)\n    if torch.cuda.is_available(): torch.cuda.empty_cache()"}, {"cell_type": "code", "execution_count": null, "id": "05-strong-ablation-official-mini-08", "metadata": {}, "outputs": [], "source": "def eval_existing_checkpoint_stage(label, ckpt_name):\n    H = 16; pr = root(H); fr = feature_root(H)\n    if pr is None or fr is None:\n        return {\"run\": label, \"status\": \"skipped\", \"type\": \"checkpoint_stage\", \"reason\": \"missing prepared root or feature root\"}\n    ckpt_path = fr / ckpt_name\n    if not ckpt_path.exists():\n        return {\"run\": label, \"status\": \"skipped\", \"type\": \"checkpoint_stage\", \"reason\": f\"missing {ckpt_name}\"}\n    try:\n        ev = DS(pr, fr, \"test\", H, True, MAX_EVAL_SAMPLES)\n        el = DataLoader(ev, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,\n                        pin_memory=torch.cuda.is_available(), collate_fn=collate)\n        ckpt = torch.load(ckpt_path, map_location=device); cfg = ckpt[\"config\"]\n        model = OfficialMiniVLDiT(44,44,cfg[\"state_horizon\"],cfg[\"action_horizon\"],cfg[\"vl_feature_dim\"],cfg[\"hidden_dim\"],cfg[\"num_layers\"],cfg[\"num_heads\"],cfg[\"dropout\"]).to(device)\n        model.load_state_dict(ckpt[\"model_state_dict\"]); model.eval()\n        am=torch.tensor(ev.am,device=device)[None,None,:]; ast=torch.tensor(ev.asd,device=device)[None,None,:]\n        def sample_stage(state,vl,K=4):\n            x=torch.randn(state.size(0),H,44,device=state.device,dtype=state.dtype); dt=1.0/K\n            for k in range(K): x=x+dt*model(x,state,vl,torch.full((state.size(0),),k/K,device=state.device,dtype=state.dtype))\n            return x\n        sq=ab=count=samples=0\n        with torch.no_grad():\n            for b in tqdm(el,desc=f\"eval {label}\"):\n                state=b[\"state\"].to(device); action=b[\"action\"].to(device); mask=b[\"mask\"].to(device); vl=b[\"vl\"].to(device)\n                pred=sample_stage(state,vl,4); diff=(pred.float()*ast+am)-(action.float()*ast+am)\n                sq+=float((diff.square()*mask).sum().cpu()); ab+=float((diff.abs()*mask).sum().cpu()); count+=int(mask.sum().cpu()); samples+=int(action.size(0))\n        mse=sq/max(count,1)\n        return {\"run\": label, \"status\": \"completed\", \"type\": \"checkpoint_stage\", \"use_vl\": True, \"H\": H,\n                \"layers\": cfg.get(\"num_layers\"), \"hidden_dim\": cfg.get(\"hidden_dim\"), \"lr\": cfg.get(\"learning_rate\"),\n                \"global_steps\": int(ckpt.get(\"global_step\", 0)), \"raw_mse\": mse, \"raw_mae\": ab/max(count,1),\n                \"raw_rmse\": math.sqrt(mse), \"eval_samples\": samples, \"K\": 4, \"checkpoint\": ckpt_name}\n    except Exception as exc:\n        return {\"run\": label, \"status\": \"skipped\", \"type\": \"checkpoint_stage\", \"reason\": repr(exc)}\n    finally:\n        if torch.cuda.is_available(): torch.cuda.empty_cache()\n\nstage_results = [\n    eval_existing_checkpoint_stage(\"pretrain_only\", \"pretrain_checkpoint.pt\"),\n    eval_existing_checkpoint_stage(\"posttrain_task_specific\", \"posttrain_checkpoint.pt\"),\n]\nstage_df = pd.DataFrame(stage_results)\nstage_df.to_csv(OUTPUT_ROOT/\"pretrain_posttrain_comparison.csv\", index=False)\n\nsummary_df=pd.DataFrame([{k:v for k,v in r.items() if k!=\"k_results\"} for r in results])\nsummary_df = pd.concat([summary_df, stage_df], ignore_index=True, sort=False)\nsummary_df.to_csv(OUTPUT_ROOT/\"ablation_summary.csv\",index=False)\nk_rows=[]\nfor r in results:\n    if r.get(\"status\")==\"completed\":\n        for kr in r[\"k_results\"]: k_rows.append({\"run\":r[\"run\"],**kr})\npd.DataFrame(k_rows).to_csv(OUTPUT_ROOT/\"k_step_tradeoff.csv\",index=False)\ndisplay(summary_df); display(stage_df)\ncompleted=summary_df[summary_df.status==\"completed\"] if \"status\" in summary_df else pd.DataFrame()\nplot_df=completed[completed.type != \"checkpoint_stage\"] if len(completed) and \"type\" in completed else completed\nif len(plot_df):\n    plt.figure(figsize=(9,4)); plt.bar(plot_df.run, plot_df.raw_mse); plt.xticks(rotation=30, ha=\"right\"); plt.ylabel(\"Raw MSE @K=4\"); plt.tight_layout(); plt.savefig(OUTPUT_ROOT/\"ablation_bar_chart.png\",dpi=160); plt.show()\nsummary={\"status\":\"completed\",\"num_runs\":len(results),\"num_completed\":int(len(completed)),\n         \"required_ablation_types\":[\"component\",\"model\",\"inference_K\",\"hyperparam\",\"horizon_H8_vs_H16\"],\n         \"additional_analysis\":[\"pretrain_only_vs_posttrain_task_specific\"],\n         \"metrics_space\":\"denormalized_raw_action_space\",\"uses_action_mask\":True,\n         \"results\":results, \"pretrain_posttrain_comparison\":stage_results}\n(OUTPUT_ROOT/\"ablation_summary.json\").write_text(json.dumps(summary,indent=2),encoding=\"utf-8\")\nprint(json.dumps({\"num_completed\":len(completed),\"output_root\":str(OUTPUT_ROOT),\"stage_analysis\":stage_results}, indent=2))\n"}]')

write("02_prepare/02_prepare_official_sequence_data.ipynb", NOTEBOOK_02)
write("03_train/03_encode_vlm_and_train_official_mini_vldit.ipynb", NOTEBOOK_03)
write("04_eval/04_evaluate_official_mini_denormalized.ipynb", NOTEBOOK_04)
write("05_ablation/05_strong_ablation_official_mini.ipynb", NOTEBOOK_05)
print("Generated notebooks:")
for p in sorted(OUT.rglob("*.ipynb")):
    print("-", p.name, p.stat().st_size)

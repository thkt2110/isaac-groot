import json
import re
from pathlib import Path
from textwrap import dedent


OUT = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(True),
    }


def write(name, cells):
    stem = Path(name).stem.replace("_", "-")
    for i, cell in enumerate(cells, 1):
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


DOWNLOAD_TARGETS = [
    # Already downloaded before, listed here for context only. The generated notebooks below focus on new data.
    # ("gr1_arms_only.CanSort", True, "already_done"),
    # ("gr1_arms_waist.CanToDrawer", True, "already_done"),
    # ("gr1_arms_waist.CupToDrawer", True, "already_done"),

    # P1: diverse GR-1 arms+waist tasks. These are the safest for current Notebook 02/03 because
    # the official-mini pipeline assumes state/action dim = 44.
    ("gr1_arms_waist.PlaceMilkToMicrowave", True, "p1_gr1_safe"),
    ("gr1_arms_waist.PotatoToMicrowave", True, "p1_gr1_safe"),
    ("gr1_arms_waist.PlateToBowl", True, "p1_gr1_safe"),
    ("gr1_arms_waist.PlateToPlate", True, "p1_gr1_safe"),
    ("gr1_arms_waist.TrayToPlate", True, "p1_gr1_safe"),
    ("gr1_arms_waist.WineToCabinet", True, "p1_gr1_safe"),
    ("gr1_arms_waist.CuttingboardToBasket", True, "p1_gr1_safe"),
    ("gr1_arms_waist.CuttingboardToPan", True, "p1_gr1_safe"),
    ("gr1_arms_waist.PlaceBottleToCabinet", True, "p1_gr1_safe"),
    ("gr1_arms_waist.PlacematToBowl", True, "p1_gr1_safe"),
    ("gr1_arms_waist.TrayToPot", True, "p1_gr1_safe"),
    ("gr1_arms_waist.TrayToTieredShelf", True, "p1_gr1_safe"),

    # P2: more diverse GR-1 variants. Downloadable for seminar evidence; keep out of Notebook 02
    # auto-plan unless you verify state/action dim is 44.
    ("gr1_full_upper_body.Coffee", False, "p2_gr1_experimental"),
    ("gr1_full_upper_body.Pouring", False, "p2_gr1_experimental"),
    ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", False, "p2_gr1_experimental"),
    ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", False, "p2_gr1_experimental"),
    ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", False, "p2_gr1_experimental"),

    # P3: cross-embodiment evidence. Current Notebook 02/03 will not auto-train these because
    # state/action dimensions may differ and need padding/embodiment-specific masking.
    ("bimanual_panda_gripper.Threading", False, "p3_cross_embodiment"),
    ("bimanual_panda_gripper.Transport", False, "p3_cross_embodiment"),
    ("single_panda_gripper.PnPCounterToCab", False, "p3_cross_embodiment"),
    ("single_panda_gripper.OpenDrawer", False, "p3_cross_embodiment"),
    ("single_panda_gripper.CoffeeServeMug", False, "p3_cross_embodiment"),
    ("sim_behavior_r1_pro.task-0001_picking_up_trash", False, "p3_cross_embodiment"),
    ("sim_behavior_r1_pro.task-0020_sorting_vegetables", False, "p3_cross_embodiment"),
    ("unitree_g1.LMPnPAppleToPlateDC", False, "p3_cross_embodiment"),
]


def safe_filename(subset):
    return re.sub(r"[^A-Za-z0-9]+", "_", subset).strip("_")


def make_notebook(subset, pipeline_safe, priority):
    report_safe = safe_filename(subset)
    return [
        md(f"""
        # 01 - Download Full Subset: `{subset}`

        Notebook này chỉ tải **1 subset** để tránh một notebook ôm quá nhiều data và làm đầy disk Kaggle.

        - Download mode mặc định: `full_subset`, tức là tải `meta/`, `data/`, và toàn bộ `videos/`.
        - Nếu hết disk, sửa riêng notebook này sang `DOWNLOAD_MODE = "data_meta_ego_chunks"` hoặc `data_meta_only`.
        - `PIPELINE_SAFE_FOR_NOTEBOOK02 = {pipeline_safe}`.

        Gợi ý vận hành:

        1. Chạy notebook này trên Kaggle CPU.
        2. Nếu output thành công, save output thành Kaggle Dataset.
        3. Add nhiều output download notebook vào Notebook 02.
        4. Notebook 02 sẽ merge các file `selected_subsets_for_notebook02.json`.
        """),
        code("!pip install -q huggingface_hub tqdm"),
        code(f'''
        from pathlib import Path
        import json, os, shutil, time, traceback

        try:
            from kaggle_secrets import UserSecretsClient
        except Exception:
            UserSecretsClient = None

        from huggingface_hub import HfApi, snapshot_download

        REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
        SUBSET_NAME = "{subset}"
        PRIORITY_GROUP = "{priority}"
        PIPELINE_SAFE_FOR_NOTEBOOK02 = {str(pipeline_safe)}

        LOCAL_DIR = Path("/kaggle/working/gr00t_x_embodiment_sim")
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)

        # Bạn muốn tải full video trước. Nếu disk không đủ, chỉ cần sửa dòng này trong notebook subset bị lỗi.
        DOWNLOAD_MODE = "full_subset"  # "full_subset" | "data_meta_ego_chunks" | "data_meta_only"
        CAMERA_KEY = "observation.images.ego_view"
        VIDEO_CHUNKS = [0, 1, 2]
        CLEAN_HF_CACHE_AFTER_DOWNLOAD = True

        REPORT_PATH = Path("/kaggle/working/download_report_{report_safe}.json")
        SELECTED_PATH = Path("/kaggle/working/selected_subsets_for_notebook02.json")

        print("SUBSET_NAME:", SUBSET_NAME)
        print("PRIORITY_GROUP:", PRIORITY_GROUP)
        print("PIPELINE_SAFE_FOR_NOTEBOOK02:", PIPELINE_SAFE_FOR_NOTEBOOK02)
        print("DOWNLOAD_MODE:", DOWNLOAD_MODE)
        print("LOCAL_DIR:", LOCAL_DIR)
        '''),
        code(r'''
        def get_token():
            # Không hard-code HF token. Dùng Kaggle Secret HF_TOKEN hoặc env HF_TOKEN.
            token = os.environ.get("HF_TOKEN")
            if token:
                return token
            if UserSecretsClient is not None:
                try:
                    return UserSecretsClient().get_secret("HF_TOKEN")
                except Exception as exc:
                    print("WARN: cannot read Kaggle Secret HF_TOKEN:", exc)
            return None

        def disk_free_gb(path="/kaggle/working"):
            return shutil.disk_usage(path).free / (1024 ** 3)

        def folder_size_gb(path):
            path = Path(path)
            if not path.exists():
                return 0.0
            total = 0
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
            return total / (1024 ** 3)

        def count_files(path, suffix=None):
            path = Path(path)
            if not path.exists():
                return 0
            if suffix is None:
                return sum(1 for p in path.rglob("*") if p.is_file())
            return sum(1 for p in path.rglob(f"*{suffix}") if p.is_file())

        def clean_hf_cache():
            # Full subset co the tao cache lon; xoa cache sau download de output gon hon.
            for p in [LOCAL_DIR / ".cache", Path("/kaggle/working/.cache/huggingface")]:
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)

        def allow_patterns():
            if DOWNLOAD_MODE == "full_subset":
                return [f"{SUBSET_NAME}/**"]
            patterns = [f"{SUBSET_NAME}/meta/**", f"{SUBSET_NAME}/data/**"]
            if DOWNLOAD_MODE == "data_meta_ego_chunks":
                for chunk_id in VIDEO_CHUNKS:
                    patterns.append(f"{SUBSET_NAME}/videos/chunk-{chunk_id:03d}/{CAMERA_KEY}/**")
            return patterns

        token = get_token()
        api = HfApi(token=token)

        # Kiem tra subset co ton tai tren Hugging Face truoc khi tai.
        try:
            root_items = list(api.list_repo_tree(repo_id=REPO_ID, repo_type="dataset", path_in_repo="", recursive=False))
            available = {x.path for x in root_items if getattr(x, "path", None)}
            if SUBSET_NAME not in available:
                raise ValueError(f"Subset not found in HF dataset: {SUBSET_NAME}")
        except Exception as exc:
            print("WARN: repo listing failed or subset not listed. Will still try snapshot_download.")
            print(repr(exc))

        print("HF token:", "OK" if token else "NONE/PUBLIC")
        print("Free disk before:", round(disk_free_gb(), 3), "GB")
        print("Allow patterns:")
        for pat in allow_patterns():
            print(" -", pat)
        '''),
        code(r'''
        subset_path = LOCAL_DIR / SUBSET_NAME
        started = time.time()
        status = "unknown"
        err = None

        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=str(LOCAL_DIR),
                allow_patterns=allow_patterns(),
                token=token,
                resume_download=True,
                local_dir_use_symlinks=False,
            )
            status = "downloaded"
        except Exception as exc:
            status = "error"
            err = {"repr": repr(exc), "traceback": traceback.format_exc()[-4000:]}
            print("ERROR:", repr(exc))

        if CLEAN_HF_CACHE_AFTER_DOWNLOAD:
            clean_hf_cache()

        report = {
            "repo_id": REPO_ID,
            "subset_name": SUBSET_NAME,
            "priority_group": PRIORITY_GROUP,
            "pipeline_safe_for_notebook02": PIPELINE_SAFE_FOR_NOTEBOOK02,
            "download_mode": DOWNLOAD_MODE,
            "allow_patterns": allow_patterns(),
            "local_dir": str(LOCAL_DIR),
            "subset_path": str(subset_path),
            "status": status,
            "error": err,
            "elapsed_sec": round(time.time() - started, 2),
            "exists": subset_path.exists(),
            "size_gb": round(folder_size_gb(subset_path), 3),
            "has_data_dir": (subset_path / "data").exists(),
            "has_meta_dir": (subset_path / "meta").exists(),
            "has_videos_dir": (subset_path / "videos").exists(),
            "num_parquet_files": count_files(subset_path / "data", ".parquet"),
            "num_video_files": count_files(subset_path / "videos", ".mp4"),
            "free_gb_after": round(disk_free_gb(), 3),
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        '''),
        code(r'''
        downloaded_ok = (
            report["status"] == "downloaded"
            and report["has_data_dir"]
            and report["has_meta_dir"]
        )

        selected = {
            "repo_id": REPO_ID,
            "subset_name": SUBSET_NAME,
            "priority_group": PRIORITY_GROUP,
            "download_mode": DOWNLOAD_MODE,
            "pipeline_safe_for_notebook02": PIPELINE_SAFE_FOR_NOTEBOOK02,
            "downloaded_ok": downloaded_ok,
            "pipeline_safe_subsets": [SUBSET_NAME] if downloaded_ok and PIPELINE_SAFE_FOR_NOTEBOOK02 else [],
            "downloaded_subsets": [SUBSET_NAME] if downloaded_ok else [],
            "note": "Notebook 02 will merge pipeline_safe_subsets from all added download outputs.",
        }
        SELECTED_PATH.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        Path("/kaggle/working/downloaded_subsets.txt").write_text((SUBSET_NAME + "\n") if downloaded_ok else "", encoding="utf-8")

        print("Selected plan:")
        print(json.dumps(selected, indent=2))
        '''),
        md("""
        ## Nếu notebook này hết disk

        Sửa riêng notebook này:

        ```python
        DOWNLOAD_MODE = "data_meta_ego_chunks"
        VIDEO_CHUNKS = [0, 1, 2]
        ```

        Nếu vẫn đầy disk:

        ```python
        DOWNLOAD_MODE = "data_meta_only"
        ```

        Sau đó rerun notebook subset này, không cần sửa các notebook subset khác.
        """),
    ]


index_rows = []
for idx, (subset, pipeline_safe, priority) in enumerate(DOWNLOAD_TARGETS, 1):
    fname = f"01_download/01_download_full_{idx:02d}_{safe_filename(subset)}.ipynb"
    write(fname, make_notebook(subset, pipeline_safe, priority))
    index_rows.append({
        "order": idx,
        "notebook": fname,
        "subset": subset,
        "priority_group": priority,
        "pipeline_safe_for_notebook02": pipeline_safe,
        "default_download_mode": "full_subset",
    })

write(
    "01_download/01_download_full_subset_INDEX.ipynb",
    [
        md("""
        # 01 - Full Subset Download Index

        File này chỉ là index cho các notebook download full subset riêng lẻ.

        Chạy theo thứ tự:

        1. P1 `p1_gr1_safe` trước, vì dùng trực tiếp được với Notebook 02/03 hiện tại.
        2. P2 `p2_gr1_experimental` nếu còn disk/GPU time.
        3. P3 `p3_cross_embodiment` để làm bằng chứng data diversity, nhưng chưa đưa thẳng vào Notebook 02/03 hiện tại.
        """),
        code("import pandas as pd\nrows = " + repr(index_rows) + "\ndf = pd.DataFrame(rows)\ndisplay(df)\ndf.to_csv('/kaggle/working/download_full_subset_index.csv', index=False)"),
    ],
)

(OUT / "01_download").mkdir(parents=True, exist_ok=True)
(OUT / "01_download" / "download_full_subset_index.json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
print("Generated full-subset download notebooks:", len(index_rows))
for row in index_rows:
    print(row["notebook"])

"""
Local ASL pose generator — reads from data/asl-signs/ on disk.

Usage:
    cd asl-backend
    pip install pose-format numpy pandas pyarrow kaggle
    python scripts/pose_generation/generate_poses_local.py

Required files in data/asl-signs/:
    label_classes.npy              (already present)
    train_landmark_files/          (already present)
    sign_to_prediction_index_map.json  (auto-generated from label_classes.npy)
    train.csv                      (auto-downloaded via Kaggle API if missing)

Writes : ../asl-frontend/src/assets/poses/
"""

import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
BACKEND_DIR  = SCRIPT_DIR.parent.parent
FRONTEND_DIR = BACKEND_DIR.parent / 'asl-frontend'

DATASET_DIR  = BACKEND_DIR / 'data' / 'asl-signs'
OUTPUT_DIR   = BACKEND_DIR / 'data' / 'generated_poses'
POSES_DIR    = OUTPUT_DIR / 'poses'
ASSETS_DIR   = FRONTEND_DIR / 'src' / 'assets' / 'poses'

LIMIT  = None   # int → first N words only; None → all 250
RESUME = True   # skip words whose .pose already exists
# ──────────────────────────────────────────────────────────────────────────────

ASL_VOCABULARY = None  # filled from sign_to_prediction_index_map.json in main()

NUM_LANDMARKS = 543
LANDMARK_TYPES = [
    ('face',       468),
    ('left_hand',   21),
    ('pose',        33),
    ('right_hand',  21),
]


# ── Parquet parser ─────────────────────────────────────────────────────────────

def load_parquet_landmarks(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    if 'x' in df.columns and 'type' in df.columns:
        return _parse_long(df)
    return _parse_wide(df)


def _parse_long(df: pd.DataFrame):
    type_offset = {'face': 0, 'left_hand': 468, 'pose': 489, 'right_hand': 522}
    frames = sorted(df['frame'].unique())
    T = len(frames)
    frame_to_i = {f: i for i, f in enumerate(frames)}

    landmarks  = np.zeros((T, NUM_LANDMARKS, 3), dtype=np.float32)
    confidence = np.zeros((T, NUM_LANDMARKS),    dtype=np.float32)

    df = df.copy()
    df['_offset'] = df['type'].map(type_offset)
    df = df.dropna(subset=['_offset'])
    df['_gi'] = df['_offset'].astype(int) + df['landmark_index'].astype(int)
    df = df[df['_gi'] < NUM_LANDMARKS]
    df['_fi'] = df['frame'].map(frame_to_i)

    valid = df[['x', 'y', 'z']].notna().all(axis=1)
    df = df[valid]

    fi = df['_fi'].to_numpy(dtype=int)
    li = df['_gi'].to_numpy(dtype=int)
    landmarks[fi, li, 0] = df['x'].to_numpy(dtype=np.float32)
    landmarks[fi, li, 1] = df['y'].to_numpy(dtype=np.float32)
    landmarks[fi, li, 2] = df['z'].to_numpy(dtype=np.float32)
    confidence[fi, li]   = 1.0

    return landmarks, confidence, 30.0


def _parse_wide(df: pd.DataFrame):
    T = len(df)
    landmarks  = np.zeros((T, NUM_LANDMARKS, 3), dtype=np.float32)
    confidence = np.zeros((T, NUM_LANDMARKS),    dtype=np.float32)
    offset = 0
    for ltype, count in LANDMARK_TYPES:
        for li in range(count):
            cx, cy, cz = f'x_{ltype}_{li}', f'y_{ltype}_{li}', f'z_{ltype}_{li}'
            if cx not in df.columns:
                break
            xv = df[cx].to_numpy(dtype=np.float32)
            yv = df[cy].to_numpy(dtype=np.float32) if cy in df.columns else np.full(T, np.nan, np.float32)
            zv = df[cz].to_numpy(dtype=np.float32) if cz in df.columns else np.full(T, np.nan, np.float32)
            idx = offset + li
            landmarks[:, idx] = np.stack([xv, yv, zv], axis=1)
            confidence[:, idx] = (~(np.isnan(xv) | np.isnan(yv) | np.isnan(zv))).astype(np.float32)
        offset += count
    np.nan_to_num(landmarks, copy=False, nan=0.0)
    return landmarks, confidence, 30.0


def quality_score(confidence: np.ndarray) -> float:
    left  = float(confidence[:, 468:489].mean())   # left hand (21 pts)
    right = float(confidence[:, 522:543].mean())   # right hand (21 pts)
    frame_factor = min(confidence.shape[0], 60) / 60.0
    # Reward the better-detected hand; bonus for having the second hand too
    return (max(left, right) + 0.5 * min(left, right)) * frame_factor


# ── Pose writer ────────────────────────────────────────────────────────────────

def build_pose_header():
    from pose_format.pose_header import PoseHeader, PoseHeaderDimensions, PoseHeaderComponent
    dims = PoseHeaderDimensions(width=1, height=1, depth=1)
    components = [
        PoseHeaderComponent(
            name='FACE_LANDMARKS',
            points=[f'face_{i}' for i in range(468)],
            limbs=[], colors=[(120, 120, 120)], point_format='XYZC',
        ),
        PoseHeaderComponent(
            name='LEFT_HAND_LANDMARKS',
            points=[f'left_hand_{i}' for i in range(21)],
            limbs=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
                   (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                   (13,17),(17,18),(18,19),(19,20),(0,17)],
            colors=[(0, 200, 0)], point_format='XYZC',
        ),
        PoseHeaderComponent(
            name='POSE_LANDMARKS',
            points=[f'pose_{i}' for i in range(33)],
            limbs=[(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
                   (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
                   (23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),
                   (29,31),(30,32)],
            colors=[(200, 80, 0)], point_format='XYZC',
        ),
        PoseHeaderComponent(
            name='RIGHT_HAND_LANDMARKS',
            points=[f'right_hand_{i}' for i in range(21)],
            limbs=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
                   (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                   (13,17),(17,18),(18,19),(19,20),(0,17)],
            colors=[(0, 80, 200)], point_format='XYZC',
        ),
    ]
    return PoseHeader(version=0.1, dimensions=dims, components=components)


def write_pose_file(landmarks, confidence, fps, out_path: Path):
    from pose_format import Pose
    from pose_format.numpy import NumPyPoseBody
    data_4d = landmarks[:, np.newaxis, :, :]
    conf_3d = confidence[:, np.newaxis, :]
    mask = np.broadcast_to((conf_3d < 0.5)[:, :, :, np.newaxis], data_4d.shape).copy()
    masked = np.ma.MaskedArray(data_4d, mask=mask)
    body = NumPyPoseBody(fps=int(round(fps)), data=masked, confidence=conf_3d)
    pose = Pose(header=_HEADER, body=body)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pose.write(f)


# ── Setup helpers ──────────────────────────────────────────────────────────────

def ensure_sign_map():
    """Generate sign_to_prediction_index_map.json from label_classes.npy if missing."""
    map_path = DATASET_DIR / 'sign_to_prediction_index_map.json'
    if map_path.exists():
        return
    npy_path = DATASET_DIR / 'label_classes.npy'
    if not npy_path.exists():
        print(f'ERROR: missing both {map_path} and {npy_path}')
        sys.exit(1)
    labels = np.load(str(npy_path), allow_pickle=True)
    sign_map = {str(label): int(i) for i, label in enumerate(labels)}
    with open(map_path, 'w') as f:
        json.dump(sign_map, f, indent=2)
    print(f'✓ Generated {map_path.name} from label_classes.npy ({len(sign_map)} signs)')


def ensure_train_csv():
    """Download train.csv via Kaggle API if missing (small file, ~3 MB)."""
    train_csv = DATASET_DIR / 'train.csv'
    if train_csv.exists():
        return
    print('train.csv not found — downloading via Kaggle API...')
    try:
        import subprocess
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', 'asl-signs',
             '-f', 'train.csv', '-p', str(DATASET_DIR)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        # Kaggle may zip even a single file
        zip_path = DATASET_DIR / 'train.csv.zip'
        if zip_path.exists():
            import zipfile as zf
            with zf.ZipFile(zip_path) as z:
                z.extractall(DATASET_DIR)
            zip_path.unlink()
        if not train_csv.exists():
            raise RuntimeError('train.csv still not found after download')
        print('✓ train.csv downloaded')
    except FileNotFoundError:
        print('ERROR: kaggle CLI not found. Install it with: pip install kaggle')
        print('Then place kaggle.json at: ~/.kaggle/kaggle.json')
        print('Or manually download train.csv from:')
        print('  https://www.kaggle.com/competitions/asl-signs/data')
        sys.exit(1)
    except Exception as e:
        print(f'ERROR downloading train.csv: {e}')
        print('Manually download train.csv from:')
        print('  https://www.kaggle.com/competitions/asl-signs/data')
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not DATASET_DIR.exists():
        print(f'ERROR: Dataset folder not found:\n  {DATASET_DIR}')
        sys.exit(1)

    ensure_sign_map()
    ensure_train_csv()

    global _HEADER
    _HEADER = build_pose_header()

    train_df = pd.read_csv(DATASET_DIR / 'train.csv')
    with open(DATASET_DIR / 'sign_to_prediction_index_map.json') as f:
        sign_to_idx = json.load(f)

    global ASL_VOCABULARY
    ASL_VOCABULARY = sorted(sign_to_idx.keys())

    print(f'Vocabulary: {len(ASL_VOCABULARY)} words (all Kaggle signs)')
    print()

    words_to_process = ASL_VOCABULARY[:LIMIT] if LIMIT else ASL_VOCABULARY
    total = len(words_to_process)

    generated, skipped, failed = [], [], []
    t_start = time.time()

    for word_idx, word in enumerate(words_to_process, 1):
        out_path = POSES_DIR / f'{word}.pose'

        if RESUME and out_path.exists():
            skipped.append(word)
            print(f'[{word_idx:3d}/{total}] SKIP  {word}')
            continue

        try:
            if 'sign' in train_df.columns:
                samples = train_df[train_df['sign'] == word]
            elif 'label' in train_df.columns:
                sign_idx = sign_to_idx.get(word)
                samples = train_df[train_df['label'] == sign_idx] if sign_idx is not None else train_df.iloc[0:0]
            else:
                raise ValueError('train.csv has neither "sign" nor "label" column')
            if len(samples) == 0:
                raise ValueError('no samples in train.csv')

            best_lm, best_conf, best_fps, best_score = None, None, 30.0, -1.0

            for _, row in samples.head(20).iterrows():
                parquet_path = DATASET_DIR / row.get('path', '')
                if not parquet_path.exists():
                    continue
                try:
                    lm, conf, fps = load_parquet_landmarks(parquet_path)
                    score = quality_score(conf)
                    if score > best_score:
                        best_lm, best_conf, best_fps, best_score = lm, conf, fps, score
                except Exception:
                    pass

            if best_lm is None:
                raise ValueError('no readable parquet files')

            write_pose_file(best_lm, best_conf, best_fps, out_path)
            size_kb = out_path.stat().st_size / 1024
            generated.append(word)
            print(f'[{word_idx:3d}/{total}] OK    {word:<20s}  '
                  f'frames={best_lm.shape[0]:3d}  score={best_score:.2f}  {size_kb:.1f} KB')

        except Exception as e:
            failed.append((word, str(e)))
            print(f'[{word_idx:3d}/{total}] FAIL  {word:<20s}  {e}')

    elapsed = time.time() - t_start

    # Manifest
    available_words = sorted(p.stem for p in POSES_DIR.glob('*.pose'))
    missing = [w for w in ASL_VOCABULARY if w not in set(available_words)]

    manifest = {
        'total': len(ASL_VOCABULARY),
        'available': len(available_words),
        'words': available_words,
        'missing': missing,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    # Copy to frontend assets
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    for pose_file in POSES_DIR.glob('*.pose'):
        shutil.copy2(pose_file, ASSETS_DIR / pose_file.name)
    shutil.copy2(OUTPUT_DIR / 'manifest.json', ASSETS_DIR / 'manifest.json')

    print()
    print('=' * 60)
    print(f'  Generated : {len(generated)}')
    print(f'  Skipped   : {len(skipped)}')
    print(f'  Failed    : {len(failed)}')
    print(f'  Coverage  : {len(available_words)}/{len(ASL_VOCABULARY)}')
    print(f'  Elapsed   : {elapsed:.1f}s')
    print(f'  Copied to : {ASSETS_DIR}')
    print('=' * 60)

    if failed:
        print('\nFailed words:')
        for w, r in failed:
            print(f'  {w}: {r}')


if __name__ == '__main__':
    main()
